"""Check the FK derivatives that drive TCP-loss backpropagation."""

import unittest

import numpy as np

from scripts.tcp_loss import tcp_and_jacobian, tcp_loss_and_gradient
from scripts.train_mlp import FORWARD_KINEMATICS, JOINT_LIMITS, MLP, denormalise


class TCPLossTests(unittest.TestCase):
    def test_batched_fk_matches_existing_robot_equations(self):
        rng = np.random.default_rng(42)
        for robot, fk in FORWARD_KINEMATICS.items():
            with self.subTest(robot=robot):
                normalised = np.vstack((rng.uniform(-1, 1, (50, 3)), [-1] * 3, [1] * 3))
                joints = denormalise(normalised, JOINT_LIMITS[robot])
                positions, _ = tcp_and_jacobian(robot, joints)
                np.testing.assert_allclose(positions, [fk(row) for row in joints], atol=1e-14)

    def test_physical_jacobians_match_finite_differences(self):
        rng = np.random.default_rng(7)
        delta = 1e-6
        for robot, fk in FORWARD_KINEMATICS.items():
            with self.subTest(robot=robot):
                joints = denormalise(rng.uniform(-0.9, 0.9, (10, 3)), JOINT_LIMITS[robot])
                _, analytical = tcp_and_jacobian(robot, joints)
                numerical = np.zeros_like(analytical)
                for i, row in enumerate(joints):
                    for joint in range(3):
                        step = np.eye(3)[joint] * delta
                        numerical[i, :, joint] = (np.array(fk(row + step)) - fk(row - step)) / (2 * delta)
                np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-9)

    def test_full_loss_gradient_including_scaling_clipping_and_limits(self):
        # Mixed robots, in-range predictions and predictions clipped at either bound.
        prediction = np.array([[0.2, -0.4, 0.5], [-0.3, 0.4, -0.2],
                               [1.2, 0.3, -1.4], [-1.3, 1.2, 0.4]])
        inputs = np.array([[1, 0, 0.3, 0.2, 0.1], [0, 1, 0.1, 0.1, 0.3],
                           [1, 0, 0.5, 0.0, 0.15], [0, 1, -0.1, 0.2, 0.4]])
        for indices in ([0, 2], [1, 3], [0, 1, 2, 3]):
            with self.subTest(indices=indices):
                q, x = prediction[indices], inputs[indices]
                _, analytical = tcp_loss_and_gradient(q, x, JOINT_LIMITS)
                numerical = np.zeros_like(q)
                for index in np.ndindex(q.shape):
                    plus, minus = q.copy(), q.copy()
                    plus[index] += 1e-6
                    minus[index] -= 1e-6
                    numerical[index] = (
                        tcp_loss_and_gradient(plus, x, JOINT_LIMITS)[0]
                        - tcp_loss_and_gradient(minus, x, JOINT_LIMITS)[0]
                    ) / 2e-6
                np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-9)

    def test_exact_endpoint_has_zero_loss_without_joint_labels(self):
        prediction = np.array([[0.2, -0.3, 0.4], [-0.4, 0.1, 0.2]])
        inputs = np.zeros((2, 5))
        for i, robot in enumerate(("scara", "ned")):
            inputs[i, i] = 1
            inputs[i, 2:] = FORWARD_KINEMATICS[robot](denormalise(prediction[i], JOINT_LIMITS[robot]))
        loss, gradient = tcp_loss_and_gradient(prediction, inputs, JOINT_LIMITS)
        self.assertLess(loss, 1e-28)
        np.testing.assert_allclose(gradient, 0, atol=1e-14)

    def test_tcp_loss_backpropagates_through_mlp_weights(self):
        inputs = np.array([[1, 0, 0.4, 0.1, 0.1], [0, 1, -0.1, 0.2, 0.3]])
        model = MLP((5, 4, 3), np.random.default_rng(8))

        def objective(prediction):
            return tcp_loss_and_gradient(prediction, inputs, JOINT_LIMITS)

        # lr=0 records the gradients in Adam's first moment without moving weights.
        model.train_batch(inputs, None, lr=0, loss_fn=objective)
        for key, values in model.params.items():
            with self.subTest(parameter=key):
                numerical = np.zeros_like(values)
                for index in np.ndindex(values.shape):
                    original = values[index]
                    values[index] = original + 1e-6
                    plus = objective(model.predict(inputs))[0]
                    values[index] = original - 1e-6
                    minus = objective(model.predict(inputs))[0]
                    values[index] = original
                    numerical[index] = (plus - minus) / 2e-6
                np.testing.assert_allclose(model.m[key] / 0.1, numerical, rtol=1e-5, atol=1e-9)

    def test_fallback_targets_replace_unreachable_request_in_loss_and_gradient(self):
        q = np.array([[0.2, -0.3, 0.4], [-0.4, 0.1, 0.2]])
        inputs = np.array([[1, 0, 2.0, 2.0, 2.0], [0, 1, -2.0, 2.0, 2.0]])
        targets = np.array([FORWARD_KINEMATICS[robot](denormalise(row, JOINT_LIMITS[robot]))
                            for robot, row in zip(("scara", "ned"), q)])
        loss, gradient = tcp_loss_and_gradient(q, inputs, JOINT_LIMITS, target_xyz=targets)
        self.assertLess(loss, 1e-28)
        np.testing.assert_allclose(gradient, 0, atol=1e-14)
        self.assertGreater(tcp_loss_and_gradient(q, inputs, JOINT_LIMITS)[0], 1)
        targets += np.array([.01, -.02, .03])
        _, analytical = tcp_loss_and_gradient(q, inputs, JOINT_LIMITS, target_xyz=targets)
        numerical = np.zeros_like(q)
        for index in np.ndindex(q.shape):
            plus, minus = q.copy(), q.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numerical[index] = (tcp_loss_and_gradient(plus, inputs, JOINT_LIMITS, target_xyz=targets)[0]
                                - tcp_loss_and_gradient(minus, inputs, JOINT_LIMITS, target_xyz=targets)[0]) / 2e-6
        np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
