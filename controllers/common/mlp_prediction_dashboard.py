"""Live Matplotlib explanation view used by the Webots MLP diagnostic controller."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from scripts.forward_kinematics_ned import (
    SHOULDER_HEIGHT,
    UPPER_ARM_REACH,
    UPPER_ARM_RISE,
    forward_kinematics_ned,
)
from scripts.forward_kinematics_scara import (
    BASE_TO_JOINT1_X,
    BASE_TO_JOINT1_Z,
    JOINT1_TO_JOINT2_X,
    JOINT1_TO_JOINT2_Y,
    JOINT1_TO_JOINT2_Z,
    JOINT2_TO_TCP_X,
    JOINT2_TO_TCP_Y,
    TCP_Z_OFFSET,
    forward_kinematics_scara,
)
from scripts.train_mlp import INPUTS, JOINT_LIMITS, OUTPUTS


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "nn" / "splits"
NEIGHBOUR_COUNT = 5
MAX_CLOUD_POINTS = 4000


def _arm_points(robot, q):
    """Return base, arm joints, and predicted TCP in the robot's base frame."""
    if robot == "ned":
        q1, q2, q3 = q

        def rotate_radius(radius, height):
            return -radius * np.sin(q1), radius * np.cos(q1), height

        shoulder = (0.0, 0.0, SHOULDER_HEIGHT)
        elbow_radius = UPPER_ARM_REACH * np.cos(q2) + UPPER_ARM_RISE * np.sin(q2)
        elbow_height = SHOULDER_HEIGHT - UPPER_ARM_REACH * np.sin(q2) + UPPER_ARM_RISE * np.cos(q2)
        elbow = rotate_radius(elbow_radius, elbow_height)
        tcp = forward_kinematics_ned(q)
        return np.asarray([(0.0, 0.0, 0.0), shoulder, elbow, tcp])

    q1, q2, lift = q
    c1, s1 = np.cos(q1), np.sin(q1)
    pivot = (BASE_TO_JOINT1_X, 0.0, BASE_TO_JOINT1_Z)
    elbow = (
        pivot[0] + c1 * JOINT1_TO_JOINT2_X - s1 * JOINT1_TO_JOINT2_Y,
        pivot[1] + s1 * JOINT1_TO_JOINT2_X + c1 * JOINT1_TO_JOINT2_Y,
        pivot[2] + JOINT1_TO_JOINT2_Z,
    )
    c12, s12 = np.cos(q1 + q2), np.sin(q1 + q2)
    tcp = (
        elbow[0] + c12 * JOINT2_TO_TCP_X - s12 * JOINT2_TO_TCP_Y,
        elbow[1] + s12 * JOINT2_TO_TCP_X + c12 * JOINT2_TO_TCP_Y,
        elbow[2] + lift + TCP_Z_OFFSET,
    )
    return np.asarray([(0.0, 0.0, 0.0), pivot, elbow, tcp])


def _raw_output_sensitivity(ik, target):
    """Exact local derivative of raw normalized MLP outputs against raw XYZ."""
    flags = (1.0, 0.0) if ik.robot_name == "scara" else (0.0, 1.0)
    activation = (np.array([[*flags, *target]]) - ik.mean) / ik.std
    jacobian = np.eye(len(INPUTS))
    for layer in range(1, ik.model.n_layers + 1):
        weights = ik.model.params[f"W{layer}"]
        pre_activation = activation @ weights + ik.model.params[f"b{layer}"]
        jacobian = jacobian @ weights
        if layer < ik.model.n_layers:
            active = pre_activation[0] > 0.0
            activation = np.maximum(pre_activation, 0.0)
            jacobian *= active[None, :]
        else:
            activation = pre_activation

    return jacobian[2:, :].T / ik.std[None, 2:]


class MLPPredictionDashboard:
    """Update a live view with the target and network result from Webots."""

    def __init__(self, ik, data_size="large"):
        self.ik = ik
        self.robot = ik.robot_name
        data_path = DATA / self.robot / f"train_{data_size}.csv"
        with data_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.xyz = np.array([[float(row[n]) for n in ("x", "y", "z")] for row in rows])
        self.q_train = np.array([[float(row[n]) for n in OUTPUTS] for row in rows])
        # Compare a query's nearest-sample distance with the training set's own
        # nearest-neighbour spacing. This describes data support, not reachability.
        internal_distances, _ = cKDTree(self.xyz).query(self.xyz, k=2)
        self.support_distance_95 = float(np.percentile(internal_distances[:, 1], 95))

        rng = np.random.default_rng(42)
        indices = np.arange(len(self.xyz))
        if len(indices) > MAX_CLOUD_POINTS:
            indices = rng.choice(indices, MAX_CLOUD_POINTS, replace=False)
        self.cloud = self.xyz[indices]

        self.figure = plt.figure(figsize=(14, 8))
        self.figure.canvas.manager.set_window_title(f"Live MLP explanation: {self.robot}")
        self.ax_cloud = self.figure.add_subplot(221, projection="3d")
        self.ax_arm = self.figure.add_subplot(222, projection="3d")
        self.ax_joints = self.figure.add_subplot(223)
        self.ax_sensitivity = self.figure.add_subplot(224)
        self.figure.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.15,
                                    wspace=0.25, hspace=0.34)

        self.ax_cloud.scatter(*self.cloud.T, s=5, c="#b7c4ce", alpha=0.35,
                              label="training TCPs")
        self.ax_cloud.set_title("How well does the training data cover this target?")
        self.ax_cloud.set_xlabel("x (m)")
        self.ax_cloud.set_ylabel("y (m)")
        self.ax_cloud.set_zlabel("z (m)")
        self.target_artist = self.ax_cloud.scatter([], [], [], c="red", marker="x", s=100,
                                                   linewidths=3, label="Webots target")
        self.predicted_artist = self.ax_cloud.scatter([], [], [], c="#f28e2b", s=65,
                                                      label="MLP predicted TCP")
        self.measured_artist = self.ax_cloud.scatter([], [], [], c="#59a14f", marker="^", s=55,
                                                     label="measured Webots TCP")
        self.nearest_artist = self.ax_cloud.scatter([], [], [], c="#4e79a7", s=34,
                                                     label="5 nearest training TCPs")
        self.error_line, = self.ax_cloud.plot([], [], [], c="#f28e2b", linewidth=2)
        self.support_line, = self.ax_cloud.plot(
            [], [], [], c="#4e79a7", linestyle="--", linewidth=1.5,
            label="target-to-nearest data gap",
        )
        self.ax_cloud.legend(loc="upper left", fontsize=8)

        self.arm_artist, = self.ax_arm.plot([], [], [], "o-", linewidth=4, markersize=7,
                                            color="#4e79a7", label="predicted arm pose")
        self.arm_target_artist = self.ax_arm.scatter([], [], [], c="red", marker="x", s=100,
                                                      linewidths=3, label="Webots target")
        self.ax_arm.set_title("Pose commanded by the controller")
        self.ax_arm.set_xlabel("x (m)")
        self.ax_arm.set_ylabel("y (m)")
        self.ax_arm.set_zlabel("z (m)")
        self.ax_arm.legend(loc="upper left", fontsize=8)

        self.joint_names = ("q1", "q2", "q3")
        self.joint_units = ("rad", "rad", "m") if self.robot == "scara" else ("rad",) * 3
        self.joint_x = np.arange(3)
        self.ax_joints.set_title("Joint commands after output clipping")
        self.ax_joints.set_xticks(self.joint_x, self.joint_names)
        self.ax_joints.set_ylabel("normalised command (allowed: −1 to +1)")
        self.ax_joints.set_ylim(-1.15, 1.15)
        self.ax_joints.axhline(0.0, color="black", linewidth=0.6)
        self.joint_bars = self.ax_joints.bar(
            self.joint_x, np.zeros(3), color="#f28e2b", label="clipped controller command"
        )
        self.neighbour_markers = [
            self.ax_joints.plot([], [], "o", color="#4e79a7", alpha=0.7, markersize=4,
                                label="nearest examples" if i == 0 else None)[0]
            for i in range(NEIGHBOUR_COUNT)
        ]
        self.ax_joints.legend(loc="upper right", fontsize=8)

        self.ax_sensitivity.set_title("How the raw network output changes nearby")
        self.ax_sensitivity.set_xlabel("target coordinate")
        self.ax_sensitivity.set_ylabel("raw normalized output change / metre")
        self.ax_sensitivity.set_xticks(np.arange(3), ("x", "y", "z"))
        self.sensitivity_bars = [
            self.ax_sensitivity.bar(np.arange(3) + (i - 1) * 0.24, np.zeros(3),
                                    width=0.22, label=self.joint_names[i])
            for i in range(3)
        ]
        self.ax_sensitivity.axhline(0.0, color="black", linewidth=0.6)
        self.ax_sensitivity.legend(loc="upper right", fontsize=8)
        self.status = self.figure.text(0.07, 0.075, "Waiting for a target from Webots...",
                                       ha="left", va="center", fontsize=8,
                                       family="monospace")
        self.figure.text(
            0.07, 0.02,
            "Move the target in Webots. A far training-data distance means weak model support, not proof the point is physically unreachable.",
            fontsize=8,
        )
        self.figure.suptitle(f"{self.robot.upper()} | {ik.description}", fontsize=12)
        plt.show(block=False)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def update(self, target, q_physical, measured_tcp=None):
        target = np.asarray(target, dtype=float)
        q_physical = np.asarray(q_physical, dtype=float)
        fk = forward_kinematics_scara if self.robot == "scara" else forward_kinematics_ned
        predicted = np.asarray(fk(q_physical), dtype=float)
        flags = (1.0, 0.0) if self.robot == "scara" else (0.0, 1.0)
        input_row = (np.array([[*flags, *target]]) - self.ik.mean) / self.ik.std
        q_raw = self.ik.model.predict(input_row)[0]
        q_norm = np.clip(q_raw, -1.0, 1.0)

        distances = np.linalg.norm(self.xyz - target, axis=1)
        nearest = np.argsort(distances)[:NEIGHBOUR_COUNT]
        nearest_xyz = self.xyz[nearest]
        self.target_artist._offsets3d = ([target[0]], [target[1]], [target[2]])
        self.predicted_artist._offsets3d = ([predicted[0]], [predicted[1]], [predicted[2]])
        self.nearest_artist._offsets3d = (nearest_xyz[:, 0], nearest_xyz[:, 1], nearest_xyz[:, 2])
        nearest_point = nearest_xyz[0]
        self.support_line.set_data_3d(
            [target[0], nearest_point[0]],
            [target[1], nearest_point[1]],
            [target[2], nearest_point[2]],
        )
        self.error_line.set_data_3d([target[0], predicted[0]],
                                    [target[1], predicted[1]],
                                    [target[2], predicted[2]])
        if measured_tcp is not None:
            measured_tcp = np.asarray(measured_tcp, dtype=float)
            self.measured_artist._offsets3d = (
                [measured_tcp[0]], [measured_tcp[1]], [measured_tcp[2]]
            )
        else:
            self.measured_artist._offsets3d = ([], [], [])

        arm = _arm_points(self.robot, q_physical)
        self.arm_artist.set_data_3d(arm[:, 0], arm[:, 1], arm[:, 2])
        self.arm_target_artist._offsets3d = ([target[0]], [target[1]], [target[2]])
        for axis, bounds in (
            (self.ax_cloud, np.vstack((self.cloud, nearest_xyz, [target, predicted]))),
            (self.ax_arm, np.vstack((arm, [target, predicted]))),
        ):
            lows, highs = bounds.min(axis=0), bounds.max(axis=0)
            center = (lows + highs) / 2.0
            radius = max(float(np.max(highs - lows)) * 0.55, 0.15)
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(max(-0.15, center[2] - radius), center[2] + radius)
            axis.set_box_aspect((1, 1, 1))

        for index, bar in enumerate(self.joint_bars):
            bar.set_height(float(q_norm[index]))
        for row, marker in enumerate(self.neighbour_markers):
            marker.set_data(self.joint_x, self.q_train[nearest[row]])

        jacobian = _raw_output_sensitivity(self.ik, target)
        max_abs = max(float(np.max(np.abs(jacobian))), 0.1)
        self.ax_sensitivity.set_ylim(-max_abs * 1.2, max_abs * 1.2)
        for joint, group in enumerate(self.sensitivity_bars):
            for coordinate, bar in enumerate(group.patches):
                bar.set_height(float(jacobian[joint, coordinate]))

        nearest_distance = float(distances[nearest[0]])
        weak_support = nearest_distance > self.support_distance_95
        support_label = "WEAK TRAINING SUPPORT" if weak_support else "near training samples"
        measured_error = ""
        if measured_tcp is not None:
            measured_error = f" | measured TCP error {np.linalg.norm(measured_tcp - target) * 1000:.1f} mm"
        saturated = [
            name for name, value in zip(self.joint_names, q_raw)
            if value <= -1.0 or value >= 1.0
        ]
        saturation_label = ", ".join(saturated) if saturated else "none"
        physical_joints = ", ".join(
            f"{name}={value:+.3f} {unit}"
            for name, value, unit in zip(self.joint_names, q_physical, self.joint_units)
        )
        limits = JOINT_LIMITS[self.robot]
        limit_text = ", ".join(
            f"{name}[{low:+.2f},{high:+.2f}] {unit}"
            for name, (low, high), unit in zip(self.joint_names, limits, self.joint_units)
        )
        self.status.set_text(
            f"1. DATA SUPPORT: {support_label}; nearest example {nearest_distance * 1000:.0f} mm away "
            f"(95% of training points have a closer peer within {self.support_distance_95 * 1000:.0f} mm).\n"
            f"2. MLP → CONTROLLER: raw normalized output "
            f"[{q_raw[0]:+.2f}, {q_raw[1]:+.2f}, {q_raw[2]:+.2f}] → clipped "
            f"[{q_norm[0]:+.2f}, {q_norm[1]:+.2f}, {q_norm[2]:+.2f}]; at limit: {saturation_label}.\n"
            f"3. POSE: {physical_joints} | limits {limit_text}\n"
            f"Predicted TCP error {np.linalg.norm(predicted - target) * 1000:.0f} mm"
            f"{measured_error}\n"
            f"Target ({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}) m | "
            f"standardized network input xyz ({input_row[0, 2]:+.2f}, "
            f"{input_row[0, 3]:+.2f}, {input_row[0, 4]:+.2f})"
        )
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def process_events(self):
        """Let the GUI toolkit process redraw and window events between steps."""
        if plt.fignum_exists(self.figure.number):
            plt.pause(0.001)
