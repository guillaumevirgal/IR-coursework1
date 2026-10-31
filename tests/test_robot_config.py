"""Checks for the shared joint limits and (de)normalisation used everywhere."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.robot_config import FORWARD_KINEMATICS, JOINT_LIMITS, ROBOTS, denormalise, normalise


class RobotConfigTest(unittest.TestCase):
    def test_every_robot_has_limits_and_kinematics(self):
        for robot in ROBOTS:
            self.assertEqual(len(JOINT_LIMITS[robot]), 3)
            self.assertTrue(all(low < high for low, high in JOINT_LIMITS[robot]))
            self.assertEqual(len(FORWARD_KINEMATICS[robot](tuple(low for low, _ in JOINT_LIMITS[robot]))), 3)

    def test_limits_map_to_minus_one_and_one(self):
        for robot in ROBOTS:
            limits = JOINT_LIMITS[robot]
            lower = [low for low, _ in limits]
            upper = [high for _, high in limits]
            np.testing.assert_allclose(normalise(lower, limits), [-1, -1, -1])
            np.testing.assert_allclose(normalise(upper, limits), [1, 1, 1])

    def test_normalise_then_denormalise_round_trips(self):
        rng = np.random.default_rng(0)
        for robot in ROBOTS:
            limits = JOINT_LIMITS[robot]
            lower = np.array([low for low, _ in limits])
            upper = np.array([high for _, high in limits])
            q = lower + rng.random((50, 3)) * (upper - lower)
            np.testing.assert_allclose(denormalise(normalise(q, limits), limits), q, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
