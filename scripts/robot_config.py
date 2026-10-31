"""Single source of truth for the Ned and SCARA joint limits and kinematics.

Every training, evaluation and controller script imports these definitions, so
the limits used to normalise the dataset are the ones used to denormalise the
network output (the dataset generator controller imports them from here too).

Joints are the three active joints per arm, in physical units (rad, or metres
for the SCARA lift):
    Ned:   joint_1 (waist), joint_2 (shoulder), joint_3 (elbow)
    SCARA: base_arm_motor (shoulder), arm_motor (elbow), shaft_linear_motor (lift)
"""

import numpy as np

from scripts.forward_kinematics_ned import forward_kinematics_ned
from scripts.forward_kinematics_scara import forward_kinematics_scara

ROBOTS = ("ned", "scara")

# (lower, upper) per active joint. The Ned elbow and the SCARA elbow are limited
# to the elbow-up configuration used for data generation.
JOINT_LIMITS = {
    "ned": ((-2.8, 2.8), (-0.8, 0.8), (0.0, 1.3)),
    "scara": ((-0.73, 0.73), (0.0, 0.83), (-0.2, 0.0)),
}

FORWARD_KINEMATICS = {"ned": forward_kinematics_ned, "scara": forward_kinematics_scara}


def normalise(q_physical, limits):
    """Map physical joint values (rad or m) to [-1, 1] using the arm's limits."""

    lower = np.array([low for low, _ in limits])
    upper = np.array([high for _, high in limits])
    return 2.0 * (np.asarray(q_physical) - lower) / (upper - lower) - 1.0


def denormalise(q_norm, limits):
    """Map normalised joints in [-1, 1] back to motor units (rad or m)."""

    lower = np.array([low for low, _ in limits])
    upper = np.array([high for _, high in limits])
    return (np.asarray(q_norm) + 1.0) / 2.0 * (upper - lower) + lower
