"""TCP loss and analytical FK gradients for the existing NumPy MLP.

Positions use metres in the same arm frame as the dataset. Jacobians have
shape (batch, xyz, joint) and differentiate physical joint values (rad or m).
The vectorised equations mirror the existing scalar FK implementations.
"""

import numpy as np

from scripts import forward_kinematics_ned as ned
from scripts import forward_kinematics_scara as scara


def tcp_and_jacobian(robot, joints):
    """Return batched TCP positions and d(TCP)/d(physical joints)."""
    q1, q2, q3 = np.asarray(joints, dtype=float).T
    jacobian = np.zeros((len(q1), 3, 3))
    if robot == "ned":
        rise = ned.UPPER_ARM_RISE + ned.FOREARM_RISE * np.cos(q3) - ned.FOREARM_REACH * np.sin(q3)
        reach = ned.UPPER_ARM_REACH + ned.FOREARM_RISE * np.sin(q3) + ned.FOREARM_REACH * np.cos(q3)
        d_rise = -ned.FOREARM_RISE * np.sin(q3) - ned.FOREARM_REACH * np.cos(q3)
        d_reach = ned.FOREARM_RISE * np.cos(q3) - ned.FOREARM_REACH * np.sin(q3)
        radius = reach * np.cos(q2) + rise * np.sin(q2)
        relative_height = -reach * np.sin(q2) + rise * np.cos(q2)
        d_radius = d_reach * np.cos(q2) + d_rise * np.sin(q2)
        d_height = -d_reach * np.sin(q2) + d_rise * np.cos(q2)
        positions = np.column_stack((-radius * np.sin(q1), radius * np.cos(q1),
                                     ned.SHOULDER_HEIGHT + relative_height))
        jacobian[:, :, 0] = np.column_stack((-radius * np.cos(q1), -radius * np.sin(q1),
                                             np.zeros_like(q1)))
        jacobian[:, :, 1] = np.column_stack((-relative_height * np.sin(q1),
                                             relative_height * np.cos(q1), -radius))
        jacobian[:, :, 2] = np.column_stack((-d_radius * np.sin(q1), d_radius * np.cos(q1), d_height))
    elif robot == "scara":
        tcp_x = scara.JOINT2_TO_TCP_X * np.cos(q2) - scara.JOINT2_TO_TCP_Y * np.sin(q2)
        tcp_y = scara.JOINT2_TO_TCP_X * np.sin(q2) + scara.JOINT2_TO_TCP_Y * np.cos(q2)
        arm_x = scara.JOINT1_TO_JOINT2_X + tcp_x
        arm_y = scara.JOINT1_TO_JOINT2_Y + tcp_y
        x_offset = arm_x * np.cos(q1) - arm_y * np.sin(q1)
        y = arm_x * np.sin(q1) + arm_y * np.cos(q1)
        positions = np.column_stack((scara.BASE_TO_JOINT1_X + x_offset, y,
            scara.BASE_TO_JOINT1_Z + scara.JOINT1_TO_JOINT2_Z + q3 + scara.TCP_Z_OFFSET))
        jacobian[:, :, 0] = np.column_stack((-y, x_offset, np.zeros_like(q1)))
        jacobian[:, :, 1] = np.column_stack((-tcp_y * np.cos(q1) - tcp_x * np.sin(q1),
                                             -tcp_y * np.sin(q1) + tcp_x * np.cos(q1),
                                             np.zeros_like(q1)))
        jacobian[:, 2, 2] = 1.0  # Third output is the SCARA prismatic joint, in metres.
    else:
        raise ValueError(f"Unknown robot: {robot}")
    return positions, jacobian


def tcp_loss_and_gradient(q_norm, x_raw, joint_limits, limit_weight=0.01):
    """Mean squared TCP distance (m²) plus a joint-limit penalty.

    The penalty is limit_weight * mean(sum((raw - clipped)**2)); its weight
    has units m² per squared normalised joint unit. It supplies a gradient
    where clipping makes the TCP gradient zero. No joint labels are used.
    """
    q_norm = np.asarray(q_norm)
    x_raw = np.asarray(x_raw)
    flags = x_raw[:, :2]
    if not np.all((flags == 0.0) | (flags == 1.0)) or not np.all(flags.sum(axis=1) == 1.0):
        raise ValueError("Expected one-hot [is_scara, is_ned] robot flags")
    if limit_weight < 0:
        raise ValueError("Joint-limit penalty weight must be nonnegative")
    clipped = np.clip(q_norm, -1.0, 1.0)
    outside = q_norm - clipped
    count = len(q_norm)
    loss = limit_weight * float(np.mean(np.sum(outside ** 2, axis=1)))
    gradient = (2.0 * limit_weight / count) * outside
    for flag, robot in enumerate(("scara", "ned")):
        mask = flags[:, flag] == 1.0
        if not np.any(mask):
            continue
        limits = np.asarray(joint_limits[robot])
        half_range = (limits[:, 1] - limits[:, 0]) / 2.0
        physical_q = limits[:, 0] + (clipped[mask] + 1.0) * half_range
        positions, jacobian = tcp_and_jacobian(robot, physical_q)
        error = positions - x_raw[mask, 2:5]
        loss += float(np.sum(error ** 2) / count)
        d_physical = (2.0 / count) * np.einsum("bij,bi->bj", jacobian, error)
        clip_gradient = (q_norm[mask] > -1.0) & (q_norm[mask] < 1.0)
        gradient[mask] += d_physical * half_range * clip_gradient
    return loss, gradient
