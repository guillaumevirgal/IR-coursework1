"""Forward kinematics for the SCARA arm."""

import math


BASE_TO_JOINT1_X = 0.060
BASE_TO_JOINT1_Z = 0.220

JOINT1_TO_JOINT2_X = 0.310
JOINT1_TO_JOINT2_Y = 0.005
JOINT1_TO_JOINT2_Z = -0.004

JOINT2_TO_TCP_X = 0.2545
JOINT2_TO_TCP_Y = -0.005
TCP_Z_OFFSET = -0.034


def forward_kinematics_scara(joint_values):
    q1, q2, q4 = joint_values

    # Rotate the joint 2 to TCP vector by q2
    tcp_x = (
        JOINT2_TO_TCP_X * math.cos(q2)
        - JOINT2_TO_TCP_Y * math.sin(q2)
    )

    tcp_y = (
        JOINT2_TO_TCP_X * math.sin(q2)
        + JOINT2_TO_TCP_Y * math.cos(q2)
    )

    # Add the joint 1 to joint 2 offset
    arm_x = JOINT1_TO_JOINT2_X + tcp_x
    arm_y = JOINT1_TO_JOINT2_Y + tcp_y
    arm_z = JOINT1_TO_JOINT2_Z + q4 + TCP_Z_OFFSET

    # Rotate the arm by the base joint angle q1
    x = (
        BASE_TO_JOINT1_X
        + arm_x * math.cos(q1)
        - arm_y * math.sin(q1)
    )

    y = (
        arm_x * math.sin(q1)
        + arm_y * math.cos(q1)
    )

    z = BASE_TO_JOINT1_Z + arm_z

    return x, y, z

if __name__ == "__main__":
    test_joints = [0.5, -0.5, -0.1]

    x, y, z = forward_kinematics_scara(test_joints)

    print(f"SCARA FK: x={x:.6f}, y={y:.6f}, z={z:.6f}")
