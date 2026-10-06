"""
Given the three active joint values of the Ned arm, it computes the resulting TCP
(x, y, z) position in metres, expressed in the robot's own base frame. This
matches the "arm" frame returned by controllers/common/arm_controller.py's
get_tcp_position(), which is what Task 1.C uses (together with the Task 1.A
controller) to check this script's precision against the real simulator.

Active joint vector, per the course handout's "Motors to be used for each
arm" table: [joint_1, joint_2, joint_3], all in radians.
    joint_1 - waist yaw (rotation about the vertical/Z axis)
    joint_2 - shoulder pitch
    joint_3 - elbow pitch
joint_4, joint_5 and joint_6 (the wrist) are fixed/ignored for this
coursework, so they are not part of the input vector.

The link geometry constants below (offsets and the TCP location) are taken
from the course's "Robot specifications" appendix for the Ned, which in turn
matches the anchor/translation values defined in protos/TeachingNed.proto.
"""

import math


# The real robot links are not perfectly straight
# Vertical offset from the robot's base frame up to the joint_1 (waist) axis.
BASE_TO_JOINT1 = 0.1065  # metres
# Vertical offset from the joint_1 axis up to the joint_2 (shoulder) axis.
JOINT1_TO_JOINT2 = 0.0650  # metres


SHOULDER_HEIGHT = BASE_TO_JOINT1 + JOINT1_TO_JOINT2  # 0.1715 m


# Shoulder to elbow vector (when joint_2 = 0)
UPPER_ARM_RISE = 0.2210  # metres, vertical
UPPER_ARM_REACH = 0.0120  # metres, horizontal

# Elbow to TCP vector (joint_3 = 0)
FOREARM_RISE = 0.03185  # metres, vertical
FOREARM_REACH = 0.23000  # metres, horizontal


def forward_kinematics_ned(joint_values):
    # Compute the Ned TCP position from its three active joint values.

    joint_1, joint_2, joint_3 = joint_values  # waist yaw, shoulder pitch, elbow pitch

    # find where the TCP sits relative to the shoulder (joint_2) axis once the elbow has bent.
    rise_from_shoulder = (
        UPPER_ARM_RISE
        + FOREARM_RISE * math.cos(joint_3)  # forearm's own rise
        - FOREARM_REACH * math.sin(joint_3)  # forearm's own reach
    )
    reach_from_shoulder = (
        UPPER_ARM_REACH
        + FOREARM_RISE * math.sin(joint_3)  # forearm's own rise
        + FOREARM_REACH * math.cos(joint_3)  # forearm's own reach
    )
    # Rotate that combined upper-arm+forearm vector by the shoulder pitch joint_2 to get the TCP's horizontal reach
    radius = (
        reach_from_shoulder * math.cos(joint_2)
        + rise_from_shoulder * math.sin(joint_2)
    )  # horizontal distance from the waist (joint_1) axis
    height = (
        SHOULDER_HEIGHT
        - reach_from_shoulder * math.sin(joint_2)
        + rise_from_shoulder * math.cos(joint_2)
    )  # height of the TCP above the robot base frame

    # Rotate the horizontal reach by the waist yaw joint_1 to split it into x and y.
    # Webots Ned defines joint_1 = 0 pointing along +Y, so x is built from -sin and y from cos
    x = -radius * math.sin(joint_1)  # x coordinate of the TCP in the base frame
    y = radius * math.cos(joint_1)  # y coordinate of the TCP in the base frame
    z = height  # z coordinate

    return x, y, z


if __name__ == "__main__":
    # Quick manual sanity check: print the TCP position for a few example
    # joint configurations, including the all-zero (home) pose.
    example_joint_sets = [
        (0.0, 0.0, 0.0),  # home pose: all joints at zero
        (0.5, 0.2, 0.3),  # arbitrary pose with a positive (elbow-up) joint_3
        (-0.3, -0.1, 0.8),  # arbitrary pose with negative waist/shoulder angles
    ]
    for joint_1, joint_2, joint_3 in example_joint_sets:
        # Run the forward kinematics for this one example joint configuration.
        x, y, z = forward_kinematics_ned([joint_1, joint_2, joint_3])
        # Report the joint configuration and the TCP position it produces.
        print(
            f"joints=({joint_1:.3f}, {joint_2:.3f}, {joint_3:.3f}) rad -> "
            f"TCP=({x:.4f}, {y:.4f}, {z:.4f}) m"
        )
