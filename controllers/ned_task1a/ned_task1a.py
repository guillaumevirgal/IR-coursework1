import random
import sys
from pathlib import Path

# Webots starts each controller from its own folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.arm_controller import ArmController, Joint, round_pose
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.forward_kinematics_ned import forward_kinematics_ned

TOLERANCE = 1e-3

# Fill with your own poses, or leave empty to generate random poses.
POSES = [
    # [-2.1146, 0.7782, 0.9575], # error: 0.007892
    # [-1.9301, 0.7326, 0.8255], # error: 0.004553
    # [1.4876, -0.0048, 0.5218],
    # [-1.1322, -0.6295, 1.0351],
    # [-0.1111, -0.0788, 0.5624],
]

# If filled it will automatically check if the poses matched the TCP targets. If empty, no check will be performed.
TCP_TARGETS = [
    [-0.2258, 0.0189, 0.3065],
    [0.0937, 0.0439, 0.2957],
    [0.0237, 0.2117, 1],
]

JOINTS = {
    "waist": Joint("joint_1", speed=2.0, tolerance=TOLERANCE),
    "shoulder": Joint("joint_2", speed=2.0, tolerance=TOLERANCE),
    "elbow": Joint("joint_3", speed=2.0, tolerance=TOLERANCE), # positive only
}

JOINT_LIMITS = {
    "joint_1": (-2.8, 2.8),
    "joint_2": (-0.8, 0.8),
    "joint_3": (-1.5, 1.3), # limited to positive only from -1.5
}

def generate_random_pose():
    return [
        random.uniform(*JOINT_LIMITS["joint_1"]),
        random.uniform(*JOINT_LIMITS["joint_2"]),
        random.uniform(*JOINT_LIMITS["joint_3"])
    ]

def check_tcp(measured, target, tolerance=TOLERANCE):
    return all(
        abs(m - t) <= tolerance
        for m, t in zip(measured, target)
    )

import math

def print_tcp_error(expected, actual):
    dx = actual[0] - expected[0]
    dy = actual[1] - expected[1]
    dz = actual[2] - expected[2]

    distance = math.sqrt(dx**2 + dy**2 + dz**2)

    print(f"Expected TCP in world frame (m): "
          f"({expected[0]:+.4f}, {expected[1]:+.4f}, {expected[2]:+.4f})")
    print(f"Actual   TCP in world frame (m): "
          f"({actual[0]:+.4f}, {actual[1]:+.4f}, {actual[2]:+.4f})")
    print(f"Position error: {distance:.6f} m ({distance * 1000:.3f} mm)")

def move_and_report(arm,pose,target=None):
    x, y, z = forward_kinematics_ned(pose)
    print(f"Moving to pose: {round_pose(pose)}, expected TCP in world frame (m): ({x:+.4f}, {y:+.4f}, {z:+.4f})")
    try:
        tcp = arm.move_to(pose)
    except TimeoutError as e:
        print(f"Is the arm stuck? TimeoutError: {e}")
        return
    print(f"Pose {round_pose(pose)} settled joint values:", arm.read_joints(precision=4))
    # print(f"Expected TCP in world frame (m): ({x:+.4f}, {y:+.4f}, {z:+.4f})")
    # print(f"Actual   TCP in world frame (m): ({tcp['world'][0]:+.4f}, {tcp['world'][1]:+.4f}, {tcp['world'][2]:+.4f})")

    print_tcp_error(
        (x, y, z),
        tcp["world"]
    )

    if check_tcp(tcp["world"], [x, y, z]):
        print("TCP target matched!")
    else:
        print("ERROR: TCP target didn't match.")

def main():
    arm = ArmController(JOINTS, tcp_slot="tcpSlot")
    if len(POSES):
        for i, pose in enumerate(POSES, start=1):
            target = TCP_TARGETS[i - 1] if i <= len(TCP_TARGETS) else None
            move_and_report(arm, pose, target)
    else:
        while True:
            move_and_report(arm, generate_random_pose())

if __name__ == "__main__":
    main()