"""SCARA configuration and joint targets for Task 1.A."""

import sys
from pathlib import Path

# Webots starts each controller from its own folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.arm_controller import ArmController, Joint


# Each joint specifies its Webots motor, movement speed and position tolerance.
# The names on the left are the simple names used when giving target values.
# Rotating joint positions/tolerances use radians; the lift uses metres.
JOINTS = {
    "shoulder": Joint("base_arm_motor", speed=0.7, tolerance=0.001),
    "elbow": Joint("arm_motor", speed=0.7, tolerance=0.001),
    "rotation": Joint("shaft_rotation_motor", speed=1.0, tolerance=0.001),
    "lift": Joint("shaft_linear_motor", speed=0.05, tolerance=0.0001),
}

# Pose format: [shoulder (rad), elbow (rad), rotation (rad), lift (m)].
# The order must match the JOINTS list above. Add more rows to run more poses.
POSES = [
    [0.5, -0.5, 0.0, -0.1],
    [0.2, 0.3, 0.0, -0.05],
]


if __name__ == "__main__":
    # By default, move through every pose in POSES, one after another.
    poses = POSES
    if sys.argv[1:]:
        # Optional arguments look like shoulder=0.5 or lift=-0.1; these run one pose.
        targets = {}
        for argument in sys.argv[1:]:
            name, value = argument.split("=", 1)
            targets[name] = float(value)
        poses = [targets]

    # Connect the shared controller to this SCARA's motor names and TCP slot.
    arm = ArmController(JOINTS, tcp_slot="handSlot")
    for pose_number, pose in enumerate(poses, start=1):
        # Move to this row of joint values, wait for it to settle, then report the result.
        tcp = arm.move_to(pose)
        print(f"Pose {pose_number} settled joint values:", arm.read_joints())
        print("Measured TCP in world frame (m):", tcp["world"])
        print("Measured TCP in arm frame (m):", tcp["arm"])
    # Keep the simulation running with the arm holding its final pose.
    arm.hold_position()
