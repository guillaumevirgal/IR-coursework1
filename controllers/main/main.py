"""Main Webots dataset controller for both the Ned and SCARA robots."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "controllers"))
sys.path.insert(0, str(ROOT))

from common.arm_controller import Joint
from common.dataset_generator import DatasetConfig, generate_dataset
from scripts.forward_kinematics_ned import forward_kinematics_ned
from scripts.forward_kinematics_scara import forward_kinematics_scara


CONFIGS = {
    "ned": DatasetConfig(
        name="ned",
        joints={
            "waist": Joint("joint_1", speed=2.0, tolerance=1e-3),
            "shoulder": Joint("joint_2", speed=2.0, tolerance=1e-3),
            "elbow": Joint("joint_3", speed=2.0, tolerance=1e-3),
        },
        tcp_slot="tcpSlot",
        joint_limits=((-2.8, 2.8), (-0.8, 0.8), (0.0, 1.3)),
        forward_kinematics=forward_kinematics_ned,
        command_pose=lambda pose: pose,
    ),
    "scara": DatasetConfig(
        name="scara",
        joints={
            "shoulder": Joint("base_arm_motor", speed=0.7, tolerance=1e-3),
            "elbow": Joint("arm_motor", speed=0.7, tolerance=1e-3),
            # Tool rotation is ignored by the coursework and fixed at zero.
            "rotation": Joint("shaft_rotation_motor", speed=1.0, tolerance=1e-3),
            "lift": Joint("shaft_linear_motor", speed=0.05, tolerance=1e-4),
        },
        tcp_slot="handSlot",
        joint_limits=((-0.73, 0.73), (0.0, 0.83), (-0.2, 0.0)),
        forward_kinematics=forward_kinematics_scara,
        command_pose=lambda pose: (pose[0], pose[1], 0.0, pose[2]),
    ),
}

# Add active-joint poses for repeatable checks, keyed by robot name.
POSES = {"ned": [], "scara": []}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", choices=CONFIGS)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or ROOT / "data" / "csv" / f"{args.robot}_robot_data_1e-3.csv"
    generate_dataset(
        CONFIGS[args.robot],
        output,
        samples=args.samples,
        seed=args.seed,
        poses=POSES[args.robot],
    )


if __name__ == "__main__":
    main()
