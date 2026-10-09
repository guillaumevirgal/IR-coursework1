"""Generate normalised Task 2 datasets directly from forward kinematics."""

import argparse
from pathlib import Path

from dataset_generation.common import RobotConfig, generate_dataset_files
from forward_kinematics_ned import forward_kinematics_ned
from forward_kinematics_scara import forward_kinematics_scara


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = (1_000, 5_000, 10_000)

ROBOTS = {
    "ned": RobotConfig(
        name="ned",
        joint_limits=((-2.8, 2.8), (-0.8, 0.8), (0.0, 1.3)),
        minimum_z=0.14,
        forward_kinematics=forward_kinematics_ned,
    ),
    "scara": RobotConfig(
        name="scara",
        joint_limits=((-0.73, 0.73), (0.0, 0.83), (-0.2, 0.0)),
        minimum_z=0.02,
        forward_kinematics=forward_kinematics_scara,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=("ned", "scara", "both"))
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "datasets",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    robot_names = ROBOTS if args.robot == "both" else (args.robot,)
    for robot_name in robot_names:
        paths = generate_dataset_files(
            ROBOTS[robot_name],
            args.output_dir / robot_name,
            args.sizes,
            args.test_size,
            args.seed,
        )
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
