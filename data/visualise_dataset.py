"""Visualise a generated Ned or SCARA dataset."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from visualisation.common import (
    load_dataset,
    robot_points,
    set_equal_3d_axes,
    tcp_positions,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=("ned", "scara"))
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--tcp-source",
        choices=("actual", "expected"),
        default="expected",
        help="TCP coordinates used in the distribution plot (actual omits attempts without a measurement)",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=1,
        help="one-based dataset row drawn in the robot-pose panel",
    )
    parser.add_argument("--save", type=Path, help="save the figure instead of opening it")
    return parser.parse_args()


def draw_distribution(axis, rows, source):
    accepted = [row for row in rows if row["accepted"]]
    rejected = [row for row in rows if not row["accepted"]]
    accepted_positions = tcp_positions(accepted, source)
    rejected_positions = tcp_positions(rejected, source)
    errors_mm = [row["error"] * 1000 for row in accepted if row["error"] is not None]
    scatter = None
    if accepted_positions:
        x, y, z = zip(*accepted_positions)
        scatter = axis.scatter(x, y, z, c=errors_mm, cmap="RdYlGn_r", s=8, label="accepted")
    if rejected_positions:
        x, y, z = zip(*rejected_positions)
        axis.scatter(x, y, z, color="tab:orange", marker="x", s=16, label="rejected")
    axis.set_title(
        f"{source.title()} TCP: {len(accepted)} accepted, {len(rejected)} rejected"
    )
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    if accepted_positions or rejected_positions:
        axis.legend()
    return scatter


def draw_pose(axis, robot, row, row_number):
    points = robot_points(robot, row)
    x, y, z = zip(*points)
    axis.plot(x, y, z, "-o", linewidth=3, markersize=6)
    axis.scatter([x[-1]], [y[-1]], [z[-1]], s=80, color="tab:red", label="TCP")
    set_equal_3d_axes(axis, points)
    axis.set_title(f"{robot.upper()} pose at row {row_number}")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.legend()


def draw_error_histogram(axis, rows):
    errors_mm = [row["error"] * 1000 for row in rows if row["error"] is not None]
    axis.hist(errors_mm, bins=min(50, max(5, len(rows) // 20)))
    axis.axvline(1.0, color="tab:red", linestyle="--", label="1 mm acceptance limit")
    rejected = {}
    for row in rows:
        if not row["accepted"]:
            rejected[row["rejection_reason"]] = rejected.get(row["rejection_reason"], 0) + 1
    reasons = ", ".join(f"{key}: {value}" for key, value in sorted(rejected.items())) or "none"
    axis.set_title(f"FK versus Webots error\nRejected: {reasons}")
    axis.set_xlabel("Position error (mm)")
    axis.set_ylabel("Samples")
    axis.legend()


def main():
    args = parse_args()
    rows = load_dataset(args.input)
    accepted_rows = [row for row in rows if row["accepted"]]
    if not 1 <= args.row <= len(accepted_rows):
        raise ValueError(f"--row must be between 1 and {len(accepted_rows)} accepted rows")

    figure = plt.figure(figsize=(16, 5))
    distribution_axis = figure.add_subplot(1, 3, 1, projection="3d")
    pose_axis = figure.add_subplot(1, 3, 2, projection="3d")
    error_axis = figure.add_subplot(1, 3, 3)

    scatter = draw_distribution(distribution_axis, rows, args.tcp_source)
    if scatter is not None:
        figure.colorbar(scatter, ax=distribution_axis, label="TCP error (mm)", shrink=0.7)
    draw_pose(pose_axis, args.robot, accepted_rows[args.row - 1], args.row)
    draw_error_histogram(error_axis, rows)
    figure.tight_layout()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=180)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
