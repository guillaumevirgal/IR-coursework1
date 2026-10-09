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
        default="actual",
        help="TCP coordinates used in the distribution plot",
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
    positions = tcp_positions(rows, source)
    x, y, z = zip(*positions)
    has_errors = rows[0]["error"] is not None
    if has_errors:
        errors_mm = [row["error"] * 1000 for row in rows]
        scatter = axis.scatter(x, y, z, c=errors_mm, cmap="RdYlGn_r", s=8)
    else:
        scatter = axis.scatter(x, y, z, color="tab:blue", s=8)
    axis.set_title(f"{source.title()} TCP distribution ({len(rows)} points)")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    return scatter, has_errors


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


def draw_quality(axis, rows):
    bins = min(50, max(5, len(rows) // 20))
    if rows[0]["error"] is not None:
        errors_mm = [row["error"] * 1000 for row in rows]
        axis.hist(errors_mm, bins=bins)
        axis.axvline(1.0, color="tab:red", linestyle="--", label="1 mm limit")
        axis.set_title("FK versus Webots error")
        axis.set_xlabel("Position error (mm)")
        axis.legend()
    else:
        for joint in ("q1", "q2", "q3"):
            axis.hist([row[joint] for row in rows], bins=bins, alpha=0.5, label=joint)
        axis.set_title("Normalised joint distribution")
        axis.set_xlabel("Normalised joint value")
        axis.legend()
    axis.set_ylabel("Samples")


def main():
    args = parse_args()
    rows = load_dataset(args.input, args.robot)
    if not 1 <= args.row <= len(rows):
        raise ValueError(f"--row must be between 1 and {len(rows)}")

    figure = plt.figure(figsize=(16, 5))
    distribution_axis = figure.add_subplot(1, 3, 1, projection="3d")
    pose_axis = figure.add_subplot(1, 3, 2, projection="3d")
    error_axis = figure.add_subplot(1, 3, 3)

    scatter, has_errors = draw_distribution(distribution_axis, rows, args.tcp_source)
    if has_errors:
        figure.colorbar(scatter, ax=distribution_axis, label="TCP error (mm)", shrink=0.7)
    draw_pose(pose_axis, args.robot, rows[args.row - 1], args.row)
    draw_quality(error_axis, rows)
    figure.tight_layout()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=180)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
