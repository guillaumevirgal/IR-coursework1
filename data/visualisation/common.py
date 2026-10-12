"""Shared CSV loading and robot geometry for dataset visualisation."""

import csv
import math


CSV_FIELDS = (
    "robot", "seed", "attempt", "accepted", "rejection_reason",
    "joint1", "joint2", "joint3",
    "ex", "ey", "ez",
    "ax", "ay", "az",
    "error",
)


def load_dataset(path):
    """Load and validate a dataset produced by the shared generator."""

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(CSV_FIELDS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = []
        for row in reader:
            values = {
                field: (float(row[field]) if row[field] != "" else None)
                for field in CSV_FIELDS
                if field not in ("robot", "seed", "attempt", "accepted", "rejection_reason")
            }
            values.update({
                "robot": row["robot"],
                "seed": int(row["seed"]),
                "attempt": int(row["attempt"]),
                "accepted": row["accepted"] == "1",
                "rejection_reason": row["rejection_reason"],
            })
            rows.append(values)

    if not rows:
        raise ValueError(f"{path} contains no dataset rows")
    return rows


def tcp_positions(rows, source):
    """Return expected or Webots-measured TCP coordinates."""

    prefix = "e" if source == "expected" else "a"
    return [
        (row[f"{prefix}x"], row[f"{prefix}y"], row[f"{prefix}z"])
        for row in rows
        if row[f"{prefix}x"] is not None
    ]


def ned_points(joints):
    """Return base, waist, shoulder, elbow and TCP points for the Ned."""

    q1, q2, q3 = joints
    shoulder_height = 0.1065 + 0.0650

    elbow_radius = 0.0120 * math.cos(q2) + 0.2210 * math.sin(q2)
    elbow_height = (
        shoulder_height
        - 0.0120 * math.sin(q2)
        + 0.2210 * math.cos(q2)
    )

    rise = 0.2210 + 0.03185 * math.cos(q3) - 0.23000 * math.sin(q3)
    reach = 0.0120 + 0.03185 * math.sin(q3) + 0.23000 * math.cos(q3)
    tcp_radius = reach * math.cos(q2) + rise * math.sin(q2)
    tcp_height = shoulder_height - reach * math.sin(q2) + rise * math.cos(q2)

    return [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.1065),
        (0.0, 0.0, shoulder_height),
        (-elbow_radius * math.sin(q1), elbow_radius * math.cos(q1), elbow_height),
        (-tcp_radius * math.sin(q1), tcp_radius * math.cos(q1), tcp_height),
    ]


def scara_points(joints):
    """Return base, first joint, second joint and TCP points for SCARA."""

    q1, q2, q4 = joints
    joint1 = (0.060, 0.0, 0.220)
    joint2 = (
        joint1[0] + 0.310 * math.cos(q1) - 0.005 * math.sin(q1),
        joint1[1] + 0.310 * math.sin(q1) + 0.005 * math.cos(q1),
        joint1[2] - 0.004,
    )
    tool_angle = q1 + q2
    tcp = (
        joint2[0] + 0.2545 * math.cos(tool_angle) + 0.005 * math.sin(tool_angle),
        joint2[1] + 0.2545 * math.sin(tool_angle) - 0.005 * math.cos(tool_angle),
        joint2[2] + q4 - 0.034,
    )
    return [(0.0, 0.0, 0.0), joint1, joint2, tcp]


ROBOT_POINTS = {"ned": ned_points, "scara": scara_points}


def robot_points(robot, row):
    """Build drawing points for one dataset row and robot type."""

    joints = (row["joint1"], row["joint2"], row["joint3"])
    return ROBOT_POINTS[robot](joints)


def set_equal_3d_axes(axis, points):
    """Use equal scale on all axes so robot geometry is not distorted."""

    columns = list(zip(*points))
    centres = [(min(values) + max(values)) / 2 for values in columns]
    radius = max(max(values) - min(values) for values in columns) / 2
    radius = max(radius, 0.05)
    axis.set_xlim(centres[0] - radius, centres[0] + radius)
    axis.set_ylim(centres[1] - radius, centres[1] + radius)
    axis.set_zlim(centres[2] - radius, centres[2] + radius)
