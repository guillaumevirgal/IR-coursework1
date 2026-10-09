"""Shared CSV loading and robot geometry for dataset visualisation."""

import csv
import math


RAW_FIELDS = {
    "joint1", "joint2", "joint3",
    "ex", "ey", "ez",
    "ax", "ay", "az",
    "error",
}
NORMALISED_FIELDS = {"q1", "q2", "q3", "x", "y", "z"}
JOINT_LIMITS = {
    "ned": ((-2.8, 2.8), (-0.8, 0.8), (0.0, 1.3)),
    "scara": ((-0.73, 0.73), (0.0, 0.83), (-0.2, 0.0)),
}


def denormalise(value, lower, upper):
    return (value + 1.0) / 2.0 * (upper - lower) + lower


def load_dataset(path, robot):
    """Load either a current normalised dataset or a legacy raw dataset."""

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fields = set(reader.fieldnames or ())
        source_rows = list(reader)

    if RAW_FIELDS.issubset(fields):
        rows = [{field: float(row[field]) for field in RAW_FIELDS} for row in source_rows]
    elif NORMALISED_FIELDS.issubset(fields):
        rows = []
        for row in source_rows:
            normalised = [float(row[f"q{index}"]) for index in range(1, 4)]
            joints = [
                denormalise(value, lower, upper)
                for value, (lower, upper) in zip(normalised, JOINT_LIMITS[robot])
            ]
            x, y, z = (float(row[axis]) for axis in ("x", "y", "z"))
            rows.append({
                "joint1": joints[0], "joint2": joints[1], "joint3": joints[2],
                "q1": normalised[0], "q2": normalised[1], "q3": normalised[2],
                "ex": x, "ey": y, "ez": z,
                "ax": x, "ay": y, "az": z,
                "error": None,
            })
    else:
        expected = sorted(RAW_FIELDS), sorted(NORMALISED_FIELDS)
        raise ValueError(f"{path} does not match a supported CSV schema: {expected}")

    if not rows:
        raise ValueError(f"{path} contains no dataset rows")
    return rows


def tcp_positions(rows, source):
    """Return expected or Webots-measured TCP coordinates."""

    prefix = "e" if source == "expected" else "a"
    return [
        (row[f"{prefix}x"], row[f"{prefix}y"], row[f"{prefix}z"])
        for row in rows
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
