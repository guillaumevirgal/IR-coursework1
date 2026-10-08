import numpy as np
import csv

def read_joints_from_csv(line_number, filename="robot_data.csv"):
    with open(filename, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for current_line, row in enumerate(reader, start=1):
            if current_line == line_number:
                joint1 = float(row["joint1"])
                joint2 = float(row["joint2"])
                joint3 = float(row["joint3"])

                return joint1, joint2, joint3

    raise ValueError(f"Line {line_number} not found in {filename}")

# Robot dimensions (metres)
BASE_TO_JOINT1 = 0.1065
JOINT1_TO_JOINT2 = 0.0650

SHOULDER_HEIGHT = BASE_TO_JOINT1 + JOINT1_TO_JOINT2

UPPER_ARM_RISE = 0.2210
UPPER_ARM_REACH = 0.0120

FOREARM_RISE = 0.03185
FOREARM_REACH = 0.23000

import numpy as np
import math


def draw_robot(ax, joint_1, joint_2, joint_3):

    # ---------------------------------------------------------
    # Fixed points
    # ---------------------------------------------------------

    base = np.array([0.0, 0.0, 0.0])

    joint1 = np.array([
        0.0,
        0.0,
        BASE_TO_JOINT1
    ])

    joint2 = np.array([
        0.0,
        0.0,
        SHOULDER_HEIGHT
    ])

    # ---------------------------------------------------------
    # Calculate elbow position
    #
    # At joint_3 = 0:
    #   upper arm  = (REACH, RISE)
    #   forearm    = (REACH, RISE)
    #
    # Joint 3 bends the forearm relative to the upper arm.
    # ---------------------------------------------------------

    elbow_radius = (
        UPPER_ARM_REACH * math.cos(joint_2)
        + UPPER_ARM_RISE * math.sin(joint_2)
    )

    elbow_height = (
        SHOULDER_HEIGHT
        - UPPER_ARM_REACH * math.sin(joint_2)
        + UPPER_ARM_RISE * math.cos(joint_2)
    )

    # ---------------------------------------------------------
    # Calculate TCP position relative to shoulder
    #
    # This is exactly the same calculation as your FK.
    # ---------------------------------------------------------

    rise_from_shoulder = (
        UPPER_ARM_RISE
        + FOREARM_RISE * math.cos(joint_3)
        - FOREARM_REACH * math.sin(joint_3)
    )

    reach_from_shoulder = (
        UPPER_ARM_REACH
        + FOREARM_RISE * math.sin(joint_3)
        + FOREARM_REACH * math.cos(joint_3)
    )

    tcp_radius = (
        reach_from_shoulder * math.cos(joint_2)
        + rise_from_shoulder * math.sin(joint_2)
    )

    tcp_height = (
        SHOULDER_HEIGHT
        - reach_from_shoulder * math.sin(joint_2)
        + rise_from_shoulder * math.cos(joint_2)
    )

    # ---------------------------------------------------------
    # Convert radial positions into world X/Y
    #
    # Webots convention:
    # joint_1 = 0 -> arm points +Y
    # ---------------------------------------------------------

    elbow = np.array([
        -elbow_radius * math.sin(joint_1),
         elbow_radius * math.cos(joint_1),
         elbow_height
    ])

    tcp = np.array([
        -tcp_radius * math.sin(joint_1),
         tcp_radius * math.cos(joint_1),
         tcp_height
    ])

    # ---------------------------------------------------------
    # Draw robot
    # ---------------------------------------------------------

    points = np.array([
        base,
        joint1,
        joint2,
        elbow,
        tcp
    ])

    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        "-o",
        linewidth=3,
        markersize=6
    )

    # Highlight TCP
    ax.scatter(
        tcp[0],
        tcp[1],
        tcp[2],
        s=60
    )