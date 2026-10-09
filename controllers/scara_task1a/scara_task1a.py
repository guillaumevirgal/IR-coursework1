"""Generate SCARA FK validation data using measurements from Webots."""

import csv
import math
import random
import sys
from pathlib import Path

# Webots starts each controller from its own directory.
CONTROLLERS_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CONTROLLERS_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from common.arm_controller import ArmController, Joint, round_pose
from scripts.forward_kinematics_scara import forward_kinematics_scara


TOLERANCE = 1e-3
CSV_FILE = "scara_robot_data_1e-3.csv"  # Set to False to disable logging.

# Add specific active-joint poses here, or leave empty for continuous random
# generation. Each pose is [base_arm_motor, arm_motor, shaft_linear_motor].
POSES = []

# The coursework requires the arm_motor range to be positive (elbow-up).
JOINT_LIMITS = {
    "base_arm_motor": (-0.73, 0.73),
    "arm_motor": (0.0, 0.83),
    "shaft_linear_motor": (-0.2, 0.0),
}

JOINTS = {
    "shoulder": Joint("base_arm_motor", speed=0.7, tolerance=TOLERANCE),
    "elbow": Joint("arm_motor", speed=0.7, tolerance=TOLERANCE),
    # Tool rotation is fixed because orientation is ignored in the coursework.
    "rotation": Joint("shaft_rotation_motor", speed=1.0, tolerance=TOLERANCE),
    "lift": Joint("shaft_linear_motor", speed=0.05, tolerance=1e-4),
}

CSV_HEADER = (
    "joint1",
    "joint2",
    "joint4",
    "ex",
    "ey",
    "ez",
    "ax",
    "ay",
    "az",
    "error",
)


def generate_random_pose():
    """Sample one active-joint pose in the coursework ranges."""

    return [
        random.uniform(*JOINT_LIMITS["base_arm_motor"]),
        random.uniform(*JOINT_LIMITS["arm_motor"]),
        random.uniform(*JOINT_LIMITS["shaft_linear_motor"]),
    ]


def initialise_csv():
    """Create the output file and header without overwriting existing data."""

    path = Path(CSV_FILE)
    if not path.is_file():
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerow(CSV_HEADER)


def write_csv_line(pose, expected, actual, error):
    """Append one settled pose and its FK/Webots comparison."""

    with Path(CSV_FILE).open("a", newline="", encoding="utf-8") as csv_file:
        csv.writer(csv_file).writerow((*pose, *expected, *actual, error))


def position_error(expected, actual):
    """Return the Euclidean distance between two TCP positions."""

    return math.sqrt(sum((measured - calculated) ** 2 for calculated, measured in zip(expected, actual)))


def move_and_report(arm, pose):
    """Move to an active-joint pose, report it, and log valid measurements."""

    expected = forward_kinematics_scara(pose)

    # ArmController also controls the ignored shaft rotation, fixed at zero.
    commanded_pose = [pose[0], pose[1], 0.0, pose[2]]
    print(
        f"Moving to pose: {round_pose(pose)}, expected TCP in arm frame (m): "
        f"({expected[0]:+.4f}, {expected[1]:+.4f}, {expected[2]:+.4f})"
    )

    try:
        tcp = arm.move_to(commanded_pose)
    except TimeoutError as error:
        print(f"Is the arm stuck? TimeoutError: {error}")
        return

    settled_pose = [
        arm.sensors["shoulder"].getValue(),
        arm.sensors["elbow"].getValue(),
        arm.sensors["lift"].getValue(),
    ]
    # Calculate FK from the measured joints, matching what Webots actually
    # reached instead of the requested target during the precision check.
    expected = forward_kinematics_scara(settled_pose)
    actual = tcp["arm"]
    error = position_error(expected, actual)

    print(f"Settled joint values: {round_pose(settled_pose)}")
    print(
        "Expected TCP in arm frame (m): "
        f"({expected[0]:+.4f}, {expected[1]:+.4f}, {expected[2]:+.4f})"
    )
    print(
        "Actual   TCP in arm frame (m): "
        f"({actual[0]:+.4f}, {actual[1]:+.4f}, {actual[2]:+.4f})"
    )
    print(f"Position error: {error:.6f} m ({error * 1000:.3f} mm)")

    if CSV_FILE and error <= TOLERANCE:
        write_csv_line(settled_pose, expected, actual, error)


def main():
    arm = ArmController(JOINTS, tcp_slot="handSlot")

    if CSV_FILE:
        initialise_csv()

    if POSES:
        for pose in POSES:
            move_and_report(arm, pose)
    else:
        while True:
            move_and_report(arm, generate_random_pose())


if __name__ == "__main__":
    main()
