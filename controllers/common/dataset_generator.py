"""Shared Webots dataset generation for the Ned and SCARA controllers."""

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

from common.arm_controller import ArmController, round_pose


CSV_FIELDS = (
    "joint1", "joint2", "joint3",
    "ex", "ey", "ez",
    "ax", "ay", "az",
    "error",
)


@dataclass(frozen=True)
class DatasetConfig:
    """Everything the shared generator needs to control one robot."""

    name: str
    joints: dict
    tcp_slot: str
    joint_limits: tuple
    forward_kinematics: object
    command_pose: object


def random_pose(config, rng):
    """Sample the three active joints within their configured limits."""

    return [rng.uniform(lower, upper) for lower, upper in config.joint_limits]


def initialise_csv(path):
    """Create a CSV or confirm that an existing file has the expected schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerow(CSV_FIELDS)
        return

    with path.open(newline="", encoding="utf-8") as csv_file:
        header = next(csv.reader(csv_file), None)
    if header != list(CSV_FIELDS):
        raise ValueError(f"{path} has an incompatible CSV header: {header}")


def append_row(path, pose, expected, actual, error):
    """Append one joint/FK/Webots result and flush it immediately."""

    with path.open("a", newline="", encoding="utf-8") as csv_file:
        csv.writer(csv_file).writerow((*pose, *expected, *actual, error))
        csv_file.flush()


def measure_pose(arm, config, pose, output_path, tolerance):
    """Command one pose and save it when FK agrees with Webots."""

    expected = config.forward_kinematics(pose)
    print(
        f"{config.name}: moving to {round_pose(pose)}, expected TCP (m): "
        f"({expected[0]:+.4f}, {expected[1]:+.4f}, {expected[2]:+.4f})"
    )

    try:
        actual = arm.move_to(config.command_pose(pose))["arm"]
    except TimeoutError as error:
        print(f"{config.name}: skipped pose that did not settle: {error}")
        return False

    error = math.dist(expected, actual)
    print(
        f"{config.name}: actual TCP (m): "
        f"({actual[0]:+.4f}, {actual[1]:+.4f}, {actual[2]:+.4f}); "
        f"error={error * 1000:.3f} mm"
    )

    if error > tolerance:
        print(
            f"{config.name}: skipped row because error exceeds "
            f"{tolerance * 1000:.3f} mm"
        )
        return False

    append_row(output_path, pose, expected, actual, error)
    return True


def generate_dataset(
    config,
    output_path,
    *,
    samples=None,
    seed=None,
    poses=None,
    tolerance=1e-3,
):
    """Generate valid rows until the requested count is reached.

    ``samples=None`` runs continuously. Explicit ``poses`` are each attempted
    once and are useful for repeatable manual checks.
    """

    if samples is not None and samples < 1:
        raise ValueError("samples must be positive or None")
    if tolerance <= 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be a positive finite number")

    output_path = Path(output_path)
    initialise_csv(output_path)
    arm = ArmController(config.joints, tcp_slot=config.tcp_slot)
    rng = random.Random(seed)
    saved = 0

    candidates = iter(poses) if poses else iter(lambda: random_pose(config, rng), None)
    for pose in candidates:
        if measure_pose(arm, config, pose, output_path, tolerance):
            saved += 1
            print(f"{config.name}: saved {saved} row(s) to {output_path}")
        if samples is not None and saved >= samples:
            arm.robot.simulationQuit(0)
            return saved

    if samples is not None and saved < samples:
        print(f"{config.name}: saved {saved} of {samples} requested row(s)")
    arm.robot.simulationQuit(0)
    return saved
