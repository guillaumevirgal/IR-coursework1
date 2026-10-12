"""Shared Webots dataset generation for the Ned and SCARA controllers."""

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

from common.arm_controller import ArmController, round_pose


CSV_FIELDS = (
    "attempt", "accepted", "rejection_reason",
    "joint1", "joint2", "joint3",
    "ex", "ey", "ez",
    "ax", "ay", "az",
    "error",
)
NN_FIELDS = ("is_scara", "is_ned", "x", "y", "z", "q1", "q2", "q3")


@dataclass(frozen=True)
class DatasetConfig:
    """Everything the shared generator needs to control one robot."""

    name: str
    joints: dict
    tcp_slot: str
    joint_limits: tuple
    forward_kinematics: object
    command_pose: object
    minimum_z: float


def random_pose(config, rng):
    """Sample one joint vector; all rejection decisions are recorded later."""

    return [rng.uniform(lower, upper) for lower, upper in config.joint_limits]


def initialise_csv(path):
    """Create an audit CSV or confirm that an existing file has its schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerow(CSV_FIELDS)
        return

    with path.open(newline="", encoding="utf-8") as csv_file:
        header = next(csv.reader(csv_file), None)
    if header != list(CSV_FIELDS):
        raise ValueError(f"{path} has an incompatible CSV header: {header}")


def append_audit_row(path, attempt, pose, expected, actual, error, reason):
    """Append one attempted pose, including failed/rejected attempts."""

    with path.open("a", newline="", encoding="utf-8") as csv_file:
        accepted = reason == ""
        actual_values = actual if actual is not None else ("", "", "")
        csv.writer(csv_file).writerow((
            attempt, int(accepted), reason, *pose, *expected, *actual_values,
            error if error is not None else "",
        ))
        csv_file.flush()


def normalise_joint(value, limits):
    lower, upper = limits
    return 2.0 * (value - lower) / (upper - lower) - 1.0


def build_nn_dataset(audit_path, nn_path, robot, joint_limits):
    """Atomically derive the ready-to-train file from accepted audit rows."""

    nn_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = nn_path.with_suffix(nn_path.suffix + ".tmp")
    flags = (1, 0) if robot == "scara" else (0, 1)
    try:
        with audit_path.open(newline="", encoding="utf-8") as source, temporary_path.open(
            "w", newline="", encoding="utf-8"
        ) as destination:
            reader = csv.DictReader(source)
            writer = csv.writer(destination)
            writer.writerow(NN_FIELDS)
            for row in reader:
                if row["accepted"] != "1":
                    continue
                joints = [float(row[f"joint{index}"]) for index in range(1, 4)]
                normalised = [
                    normalise_joint(value, limits)
                    for value, limits in zip(joints, joint_limits)
                ]
                if any(value < -1.000000001 or value > 1.000000001 for value in normalised):
                    raise ValueError(f"Accepted joint outside {robot} limits: {joints}")
                position = [float(row[f"e{axis}"]) for axis in "xyz"]
                writer.writerow((*flags, *position, *normalised))
        temporary_path.replace(nn_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def measure_pose(arm, config, pose, audit_path, attempt, tolerance):
    """Attempt a pose and audit it whether it succeeds or is rejected."""

    expected = config.forward_kinematics(pose)
    if expected[2] < config.minimum_z:
        print(
            f"{config.name}: skipped pose because FK z={expected[2]:.4f} m "
            f"is below the {config.minimum_z:.4f} m floor-clearance limit"
        )
        append_audit_row(audit_path, attempt, pose, expected, None, None, "below_clearance")
        return False

    print(
        f"{config.name}: moving to {round_pose(pose)}, expected TCP (m): "
        f"({expected[0]:+.4f}, {expected[1]:+.4f}, {expected[2]:+.4f})"
    )

    try:
        actual = arm.move_to(config.command_pose(pose))["arm"]
    except TimeoutError as error:
        print(f"{config.name}: skipped pose that did not settle: {error}")
        append_audit_row(audit_path, attempt, pose, expected, None, None, "timeout")
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
        append_audit_row(audit_path, attempt, pose, expected, actual, error, "fk_error")
        return False

    append_audit_row(audit_path, attempt, pose, expected, actual, error, "")
    return True


def generate_dataset(
    config,
    output_path,
    nn_path,
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
    if nn_path.exists():
        raise FileExistsError(f"Refusing to overwrite NN dataset: {nn_path}")
    initialise_csv(output_path)
    arm = ArmController(config.joints, tcp_slot=config.tcp_slot)
    rng = random.Random(seed)
    saved = 0

    candidates = iter(poses) if poses else iter(lambda: random_pose(config, rng), None)
    attempt = 0
    for pose in candidates:
        attempt += 1
        if measure_pose(arm, config, pose, output_path, attempt, tolerance):
            saved += 1
            print(f"{config.name}: saved {saved} row(s) to {output_path}")
        if samples is not None and saved >= samples:
            build_nn_dataset(output_path, nn_path, config.name, config.joint_limits)
            print(f"{config.name}: accepted={saved}, attempted={attempt}, rejected={attempt - saved}")
            print(f"{config.name}: NN dataset saved to {nn_path}")
            arm.robot.simulationQuit(0)
            return saved

    if samples is not None and saved < samples:
        print(f"{config.name}: saved {saved} of {samples} requested row(s)")
    if samples is None:
        build_nn_dataset(output_path, nn_path, config.name, config.joint_limits)
        print(f"{config.name}: accepted={saved}, attempted={attempt}, rejected={attempt - saved}")
    arm.robot.simulationQuit(0)
    return saved
