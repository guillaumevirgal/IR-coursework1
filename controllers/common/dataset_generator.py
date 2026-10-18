"""Shared Webots dataset generation for the Ned and SCARA controllers."""

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

from common.arm_controller import ArmController, round_pose


CSV_FIELDS = (
    "robot", "seed", "attempt", "accepted", "rejection_reason",
    "joint1", "joint2", "joint3",
    "ex", "ey", "ez",
    "ax", "ay", "az",
    "error",
)
NN_FIELDS = ("is_scara", "is_ned", "x", "y", "z", "q1", "q2", "q3")
LEGACY_AUDIT_FIELDS = CSV_FIELDS[2:]


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


def prepare_audit(path, robot, seed):
    """Create a checkpoint or return accepted/attempt counts for resuming it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file, lineterminator="\n").writerow(CSV_FIELDS)
        return 0, 0

    # A killed process may leave a final, incomplete CSV line. Discard only
    # that unterminated line; all completed rows were flushed individually.
    with path.open("rb+") as csv_file:
        content = csv_file.read()
        if content and not content.endswith(b"\n"):
            last_newline = content.rfind(b"\n")
            csv_file.truncate(last_newline + 1 if last_newline >= 0 else 0)

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        header = reader.fieldnames
        if header == list(LEGACY_AUDIT_FIELDS):
            legacy_rows = list(reader)
            temporary_path = path.with_suffix(path.suffix + ".resume.tmp")
            with temporary_path.open("w", newline="", encoding="utf-8") as upgraded:
                writer = csv.writer(upgraded, lineterminator="\n")
                writer.writerow(CSV_FIELDS)
                for row in legacy_rows:
                    writer.writerow((robot, seed, *(row[field] for field in LEGACY_AUDIT_FIELDS)))
            temporary_path.replace(path)
            print(
                f"{robot}: upgrading existing audit; it has no saved seed, "
                f"so continuing with supplied seed {seed}",
                flush=True,
            )
        elif header != list(CSV_FIELDS):
            raise ValueError(f"{path} has an incompatible audit schema")

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        accepted = 0
        attempts = 0
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path} has an incomplete row at line {line_number}")
            if row["robot"] != robot or int(row["seed"]) != seed:
                raise ValueError(
                    f"{path} belongs to robot={row['robot']} seed={row['seed']}; "
                    f"resume with the same robot and seed"
                )
            if int(row["attempt"]) != attempts + 1:
                raise ValueError(f"{path} has a non-sequential attempt at line {line_number}")
            attempts += 1
            accepted += row["accepted"] == "1"
    return accepted, attempts


def append_audit_row(path, robot, seed, attempt, pose, expected, actual, error, reason):
    """Append one attempted pose, including failed/rejected attempts."""

    with path.open("a", newline="", encoding="utf-8") as csv_file:
        accepted = reason == ""
        actual_values = actual if actual is not None else ("", "", "")
        csv.writer(csv_file, lineterminator="\n").writerow((
            robot, seed, attempt, int(accepted), reason, *pose, *expected, *actual_values,
            error if error is not None else "",
        ))
        csv_file.flush()


def normalise_joint(value, limits):
    lower, upper = limits
    return 2.0 * (value - lower) / (upper - lower) - 1.0


def build_nn_dataset(audit_path, nn_path, robot, joint_limits, accepted_limit=None):
    """Atomically derive the ready-to-train file from accepted audit rows."""

    nn_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = nn_path.with_suffix(nn_path.suffix + ".tmp")
    flags = (1, 0) if robot == "scara" else (0, 1)
    try:
        with audit_path.open(newline="", encoding="utf-8") as source, temporary_path.open(
            "w", newline="", encoding="utf-8"
        ) as destination:
            reader = csv.DictReader(source)
            writer = csv.writer(destination, lineterminator="\n")
            writer.writerow(NN_FIELDS)
            accepted_count = 0
            for row in reader:
                if row["accepted"] != "1":
                    continue
                if accepted_limit is not None and accepted_count >= accepted_limit:
                    break
                joints = [float(row[f"joint{index}"]) for index in range(1, 4)]
                normalised = [
                    normalise_joint(value, limits)
                    for value, limits in zip(joints, joint_limits)
                ]
                if any(value < -1.000000001 or value > 1.000000001 for value in normalised):
                    raise ValueError(f"Accepted joint outside {robot} limits: {joints}")
                position = [float(row[f"e{axis}"]) for axis in "xyz"]
                writer.writerow((*flags, *position, *normalised))
                accepted_count += 1
            if accepted_limit is not None and accepted_count < accepted_limit:
                raise ValueError(
                    f"Audit contains {accepted_count} accepted rows; expected {accepted_limit}"
                )
        temporary_path.replace(nn_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def measure_pose(arm, config, pose, audit_path, attempt, seed, tolerance):
    """Attempt a pose and audit it whether it succeeds or is rejected."""

    expected = config.forward_kinematics(pose)
    if expected[2] < config.minimum_z:
        print(
            f"{config.name}: skipped pose because FK z={expected[2]:.4f} m "
            f"is below the {config.minimum_z:.4f} m floor-clearance limit"
        )
        append_audit_row(audit_path, config.name, seed, attempt, pose, expected, None, None, "below_clearance")
        return False

    print(
        f"{config.name}: moving to {round_pose(pose)}, expected TCP (m): "
        f"({expected[0]:+.4f}, {expected[1]:+.4f}, {expected[2]:+.4f})"
    )

    try:
        actual = arm.move_to(config.command_pose(pose))["arm"]
    except TimeoutError as error:
        print(f"{config.name}: skipped pose that did not settle: {error}")
        append_audit_row(audit_path, config.name, seed, attempt, pose, expected, None, None, "timeout")
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
        append_audit_row(audit_path, config.name, seed, attempt, pose, expected, actual, error, "fk_error")
        return False

    append_audit_row(audit_path, config.name, seed, attempt, pose, expected, actual, error, "")
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed = 0 if seed is None else int(seed)
    saved, attempts = prepare_audit(output_path, config.name, seed)
    arm = ArmController(config.joints, tcp_slot=config.tcp_slot)
    rng = random.Random(seed)
    if samples is None:
        print(
            f"{config.name}: continuing from {saved} accepted rows after {attempts} attempts",
            flush=True,
        )
    else:
        print(
            f"{config.name}: resuming at {saved}/{samples} accepted rows "
            f"after {attempts} attempts; {max(0, samples - saved)} remaining",
            flush=True,
        )

    if samples is not None and saved >= samples:
        build_nn_dataset(output_path, nn_path, config.name, config.joint_limits, samples)
        print(
            f"{config.name}: target reached: requested {samples}, already have "
            f"{saved} accepted samples after {attempts} attempts; no more poses needed"
        )
        arm.robot.simulationQuit(0)
        return saved

    if poses:
        candidates = iter(poses[attempts:])
    else:
        for _ in range(attempts):
            random_pose(config, rng)
        candidates = iter(lambda: random_pose(config, rng), None)
    attempt = attempts
    for pose in candidates:
        attempt += 1
        if measure_pose(arm, config, pose, output_path, attempt, seed, tolerance):
            saved += 1
            print(f"{config.name}: saved {saved} row(s) to {output_path}")
        if samples is not None and saved >= samples:
            build_nn_dataset(output_path, nn_path, config.name, config.joint_limits, samples)
            print(
                f"{config.name}: target reached ({samples} accepted samples); "
                f"audit attempts={attempt}, rejected={attempt - saved}"
            )
            print(f"{config.name}: NN dataset saved to {nn_path}")
            arm.robot.simulationQuit(0)
            return saved

    if samples is not None and saved < samples:
        print(f"{config.name}: only {saved} accepted rows; requested target is {samples}")
    if samples is None:
        build_nn_dataset(output_path, nn_path, config.name, config.joint_limits)
        print(f"{config.name}: accepted={saved}, attempted={attempt}, rejected={attempt - saved}")
    arm.robot.simulationQuit(0)
    return saved
