"""Fast FK-based dataset generation shared by Ned and SCARA."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path


CSV_FIELDS = ("q1", "q2", "q3", "x", "y", "z")
TRAIN_FRACTION = 0.8


@dataclass(frozen=True)
class RobotConfig:
    name: str
    joint_limits: tuple
    minimum_z: float
    forward_kinematics: object


def normalise(value, lower, upper):
    """Map a joint value from its sampled range to [-1, 1]."""

    return 2.0 * (value - lower) / (upper - lower) - 1.0


def sample_row(config, rng):
    """Sample one collision-filtered pose and calculate its normalised row."""

    while True:
        joints = [rng.uniform(lower, upper) for lower, upper in config.joint_limits]
        position = config.forward_kinematics(joints)
        if position[2] >= config.minimum_z:
            normalised = [
                normalise(value, lower, upper)
                for value, (lower, upper) in zip(joints, config.joint_limits)
            ]
            return tuple(normalised + list(position))


def unique_rows(config, count, rng, used_positions=None):
    """Generate rows with unique TCP positions."""

    used = set() if used_positions is None else set(used_positions)
    rows = []
    while len(rows) < count:
        row = sample_row(config, rng)
        position = row[3:]
        if position in used:
            continue
        used.add(position)
        rows.append(row)
    return rows, used


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_FIELDS)
        writer.writerows(rows)


def size_labels(sizes):
    if len(sizes) == 3:
        return ("small", "medium", "large")
    return tuple(str(size) for size in sizes)


def generate_dataset_files(config, output_dir, sizes, test_size, seed):
    """Create nested train/validation files and a separate test file."""

    sizes = tuple(sorted(sizes))
    if not sizes or any(size < 2 for size in sizes):
        raise ValueError("dataset sizes must all be at least 2")
    if len(set(sizes)) != len(sizes):
        raise ValueError("dataset sizes must be different")
    if test_size < 1:
        raise ValueError("test size must be at least 1")

    rng = random.Random(seed)
    largest_train = round(sizes[-1] * TRAIN_FRACTION)
    largest_validation = sizes[-1] - largest_train
    train_rows, used = unique_rows(config, largest_train, rng)
    validation_rows, used = unique_rows(config, largest_validation, rng, used)
    test_rows, _ = unique_rows(config, test_size, rng, used)

    written = []
    for label, size in zip(size_labels(sizes), sizes):
        train_size = round(size * TRAIN_FRACTION)
        validation_size = size - train_size
        train_path = output_dir / f"train_{label}.csv"
        validation_path = output_dir / f"validation_{label}.csv"
        write_csv(train_path, train_rows[:train_size])
        write_csv(validation_path, validation_rows[:validation_size])
        written.extend((train_path, validation_path))

    test_path = output_dir / "test.csv"
    write_csv(test_path, test_rows)
    written.append(test_path)
    return written
