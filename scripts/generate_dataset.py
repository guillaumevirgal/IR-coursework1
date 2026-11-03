"""Generate Ned and SCARA training data in Webots, then split it for training.

Usage, from the repository root:

    python scripts/generate_dataset.py [ned | scara | both] [--config PATH] [--webots PATH]

Every setting (sample count, seed, split sizes, output folders, worlds) comes from
the config file, default configs/dataset_generation.json. Each run:

1. Copies the robot's world with the dataset controller and its arguments.
2. Runs Webots in fast headless mode. The controller (controllers/generate_dataset)
   writes the audit of every attempt and the normalised NN file of accepted rows.
3. Splits the NN file into a test set and nested 80/20 train/validation sets.

Generation is resumable: rerunning with the same config continues the audit.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "dataset_generation.json"
ROBOTS = ("ned", "scara")
NN_FIELDS = ("is_scara", "is_ned", "x", "y", "z", "q1", "q2", "q3")
SPLIT_UNIT = 5  # every fifth development row is validation, so sizes must divide by 5
CONTROLLER_LINE = '  controller "mlp_ik"'


@dataclass(frozen=True)
class Splits:
    small: int
    medium: int
    large: int
    test: int


@dataclass(frozen=True)
class Settings:
    samples: int
    seed: int
    splits: Splits
    audit_dir: Path
    nn_dir: Path
    splits_root: Path
    worlds: dict


@dataclass(frozen=True)
class Outputs:
    audit: Path
    nn: Path
    splits: Path


def load_settings(path):
    """Read the config file, resolve its paths against the repository, and validate it."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    settings = Settings(
        samples=int(raw["samples"]),
        seed=int(raw["seed"]),
        splits=Splits(**{name: int(raw["splits"][name]) for name in Splits.__dataclass_fields__}),
        audit_dir=ROOT / raw["audit_dir"],
        nn_dir=ROOT / raw["nn_dir"],
        splits_root=ROOT / raw["splits_root"],
        worlds={robot: ROOT / raw["worlds"][robot] for robot in ROBOTS},
    )
    validate(settings)
    return settings


def validate(settings):
    splits = settings.splits
    if settings.samples < 1:
        raise ValueError("samples must be positive")
    sizes = (splits.small, splits.medium, splits.large)
    if any(size < SPLIT_UNIT or size % SPLIT_UNIT for size in sizes):
        raise ValueError("Split sizes must be positive multiples of 5 for an exact 80/20 split")
    if not splits.small <= splits.medium <= splits.large or splits.test < 1:
        raise ValueError("Require small <= medium <= large and a positive test size")
    needed = splits.large + splits.test
    if settings.samples < needed:
        raise ValueError(f"Need at least {needed} accepted samples for the requested splits")


def outputs_for(settings, robot):
    return Outputs(
        audit=(settings.audit_dir / f"{robot}_master_audit.csv").resolve(),
        nn=(settings.nn_dir / f"{robot}_master.csv").resolve(),
        splits=(settings.splits_root / robot).resolve(),
    )


def check_outputs(outputs, robot):
    """Refuse to split or overwrite derived files that lack their raw audit."""
    if not outputs.audit.exists() and (outputs.nn.exists() or outputs.splits.exists()):
        raise FileExistsError(
            f"Found derived outputs without their raw audit for {robot}; "
            "restore the audit or move the derived outputs before generating."
        )
    for path in (outputs.audit, outputs.nn, outputs.splits):
        path.parent.mkdir(parents=True, exist_ok=True)


def find_webots(explicit=None):
    """Find the Webots launcher on Windows, Linux or macOS."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))

    executable = shutil.which("webots") or shutil.which("webots.exe")
    if executable:
        candidates.append(Path(executable))

    webots_home = os.environ.get("WEBOTS_HOME")
    if webots_home:
        home = Path(webots_home)
        candidates.extend((
            home / "webots.exe",
            home / "msys64" / "mingw64" / "bin" / "webots.exe",
            home / "webots",
        ))

    program_files = os.environ.get("ProgramFiles")
    if program_files:
        install = Path(program_files) / "Webots"
        candidates.extend((
            install / "webots.exe",
            install / "msys64" / "mingw64" / "bin" / "webots.exe",
        ))

    candidates.extend((
        Path("/snap/bin/webots"),
        Path("/Applications/Webots.app/Contents/MacOS/webots"),
    ))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find Webots. Add it to PATH, set WEBOTS_HOME, or pass --webots."
    )


def make_world(source, robot, settings, outputs):
    """Write a copy of the robot's world that runs the dataset controller.

    The copy sits beside the original so relative PROTO paths still resolve.
    """
    text = source.read_text(encoding="utf-8")
    if text.count(CONTROLLER_LINE) != 1:
        raise RuntimeError(f"Expected one MLP controller in {source}")
    text = text.replace(CONTROLLER_LINE, '  controller "generate_dataset"', 1)

    marker = f'    "{robot}"'
    if text.count(marker) != 1:
        raise RuntimeError(f"Expected one {robot!r} controller argument in {source}")
    arguments = (
        f"{marker}\n"
        f'    "--samples={settings.samples}"\n'
        f'    "--seed={settings.seed}"\n'
        f'    "--output={outputs.audit.as_posix()}"\n'
        f'    "--nn-output={outputs.nn.as_posix()}"'
    )
    text = text.replace(marker, arguments, 1)

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".wbt", prefix=f"dataset_{robot}_",
        dir=source.parent, delete=False,
    ) as handle:
        handle.write(text)
    return Path(handle.name)


def remove_world(path):
    """Delete the temporary world and the Webots project files it may create."""
    for project_file in (path, path.with_suffix(".wbproj"), path.with_name(f".{path.stem}.wbproj")):
        try:
            project_file.unlink()
        except FileNotFoundError:
            pass


def run_webots(webots, world):
    command = [str(webots), "--batch", "--mode=fast", "--no-rendering", "--stdout", "--stderr", str(world)]
    subprocess.run(command, check=True)


def read_master(path, robot):
    """Read the NN file and check every row belongs to this robot and is in range."""
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != NN_FIELDS:
            raise ValueError(f"Expected NN columns {NN_FIELDS} in {path}")
        rows = list(reader)
    expected_flags = ("1", "0") if robot == "scara" else ("0", "1")
    for line, row in enumerate(rows, start=2):
        if (row["is_scara"], row["is_ned"]) != expected_flags:
            raise ValueError(f"Wrong robot flag on line {line} of {path}")
        if any(not -1.0 <= float(row[name]) <= 1.0 for name in ("q1", "q2", "q3")):
            raise ValueError(f"Normalised joint outside [-1, 1] on line {line}")
    return rows


def split_rows(rows, splits):
    """Return the test rows and the nested train/validation rows for each size.

    Rows are already random, so their order is kept. The first rows form the test
    set. In the rest, every fifth row is validation and the others are training.
    Smaller sizes take prefixes of the larger lists, so they nest.
    """
    needed = splits.test + splits.large
    if len(rows) < needed:
        raise ValueError(f"Need {needed} master rows for splits, found {len(rows)}")
    test = rows[:splits.test]
    development = rows[splits.test:needed]
    training = [row for index, row in enumerate(development) if index % SPLIT_UNIT != 0]
    validation = [row for index, row in enumerate(development) if index % SPLIT_UNIT == 0]

    files = {"test.csv": test}
    for label, size in (("small", splits.small), ("medium", splits.medium), ("large", splits.large)):
        count = size // SPLIT_UNIT
        files[f"train_{label}.csv"] = training[: count * 4]
        files[f"validation_{label}.csv"] = validation[:count]
    return files


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=NN_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate(robot, settings, outputs, webots):
    source = settings.worlds[robot]
    world = make_world(source, robot, settings, outputs)
    print(f"Generating {settings.samples} {robot} rows in Webots fast mode...", flush=True)
    try:
        run_webots(webots, world)
    finally:
        remove_world(world)

    rows = read_master(outputs.nn, robot)
    files = split_rows(rows, settings.splits)
    outputs.splits.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        write_rows(outputs.splits / name, data)
    print(f"Wrote {robot} train/validation/test files to {outputs.splits}")
    print(f"Saved raw audit: {outputs.audit}")
    print(f"Saved NN dataset: {outputs.nn}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("robot", nargs="?", default="both", choices=(*ROBOTS, "both"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="dataset settings JSON")
    parser.add_argument("--webots", type=Path, help="path to webots or webots.exe")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings(args.config)
    robots = ROBOTS if args.robot == "both" else (args.robot,)
    plans = {robot: outputs_for(settings, robot) for robot in robots}
    for robot in robots:
        check_outputs(plans[robot], robot)
    webots = find_webots(args.webots)
    for robot in robots:
        generate(robot, settings, plans[robot], webots)


if __name__ == "__main__":
    sys.exit(main())
