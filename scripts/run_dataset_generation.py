"""Run physics-based dataset generation in headless Webots fast mode."""

import argparse
import csv
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORLDS = {"ned": "articulated.wbt", "scara": "scara.wbt"}
NN_FIELDS = ("is_scara", "is_ned", "x", "y", "z", "q1", "q2", "q3")


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=(*WORLDS, "both"))
    parser.add_argument("samples", nargs="?", type=int, default=16000)
    parser.add_argument("seed", nargs="?", type=int, default=2026)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--webots", type=Path, help="path to webots or webots.exe")
    parser.add_argument("--small", type=int, default=1000, help="small dataset size before 80/20 split")
    parser.add_argument("--medium", type=int, default=5000, help="medium dataset size before 80/20 split")
    parser.add_argument("--large", type=int, default=15000, help="large dataset size before 80/20 split")
    parser.add_argument("--test", type=int, default=1000, help="independent test-set size")
    return parser.parse_args()


def write_dataset(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=NN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def create_splits(master_path, output_dir, robot, *, small, medium, large, test, seed):
    """Create nested 80/20 train/validation sets and a separate test set."""

    sizes = (small, medium, large)
    if any(size < 5 or size % 5 for size in sizes):
        raise ValueError("Split sizes must be positive multiples of 5 for an exact 80/20 split")
    if not small <= medium <= large or test < 1:
        raise ValueError("Require small <= medium <= large and a positive test size")

    with master_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != NN_FIELDS:
            raise ValueError(f"Expected NN columns {NN_FIELDS} in {master_path}")
        rows = list(reader)
    expected_flags = ("1", "0") if robot == "scara" else ("0", "1")
    for line, row in enumerate(rows, start=2):
        if (row["is_scara"], row["is_ned"]) != expected_flags:
            raise ValueError(f"Wrong robot flag on line {line} of {master_path}")
        joints = [float(row[name]) for name in ("q1", "q2", "q3")]
        if any(value < -1.0 or value > 1.0 for value in joints):
            raise ValueError(f"Normalised joint outside [-1, 1] on line {line}")

    needed = test + large
    if len(rows) < needed:
        raise ValueError(f"Need {needed} master rows for splits, found {len(rows)}")
    random.Random(seed).shuffle(rows)
    test_rows = rows[:test]
    development = rows[test:needed]
    training = [row for index, row in enumerate(development) if index % 5 != 0]
    validation = [row for index, row in enumerate(development) if index % 5 == 0]

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite split directory: {output_dir}")
    write_dataset(output_dir / "test.csv", test_rows)
    for label, size in (("small", small), ("medium", medium), ("large", large)):
        count = size // 5
        write_dataset(output_dir / f"train_{label}.csv", training[: count * 4])
        write_dataset(output_dir / f"validation_{label}.csv", validation[:count])
    print(f"Wrote {robot} train/validation/test files to {output_dir}")


def temporary_world(robot, samples, seed, output, nn_output):
    """Create a world beside the originals so relative PROTO paths still work."""

    source = ROOT / "worlds" / WORLDS[robot]
    text = source.read_text(encoding="utf-8")
    marker = f'    "{robot}"'
    if text.count(marker) != 1:
        raise RuntimeError(f"Expected one {robot!r} controller argument in {source}")

    extra_arguments = (
        f'{marker}\n'
        f'    "--samples={samples}"\n'
        f'    "--seed={seed}"\n'
        f'    "--output={output.as_posix()}"\n'
        f'    "--nn-output={nn_output.as_posix()}"'
    )
    text = text.replace(marker, extra_arguments, 1)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".wbt",
        prefix=f"dataset_{robot}_",
        dir=source.parent,
        delete=False,
    )
    with handle:
        handle.write(text)
    return Path(handle.name)


def remove_temporary_world(path):
    """Remove the temporary world and Webots project metadata it may create."""

    project_files = (
        path,
        path.with_suffix(".wbproj"),
        path.with_name(f".{path.stem}.wbproj"),
    )
    for project_file in project_files:
        try:
            project_file.unlink()
        except FileNotFoundError:
            pass


def main():
    args = parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")
    sizes = (args.small, args.medium, args.large)
    if any(size < 5 or size % 5 for size in sizes):
        raise ValueError("Split sizes must be positive multiples of 5")
    if not args.small <= args.medium <= args.large or args.test < 1:
        raise ValueError("Require small <= medium <= large and a positive test size")
    if args.samples < args.large + args.test:
        raise ValueError(
            f"Need at least {args.large + args.test} accepted samples for requested splits"
        )

    webots = find_webots(args.webots)
    robots = tuple(WORLDS) if args.robot == "both" else (args.robot,)
    if args.output and len(robots) != 1:
        raise ValueError("A custom raw output path can only be used for one robot")

    outputs = {}
    for robot in robots:
        raw_path = args.output if args.output else ROOT / "data" / "raw" / f"{robot}_master_audit.csv"
        raw_path = raw_path.resolve()
        nn_path = ROOT / "data" / "nn" / f"{robot}_master.csv"
        if args.output:
            nn_path = raw_path.with_name(f"{raw_path.stem.removesuffix('_audit')}_nn.csv")
        nn_path = nn_path.resolve()
        split_dir = ROOT / "data" / "nn" / "splits" / robot
        for destination in (raw_path, nn_path, split_dir):
            if destination.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing dataset: {destination}. Choose another path."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
        outputs[robot] = (raw_path, nn_path, split_dir)

    for robot in robots:
        raw_path, nn_path, split_dir = outputs[robot]
        world = temporary_world(robot, args.samples, args.seed, raw_path, nn_path)
        command = [
            str(webots),
            "--batch",
            "--mode=fast",
            "--no-rendering",
            "--stdout",
            "--stderr",
            str(world),
        ]
        print(
            f"Generating {args.samples} {robot} rows in Webots fast mode...",
            flush=True,
        )
        try:
            subprocess.run(command, check=True)
        finally:
            remove_temporary_world(world)
        create_splits(
            nn_path,
            split_dir,
            robot,
            small=args.small,
            medium=args.medium,
            large=args.large,
            test=args.test,
            seed=args.seed + 1,
        )
        print(f"Saved raw audit: {raw_path}")
        print(f"Saved NN dataset: {nn_path}")


if __name__ == "__main__":
    main()
