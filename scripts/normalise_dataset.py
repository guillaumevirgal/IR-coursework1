"""Normalise raw Ned or SCARA Webots dataset joint values to [-1, 1]."""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOINT_LIMITS = {
    "ned": ((-2.8, 2.8), (-0.8, 0.8), (0.0, 1.3)),
    "scara": ((-0.73, 0.73), (0.0, 0.83), (-0.2, 0.0)),
}


def normalise(value, lower, upper):
    return 2.0 * (value - lower) / (upper - lower) - 1.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=JOINT_LIMITS)
    parser.add_argument("input", type=Path)
    parser.add_argument("--tcp-source", choices=("actual", "expected"), default="actual")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.input if args.input.is_absolute() else ROOT / args.input
    output = args.output or (
        ROOT / "data" / "normalised"
        / f"{args.robot}_{source.stem}_{args.tcp_source}_normalised.csv"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    prefix = "a" if args.tcp_source == "actual" else "e"
    with source.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        with output.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(("q1", "q2", "q3", "x", "y", "z"))
            count = 0
            for line_number, row in enumerate(reader, start=2):
                joints = [float(row[f"joint{index}"]) for index in range(1, 4)]
                values = [
                    normalise(value, lower, upper)
                    for value, (lower, upper) in zip(joints, JOINT_LIMITS[args.robot])
                ]
                if any(not -1.0 <= value <= 1.0 for value in values):
                    raise ValueError(f"line {line_number}: joint outside {args.robot} range")
                position = [float(row[f"{prefix}{axis}"]) for axis in "xyz"]
                writer.writerow(values + position)
                count += 1

    print(f"Normalised {count} rows from {source} to {output}")


if __name__ == "__main__":
    main()
