"""Task 4.B: execute the first 100 test samples per arm in Webots.

Uses the saved networks, controller-equivalent clipping, joint sensors and
Supervisor TCP measurements. No FK values substitute for measured TCPs.
The models evaluated are the ones listed in configs/evaluate_task4a.json.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.task4_common import (
    ROOT, ROBOTS, digest, load_test, predict_model,
    position_statistics, write_csv, draw_graphs,
)
from scripts.robot_config import JOINT_LIMITS, denormalise
from scripts.webots_runner import run_webots


def first_100(test_root, robot):
    path = test_root / robot / "test.csv"
    x, _ = load_test(path, robot, count=100)
    with path.open(newline="", encoding="utf-8") as source:
        reduced_rows = list(csv.DictReader(source))[:100]
    cases = [{"label": f"test_{i + 1:04d}", "kind": "coursework_first_100",
              "point": tuple(row[2:5])} for i, row in enumerate(x)]
    return x, cases, reduced_rows


def validate_measurements(rows, cases, expected_raw, robot):
    """Reject incomplete, reordered or stale output instead of scoring it."""
    if len(rows) != 100 or len(cases) != 100:
        raise RuntimeError(f"Task 4.B requires 100 measured rows, received {len(rows)}")
    numeric = ["x", "y", "z", "raw_q1", "raw_q2", "raw_q3",
               "command_q1", "command_q2", "command_q3", "measured_q1", "measured_q2", "measured_q3",
               "fk_x", "fk_y", "fk_z", "webots_x", "webots_y", "webots_z", "settle_seconds"]
    expected_command = denormalise(np.clip(expected_raw, -1, 1), JOINT_LIMITS[robot])
    for i, (row, case) in enumerate(zip(rows, cases)):
        if row["label"] != case["label"] or row["robot"] != robot:
            raise RuntimeError(f"Wrong order or robot in Webots row {i + 1}")
        if not all(math.isfinite(float(row[key])) for key in numeric):
            raise RuntimeError(f"Nonfinite measurement in Webots row {i + 1}")
        if row["settled"].lower() not in ("true", "false"):
            raise RuntimeError(f"Invalid settling status in row {i + 1}")
        if not np.allclose([float(row[a]) for a in "xyz"], case["point"], atol=1e-12, rtol=0):
            raise RuntimeError(f"Wrong target in Webots row {i + 1}")
        if not np.allclose([float(row[f"raw_q{j}"]) for j in (1, 2, 3)], expected_raw[i], atol=1e-9, rtol=1e-9):
            raise RuntimeError(f"Webots loaded different model predictions in row {i + 1}")
        if not np.allclose([float(row[f"command_q{j}"]) for j in (1, 2, 3)], expected_command[i], atol=1e-9, rtol=1e-9):
            raise RuntimeError(f"Wrong joint command in Webots row {i + 1}")


def summarize(size, variant, name, robot, measured):
    targets = np.array([[float(row[a]) for a in "xyz"] for row in measured])
    actual = np.array([[float(row[f"webots_{a}"]) for a in "xyz"] for row in measured])
    fk = np.array([[float(row[f"fk_{a}"]) for a in "xyz"] for row in measured])
    errors = np.linalg.norm(actual - targets, axis=1) * 1000
    fk_errors = np.linalg.norm(fk - targets, axis=1) * 1000
    settled = sum(row["settled"].lower() == "true" for row in measured)
    return {"size": size, "variant": variant, "model": name, "robot": robot, "samples": len(measured),
        **position_statistics(errors), "settled_count": settled, "timeout_count": len(measured) - settled,
        "all_samples_settled": settled == len(measured),
        "fk_first100_mean_mm": float(fk_errors.mean()), "fk_first100_std_mm": float(fk_errors.std(ddof=0)),
        "mean_fk_webots_gap_mm": float(np.linalg.norm(fk - actual, axis=1).mean() * 1000)}


def save_results(args, summary, predictions, metadata):
    if summary:
        write_csv(args.output / "summary.csv", summary)
        write_csv(args.output / "predictions.csv", predictions)
    else:
        # Do not leave prior measurements beside a new, failed run's manifest.
        for name in ("summary.csv", "predictions.csv"):
            (args.output / name).unlink(missing_ok=True)
    metadata = {**metadata, "completed_position_summaries": len(summary),
                "all_samples_settled": bool(summary) and all(row["all_samples_settled"] for row in summary)}
    (args.output / "metrics.json").write_text(json.dumps({**metadata, "metrics": summary}, indent=2) + "\n", encoding="utf-8")
    lines = ["# Task 4.B: Webots endpoint evaluation", "",
        "Exactly the first 100 rows of each original test.csv, in file order. All measured samples, including motor timeouts, are included. No FK substitution. Standard deviation uses ddof=0.", "",
        "| Size | Model | Robot | Mean TCP (mm) | Std TCP (mm) | Settled / 100 | FK mean on same 100 (mm) |",
        "|---|---|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['size']} | {row['variant']} | {row['robot']} | {row['mean_tcp_mm']:.4f} | "
                     f"{row['std_tcp_mm']:.4f} | {row['settled_count']} | {row['fk_first100_mean_mm']:.4f} |")
    lines += ["", f"Execution complete: {metadata['execution_complete']}. See metrics.json for missing checkpoints, robot coverage and any run failure. Timeouts are explicitly counted; results with timeouts are not all settled poses.", ""]
    (args.output / "task4b.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--test-root", type=Path, default=ROOT / "data/nn/splits")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluate_task4a.json",
                        help="JSON list of models to evaluate (shared with evaluate_task4a.py)")
    parser.add_argument("--robot", choices=(*ROBOTS, "both"), default="both")
    parser.add_argument("--webots", type=Path, help="Webots launcher, including Windows webots.exe")
    parser.add_argument("--settle-time", type=float, default=0.25, help="continuous simulated seconds within joint tolerance")
    parser.add_argument("--timeout", type=float, default=10.0, help="maximum simulated seconds per target")
    parser.add_argument("--wall-timeout", type=float, default=600.0, help="maximum real seconds per model/robot process")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/task4b")
    args = parser.parse_args()
    if not 0 < args.settle_time <= args.timeout or not all(math.isfinite(v) for v in (args.timeout, args.wall_timeout)) or args.wall_timeout <= 0:
        parser.error("Require 0 < settle-time <= finite timeout and a positive finite wall-timeout")
    args.output = args.output.resolve()
    args.model_dir = args.model_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    models = json.loads(args.config.read_text(encoding="utf-8"))["models"]
    available = [m for m in models if all((args.model_dir / f"{m['name']}.{ext}").is_file() for ext in ("json", "npz"))]
    missing = [m for m in models if m not in available]
    if not available:
        raise FileNotFoundError(f"No configured checkpoints found in {args.model_dir}")
    robots = ROBOTS if args.robot == "both" else (args.robot,)
    for model in missing:
        print(f"NOT EVALUATED: missing checkpoint {model['name']}; recorded in coverage")
    inputs, cases = {}, {}
    for robot in robots:
        inputs[robot], cases[robot], reduced = first_100(args.test_root, robot)
        write_csv(args.output / "test_sets" / f"{robot}_test_first100.csv", reduced)
    # Load every model before starting any simulation to catch invalid files early.
    expected = {(m["name"], robot): predict_model(args.model_dir, m["size"], m["variant"], robot, inputs[robot], name=m["name"])[0]
                for m in available for robot in robots}
    summary, predictions = [], []
    metadata = {"task": "4.B", "execution_complete": False, "robots": list(robots), "std_ddof": 0,
        "test_selection": "first 100 file rows, no shuffle, no substitution",
        "config": str(args.config.resolve().relative_to(ROOT)) if args.config.resolve().is_relative_to(ROOT) else str(args.config),
        "both_arms_selected": set(robots) == set(ROBOTS),
        "missing_models": [{"name": m["name"]} for m in missing],
        "settle_time_s": args.settle_time, "timeout_s": args.timeout,
        "test_sha256": {r: digest(args.test_root / r / "test.csv") for r in robots},
        "weights_sha256": {m["name"]: digest(args.model_dir / f"{m['name']}.npz") for m in available}}
    save_results(args, summary, predictions, metadata)
    try:
        for model in available:
            size, variant, name = model["size"], model["variant"], model["name"]
            for robot in robots:
                measured = run_webots(robot, name, cases[robot], args)
                validate_measurements(measured, cases[robot], expected[name, robot], robot)
                summary.append(summarize(size, variant, name, robot, measured))
                predictions.extend({"size": size, "variant": variant, "model": name,
                                    "sample_index": i + 1, **row} for i, row in enumerate(measured))
                save_results(args, summary, predictions, metadata)
                row = summary[-1]
                print(f"{name} {robot}: {row['mean_tcp_mm']:.3f} ± {row['std_tcp_mm']:.3f} mm; "
                      f"{row['settled_count']}/100 settled", flush=True)
    except Exception as error:
        metadata["failure"] = str(error)
        save_results(args, summary, predictions, metadata)
        raise
    metadata["execution_complete"] = True
    save_results(args, summary, predictions, metadata)
    drawn = draw_graphs("plot_task4b", summary, predictions, args.output, args.model_dir)
    if drawn:
        with (args.output / "task4b.md").open("a", encoding="utf-8") as destination:
            destination.write("\n[Graphs and how to read them](graphs.md)\n")
    print(f"Saved Task 4.B results to {args.output}")


if __name__ == "__main__":
    main()
