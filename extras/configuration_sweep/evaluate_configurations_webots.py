"""Task 4.B: measure the Task 4.A models in Webots on the first 100 test rows of each arm.

Steps (handout page 12):
  1. take the first 100 rows of each arm's test.csv, in file order;
  2. predict the joints with each model (clipped to [-1, 1] and de-normalised,
     as the live controller does);
  3. command them in headless Webots and read the TCP from the Supervisor;
  4. compare the measured (x, y, z) with the test set's (x, y, z).

By default the models are the three Task 4.A picks saved by
evaluate_configurations.py. Add others with --models.

Outputs in reports/configuration_evaluation/:
  task4b_position.csv     6 position-error sets (model x arm), with the FK error
                          on the same 100 points and the Task 4.A full-test result
  task4b_measurements.csv every commanded, sensed and measured value per point
  test_sets/<robot>_test_first100.csv   the two reduced test sets

Usage:
    python extras/configuration_sweep/evaluate_configurations.py      # Task 4.A first: picks the models
    python extras/configuration_sweep/evaluate_configurations_webots.py
    python extras/configuration_sweep/evaluate_configurations_webots.py --models mlp_ned_scara_large_h64x64x64x64_cosine_per-robot
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.train_mlp import SPLITS
from scripts.webots_runner import run_webots

SWEEP_MODELS = ROOT / "extras" / "configuration_sweep" / "models"

RESULTS = ROOT / "reports" / "configuration_evaluation"
ROBOTS = ("ned", "scara")
SAMPLES = 100


def first_rows(robot):
    with (SPLITS / robot / "test.csv").open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = [row for _, row in zip(range(SAMPLES), reader)]
    if len(rows) != SAMPLES:
        raise ValueError(f"{robot} test.csv has only {len(rows)} rows; Task 4.B needs {SAMPLES}")
    return reader.fieldnames, rows


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=[], help="extra model names (file stems) to measure")
    parser.add_argument("--only", action="store_true", help="measure only --models, not the Task 4.A picks")
    parser.add_argument("--webots", help="Webots launcher, if not found on PATH or WEBOTS_HOME")
    parser.add_argument("--settle-time", type=float, default=0.25, help="simulated seconds the joints must stay within 1e-3 of the command")
    parser.add_argument("--timeout", type=float, default=10.0, help="maximum simulated seconds per target")
    return parser.parse_args()


def main():
    args = parse_args()
    results = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))
    by_name = {m["name"]: m for m in results["models"]}
    picks = {name: size for size, name in results["task4a_selection"].items()}
    names = ([] if args.only else list(picks)) + [n for n in args.models if n not in picks]
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise SystemExit(f"Not in results.json (run evaluate_configurations.py first): {unknown}")

    cases = {}
    for robot in ROBOTS:
        fieldnames, rows = first_rows(robot)
        write_csv(RESULTS / "test_sets" / f"{robot}_test_first100.csv", fieldnames, rows)
        cases[robot] = [{"label": f"test_{i + 1:04d}", "kind": "task4b_first_100",
                         "point": (row["x"], row["y"], row["z"])} for i, row in enumerate(rows)]

    summary, measurements = [], []
    for name in names:
        model = by_name[name]
        # Sweep checkpoints live in extras/configuration_sweep/models/ (sometimes in tanh/ or bigger/); the rest in models/.
        model_path = name if model["group"] == "." else f"{model['group']}/{name}"
        for robot in ROBOTS:
            print(f"{name} · {robot}: running {SAMPLES} targets in headless Webots...", flush=True)
            webots_args = SimpleNamespace(
                webots=args.webots, output=RESULTS / "webots_work",
                settle_time=args.settle_time, timeout=args.timeout,
                model_dir=SWEEP_MODELS if (SWEEP_MODELS / f"{model_path}.json").is_file() else None, wall_timeout=None,
            )
            rows = run_webots(robot, model_path, cases[robot], webots_args)
            if [row["label"] for row in rows] != [case["label"] for case in cases[robot]]:
                raise RuntimeError(f"Webots returned {len(rows)} rows for {name} · {robot}; expected all {SAMPLES} in order")

            webots_mm = np.array([float(row["webots_error_mm"]) for row in rows])
            fk_mm = np.array([float(row["fk_target_error_mm"]) for row in rows])
            gap_mm = np.array([float(row["fk_webots_gap_mm"]) for row in rows])
            settled = sum(row["settled"] == "True" for row in rows)
            task4a = model["robots"][robot]["position_mm"]
            summary.append({
                "task4a_pick": picks.get(name, ""), "model": name, "robot": robot, "samples": len(rows),
                "mean_mm": webots_mm.mean(), "std_mm": webots_mm.std(ddof=0),
                "median_mm": float(np.median(webots_mm)), "p95_mm": float(np.percentile(webots_mm, 95)),
                "max_mm": webots_mm.max(), "settled": settled,
                "fk_first100_mean_mm": fk_mm.mean(), "fk_first100_std_mm": fk_mm.std(ddof=0),
                "fk_webots_gap_mean_mm": gap_mm.mean(), "fk_webots_gap_max_mm": gap_mm.max(),
                "task4a_full_test_mean_mm": task4a["mean"], "task4a_full_test_std_mm": task4a["std"],
            })
            measurements += [{"model": name, **row} for row in rows]
            s = summary[-1]
            print(f"  Webots {s['mean_mm']:.3f} ± {s['std_mm']:.3f} mm · FK on same points {s['fk_first100_mean_mm']:.3f} mm"
                  f" · {settled}/{SAMPLES} settled", flush=True)

    try:
        (RESULTS / "webots_work").rmdir()  # each run removes its own sub-folder; drop the empty parent
    except OSError:
        pass
    write_csv(RESULTS / "task4b_position.csv", list(summary[0]), summary)
    write_csv(RESULTS / "task4b_measurements.csv", list(measurements[0]), measurements)
    print(f"Saved task4b_position.csv, task4b_measurements.csv and test_sets/ to {RESULTS}")


if __name__ == "__main__":
    main()
