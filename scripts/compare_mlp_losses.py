"""Evaluate saved joint-loss and TCP-loss MLPs on the same independent tests.

Run after training the desired sizes with scripts/train_mlp.py. This script evaluates
fixed checkpoints; it never trains, selects epochs, or tunes on the test set.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ir_comparison_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ir_comparison_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_mlp_evaluation import endpoint_summary, plot_task4a
from scripts.train_mlp import JOINT_LIMITS, denormalise, FORWARD_KINEMATICS
from scripts.evaluation_common import (
    ROOT, SIZES, ROBOTS, digest, model_name, load_test, predict_model,
    physical_joint_errors, position_statistics, write_csv, select_models, coverage,
)


def evaluate(directory, size, variant, robot, *, test_root=None, joint_rows=None, predictions=None):
    """Task 4.A on every test sample; retain optional physical joint details."""
    path = (test_root or ROOT / "data/nn/splits") / robot / "test.csv"
    x, y = load_test(path, robot)
    prediction, config = predict_model(directory, size, variant, robot, x)
    command = denormalise(np.clip(prediction, -1, 1), JOINT_LIMITS[robot])
    true_q = denormalise(y, JOINT_LIMITS[robot])
    fk_positions = np.array([FORWARD_KINEMATICS[robot](q) for q in command])
    errors = np.linalg.norm(fk_positions - x[:, 2:5], axis=1) * 1000.0
    joint_errors, units = physical_joint_errors(robot, true_q, command)
    row = {
        "size": size, "robot": robot, "variant": variant, "samples": len(x),
        **position_statistics(errors),
        "raw_joint_rmse_normalised": float(np.sqrt(np.mean((prediction - y) ** 2))),
        "clipped_joint_rmse_normalised": float(np.sqrt(np.mean((np.clip(prediction, -1, 1) - y) ** 2))),
        "clipped_predictions_pct": float(np.mean(np.any(np.abs(prediction) > 1.0, axis=1)) * 100),
        "best_epoch": config["best_epoch"],
    }
    if joint_rows is not None:
        for j, unit in enumerate(units):
            joint_rows.append({"size": size, "robot": robot, "variant": variant,
                "joint": f"q{j + 1}", "samples": len(x), "unit": unit,
                "error_definition": "squared_linear_difference" if unit == "m^2" else "signed_wrapped_angular_difference",
                "mean_error": float(joint_errors[:, j].mean()),
                "std_error": float(joint_errors[:, j].std(ddof=0))})
    if predictions is not None:
        for i in range(len(x)):
            predictions.append({"size": size, "robot": robot, "variant": variant, "sample_index": i + 1,
                **dict(zip(("target_x_m", "target_y_m", "target_z_m"), x[i, 2:5])),
                **{f"true_q{j+1}_physical": true_q[i, j] for j in range(3)},
                **{f"raw_q{j+1}_normalised": prediction[i, j] for j in range(3)},
                **{f"command_q{j+1}_physical": command[i, j] for j in range(3)},
                **dict(zip(("fk_x_m", "fk_y_m", "fk_z_m"), fk_positions[i])),
                "position_error_mm": errors[i],
                **{f"joint{j+1}_coursework_error": joint_errors[i, j] for j in range(3)}})
    provenance = {"size": size, "variant": variant, "robot": robot,
                  "weights_sha256": digest(directory / f"{model_name(size, variant)}.npz"),
                  "test_sha256": digest(path), "config": config}
    return row, provenance


def draw_comparison(rows, output):
    """Backward-compatible entry point for the endpoint overview."""
    endpoint_summary(rows, output, "4a")


def write_task4a_report(output, rows, joints, missing):
    lines = ["# Task 4.A: independent test evaluation", "",
        "All test samples; saved checkpoints only. TCP positions are calculated with FK after controller-equivalent joint-limit clipping.", "",
        "Statistics use population standard deviation (`ddof=0`). Angular error is signed `atan2(sin(true-predicted), cos(true-predicted))` in rad. SCARA lift uses the handout's squared displacement error in m². Normalised joint RMSE is additional and does not replace these physical per-joint statistics.", "",
        "## Position error", "", "| Size | Model | Robot | Samples | Mean (mm) | Std (mm) | RMSE (mm) |",
        "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['size']} | {row['variant']} | {row['robot']} | {row['samples']} | "
                     f"{row['mean_tcp_mm']:.4f} | {row['std_tcp_mm']:.4f} | {row['rmse_tcp_mm']:.4f} |")
    lines += ["", "## Per-joint coursework errors", "", "Signed angular means describe bias; cancellation can make a mean small despite a large standard deviation.", "",
              "| Size | Model | Robot | Joint | Unit | Mean | Std |", "|---|---|---|---|---|---:|---:|"]
    for row in joints:
        lines.append(f"| {row['size']} | {row['variant']} | {row['robot']} | {row['joint']} | {row['unit']} | "
                     f"{row['mean_error']:.6g} | {row['std_error']:.6g} |")
    lines += ["", "## Coverage", "", "For a complete model family Task 4.A needs three sizes × two arms: six position summaries and eighteen joint summaries."]
    if missing:
        lines += ["", "Unavailable selected checkpoints (not evaluated): " + ", ".join(f"{s}/{v}" for s, v in missing) + "."]
    lines += ["", "See `predictions.csv` for every network prediction and error, `joint_errors.csv` for per-joint summaries, and `metrics.json` for model/data hashes and model-family coverage. Task 4.B is a separate Webots measurement; these FK results do not replace it.", ""]
    (output / "task4a.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--test-root", type=Path, default=ROOT / "data/nn/splits")
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--losses", nargs="+", choices=("joints", "tcp"), default=["joints", "tcp"])
    parser.add_argument("--joint-repeat-dir", type=Path,
                        help="optional training root containing <size>/joints/<model> artifacts")
    parser.add_argument("--include-large-baseline", action="store_true",
                        help="legacy option: large joint baseline is already included by default")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/task4a")
    args = parser.parse_args()
    available, missing = select_models(args.model_dir, args.sizes, args.losses)
    if args.include_large_baseline and ("large", "existing_joints") not in available:
        extra, _ = select_models(args.model_dir, ["large"], ["joints"])
        available.extend(extra)
    for size, variant in missing:
        print(f"NOT EVALUATED: missing checkpoint {model_name(size, variant)}; recorded in coverage")
    rows, provenance, joints, predictions = [], [], [], []
    runs = [(size, variant, args.model_dir) for size, variant in available]
    if args.joint_repeat_dir is not None:
        for size in args.sizes:
            directory = args.joint_repeat_dir / size / "joints"
            if (directory / f"{model_name(size, 'joint_repeat')}.npz").is_file():
                runs.append((size, "joint_repeat", directory))
    for size, variant, directory in runs:
        for robot in ROBOTS:
            row, source = evaluate(directory, size, variant, robot, test_root=args.test_root,
                                   joint_rows=joints, predictions=predictions)
            rows.append(row)
            provenance.append(source)
            print(f"{size:6s} {robot:5s} {variant:16s}: mean ± std "
                  f"{row['mean_tcp_mm']:.3f} ± {row['std_tcp_mm']:.3f} mm; RMSE {row['rmse_tcp_mm']:.3f} mm")
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "summary.csv", rows)
    write_csv(args.output / "joint_errors.csv", joints)
    write_csv(args.output / "predictions.csv", predictions)
    metadata = {"task": "4.A", "metrics": rows, "joint_metrics": joints, "provenance": provenance,
        "missing_models": [{"size": s, "variant": v} for s, v in missing],
        "coursework_coverage": coverage(available, args.losses),
        "completed_position_summaries": len(rows), "completed_joint_summaries": len(joints),
        "std_ddof": 0, "prediction_policy": "clip normalised joints to [-1,1] before FK and physical joint errors"}
    (args.output / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    plot_task4a(rows, joints, predictions, args.output)
    write_task4a_report(args.output, rows, joints, missing)
    with (args.output / "task4a.md").open("a", encoding="utf-8") as destination:
        destination.write("\n[Graphs and how to read them](graphs.md)\n")
    print(f"Saved Task 4.A results to {args.output}")


if __name__ == "__main__":
    main()
