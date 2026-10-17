"""Evaluate saved joint-loss and TCP-loss MLPs on the same independent tests.

Run after training both sizes with scripts/train_mlp.py. This script evaluates
fixed checkpoints; it never trains, selects epochs, or tunes on the test set.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ir_comparison_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ir_comparison_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_mlp import INPUTS, OUTPUTS, MLP, position_errors


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(directory, size, variant, robot):
    suffix = "_tcp" if variant == "tcp" else ""
    name = f"mlp_ned_scara_{size}_h64x64_cosine{suffix}"
    config = json.loads((directory / f"{name}.json").read_text())
    weights_path = directory / f"{name}.npz"
    with np.load(weights_path) as stored:
        model = MLP((len(INPUTS), *config["hidden"], len(OUTPUTS)), np.random.default_rng(0))
        model.params = {key: stored[key].copy() for key in model.params}
        mean, std = stored["mean"], stored["std"]
    path = ROOT / "data/nn/splits" / robot / "test.csv"
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    x = np.array([[float(row[key]) for key in INPUTS] for row in rows])
    y = np.array([[float(row[key]) for key in OUTPUTS] for row in rows])
    prediction = model.predict((x - mean) / std)
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite predictions: {name}")
    errors = position_errors(x, prediction, [robot]) * 1000.0
    row = {
        "size": size, "robot": robot, "variant": variant, "samples": len(rows),
        "mean_tcp_mm": float(errors.mean()), "median_tcp_mm": float(np.median(errors)),
        "p95_tcp_mm": float(np.percentile(errors, 95)), "max_tcp_mm": float(errors.max()),
        "rmse_tcp_mm": float(np.sqrt(np.mean(errors ** 2))),
        "within_5mm_pct": float(np.mean(errors <= 5.0) * 100),
        "within_10mm_pct": float(np.mean(errors <= 10.0) * 100),
        "raw_joint_rmse_normalised": float(np.sqrt(np.mean((prediction - y) ** 2))),
        "clipped_joint_rmse_normalised": float(np.sqrt(np.mean((np.clip(prediction, -1, 1) - y) ** 2))),
        "clipped_predictions_pct": float(np.mean(np.any(np.abs(prediction) > 1.0, axis=1)) * 100),
        "best_epoch": config["best_epoch"],
    }
    provenance = {"size": size, "variant": variant, "robot": robot,
                  "weights_sha256": digest(weights_path), "test_sha256": digest(path),
                  "config": config}
    return row, provenance


def draw_comparison(rows, output):
    cases = [(size, robot) for size in ("small", "medium") for robot in ("ned", "scara")]
    lookup = {(r["size"], r["robot"], r["variant"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, metric, title in zip(axes, ("mean_tcp_mm", "p95_tcp_mm"),
                               ("Average endpoint error", "95% of test points fall within this error")):
        largest = 0
        for variant, offset, color, label in (
            ("existing_joints", -0.13, "#2878b5", "Existing model: joint loss"),
            ("tcp", 0.13, "#e07a3f", "New model: TCP loss"),
        ):
            values = [lookup[(size, robot, variant)][metric] for size, robot in cases]
            largest = max(largest, max(values))
            positions = np.arange(len(cases)) + offset
            ax.scatter(values, positions, color=color, label=label, s=65)
            for value, position in zip(values, positions):
                ax.annotate(f"{value:.2f}", (value, position), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=9, color=color)
        ax.set_xlim(0, largest * 1.28)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("TCP error (mm) — smaller is better")
        ax.set_yticks(np.arange(len(cases)), [f"{s.capitalize()} · {r.upper()}" for s, r in cases])
        ax.set_ylim(len(cases) - 0.5, -0.5)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Joint-loss vs TCP-loss training: independent test accuracy", fontsize=13)
    fig.text(0.5, 0.02, "Same 1,000 test targets per robot. Endpoint errors calculated with FK; not Webots measurements.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.06, 0.99, 0.93))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--joint-repeat-dir", type=Path,
                        help="optional training root containing <size>/joints/<model> artifacts")
    parser.add_argument("--include-large-baseline", action="store_true",
                        help="also score the existing large joint-loss model as an unmatched-size reference")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/tcp_loss_comparison")
    args = parser.parse_args()
    rows, provenance = [], []
    for size in ("small", "medium"):
        variants = [("existing_joints", args.model_dir), ("tcp", args.model_dir)]
        if args.joint_repeat_dir is not None:
            variants.append(("joint_repeat", args.joint_repeat_dir / size / "joints"))
        for variant, directory in variants:
            for robot in ("ned", "scara"):
                row, source = evaluate(directory, size, variant, robot)
                rows.append(row)
                provenance.append(source)
                print(f"{size:6s} {robot:5s} {variant:16s}: mean {row['mean_tcp_mm']:.3f} mm; "
                      f"P95 {row['p95_tcp_mm']:.3f} mm; max {row['max_tcp_mm']:.3f} mm")
    if args.include_large_baseline:
        for robot in ("ned", "scara"):
            row, source = evaluate(args.model_dir, "large", "existing_joints", robot)
            rows.append(row)
            provenance.append(source)
            print(f"large  {robot:5s} existing_joints : mean {row['mean_tcp_mm']:.3f} mm; "
                  f"P95 {row['p95_tcp_mm']:.3f} mm; max {row['max_tcp_mm']:.3f} mm")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "metrics.json").write_text(json.dumps({"metrics": rows, "provenance": provenance}, indent=2) + "\n")
    draw_comparison(rows, args.output / "comparison.png")
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
