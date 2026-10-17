"""Plot Task 4.A/4.B exports without retraining or rerunning Webots.

The evaluation scripts also call these functions automatically. Run directly
with --task4a PATH and/or --task4b PATH to redraw existing results.
"""

import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ir_evaluation_plots"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ir_evaluation_plot_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator, FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluation_common import SIZES, ROBOTS

VARIANTS = (
    ("existing_joints", "Joint-loss model", "#2878b5"),
    ("tcp", "TCP-loss model", "#e07a3f"),
    ("joint_repeat", "Joint-loss repeat", "#777777"),
)


def variants_in(rows):
    present = {r["variant"] for r in rows}
    return [v for v in VARIANTS if v[0] in present]


def save(fig, path, bottom=0.11):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.01, bottom, 0.99, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def dot_panel(ax, rows, cases, metric, variants, *, signed=False, factor=1.0, precision=".2f"):
    lookup = {(r["size"], r["robot"], r["variant"]): r for r in rows}
    offsets = np.linspace(-0.14, 0.14, len(variants)) if len(variants) > 1 else [0]
    values_seen = [0.0]
    for (variant, label, color), offset in zip(variants, offsets):
        points = []
        for index, (size, robot) in enumerate(cases):
            row = lookup.get((size, robot, variant))
            if row is None:
                ax.text(0.03, index + offset, f"{'TCP' if variant == 'tcp' else 'Joint'} not evaluated", transform=ax.get_yaxis_transform(),
                        color=color, va="center", fontsize=8)
                continue
            value = float(row[metric]) * factor
            points.append((value, index + offset))
            values_seen.append(value)
        if points:
            ax.scatter(*np.array(points).T, s=55, color=color, label=label, zorder=3)
            for value, y in points:
                ax.annotate(format(value, precision), (value, y), xytext=(6, 0),
                            textcoords="offset points", va="center", color=color, fontsize=8)
    lo, hi = min(values_seen), max(values_seen)
    span = max(hi - lo, 1e-6)
    ax.set_xlim(lo - span * 0.1 if signed else 0, hi + span * 0.4)
    if signed:
        ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(cases)), [f"{s.capitalize()} · {r.upper()}" for s, r in cases])
    ax.set_ylim(len(cases) - 0.5, -0.5)
    ax.grid(axis="x", alpha=0.2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def endpoint_summary(rows, output, task):
    cases = [(s, r) for s in SIZES for r in ROBOTS if any(v["size"] == s and v["robot"] == r for v in rows)]
    variants = variants_in(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, 0.65 * len(cases) + 1)), sharey=True)
    for ax, metric, title in zip(axes, ("mean_tcp_mm", "std_tcp_mm"),
                               ("Mean endpoint error", "Standard deviation of endpoint error")):
        dot_panel(ax, rows, cases, metric, variants)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("TCP error (mm)")
    axes[0].legend(loc="lower right", fontsize=8)
    stage = "4.A — calculated with FK" if task == "4a" else "4.B — measured in Webots"
    fig.suptitle(f"Task {stage}: endpoint accuracy", fontsize=13)
    sample_note = "All test rows" if task == "4a" else "First 100 test rows per arm; motor timeouts are included"
    fig.text(0.5, 0.025, sample_note + ". Lower mean = closer to target; lower std = less variation. Missing models are labelled.",
             ha="center", fontsize=8)
    save(fig, output)


def endpoint_rmse(rows, output, task):
    robots = [r for r in ROBOTS if any(row["robot"] == r for row in rows)]
    fig, axes = plt.subplots(1, len(robots), figsize=(6 * len(robots), 4.5), squeeze=False)
    for ax, robot in zip(axes[0], robots):
        subset = [r for r in rows if r["robot"] == robot]
        cases = [(s, robot) for s in SIZES if any(r["size"] == s for r in subset)]
        dot_panel(ax, subset, cases, "rmse_tcp_mm", variants_in(rows))
        ax.set_title(robot.upper())
        ax.set_xlabel("TCP RMSE (mm) — smaller is better")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Task {'4.A: FK' if task == '4a' else '4.B: Webots'} endpoint RMSE", fontsize=13)
    fig.text(0.5, 0.025, "RMSE gives larger misses more weight. Units are millimetres, not normalised joint values.",
             ha="center", fontsize=9)
    save(fig, output)


def joint_statistics(rows, output, metric):
    robots = [r for r in ROBOTS if any(row["robot"] == r for row in rows)]
    fig, axes = plt.subplots(len(robots), 3, figsize=(15, 3.5 * len(robots)), squeeze=False)
    names = {"ned": ("waist", "shoulder", "elbow"), "scara": ("base", "elbow", "lift")}
    variants = variants_in(rows)
    for i, robot in enumerate(robots):
        for j in range(3):
            ax = axes[i, j]
            subset = [r for r in rows if r["robot"] == robot and r["joint"] == f"q{j+1}"]
            cases = [(s, robot) for s in SIZES if any(r["size"] == s for r in subset)]
            lift = robot == "scara" and j == 2
            dot_panel(ax, subset, cases, metric, variants, signed=metric == "mean_error" and not lift,
                      factor=1e6 if lift else 1, precision=".3g")
            ax.set_yticks(range(len(cases)), [size.capitalize() for size, _ in cases])
            ax.set_title(f"{robot.upper()} · q{j+1} ({names[robot][j]})", fontsize=11)
            if lift:
                ax.set_xlabel(("Mean" if metric == "mean_error" else "Std") + " of squared lift error (mm²)", fontsize=9)
            else:
                ax.set_xlabel("Mean signed angular error (rad)" if metric == "mean_error" else "Angular error std (rad)", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.09), ncol=len(variants), fontsize=9)
    fig.suptitle("Task 4.A: per-joint " + ("mean error (bias)" if metric == "mean_error" else "error standard deviation"), fontsize=13)
    note = ("Angular means can cancel: near zero means little bias, not necessarily accurate predictions. Read the std graph too."
            if metric == "mean_error" else "Lower std means less variation; read alongside mean bias. Different joints use separate axes and units.")
    fig.text(0.5, 0.04, note, ha="center", fontsize=9)
    fig.text(0.5, 0.015, "Handout definitions: signed wrapped angles; squared SCARA lift error. Lift values converted from m² to mm² (×1,000,000).",
             ha="center", fontsize=8)
    save(fig, output, bottom=0.18)


def accuracy_curves(predictions, output, task):
    field = "position_error_mm" if task == "4a" else "webots_error_mm"
    robots = [r for r in ROBOTS if any(row["robot"] == r for row in predictions)]
    sizes = [s for s in SIZES if any(row["size"] == s for row in predictions)]
    all_errors = np.array([float(row[field]) for row in predictions])
    positive = all_errors[all_errors > 0]
    lo = max(float(positive.min()) / 2, 1e-6) if len(positive) else 0.001
    hi = max(float(all_errors.max()) * 1.2, lo * 10)
    fig, axes = plt.subplots(len(sizes), len(robots), figsize=(6 * len(robots), 2.9 * len(sizes) + 1), squeeze=False)
    for i, size in enumerate(sizes):
        for j, robot in enumerate(robots):
            ax = axes[i, j]
            for variant, label, color in variants_in(predictions):
                errors = sorted(float(r[field]) for r in predictions if (r["size"], r["robot"], r["variant"]) == (size, robot, variant))
                if not errors:
                    ax.text(0.04, 0.9, f"{label}: not evaluated", transform=ax.transAxes,
                            color=color, fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
                    continue
                values = np.maximum(errors, lo)
                ax.step(np.r_[lo, values, hi], np.r_[0, np.arange(1, len(values)+1) / len(values) * 100, 100],
                        where="post", color=color, linewidth=2, label=label)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, 102)
            ax.set_title(f"{robot.upper()} · {size}")
            ax.set_xlabel("Allowed TCP error (mm; log scale)", fontsize=9)
            ax.set_ylabel("Test points within this error (%)", fontsize=9)
            ax.grid(alpha=0.2)
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Task {'4.A: FK' if task == '4a' else '4.B: Webots'}: how many targets are reached accurately?", fontsize=13)
    fig.text(0.5, 0.02, "Pick an allowed error on the horizontal axis; read the percentage reached within it. Higher is better at the same threshold.",
             ha="center", fontsize=9)
    save(fig, output, bottom=0.09)


def fk_vs_webots(predictions, output):
    robots = [r for r in ROBOTS if any(row["robot"] == r for row in predictions)]
    sizes = [s for s in SIZES if any(row["size"] == s for row in predictions)]
    fig, axes = plt.subplots(len(sizes), len(robots), figsize=(6 * len(robots), 3 * len(sizes) + 1), squeeze=False)
    for i, size in enumerate(sizes):
        for j, robot in enumerate(robots):
            ax = axes[i, j]
            limit = 0.001
            for variant, label, color in variants_in(predictions):
                rows = [r for r in predictions if (r["size"], r["robot"], r["variant"]) == (size, robot, variant)]
                if not rows:
                    ax.text(0.04, 0.9, f"{label}: not evaluated", transform=ax.transAxes,
                            color=color, fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
                    continue
                x = [float(r["fk_target_error_mm"]) for r in rows]
                y = [float(r["webots_error_mm"]) for r in rows]
                limit = max(limit, max(x), max(y))
                ax.scatter(x, y, s=14, alpha=0.65, color=color, label=label)
            limit *= 1.1
            ax.plot([0, limit], [0, limit], "--", color="#555555", linewidth=1, label="Equal error")
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)
            ax.set_title(f"{robot.upper()} · {size}")
            ax.set_xlabel("FK-calculated TCP error (mm)", fontsize=9)
            ax.set_ylabel("Webots-measured TCP error (mm)", fontsize=9)
            ax.grid(alpha=0.2)
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Task 4.B: does Webots confirm the FK endpoint error?", fontsize=13)
    fig.text(0.5, 0.02, "Each dot uses the same target in both calculations. Above the diagonal: Webots error is larger; below: smaller. Same first 100 points.",
             ha="center", fontsize=8)
    save(fig, output, bottom=0.09)


def plot_task4a(summary, joints, predictions, output):
    output = Path(output)
    endpoint_summary(summary, output / "comparison.png", "4a")
    endpoint_rmse(summary, output / "tcp_rmse.png", "4a")
    joint_statistics(joints, output / "joint_mean_errors.png", "mean_error")
    joint_statistics(joints, output / "joint_error_std.png", "std_error")
    accuracy_curves(predictions, output / "tcp_accuracy_curves.png", "4a")
    (output / "graphs.md").write_text("# Task 4.A graphs\n\n"
        "Blue = joint-loss model; orange = TCP-loss model. Missing checkpoints are not zero errors.\n\n"
        "- [Endpoint mean and standard deviation](comparison.png): closer to the target and variation across test points.\n"
        "- [Endpoint RMSE](tcp_rmse.png): gives larger misses more weight.\n"
        "- [Per-joint mean errors](joint_mean_errors.png): signed angular bias; SCARA lift uses squared error.\n"
        "- [Per-joint standard deviations](joint_error_std.png): read with mean bias; do not compare different units.\n"
        "- [Accuracy curves](tcp_accuracy_curves.png): percentage of test points below each allowed error.\n\n"
        "All test rows are included. FK positions are calculated, not measured in Webots.\n", encoding="utf-8")


def plot_task4b(summary, predictions, output):
    output = Path(output)
    endpoint_summary(summary, output / "comparison.png", "4b")
    endpoint_rmse(summary, output / "tcp_rmse.png", "4b")
    accuracy_curves(predictions, output / "tcp_accuracy_curves.png", "4b")
    fk_vs_webots(predictions, output / "fk_vs_webots.png")
    (output / "graphs.md").write_text("# Task 4.B graphs\n\n"
        "Blue = joint-loss model; orange = TCP-loss model. All measured points, including any motor timeouts, count. Check summary.csv for settling status.\n\n"
        "- [Measured endpoint mean and standard deviation](comparison.png).\n"
        "- [Measured endpoint RMSE](tcp_rmse.png).\n"
        "- [Accuracy curves](tcp_accuracy_curves.png): percentage below each allowed measured error.\n"
        "- [FK versus measured error](fk_vs_webots.png): paired errors on exactly the same first 100 test points per arm.\n\n"
        "The diagonal means equal error magnitudes, not identical positions. Do not compare first-100 Webots results directly with an FK full-test average as if they used the same samples.\n", encoding="utf-8")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task4a", type=Path, help="existing Task 4.A output folder")
    parser.add_argument("--task4b", type=Path, help="existing Task 4.B output folder")
    args = parser.parse_args()
    if args.task4a is None and args.task4b is None:
        args.task4a, args.task4b = ROOT / "outputs/task4a", ROOT / "outputs/task4b"
    if args.task4a is not None:
        path = args.task4a
        plot_task4a(read_rows(path / "summary.csv"), read_rows(path / "joint_errors.csv"),
                    read_rows(path / "predictions.csv"), path)
        print(f"Task 4.A graphs saved to {path}")
    if args.task4b is not None:
        path = args.task4b
        plot_task4b(read_rows(path / "summary.csv"), read_rows(path / "predictions.csv"), path)
        print(f"Task 4.B graphs saved to {path}")


if __name__ == "__main__":
    main()
