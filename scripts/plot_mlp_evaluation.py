"""Plot Task 4.A/4.B exports without retraining or rerunning Webots.

The evaluation scripts also call these functions automatically. Run directly
with --task4a PATH and/or --task4b PATH to redraw existing results.
"""

import argparse
import csv
import json
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
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluation_common import SIZES, ROBOTS, load_test, model_name, predict_model
from scripts.robot_config import JOINT_LIMITS
from scripts.tcp_loss import tcp_loss_and_gradient

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


def endpoint_summary(rows, output, task):
    variants = variants_in(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    handles = []
    for i, robot in enumerate(ROBOTS):
        subset = [r for r in rows if r["robot"] == robot]
        if not subset:
            continue
        sizes = [size for size in SIZES if any(r["size"] == size for r in subset)]
        positions = np.arange(len(sizes), dtype=float)
        width = 0.78 / max(len(variants), 1)
        for j, (metric, title) in enumerate((
            ("mean_tcp_mm", "Mean endpoint error"),
            ("std_tcp_mm", "Standard deviation of error"),
        )):
            ax = axes[i, j]
            for variant_index, (variant, label, color) in enumerate(variants):
                vals = [next((float(r[metric]) for r in subset
                              if r["size"] == size and r["variant"] == variant), np.nan)
                        for size in sizes]
                x = positions + (variant_index - (len(variants) - 1) / 2) * width
                valid = np.isfinite(vals)
                bars = ax.bar(x[valid], np.asarray(vals)[valid], width=width * 0.9,
                              color=color, label=label, zorder=3)
                if len(bars):
                    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
                    if i == 0 and j == 0:
                        handles.append(bars)
            ax.set_title(f"{robot.upper()} · {title}", fontsize=10)
            ax.set_xticks(positions, [size.capitalize() for size in sizes])
            ax.set_xlabel("Training dataset size")
            ax.set_ylabel("TCP error (mm)")
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
    stage = "4.A — calculated with FK" if task == "4a" else "4.B — measured in Webots"
    fig.suptitle(f"Task {stage}: endpoint error by model", fontsize=13)
    if handles:
        fig.legend(handles, [label for _, label, _ in variants], loc="lower center",
                   bbox_to_anchor=(0.5, 0.04), ncol=len(variants), fontsize=8)
    sample_note = "All test rows" if task == "4a" else "First 100 test rows per arm; all settled in this run"
    fig.text(0.5, 0.012, sample_note + ". Lower bars mean smaller error; std shows variation.",
             ha="center", fontsize=8)
    save(fig, output, bottom=0.13)


def endpoint_rmse(rows, output, task):
    robots = [r for r in ROBOTS if any(row["robot"] == r for row in rows)]
    fig, axes = plt.subplots(1, len(robots), figsize=(5.5 * len(robots), 4.5), squeeze=False)
    variants = variants_in(rows)
    handles = []
    for ax, robot in zip(axes[0], robots):
        subset = [r for r in rows if r["robot"] == robot]
        sizes = [size for size in SIZES if any(r["size"] == size for r in subset)]
        positions = np.arange(len(sizes), dtype=float)
        width = 0.78 / max(len(variants), 1)
        for variant_index, (variant, label, color) in enumerate(variants):
            vals = [next((float(r["rmse_tcp_mm"]) for r in subset
                          if r["size"] == size and r["variant"] == variant), np.nan)
                    for size in sizes]
            x = positions + (variant_index - (len(variants) - 1) / 2) * width
            valid = np.isfinite(vals)
            bars = ax.bar(x[valid], np.asarray(vals)[valid], width=width * 0.9,
                          color=color, label=label, zorder=3)
            if len(bars):
                ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
                if ax is axes[0, 0]:
                    handles.append(bars)
        ax.set_title(robot.upper())
        ax.set_xticks(positions, [size.capitalize() for size in sizes])
        ax.set_xlabel("Training dataset size")
        ax.set_ylabel("TCP RMSE (mm)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Task {'4.A: FK' if task == '4a' else '4.B: Webots'} test TCP RMSE", fontsize=13)
    if handles:
        fig.legend(handles, [label for _, label, _ in variants], loc="lower center",
                   bbox_to_anchor=(0.5, 0.04), ncol=len(variants), fontsize=8)
    fig.text(0.5, 0.012, "Lower bars are better. RMSE gives larger misses more weight.",
             ha="center", fontsize=8)
    save(fig, output, bottom=0.13)


def joint_statistics(rows, output, metric):
    robots = [r for r in ROBOTS if any(row["robot"] == r for row in rows)]
    fig, axes = plt.subplots(len(robots), 3, figsize=(15, 3.5 * len(robots)), squeeze=False)
    names = {"ned": ("waist", "shoulder", "elbow"), "scara": ("base", "elbow", "lift")}
    variants = variants_in(rows)
    marker_sizes = {"small": 38, "medium": 78, "large": 132}
    size_offsets = {"small": -0.23, "medium": 0.0, "large": 0.23}
    for i, robot in enumerate(robots):
        for j in range(3):
            ax = axes[i, j]
            subset = [r for r in rows if r["robot"] == robot and r["joint"] == f"q{j+1}"]
            lift = robot == "scara" and j == 2
            lookup = {(r["size"], r["variant"]): r for r in subset}
            values_seen = [0.0]
            for row_index, (variant, label, color) in enumerate(variants):
                for size in SIZES:
                    model = lookup.get((size, variant))
                    if model is None:
                        if size == "large" and variant == "tcp":
                            ax.text(0.02, row_index + size_offsets[size], "not evaluated",
                                    transform=ax.get_yaxis_transform(), color=color,
                                    va="center", fontsize=8)
                        continue
                    value = float(model[metric]) * (1e6 if lift else 1)
                    y = row_index + size_offsets[size]
                    values_seen.append(value)
                    ax.scatter([value], [y], s=marker_sizes[size], color=color,
                               edgecolors="white", linewidths=0.6, zorder=3)
                    ax.annotate(f"{value:.3g}", (value, y), xytext=(7, 0),
                                textcoords="offset points", va="center", color=color, fontsize=8)
            lo, hi = min(values_seen), max(values_seen)
            span = max(hi - lo, 1e-6)
            ax.set_xlim(lo - span * 0.1 if metric == "mean_error" and not lift else 0,
                        hi + span * 0.45)
            if metric == "mean_error" and not lift:
                ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
            ax.set_yticks(range(len(variants)), [label for _, label, _ in variants])
            ax.set_ylim(len(variants) - 0.5, -0.5)
            ax.grid(axis="x", alpha=0.2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
            ax.tick_params(axis="x", labelsize=9)
            ax.tick_params(axis="y", labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{robot.upper()} · q{j+1} ({names[robot][j]})", fontsize=11)
            if lift:
                ax.set_xlabel(("Mean" if metric == "mean_error" else "Std") + " of squared lift error (mm²)", fontsize=9)
            else:
                ax.set_xlabel("Mean signed angular error (rad)" if metric == "mean_error" else "Angular error std (rad)", fontsize=9)
    fig.suptitle("Task 4.A: per-joint " + ("mean error (bias)" if metric == "mean_error" else "error standard deviation"), fontsize=13)
    loss_handles = [Line2D([0], [0], marker="o", linestyle="none", color=color,
                           markersize=7, label=label)
                    for _, label, color in variants]
    size_handles = [Line2D([0], [0], marker="o", linestyle="none", color="#666666",
                           markerfacecolor="#666666", markersize=np.sqrt(marker_sizes[size]) / 1.4,
                           label=f"{size.capitalize()} dataset") for size in SIZES]
    fig.legend(handles=loss_handles, loc="lower center", bbox_to_anchor=(0.36, 0.075),
               ncol=len(loss_handles), fontsize=8, title="Colour = training loss", title_fontsize=8)
    fig.legend(handles=size_handles, loc="lower center", bbox_to_anchor=(0.73, 0.075),
               ncol=len(size_handles), fontsize=8, title="Dot size = dataset size", title_fontsize=8)
    note = ("Angular means can cancel: near zero means little bias, not necessarily accurate predictions. Read the std graph too."
            if metric == "mean_error" else "Lower std means less variation; read alongside mean bias. Different joints use separate axes and units.")
    fig.text(0.5, 0.035, note, ha="center", fontsize=9)
    fig.text(0.5, 0.01, "Handout definitions: signed wrapped angles; squared SCARA lift error. Lift values converted from m² to mm² (×1,000,000).",
             ha="center", fontsize=8)
    save(fig, output, bottom=0.2)


def heldout_objective(model_dir, size, variant):
    """Score the saved best-validation checkpoint once on the independent test split."""
    raw_inputs, labels, outputs = [], [], []
    config = None
    for robot in ROBOTS:
        x, y = load_test(ROOT / "data" / "nn" / "splits" / robot / "test.csv", robot)
        prediction, config = predict_model(model_dir, size, variant, robot, x)
        raw_inputs.append(x)
        labels.append(y)
        outputs.append(prediction)
    x = np.concatenate(raw_inputs)
    y = np.concatenate(labels)
    prediction = np.concatenate(outputs)
    if variant == "tcp":
        return tcp_loss_and_gradient(prediction, x, JOINT_LIMITS,
                                     float(config.get("tcp_limit_weight", 0.01)))[0]
    return float(np.mean((prediction - y) ** 2))


def smooth_curve(values, window=25):
    """Centered moving average to make noisy epoch histories readable."""
    values = np.asarray(values, dtype=float)
    window = min(window, len(values) if len(values) % 2 else max(1, len(values) - 1))
    if window <= 1:
        return values
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, window).mean(axis=-1)


def training_loss_curves(model_dir, output):
    """Show smoothed train/validation curves and one held-out test score per checkpoint."""
    split_colors = {"train": "#2878b5", "validation": "#e07a3f", "test": "#4a9b67"}
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), squeeze=False)
    objectives = (
        ("existing_joints", "Joint-label MSE", "train_mse", "val_mse", "MSE (normalised joint units²)"),
        ("tcp", "TCP objective", "train_loss_m2", "val_loss_m2", "Loss (m², includes limit penalty)"),
    )
    for row, size in enumerate(SIZES):
        for column, (variant, objective_title, train_key, val_key, ylabel) in enumerate(objectives):
            ax = axes[row, column]
            name = model_name(size, variant)
            history_path = Path(model_dir) / f"{name}_history.csv"
            config_path = Path(model_dir) / f"{name}.json"
            if not history_path.is_file() or not config_path.is_file():
                ax.text(0.5, 0.5, "Model history not found", transform=ax.transAxes, ha="center")
                continue
            with history_path.open(newline="", encoding="utf-8") as source:
                history = list(csv.DictReader(source))
            epochs = [int(row["epoch"]) for row in history]
            train = [max(float(row[train_key]), 1e-12) for row in history]
            validation = [max(float(row[val_key]), 1e-12) for row in history]
            ax.plot(epochs, smooth_curve(train), color=split_colors["train"], linewidth=2.0)
            ax.plot(epochs, smooth_curve(validation), color=split_colors["validation"], linewidth=2.0)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            best_epoch = int(config["best_epoch"])
            test_value = max(heldout_objective(model_dir, size, variant), 1e-12)
            ax.scatter([best_epoch], [test_value], marker="D", s=62,
                       facecolor=split_colors["test"], edgecolor="#202020", linewidth=0.9,
                       zorder=5, clip_on=False)
            ax.set_title(f"{size.capitalize()} dataset · {objective_title}")
            ax.set_xlabel("Training epoch")
            ax.set_ylabel(ylabel)
            ax.set_yscale("log")
            ax.grid(alpha=0.2, which="both")
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Training, validation and held-out test loss", fontsize=14)
    split_handles = [
        Line2D([0], [0], color=split_colors["train"], linewidth=2.4, label="Training · 25-epoch moving average"),
        Line2D([0], [0], color=split_colors["validation"], linewidth=2.4, label="Validation · 25-epoch moving average"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=split_colors["test"],
               markeredgecolor="#202020", markersize=7,
               label="Test set · once at best validation epoch"),
    ]
    fig.legend(handles=split_handles, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=3, fontsize=8, title="Colour = data split", title_fontsize=8)
    fig.text(0.5, 0.008,
             "Each row is a dataset size; compare values only within the same objective column. "
             "Test data do not select the saved checkpoint.", ha="center", fontsize=8)
    save(fig, output, bottom=0.12)


def remove_old_extra_graphs(output, filenames):
    for filename in filenames:
        (Path(output) / filename).unlink(missing_ok=True)


def plot_task4a(summary, joints, predictions, output, model_dir=ROOT / "models"):
    output = Path(output)
    endpoint_summary(summary, output / "comparison.png", "4a")
    endpoint_rmse(summary, output / "tcp_rmse.png", "4a")
    joint_statistics(joints, output / "joint_mean_errors.png", "mean_error")
    joint_statistics(joints, output / "joint_error_std.png", "std_error")
    training_loss_curves(model_dir, output / "training_loss.png")
    remove_old_extra_graphs(output, ("tcp_accuracy_curves.png",))
    (output / "graphs.md").write_text("# Task 4.A graphs\n\n"
        "Blue = joint-loss model; orange = TCP-loss model. Missing checkpoints are not zero errors.\n\n"
        "- [Endpoint mean and standard deviation](comparison.png): closer to the target and variation across test points.\n"
        "- [Endpoint RMSE](tcp_rmse.png): gives larger misses more weight.\n"
        "- [Per-joint mean errors](joint_mean_errors.png): colour shows loss type; dot size shows dataset size; signed angular bias; SCARA lift uses squared error.\n"
        "- [Per-joint standard deviations](joint_error_std.png): colour shows loss type; dot size shows dataset size; read with mean bias and do not compare different units.\n"
        "- [Training, validation and test loss](training_loss.png): blue/orange lines show the 25-epoch moving average of training/validation loss; green diamonds show independent test loss once at the checkpoint selected by validation. Rows identify dataset size.\n\n"
        "All test rows are included. FK positions are calculated, not measured in Webots.\n", encoding="utf-8")


def plot_task4b(summary, predictions, output, model_dir=ROOT / "models"):
    output = Path(output)
    endpoint_summary(summary, output / "comparison.png", "4b")
    endpoint_rmse(summary, output / "tcp_rmse.png", "4b")
    training_loss_curves(model_dir, output / "training_loss.png")
    remove_old_extra_graphs(output, ("tcp_accuracy_curves.png", "fk_vs_webots.png"))
    (output / "graphs.md").write_text("# Task 4.B graphs\n\n"
        "Blue = joint-loss model; orange = TCP-loss model. All measured points, including any motor timeouts, count. Check summary.csv for settling status.\n\n"
        "- [Measured endpoint mean and standard deviation](comparison.png).\n"
        "- [Measured endpoint RMSE](tcp_rmse.png).\n"
        "- [Training, validation and test loss](training_loss.png): blue/orange lines show the 25-epoch moving average of training/validation loss; green diamonds show independent test loss once at the checkpoint selected by validation. Rows identify dataset size.\n\n"
        "Task 4.B uses the first 100 test points per arm and measures the TCP in Webots.\n", encoding="utf-8")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task4a", type=Path, help="existing Task 4.A output folder")
    parser.add_argument("--task4b", type=Path, help="existing Task 4.B output folder")
    parser.add_argument("--models", type=Path, default=ROOT / "models", help="model folder containing training histories")
    args = parser.parse_args()
    if args.task4a is None and args.task4b is None:
        args.task4a, args.task4b = ROOT / "outputs/task4a", ROOT / "outputs/task4b"
    if args.task4a is not None:
        path = args.task4a
        plot_task4a(read_rows(path / "summary.csv"), read_rows(path / "joint_errors.csv"),
                    read_rows(path / "predictions.csv"), path, args.models)
        print(f"Task 4.A graphs saved to {path}")
    if args.task4b is not None:
        path = args.task4b
        plot_task4b(read_rows(path / "summary.csv"), read_rows(path / "predictions.csv"), path, args.models)
        print(f"Task 4.B graphs saved to {path}")


if __name__ == "__main__":
    main()
