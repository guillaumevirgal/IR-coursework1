"""
Task 3 - summarise every trained MLP on its validation set.

For each run in RUNS it reads models/<name>.npz / .json / _history.csv,
recomputes the validation position error per robot (de-normalise -> FK), and
writes, next to this script:
    validation_results.csv   one row per run (best epoch, MSE, errors in mm)
    validation_results.md    the same table, readable in GitHub
    learning_curves.png      train vs validation MSE, one panel per run
    validation_overview.png  validation MSE of all runs on one axis

The test sets are never read. Usage: python data/nn/validation_results/make_validation_results.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))

from scripts.train_mlp import INPUTS, MLP, MODELS, OUTPUTS, load_split, position_errors


# (model file stem, label) in the order the experiments were run.
RUNS = (
    ("mlp_ned_scara_small_h64", "Small · 1×64 · fixed lr"),
    ("mlp_ned_scara_small_h64x64", "Small · 2×64 · fixed lr"),
    ("mlp_ned_scara_small_h64x64_cosine", "Small · 2×64 · cosine"),
    ("mlp_ned_scara_medium_h64x64_cosine", "Medium · 2×64 · cosine"),
    ("mlp_ned_scara_large_h64x64_cosine", "Large · 2×64 · cosine"),
)

# Reference categorical palette (validated: CVD ΔE >= 9, light surface).
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
TRAIN_COLOR, VAL_COLOR = "#eb6834", "#2a78d6"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def read_history(name):
    with (MODELS / f"{name}_history.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return {key: np.array([float(row[key]) for row in rows]) for key in ("epoch", "train_mse", "val_mse")}


def load_model(name, config):
    hidden = config["hidden"] if isinstance(config["hidden"], list) else [config["hidden"]]
    model = MLP((len(INPUTS), *hidden, len(OUTPUTS)), np.random.default_rng(0), activation=config.get("activation") or "relu")
    stored = np.load(MODELS / f"{name}.npz")
    model.params = {key: stored[key] for key in model.params}
    return model, stored["mean"], stored["std"], hidden


def evaluate(name, label):
    with (MODELS / f"{name}.json").open(encoding="utf-8") as source:
        config = json.load(source)
    history = read_history(name)
    model, mean, std, hidden = load_model(name, config)

    x_val_raw, _ = load_split(config["robots"], "validation", config["size"])
    q_val = model.predict((x_val_raw - mean) / std)
    best = int(np.argmin(history["val_mse"]))

    result = {
        "run": label,
        "dataset": config["size"],
        "hidden_layers": "×".join(map(str, hidden)),
        "lr_schedule": config.get("decay", "none"),
        "train_rows": len(load_split(config["robots"], "train", config["size"])[0]),
        "val_rows": len(x_val_raw),
        "best_epoch": int(history["epoch"][best]),
        "epochs_run": int(history["epoch"][-1]),
        "train_mse_at_best": history["train_mse"][best],
        "val_mse_best": history["val_mse"][best],
    }
    for robot in config["robots"]:
        errors = position_errors(x_val_raw, q_val, [robot]) * 1000.0
        result[f"{robot}_mean_mm"] = errors.mean()
        result[f"{robot}_std_mm"] = errors.std()
        result[f"{robot}_max_mm"] = errors.max()
    return result, history


def style_axis(axis):
    axis.set_facecolor(SURFACE)
    axis.set_yscale("log")
    axis.grid(True, which="major", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
    axis.tick_params(colors=MUTED, labelsize=8)


def plot_learning_curves(results, histories):
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True, facecolor=SURFACE)
    axes = axes.ravel()
    for axis, result, history in zip(axes, results, histories):
        style_axis(axis)
        axis.plot(history["epoch"], history["train_mse"], color=TRAIN_COLOR, linewidth=1.2, label="Train")
        axis.plot(history["epoch"], history["val_mse"], color=VAL_COLOR, linewidth=1.2, label="Validation")
        best_epoch, best_mse = result["best_epoch"], result["val_mse_best"]
        axis.axvline(best_epoch, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        axis.plot([best_epoch], [best_mse], "o", markersize=8, color=VAL_COLOR,
                  markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        axis.text(0.97, 0.95, f"best epoch {best_epoch}\nval MSE {best_mse:.2e}", transform=axis.transAxes,
                  ha="right", va="top", fontsize=8, color=INK_2, zorder=6,
                  bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2})
        axis.set_title(result["run"], fontsize=10, color=INK, loc="left")
        axis.set_xlabel("Epoch", fontsize=8, color=INK_2)
    for axis in axes[::3]:
        axis.set_ylabel("MSE (normalised joints)", fontsize=8, color=INK_2)

    # Last cell holds the shared legend instead of an empty plot.
    legend_axis = axes[-1]
    legend_axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), marker="o",
                              markersize=8, markerfacecolor=VAL_COLOR, markeredgecolor=SURFACE))
    labels.append("Best validation epoch (weights kept)")
    legend_axis.legend(handles, labels, loc="center", frameon=False, fontsize=10, labelcolor=INK_2)

    figure.suptitle("Learning curves — Ned + SCARA combined MLP", fontsize=13, color=INK, x=0.06, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(OUT / "learning_curves.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_overview(results, histories):
    figure, axis = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    style_axis(axis)
    for color, result, history in zip(COLORS, results, histories):
        axis.plot(history["epoch"], history["val_mse"], color=color, linewidth=1.4, label=result["run"])
        axis.plot([result["best_epoch"]], [result["val_mse_best"]], "o", markersize=8, color=color,
                  markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    axis.set_xlabel("Epoch", fontsize=9, color=INK_2)
    axis.set_ylabel("Validation MSE (normalised joints)", fontsize=9, color=INK_2)
    axis.set_title("Validation MSE per run (dot = best epoch)", fontsize=12, color=INK, loc="left")
    axis.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    figure.tight_layout()
    figure.savefig(OUT / "validation_overview.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def write_tables(results):
    fields = list(results[0])
    with (OUT / "validation_results.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "# Validation results — Ned + SCARA combined MLP",
        "",
        "ReLU hidden layers, linear output, Adam (lr 1e-3, batch 32), MSE loss, "
        "early stopping (patience 300) keeping the best-validation weights. "
        "Position error = Euclidean distance between FK(de-normalised prediction) and the target (x, y, z).",
        "",
        "| Run | Train rows | Val rows | Best epoch | Epochs run | Train MSE @ best | Val MSE (best) "
        "| Ned error (mm) | SCARA error (mm) | Ned max (mm) | SCARA max (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['run']} | {r['train_rows']} | {r['val_rows']} | {r['best_epoch']} | {r['epochs_run']} "
            f"| {r['train_mse_at_best']:.2e} | {r['val_mse_best']:.2e} "
            f"| {r['ned_mean_mm']:.2f} ± {r['ned_std_mm']:.2f} | {r['scara_mean_mm']:.2f} ± {r['scara_std_mm']:.2f} "
            f"| {r['ned_max_mm']:.1f} | {r['scara_max_mm']:.1f} |"
        )
    lines += ["", "![Learning curves](learning_curves.png)", "", "![Validation overview](validation_overview.png)", ""]
    (OUT / "validation_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    evaluated = [evaluate(name, label) for name, label in RUNS]
    results = [result for result, _ in evaluated]
    histories = [history for _, history in evaluated]
    write_tables(results)
    plot_learning_curves(results, histories)
    plot_overview(results, histories)
    print((OUT / "validation_results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
