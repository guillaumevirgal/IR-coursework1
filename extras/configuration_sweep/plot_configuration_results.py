"""Graphs for the configuration comparison, read from evaluate_configurations.py output.

Figure-only script (matplotlib), so it is outside the coursework's library
restriction and does not need to be submitted.

Usage:
    python extras/configuration_sweep/evaluate_configurations.py
    python extras/configuration_sweep/plot_configuration_results.py
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reports" / "configuration_evaluation"
FIGURES = RESULTS / "figures"
ROBOTS = ("ned", "scara")
ROBOT_NAME = {"ned": "Ned", "scara": "SCARA"}
SIZES = ("small", "medium", "large")
COLOR = {"ned": "#2f6db3", "scara": "#d9822b", "relu": "#7a5195", "tanh": "#2a9d8f", "per-robot": "#c0392b"}


def label(m):
    text = f"{m['size']} · {'-'.join(map(str, m['hidden']))} · {m['activation']}"
    return text if m["decay"] == "cosine" else text + " · no decay"


def pos(m, robot, key="mean"):
    return m["robots"][robot]["position_mm"][key]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=160)
    plt.close(fig)
    print(f"Saved {FIGURES / name}")


def task4a_position(models, selected):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(SIZES))
    for offset, robot in zip((-0.2, 0.2), ROBOTS):
        chosen = [models[selected[size]] for size in SIZES]
        means = [pos(m, robot) for m in chosen]
        stds = [pos(m, robot, "std") for m in chosen]
        ax.bar(x + offset, means, 0.38, yerr=stds, capsize=4, color=COLOR[robot], label=ROBOT_NAME[robot])
        for xi, mean_mm, std_mm in zip(x + offset, means, stds):
            ax.annotate(f"{mean_mm:.2f} ± {std_mm:.2f}", (xi, mean_mm + std_mm), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{size}\n{label(models[selected[size]]).split(' · ', 1)[1]}" for size in SIZES])
    ax.set_ylabel("TCP position error (mm)")
    ax.set_ylim(bottom=0)
    ax.set_title("Task 4.A: position error, mean ± std on the test sets")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, "task4a_position_error.png")


def task4a_joints(models, selected):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    x = np.arange(len(SIZES))
    for row, robot in enumerate(ROBOTS):
        for col in range(3):
            ax = axes[row, col]
            joints = [models[selected[size]]["robots"][robot]["joints"][col] for size in SIZES]
            ax.bar(x, [j["mean"] for j in joints], 0.5, yerr=[j["std"] for j in joints], capsize=4, color=COLOR[robot])
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(x, SIZES)
            ax.set_title(f"{ROBOT_NAME[robot]} {joints[0]['joint']}", fontsize=10)
            ax.set_ylabel("m²" if joints[0]["unit"] == "m^2" else "rad")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Task 4.A: joint errors, mean ± std (revolute: wrapped angle; prismatic: squared difference)")
    save(fig, "task4a_joint_errors.png")


def ranked(models, selected):
    rows = sorted(models.values(), key=lambda m: (pos(m, "ned") + pos(m, "scara")) / 2, reverse=True)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(rows) + 1.2))
    y = np.arange(len(rows))
    for robot in ROBOTS:
        ax.scatter([pos(m, robot) for m in rows], y, color=COLOR[robot], label=ROBOT_NAME[robot], zorder=3)
    for i, m in enumerate(rows):
        ax.plot([pos(m, "ned"), pos(m, "scara")], [i, i], color="0.75", zorder=2)
    ax.set_yticks(y, [label(m) for m in rows], fontsize=8)
    for tick, m in zip(ax.get_yticklabels(), rows):
        if m["name"] in selected.values():
            tick.set_fontweight("bold")
    ax.set_xlabel("Mean TCP position error (mm)")
    ax.set_title("All configurations, ranked by the average of both arms (bold: Task 4.A picks)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    save(fig, "all_configurations_ranked.png")


def error_vs_parameters(models):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    size_marker = {"small": 30, "medium": 70, "large": 130}
    for m in models.values():
        ax.scatter(m["parameters"], (pos(m, "ned") + pos(m, "scara")) / 2, s=size_marker[m["size"]],
                   color=COLOR[m["activation"]], alpha=0.75, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Mean TCP error, average of both arms (mm)")
    ax.set_title("Error against network size (dot size = dataset size)")
    for act in sorted({m["activation"] for m in models.values()}):
        ax.scatter([], [], color=COLOR[act], label=act)
    for size, s in size_marker.items():
        ax.scatter([], [], s=s, color="0.6", label=size)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "error_vs_parameters.png")


def relu_vs_tanh(models):
    cosine = [m for m in models.values() if m["decay"] == "cosine"]
    pairs = []
    for relu in (m for m in cosine if m["activation"] == "relu"):
        tanh = next((m for m in cosine if m["activation"] == "tanh" and m["size"] == relu["size"] and m["hidden"] == relu["hidden"]), None)
        if tanh:
            pairs.append((f"{relu['size']} · {'-'.join(map(str, relu['hidden']))}", relu, tanh))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(pairs) + 1.6), sharey=True)
    y = np.arange(len(pairs))
    for ax, robot in zip(axes, ROBOTS):
        for i, (_, relu, tanh) in enumerate(pairs):
            ax.plot([pos(relu, robot), pos(tanh, robot)], [i, i], color="0.75", zorder=2)
        ax.scatter([pos(p[1], robot) for p in pairs], y, color=COLOR["relu"], label="ReLU", zorder=3)
        ax.scatter([pos(p[2], robot) for p in pairs], y, color=COLOR["tanh"], label="tanh", zorder=3)
        ax.set_title(ROBOT_NAME[robot])
        ax.set_xlabel("Mean TCP position error (mm)")
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [p[0] for p in pairs])
    axes[1].legend()
    fig.suptitle("ReLU vs tanh on otherwise identical configurations")
    save(fig, "relu_vs_tanh.png")


def depth_and_datasize(models):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    depth = sorted((m for m in models.values() if m["size"] == "small" and m["activation"] == "tanh"
                    and m["decay"] == "cosine" and set(m["hidden"]) == {64}), key=lambda m: m["depth"])
    for robot in ROBOTS:
        left.plot([m["depth"] for m in depth], [pos(m, robot) for m in depth], "o-", color=COLOR[robot], label=ROBOT_NAME[robot])
    left.set_xticks([m["depth"] for m in depth])
    left.set_xlabel("Hidden layers of 64 units (tanh, small dataset)")
    left.set_ylabel("Mean TCP position error (mm)")
    left.set_ylim(bottom=0)
    left.set_title("Depth")
    left.legend()
    left.grid(alpha=0.3)

    for act in ("relu", "tanh"):
        line = {m["size"]: m for m in models.values() if m["hidden"] == [64, 64] and m["decay"] == "cosine" and m["activation"] == act}
        sizes = [s for s in SIZES if s in line]
        for robot, style in zip(ROBOTS, ("-", "--")):
            right.plot(sizes, [pos(line[s], robot) for s in sizes], "o" + style, color=COLOR[act], label=f"{act} · {ROBOT_NAME[robot]}")
    right.set_xlabel("Training-set size (64-64 networks)")
    right.set_ylim(bottom=0)
    right.set_title("Dataset size")
    right.legend(fontsize=8)
    right.grid(alpha=0.3)
    save(fig, "depth_and_dataset_size.png")


def architecture_by_size(models):
    """Average of both arms' mean error for every architecture trained on more than one size."""

    grid = {}
    for m in models.values():
        if m["decay"] == "cosine":
            key = f"{'-'.join(map(str, m['hidden']))} · {m['activation']}"
            grid.setdefault(key, {})[m["size"]] = (pos(m, "ned") + pos(m, "scara")) / 2
    rows = sorted((k for k, v in grid.items() if len(v) > 1), key=lambda k: min(grid[k].values()))
    values = np.array([[grid[k].get(s, np.nan) for s in SIZES] for k in rows])
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(rows) + 1.4))
    image = ax.imshow(values, cmap="viridis_r", aspect="auto")
    for (i, j), v in np.ndenumerate(values):
        text = "—" if np.isnan(v) else f"{v:.2f}"
        shade = "white" if not np.isnan(v) and v > np.nanmean(values) else "black"
        ax.text(j, i, text, ha="center", va="center", fontsize=8, color=shade)
    ax.set_xticks(range(len(SIZES)), SIZES)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_xlabel("Training-set size")
    ax.set_title("Mean TCP error (mm, average of both arms)\nby architecture and dataset size")
    fig.colorbar(image, ax=ax, label="mm")
    save(fig, "architecture_by_dataset_size.png")


def per_robot_vs_tanh(models):
    """64-64-64-64 tanh against per-robot: validation joint MSE and test TCP error per size."""

    def find(size, act):
        return next((m for m in models.values() if m["size"] == size and m["hidden"] == [64, 64, 64, 64]
                     and m["decay"] == "cosine" and m["activation"] == act), None)

    sizes = [s for s in SIZES if find(s, "tanh") and find(s, "per-robot")]
    if not sizes:
        return
    acts = ("tanh", "per-robot")
    x = np.arange(len(sizes))
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.6]})

    for offset, act in zip((-0.2, 0.2), acts):
        values = [find(s, act)["best_val_mse"] for s in sizes]
        bars = left.bar(x + offset, values, 0.38, color=COLOR[act], label=act)
        left.bar_label(bars, labels=[f"{v:.2e}" for v in values], padding=2, fontsize=8)
    left.set_xticks(x, sizes)
    left.set_ylabel("Validation MSE (normalised joints)")
    left.set_title("Validation joint MSE (selection measure)")
    left.legend()
    left.grid(axis="y", alpha=0.3)

    width = 0.2
    for i, (robot, act) in enumerate((r, a) for r in ROBOTS for a in acts):
        values = [pos(find(s, act), robot) for s in sizes]
        bars = right.bar(x + (i - 1.5) * width, values, width, color=COLOR[act],
                         hatch="//" if robot == "scara" else None, edgecolor="white",
                         label=f"{ROBOT_NAME[robot]} · {act}")
        right.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=2, fontsize=7)
    right.set_xticks(x, sizes)
    right.set_ylabel("Mean TCP position error (mm)")
    right.set_title("Test position error (Ned plain, SCARA hatched)")
    right.legend(fontsize=8, ncol=2)
    right.grid(axis="y", alpha=0.3)

    fig.suptitle("64-64-64-64: tanh vs per-robot activation (lower is better)")
    save(fig, "per_robot_vs_tanh.png")


def task4a_vs_task4b(models):
    """Task 4.A (FK, full test) against Task 4.B (Webots, first 100) for the measured models."""

    path = RESULTS / "task4b_position.csv"
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["task4a_pick"]]
    if not rows:
        return
    sizes = [s for s in SIZES if any(row["task4a_pick"] == s for row in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    series = (("task4a_full_test_mean_mm", "task4a_full_test_std_mm", "4.A: FK, full test (1,000)", "#8ab6d6"),
              ("fk_first100_mean_mm", "fk_first100_std_mm", "FK on the same 100", "#4a7fb0"),
              ("mean_mm", "std_mm", "4.B: Webots, first 100", "#1d3557"))
    x = np.arange(len(sizes))
    for ax, robot in zip(axes, ROBOTS):
        by_size = {row["task4a_pick"]: row for row in rows if row["robot"] == robot}
        for i, (mean_key, std_key, name, color) in enumerate(series):
            means = [float(by_size[s][mean_key]) for s in sizes]
            stds = [float(by_size[s][std_key]) for s in sizes]
            ax.bar(x + (i - 1) * 0.27, means, 0.26, yerr=stds, capsize=3, color=color, label=name)
            for xi, m in zip(x + (i - 1) * 0.27, means):
                ax.annotate(f"{m:.2f}", (xi, m), xytext=(3, 2), textcoords="offset points", ha="left", fontsize=7)
        ax.set_xticks(x, [f"{s}\n{label(models[by_size[s]['model']]).split(' · ', 1)[1]}" for s in sizes])
        ax.set_title(ROBOT_NAME[robot])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("TCP position error (mm), mean ± std")
    axes[1].legend(fontsize=8)
    fig.suptitle("Task 4.A vs Task 4.B")
    save(fig, "task4a_vs_task4b.png")


def learning_curves(models, selected):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in models.values():
        if m["curve"] and m["name"] not in selected.values():
            ax.plot(m["curve"]["epoch"], m["curve"]["val_mse"], color="0.85", linewidth=0.8)
    for (size, name), color in zip(selected.items(), ("#e76f51", "#2a9d8f", "#264653")):
        c = models[name]["curve"]
        ax.plot(c["epoch"], c["train_mse"], "--", color=color, linewidth=1, alpha=0.7)
        ax.plot(c["epoch"], c["val_mse"], color=color, linewidth=2, label=f"{size}: {label(models[name]).split(' · ', 1)[1]}")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE on normalised joints")
    ax.set_title("Learning curves: Task 4.A picks (solid val, dashed train), others in grey")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    save(fig, "learning_curves.png")


def error_cdf(models, selected, levels):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    shown = list(selected.values())
    best_relu = min((m for m in models.values() if m["activation"] == "relu"), key=lambda m: pos(m, "ned") + pos(m, "scara"))
    shown.append(best_relu["name"])
    for ax, robot in zip(axes, ROBOTS):
        for name, color in zip(shown, ("#e76f51", "#2a9d8f", "#264653", COLOR["relu"])):
            ax.plot(models[name]["robots"][robot]["position_mm"]["quantiles"], levels, color=color, label=label(models[name]))
        ax.axvline(5, color="0.5", linestyle=":", linewidth=1)
        p95 = max(np.percentile(models[n]["robots"][robot]["position_mm"]["quantiles"], 95) for n in shown)
        ax.set_xlim(0, p95 * 1.3)
        ax.set_title(ROBOT_NAME[robot])
        ax.set_xlabel("TCP position error (mm)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Share of test points within error")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Cumulative error distribution (dotted line: 5 mm)")
    save(fig, "error_distribution.png")


def main():
    data = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))
    models = {m["name"]: m for m in data["models"]}
    selected = data["task4a_selection"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    task4a_position(models, selected)
    task4a_joints(models, selected)
    ranked(models, selected)
    error_vs_parameters(models)
    relu_vs_tanh(models)
    depth_and_datasize(models)
    architecture_by_size(models)
    per_robot_vs_tanh(models)
    task4a_vs_task4b(models)
    learning_curves(models, selected)
    error_cdf(models, selected, data["quantile_levels"])


if __name__ == "__main__":
    main()
