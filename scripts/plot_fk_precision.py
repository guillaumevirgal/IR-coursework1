"""
Task 1.C: plots the FK precision of the Ned and the SCARA measured by the
fk_precision controller (run scripts/run_fk_precision.sh first).

Reads data/evaluation/<robot>_fk.csv, prints a summary table, saves it as
data/evaluation/fk_summary.csv and the figure as data/evaluation/fk_precision.png.
"""

import csv
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "evaluation"
ROBOTS = {"ned": ("Ned", "#2a78d6"), "scara": ("SCARA", "#eb6834")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Horizontal position of each robot's first rotation axis, for the "error vs reach" panel.
FIRST_AXIS_X = {"ned": 0.0, "scara": 0.060}


def load_rows():
    rows = {}
    for robot in ROBOTS:
        path = EVAL_DIR / f"{robot}_fk.csv"
        if not path.exists():
            print(f"Missing {path}, skipping {robot}.")
            continue
        with open(path, newline="") as file:
            rows[robot] = list(csv.DictReader(file))
    return rows


def summarise(rows):
    summary = []
    for robot, group in rows.items():
        errors = [float(r["error"]) * 1000 for r in group if r["status"] == "ok"]
        if not errors:
            continue
        summary.append({
            "robot": ROBOTS[robot][0],
            "samples": len(errors),
            "not_settled": len(group) - len(errors),
            "mean_mm": statistics.fmean(errors),
            "median_mm": statistics.median(errors),
            "rms_mm": math.sqrt(statistics.fmean(e * e for e in errors)),
            "p95_mm": sorted(errors)[max(0, math.ceil(0.95 * len(errors)) - 1)],
            "max_mm": max(errors),
        })
    return summary


def print_summary(summary):
    header = f"{'Robot':<6} {'n':>4} {'mean':>8} {'median':>8} {'RMS':>8} {'p95':>8} {'max':>8}   (mm)"
    print(header)
    print("-" * len(header))
    for s in summary:
        print(
            f"{s['robot']:<6} {s['samples']:>4} {s['mean_mm']:>8.3f} {s['median_mm']:>8.3f} "
            f"{s['rms_mm']:>8.3f} {s['p95_mm']:>8.3f} {s['max_mm']:>8.3f}"
            + (f"   ({s['not_settled']} not settled, left out)" if s["not_settled"] else "")
        )
    with open(EVAL_DIR / "fk_summary.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_distribution(ax, ok):
    """Box plot + jittered points of the TCP position error, per robot."""
    positions, data, colors, labels = [], [], [], []
    for i, (robot, rows) in enumerate(ok.items()):
        positions.append(i)
        data.append([float(r["error"]) * 1000 for r in rows])
        colors.append(ROBOTS[robot][1])
        labels.append(ROBOTS[robot][0])

    boxes = ax.boxplot(data, positions=positions, widths=0.5, showfliers=False, patch_artist=True,
                       medianprops={"color": INK, "linewidth": 1.5},
                       whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for box, color in zip(boxes["boxes"], colors):
        box.set(facecolor=color + "33", edgecolor=color, linewidth=1.2)
    for pos, errors, color in zip(positions, data, colors):
        jitter = [pos + 0.16 * math.sin(k * 12.9898) for k in range(len(errors))]
        ax.scatter(jitter, errors, s=10, color=color, alpha=0.55, linewidths=0)

    ax.set_ylim(bottom=0)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("TCP position error (mm)", color=MUTED, fontsize=9)
    style(ax, "FK error distribution")


def plot_axis_bias(ax, ok):
    """Mean signed error (measured - FK) along x, y and z, with ±1 std."""
    width = 0.36
    for k, (robot, rows) in enumerate(ok.items()):
        label, color = ROBOTS[robot]
        means, stds = [], []
        for axis in "xyz":
            diffs = [(float(r[f"meas_{axis}"]) - float(r[f"fk_{axis}"])) * 1000 for r in rows]
            means.append(statistics.fmean(diffs))
            stds.append(statistics.pstdev(diffs))
        xs = [a + (k - 0.5) * width for a in range(3)]
        ax.bar(xs, means, width=width - 0.04, color=color, yerr=stds, label=label,
               error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(range(3), ["x", "y", "z"])
    ax.set_ylabel("Mean signed error (mm), ±1 std", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    style(ax, "Bias along each axis")


def plot_error_vs_reach(ax, ok):
    """Error against how far the TCP is from the robot's first rotation axis."""
    for robot, rows in ok.items():
        label, color = ROBOTS[robot]
        reach = [math.hypot(float(r["fk_x"]) - FIRST_AXIS_X[robot], float(r["fk_y"])) for r in rows]
        errors = [float(r["error"]) * 1000 for r in rows]
        ax.scatter(reach, errors, s=22, color=color, alpha=0.7,
                   edgecolors="white", linewidths=0.5, label=label)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Horizontal reach of the TCP (m)", color=MUTED, fontsize=9)
    ax.set_ylabel("TCP position error (mm)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    style(ax, "Error vs reach")


def main():
    rows = load_rows()
    ok = {robot: [r for r in group if r["status"] == "ok"] for robot, group in rows.items()}
    ok = {robot: group for robot, group in ok.items() if group}
    if not ok:
        raise SystemExit("No results found: run scripts/run_fk_precision.sh first.")

    print_summary(summarise(rows))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    plot_distribution(axes[0], ok)
    plot_axis_bias(axes[1], ok)
    plot_error_vs_reach(axes[2], ok)
    fig.suptitle("Forward kinematics precision against the Webots simulator", x=0.01, ha="left",
                 fontsize=13, color=INK, fontweight="bold")

    out_path = EVAL_DIR / "fk_precision.png"
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
