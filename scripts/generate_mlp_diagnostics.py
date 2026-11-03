"""Explain the MLP's end-to-end target-to-pose behavior in headless Webots."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mlp_diagnostics_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "mlp_diagnostics_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = ROOT / "data" / "nn" / "splits"
MODELS = ROOT / "models"
DEFAULT_MODEL = "mlp_ned_scara_large_h64x64x64x64_cosine_per-robot"
JOINT_NAMES = ("q1", "q2", "q3")
INPUTS = ("is_scara", "is_ned", "x", "y", "z")
WORLDS = {"ned": "articulated.wbt", "scara": "scara.wbt"}

sys.path.insert(0, str(ROOT))
from scripts.webots_mlp_runner import (  # noqa: E402
    write_cases, make_diagnostic_world, remove_world, run_webots,
)
from scripts.robot_config import FORWARD_KINEMATICS, JOINT_LIMITS, denormalise  # noqa: E402
from scripts.train_mlp import MLP  # noqa: E402
from scripts.numpy_neighbors import NumpyNearestNeighbors as cKDTree  # noqa: E402
from scripts.forward_kinematics_ned import forward_kinematics_ned  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("ned", "scara", "both"), default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size", choices=("small", "medium", "large"), default="large",
                        help="training split used for nearby-example context")
    parser.add_argument("--samples", type=int, default=2,
                        help="reachable test targets per robot")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "mlp_diagnostics")
    parser.add_argument("--points-csv", type=Path,
                        help="extra targets; columns: robot,label,x,y,z")
    parser.add_argument("--webots", type=Path, help="Webots executable, if not on PATH")
    parser.add_argument("--settle-time", type=float, default=0.25,
                        help="time all joint sensors must remain within tolerance")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="maximum simulated seconds to settle each target")
    return parser.parse_args()


def load_model(name, robot):
    weights_path = MODELS / f"{name}.npz"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Missing model weights: {weights_path}. Train it first with "
            f"python scripts/train_mlp.py --size large --hidden 64 64"
        )
    with (MODELS / f"{name}.json").open(encoding="utf-8") as source:
        config = json.load(source)
    if robot not in config["robots"]:
        raise ValueError(f"Model {name} was not trained on {robot}")
    hidden = config["hidden"] if isinstance(config["hidden"], list) else [config["hidden"]]
    activation = config.get("activation") or "relu"  # pre-activation-flag models were ReLU
    model = MLP((len(INPUTS), *hidden, len(JOINT_NAMES)), np.random.default_rng(0), activation=activation)
    with np.load(weights_path) as stored:
        model.params = {key: stored[key] for key in model.params}
        mean, std = stored["mean"], stored["std"]
    return model, mean, std


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def robot_points(robot, count):
    rows = read_rows(DATA / robot / "test.csv")
    if not rows:
        raise ValueError(f"No held-out test rows found for {robot}")
    indices = np.linspace(0, len(rows) - 1, min(count, len(rows)), dtype=int)
    cases = []
    for number, index in enumerate(indices, start=1):
        row = rows[index]
        point = tuple(float(row[axis]) for axis in ("x", "y", "z"))
        cases.append({"label": f"reachable_{number}", "kind": "reachable test point", "point": point})
    outside = (
        (("above_vertical_reach", (0.0, 0.0, 0.85)),
         ("beyond_horizontal_reach", (0.8, 0.0, 0.30))) if robot == "ned" else
        (("beyond_planar_reach", (1.0, 0.0, 0.13)),
         ("above_lift_range", (0.06, 0.0, 0.50)))
    )
    cases.extend({"label": label, "kind": "unreachable stress case", "point": point}
                 for label, point in outside)
    return cases


def extra_points(path, robot):
    if path is None:
        return []
    return [{"label": row["label"].strip(), "kind": "user-specified target",
             "point": tuple(float(row[axis]) for axis in ("x", "y", "z"))}
            for row in read_rows(path) if row["robot"].strip().lower() == robot]


def kinematic_workspace_boundary(robot, resolution=32):
    """Sample the joint-limit boundary of the FK workspace with the floor cut."""
    limits = JOINT_LIMITS[robot]
    fk = FORWARD_KINEMATICS[robot]
    axes = [np.linspace(low, high, resolution) for low, high in limits]
    floor_z = 0.02 if robot == "scara" else 0.14
    points = []
    for i, q1 in enumerate(axes[0]):
        for j, q2 in enumerate(axes[1]):
            for k, q3 in enumerate(axes[2]):
                on_joint_limit = (i in (0, resolution - 1) or j in (0, resolution - 1)
                                  or k in (0, resolution - 1))
                if on_joint_limit:
                    point = fk((q1, q2, q3))
                    if point[2] >= floor_z:
                        points.append(point)
    return np.asarray(points)


def analyse_point(robot, point, model, mean, std, train_xyz, support_95, measured):
    flags = (1.0, 0.0) if robot == "scara" else (0.0, 1.0)
    target = np.asarray(point, dtype=float)
    input_row = (np.array([[*flags, *target]]) - mean) / std
    raw = model.predict(input_row)[0]
    clipped = np.clip(raw, -1.0, 1.0)
    command = denormalise(clipped, JOINT_LIMITS[robot])
    distances = np.linalg.norm(train_xyz - target, axis=1)
    nearest = np.argsort(distances)[:5]
    fk_tcp = np.array([float(measured[f"fk_{axis}"]) for axis in "xyz"])
    webots_tcp = np.array([float(measured[f"webots_{axis}"]) for axis in "xyz"])
    return {"target": target, "raw": raw, "clipped": clipped, "command": command,
            "measured_q": np.array([float(measured[f"measured_q{i}"]) for i in range(1, 4)]),
            "fk_tcp": fk_tcp, "webots_tcp": webots_tcp,
            "fk_target_error_mm": float(measured["fk_target_error_mm"]),
            "fk_webots_gap_mm": float(measured["fk_webots_gap_mm"]),
            "webots_error_mm": float(measured["webots_error_mm"]),
            "settled": measured["settled"].lower() == "true",
            "settle_seconds": float(measured["settle_seconds"]),
            "distances": distances, "nearest": nearest,
            "weak_support": bool(distances[nearest[0]] > support_95),
            "support_95": support_95}


def draw_case(robot, model_name, case, result, workspace_boundary, output):
    target, actual = result["target"], result["webots_tcp"]
    fig = plt.figure(figsize=(15, 9))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.87, bottom=0.16, wspace=0.28, hspace=0.36)
    ax_space = fig.add_subplot(221, projection="3d")
    ax_arm = fig.add_subplot(222, projection="3d")
    ax_neighbors = fig.add_subplot(223)
    ax_flow = fig.add_subplot(224)
    ax_space.scatter(*workspace_boundary.T, s=7, c="#b8b8b8", alpha=0.32,
                     label="estimated workspace boundary")
    ax_space.scatter(*target, marker="x", s=100, c="#e15759", linewidths=3, label="requested TCP")
    ax_space.scatter(*actual, marker="o", s=65, c="#2878b5", label="Webots TCP")
    ax_space.plot([target[0], actual[0]], [target[1], actual[1]], [target[2], actual[2]],
                  color="#d62728", linestyle="--", linewidth=2, label="target-to-result error")
    ax_space.set_title("Joint-limited workspace boundary and this target")
    ax_space.set_xlabel("x (m)"); ax_space.set_ylabel("y (m)"); ax_space.set_zlabel("z (m)")
    ax_space.legend(loc="upper left", fontsize=7, framealpha=0.94,
                    title="Workspace and target", title_fontsize=8)
    ax_space.text2D(0.02, 0.02,
                    "Grey boundary: FK + joint limits + floor height; collisions not checked.",
                    transform=ax_space.transAxes, fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="#cccccc"))

    # Draw the arm links by evaluating FK at each measured joint prefix where possible.
    # The endpoint comes directly from Webots, so the figure reflects its settled pose.
    q = result["measured_q"]
    chain = _ned_chain(q) if robot == "ned" else _scara_chain(q)
    chain[-1] = actual
    arm = np.asarray(chain)
    ax_arm.plot(*arm.T, "o-", color="#555555", linewidth=5, markersize=7, label="settled arm")
    ax_arm.scatter(*target, marker="x", s=110, c="#e15759", linewidths=3, label="requested target")
    ax_arm.scatter(*actual, marker="o", s=70, c="#2878b5", label="Webots TCP")
    limits = np.vstack((arm, target))
    low, high = limits.min(axis=0), limits.max(axis=0)
    center = (low + high) / 2
    radius = max(float(np.max(high - low)) * 0.65, 0.18)
    for setter, c in ((ax_arm.set_xlim, 0), (ax_arm.set_ylim, 1), (ax_arm.set_zlim, 2)):
        setter(center[c] - radius, center[c] + radius)
    ax_arm.set_box_aspect((1, 1, 1))
    ax_arm.set_title("Arm pose after Webots settles")
    ax_arm.set_xlabel("x (m)"); ax_arm.set_ylabel("y (m)"); ax_arm.set_zlabel("z (m)")
    ax_arm.legend(loc="upper left", fontsize=8, framealpha=0.92,
                  title="Pose markers", title_fontsize=8)

    # Compare distances in physical space: zero is the requested target.
    training_mm = result["distances"][result["nearest"]] * 1000.0
    distances_mm = np.r_[training_mm, np.linalg.norm(actual - target) * 1000.0]
    labels = ["Closest training request"] + [
        f"{rank} closest training request" for rank in ("2nd", "3rd", "4th", "5th")
    ][:len(training_mm) - 1] + ["Where the robot ended up"]
    positions = np.arange(len(labels))
    colors = ["#4c956c"] * len(training_mm) + ["#2878b5"]
    for y, distance, color in zip(positions, distances_mm, colors):
        ax_neighbors.plot([0, distance], [y, y], color=color, alpha=0.35, linewidth=3)
        ax_neighbors.scatter(distance, y, s=80, color=color, zorder=3)
        ax_neighbors.annotate(f"{distance:.1f} mm away", (distance, y),
                              xytext=(8, 0), textcoords="offset points",
                              va="center", fontsize=9)
    ax_neighbors.axvline(0, color="#e15759", linewidth=2)
    ax_neighbors.text(0, 1.03, "Requested point", color="#b43c3e",
                      transform=ax_neighbors.get_xaxis_transform(), fontsize=9)
    ax_neighbors.set_yticks(positions, labels)
    ax_neighbors.set_ylim(len(labels) - 0.4, -0.6)
    ax_neighbors.set_xlim(-max(float(distances_mm.max()), 1.0) * 0.04,
                          max(float(distances_mm.max()), 1.0) * 1.55)
    ax_neighbors.set_xlabel("Distance from the requested point (mm) — closer to 0 is closer",
                            fontsize=9)
    ax_neighbors.set_title("Nearby training requests (including virtual inputs)" if result.get("synthetic_training")
                           else "Has the model seen points near this request?", pad=30, fontsize=11)
    ax_neighbors.spines[["top", "right", "left"]].set_visible(False)
    ax_neighbors.tick_params(axis="y", length=0)
    spacing_mm = result["support_95"] * 1000.0
    if result["weak_support"]:
        takeaway = (f"Even the closest training request is {training_mm[0]:.1f} mm away.\n"
                    "Few nearby examples: the model must predict beyond nearby training data.")
    else:
        takeaway = (f"The closest training request is {training_mm[0]:.1f} mm away.\n"
                    "Nearby examples exist; this alone does not guarantee an accurate prediction.")
    ax_neighbors.text(0, -0.27, takeaway +
                      f"\nFor context: 95% of training requests have a neighbour within {spacing_mm:.1f} mm.",
                      transform=ax_neighbors.transAxes, va="top", fontsize=8)

    # Prediction path: target -> network -> clipping -> physical commands -> TCP result.
    ax_flow.axis("off")
    units = ("rad", "rad", "m") if robot == "scara" else ("rad", "rad", "rad")
    physical_command = ", ".join(
        f"{value:+.3f} {unit}" for value, unit in zip(result["command"], units)
    )
    stages = [
        ("1  REQUEST", "x,y,z = " + ", ".join(f"{v:+.3f}" for v in target) + " m"),
        ("2  MLP OUTPUT", "raw normalized q = " + ", ".join(f"{v:+.2f}" for v in result["raw"])),
        ("3  LIMIT CLIP", "normalized q = " + ", ".join(f"{v:+.2f}" for v in result["clipped"])),
        ("4  MOTOR COMMAND", physical_command),
        ("5  WEBOTS TCP", "xyz = " + ", ".join(f"{v:+.3f}" for v in actual) + " m"),
    ]
    for index, (title, detail) in enumerate(stages):
        y = 0.70 - index * 0.14
        ax_flow.text(0.02, y, title, transform=ax_flow.transAxes, color="#333333", weight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#e6e6e6", edgecolor="#aaaaaa"), va="center")
        ax_flow.text(0.36, y, detail, transform=ax_flow.transAxes, va="center", family="monospace", fontsize=8)
        if index < len(stages) - 1:
            ax_flow.annotate("", xy=(0.20, y - 0.105), xytext=(0.20, y - 0.055),
                             xycoords=ax_flow.transAxes, arrowprops=dict(arrowstyle="->", color="#666"))
    motion_status = (f"settled in {result['settle_seconds']:.2f} simulated s"
                     if result["settled"] else
                     f"did not settle within {result['settle_seconds']:.2f} simulated s")
    nearest_distance = result["distances"][result["nearest"][0]] * 1000.0
    support_note = " — weak training support" if result["weak_support"] else ""
    summary = (f"WEBOTS {motion_status}  |  TCP target error: {result['webots_error_mm']:.1f} mm\n"
               f"Nearest training point: {nearest_distance:.1f} mm away{support_note}\n"
               f"95th-percentile training spacing: {result['support_95'] * 1000:.1f} mm\n"
               f"FK-to-Webots measurement difference: {result['fk_webots_gap_mm']:.2f} mm")
    ax_flow.text(0.02, 0.99, summary, transform=ax_flow.transAxes, fontsize=8, va="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f2f2f2", edgecolor="#cccccc"))

    fig.suptitle(f"{robot.upper()} MLP diagnostic | {model_name} | {case['label']}",
                 fontsize=12, y=0.99)
    if case["kind"] == "unreachable stress case":
        banner = "TARGET STATUS: UNREACHABLE STRESS CASE — outside estimated workspace"
    elif case["kind"] == "reachable test point":
        banner = "REACHABLE TARGET — held-out point from the Webots dataset"
    else:
        banner = "USER TARGET — workspace membership is not classified"
    fig.text(0.5, 0.94, banner, ha="center", va="center", fontsize=10, weight="bold", color="#333333")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def draw_reachability_comparison(robot, rows, output):
    """Compare training-data distance for reachable and unreachable targets."""
    groups = (
        ("Reachable targets", "reachable test point", "#4c956c"),
        ("Unreachable targets", "unreachable stress case", "#e07a3f"),
    )
    selected = [
        [(row, color) for row in rows if row["kind"] == kind]
        for _, kind, color in groups
    ]
    if not any(selected):
        return

    max_distance = max(
        float(row["nearest_training_distance_m"]) * 1000.0
        for group in selected for row, _ in group
    )
    max_spacing = max(float(row["training_spacing_p95_m"]) * 1000.0 for row in rows)
    right = max(max_distance, max_spacing, 1.0) * 1.45
    height = max(4.0, 1.0 + 0.55 * max(len(group) for group in selected))
    fig, axes = plt.subplots(1, 2, figsize=(12, height), sharex=True)
    for ax, (title, _, color), group in zip(axes, groups, selected):
        spacing = float(group[0][0]["training_spacing_p95_m"]) * 1000.0 if group else max_spacing
        ax.axvspan(0, spacing, color="#e9f2e8", alpha=0.9, zorder=0)
        ax.axvline(spacing, color="#555555", linestyle="--", linewidth=1.2,
                   label=f"usual training spacing: {spacing:.1f} mm")
        if group:
            for index, (row, point_color) in enumerate(group):
                distance = float(row["nearest_training_distance_m"]) * 1000.0
                ax.scatter(distance, index, s=85, color=point_color, zorder=3)
                ax.annotate(f"{distance:.1f} mm", (distance, index), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=9)
            ax.set_yticks(np.arange(len(group)), [row["label"].replace("_", " ") for row, _ in group])
            ax.set_ylim(len(group) - 0.4, -0.6)
        else:
            ax.text(0.5, 0.5, "No points", ha="center", va="center", transform=ax.transAxes)
            ax.set_yticks([])
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlim(0, right)
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.suptitle(f"{robot.upper()}: are requests near the training data?", fontsize=14, y=0.98)
    fig.supxlabel("Distance to closest training example (mm) — smaller means closer", y=0.07)
    fig.text(0.5, 0.02,
             ("Training requests include virtual unreachable inputs; proximity does not establish reachability."
              if any(row.get("training_contains_unreachable") for row in rows)
              else "Each dot is one target. Beyond the dashed line means few nearby training examples."),
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.03, 0.16, 0.98, 0.92), w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _ned_chain(q):
    q1, q2, _ = q
    from scripts.forward_kinematics_ned import SHOULDER_HEIGHT, UPPER_ARM_REACH, UPPER_ARM_RISE
    radius = UPPER_ARM_REACH * np.cos(q2) + UPPER_ARM_RISE * np.sin(q2)
    height = SHOULDER_HEIGHT - UPPER_ARM_REACH * np.sin(q2) + UPPER_ARM_RISE * np.cos(q2)
    return [np.zeros(3), np.array((0, 0, SHOULDER_HEIGHT)),
            np.array((-radius * np.sin(q1), radius * np.cos(q1), height)),
            np.asarray(forward_kinematics_ned(q))]


def _scara_chain(q):
    from scripts.forward_kinematics_scara import (BASE_TO_JOINT1_X, BASE_TO_JOINT1_Z,
        JOINT1_TO_JOINT2_X, JOINT1_TO_JOINT2_Y, JOINT1_TO_JOINT2_Z,
        JOINT2_TO_TCP_X, JOINT2_TO_TCP_Y, TCP_Z_OFFSET)
    q1, q2, lift = q
    pivot = np.array((BASE_TO_JOINT1_X, 0, BASE_TO_JOINT1_Z))
    elbow = pivot + np.array((np.cos(q1)*JOINT1_TO_JOINT2_X-np.sin(q1)*JOINT1_TO_JOINT2_Y,
                              np.sin(q1)*JOINT1_TO_JOINT2_X+np.cos(q1)*JOINT1_TO_JOINT2_Y,
                              JOINT1_TO_JOINT2_Z))
    tcp = elbow + np.array((np.cos(q1+q2)*JOINT2_TO_TCP_X-np.sin(q1+q2)*JOINT2_TO_TCP_Y,
                            np.sin(q1+q2)*JOINT2_TO_TCP_X+np.cos(q1+q2)*JOINT2_TO_TCP_Y,
                            lift+TCP_Z_OFFSET))
    return [np.zeros(3), pivot, elbow, tcp]


def main():
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if args.settle_time <= 0 or args.timeout <= 0:
        raise ValueError("--settle-time and --timeout must be positive")
    robots = ("ned", "scara") if args.robot == "both" else (args.robot,)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    config = json.loads((MODELS / f"{args.model}.json").read_text(encoding="utf-8"))
    training_dir = Path(config.get("splits_dir", DATA))
    if not training_dir.is_absolute():
        training_dir = ROOT / training_dir
        if not training_dir.exists():
            # Unreachable-target datasets moved into the extras folder.
            training_dir = ROOT / "extras/unreachable_targets" / config["splits_dir"]
    for robot in robots:
        robot_summary_rows = []
        cases = robot_points(robot, args.samples) + extra_points(args.points_csv, robot)
        labels = [case["label"] for case in cases]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate diagnostic labels for {robot}; labels must be unique")
        model, mean, std = load_model(args.model, robot)
        measured_rows = run_webots(robot, args.model, cases, args)
        training_csv = training_dir / robot / f"train_{args.size}.csv"
        train_rows = read_rows(training_csv)
        synthetic_training = any(row.get("kind") == "unreachable" for row in train_rows)
        train_xyz = np.array([[float(row[axis]) for axis in "xyz"] for row in train_rows])
        workspace_boundary = kinematic_workspace_boundary(robot)
        tree_distances, _ = cKDTree(train_xyz).query(train_xyz, k=2)
        support_95 = float(np.percentile(tree_distances[:, 1], 95))
        measured_by_label = {row["label"]: row for row in measured_rows}
        for case in cases:
            result = analyse_point(robot, case["point"], model, mean, std, train_xyz, support_95,
                                  measured_by_label[case["label"]])
            result["synthetic_training"] = synthetic_training
            filename = f"{robot}_{case['label']}.png"
            draw_case(robot, args.model, case, result, workspace_boundary,
                      args.output / filename)
            row = {"robot": robot, "label": case["label"], "kind": case["kind"],
                "training_csv": training_csv.relative_to(ROOT).as_posix() if training_csv.is_relative_to(ROOT) else str(training_csv),
                "training_contains_unreachable": synthetic_training,
                "x": result["target"][0], "y": result["target"][1], "z": result["target"][2],
                "nearest_training_distance_m": float(np.min(result["distances"])),
                "training_spacing_p95_m": support_95, "weak_training_support": result["weak_support"],
                **{f"raw_{n}": v for n, v in zip(JOINT_NAMES, result["raw"])},
                **{f"command_{n}": v for n, v in zip(JOINT_NAMES, result["command"])},
                **{f"measured_{n}": v for n, v in zip(JOINT_NAMES, result["measured_q"])},
                "fk_target_error_mm": result["fk_target_error_mm"],
                "fk_webots_gap_mm": result["fk_webots_gap_mm"],
                "webots_error_mm": result["webots_error_mm"],
                "settled": result["settled"], "settle_seconds": result["settle_seconds"], "figure": filename}
            summary_rows.append(row)
            robot_summary_rows.append(row)
        comparison_path = args.output / f"{robot}_reachable_vs_unreachable.png"
        draw_reachability_comparison(robot, robot_summary_rows, comparison_path)
        print(f"Reachable vs unreachable comparison: {comparison_path}")
    summary_path = args.output / "summary.csv"
    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=summary_rows[0].keys(), lineterminator="\n")
            writer.writeheader(); writer.writerows(summary_rows)
    print(f"Generated {len(summary_rows)} end-to-end Webots diagnostics in {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
