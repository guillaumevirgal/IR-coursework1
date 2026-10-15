"""Explain the MLP's end-to-end target-to-pose behavior in headless Webots."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
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
from scipy.spatial import cKDTree

DATA = ROOT / "data" / "nn" / "splits"
MODELS = ROOT / "models"
DEFAULT_MODEL = "mlp_ned_scara_large_h64x64_cosine"
JOINT_LIMITS = {
    "ned": ((-2.8, 2.8), (-0.8, 0.8), (0.0, 1.3)),
    "scara": ((-0.73, 0.73), (0.0, 0.83), (-0.2, 0.0)),
}
JOINT_NAMES = ("q1", "q2", "q3")
INPUTS = ("is_scara", "is_ned", "x", "y", "z")
WORLDS = {"ned": "articulated.wbt", "scara": "scara.wbt"}

sys.path.insert(0, str(ROOT))
from scripts.run_dataset_generation import find_webots  # noqa: E402
from scripts.train_mlp import MLP, denormalise  # noqa: E402
from scripts.forward_kinematics_ned import forward_kinematics_ned  # noqa: E402
from scripts.forward_kinematics_scara import forward_kinematics_scara  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("ned", "scara", "both"), default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size", choices=("small", "medium", "large"), default="large",
                        help="training split used for nearby-example context")
    parser.add_argument("--samples", type=int, default=2,
                        help="held-out reachable targets per robot")
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
    model = MLP((len(INPUTS), *hidden, len(JOINT_NAMES)), np.random.default_rng(0))
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
        cases.append({"label": f"heldout_{number}", "kind": "held-out reachable", "point": point})
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


def write_cases(path, cases):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=("label", "kind", "x", "y", "z"))
        writer.writeheader()
        for case in cases:
            writer.writerow(dict(zip(("label", "kind", "x", "y", "z"),
                                     (case["label"], case["kind"], *case["point"]))))


def make_diagnostic_world(robot, model, points, output, settle_time, timeout):
    source = ROOT / "worlds" / WORLDS[robot]
    content = source.read_text(encoding="utf-8")
    if content.count('  controller "mlp_ik"') != 1:
        raise RuntimeError(f"Expected one MLP controller in {source}")
    content = content.replace('  controller "mlp_ik"', '  controller "diagnose_mlp"', 1)
    args_block = (f'  controllerArgs [\n    "{robot}"\n    "--model={model}"\n'
                  f'    "--points={points.as_posix()}"\n    "--output={output.as_posix()}"\n'
                  f'    "--settle-time={settle_time}"\n    "--timeout={timeout}"\n  ]')
    start = content.index("  controllerArgs [", content.index('controller "diagnose_mlp"'))
    end = content.index("  ]", start) + 3
    content = content[:start] + args_block + content[end:]
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".wbt",
                                         prefix=f"diagnostic_{robot}_", dir=source.parent,
                                         delete=False)
    with handle:
        handle.write(content)
    return Path(handle.name)


def remove_world(path):
    for candidate in (path, path.with_suffix(".wbproj"), path.with_name(f".{path.stem}.wbproj")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def run_webots(robot, model, cases, args):
    webots = find_webots(args.webots)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{robot}_", dir=args.output))
    points_path, measured_path = work / "points.csv", work / "webots.csv"
    write_cases(points_path, cases)
    world = make_diagnostic_world(robot, model, points_path, measured_path,
                                  args.settle_time, args.timeout)
    command = [str(webots), "--batch", "--mode=fast", "--no-rendering", "--stdout", "--stderr", str(world)]
    print(f"Running {robot} diagnostics in headless Webots ({len(cases)} targets)...", flush=True)
    try:
        subprocess.run(command, check=True)
    finally:
        remove_world(world)
    rows = read_rows(measured_path)
    shutil.rmtree(work, ignore_errors=True)
    return rows


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


def draw_case(robot, model_name, case, result, train_xyz, train_q, output):
    target, pred, actual = result["target"], result["fk_tcp"], result["webots_tcp"]
    nearest_xyz = train_xyz[result["nearest"]]
    rng = np.random.default_rng(42)
    selection = np.arange(len(train_xyz))
    if len(selection) > 4000:
        selection = rng.choice(selection, 4000, replace=False)
    cloud = train_xyz[selection]
    fig = plt.figure(figsize=(15, 9))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.89, bottom=0.08, wspace=0.28, hspace=0.36)
    ax_space = fig.add_subplot(221, projection="3d")
    ax_arm = fig.add_subplot(222, projection="3d")
    ax_neighbors = fig.add_subplot(223)
    ax_flow = fig.add_subplot(224)
    ax_space.scatter(*cloud.T, s=5, c="#c7d0d8", alpha=0.32, label="training TCPs")
    ax_space.scatter(*nearest_xyz.T, s=35, c="#4e79a7", label="nearest 5 TCPs")
    ax_space.scatter(*target, marker="x", s=100, c="#e15759", linewidths=3, label="requested TCP")
    ax_space.scatter(*pred, marker="o", s=55, c="#f28e2b", label="command FK TCP")
    ax_space.scatter(*actual, marker="^", s=65, c="#59a14f", label="measured Webots TCP")
    ax_space.plot([target[0], actual[0]], [target[1], actual[1]], [target[2], actual[2]],
                  color="#e15759", linewidth=2, label="target-to-measured error")
    ax_space.plot([pred[0], actual[0]], [pred[1], actual[1]], [pred[2], actual[2]],
                  color="#777777", linestyle=":", linewidth=1.5, label="FK-to-Webots difference")
    ax_space.set_title("Where this request sits in the learned data")
    ax_space.set_xlabel("x (m)"); ax_space.set_ylabel("y (m)"); ax_space.set_zlabel("z (m)")
    ax_space.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.92,
                    title="Markers and lines", title_fontsize=8)

    # Draw the arm links by evaluating FK at each measured joint prefix where possible.
    # The endpoint comes directly from Webots, so the figure reflects its settled pose.
    q = result["measured_q"]
    chain = _ned_chain(q) if robot == "ned" else _scara_chain(q)
    chain[-1] = actual
    arm = np.asarray(chain)
    ax_arm.plot(*arm.T, "o-", color="#4e79a7", linewidth=5, markersize=7, label="settled Webots arm")
    ax_arm.scatter(*target, marker="x", s=110, c="#e15759", linewidths=3, label="requested target")
    ax_arm.scatter(*actual, marker="^", s=70, c="#59a14f", label="measured TCP")
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

    # Show nearby training labels beside the model response. Colors saturate at
    # normalized joint limits; cell text retains the raw model value.
    nearest_q = train_q[result["nearest"]]
    matrix = np.vstack((nearest_q, result["raw"], result["clipped"]))
    color_matrix = np.clip(matrix, -1.0, 1.0)
    nearest_labels = [
        f"Example {i + 1} ({result['distances'][row] * 1000:.1f} mm)"
        for i, row in enumerate(result["nearest"])
    ]
    row_labels = nearest_labels + ["MLP raw output", "After joint-limit clip"]
    image = ax_neighbors.imshow(color_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color_value = color_matrix[row, col]
            ink = "white" if abs(color_value) > 0.58 else "black"
            ax_neighbors.text(col, row, f"{matrix[row, col]:+.2f}", ha="center", va="center",
                              fontsize=8, color=ink)
    ax_neighbors.set_yticks(np.arange(len(row_labels)), row_labels)
    ax_neighbors.set_xticks(np.arange(3), JOINT_NAMES)
    ax_neighbors.set_title("Nearby training joint labels vs this model's output")
    ax_neighbors.set_xlabel("Normalized joint value (color scale limited to −1…+1)")
    fig.colorbar(image, ax=ax_neighbors, fraction=0.046, pad=0.03, label="normalized value")

    # Prediction path: target -> network -> clipping -> physical commands -> TCP result.
    ax_flow.axis("off")
    units = ("rad", "rad", "m") if robot == "scara" else ("rad", "rad", "rad")
    physical_command = ", ".join(
        f"{value:+.3f} {unit}" for value, unit in zip(result["command"], units)
    )
    stages = [
        ("1  REQUEST", "x,y,z = " + ", ".join(f"{v:+.3f}" for v in target) + " m", "#e15759"),
        ("2  MLP OUTPUT", "raw normalized q = " + ", ".join(f"{v:+.2f}" for v in result["raw"]), "#b07aa1"),
        ("3  LIMIT CLIP", "normalized q = " + ", ".join(f"{v:+.2f}" for v in result["clipped"]), "#f28e2b"),
        ("4  MOTOR COMMAND", physical_command, "#4e79a7"),
        ("5  WEBOTS TCP", "xyz = " + ", ".join(f"{v:+.3f}" for v in actual) + " m", "#59a14f"),
    ]
    for index, (title, detail, color) in enumerate(stages):
        y = 0.91 - index * 0.185
        ax_flow.text(0.02, y, title, transform=ax_flow.transAxes, color="white", weight="bold",
                     bbox=dict(boxstyle="round,pad=0.35", facecolor=color, edgecolor="none"), va="center")
        ax_flow.text(0.36, y, detail, transform=ax_flow.transAxes, va="center", family="monospace", fontsize=9)
        if index < len(stages) - 1:
            ax_flow.annotate("", xy=(0.20, y - 0.105), xytext=(0.20, y - 0.055),
                             xycoords=ax_flow.transAxes, arrowprops=dict(arrowstyle="->", color="#666"))
    status = "SETTLED" if result["settled"] else "TIMEOUT: did not settle"
    support = (f"nearest example {result['distances'][result['nearest'][0]] * 1000:.1f} mm; "
               f"95% training spacing {result['support_95'] * 1000:.1f} mm")
    support += " (weak support)" if result["weak_support"] else ""
    summary = (f"WEBOTS: {status} in {result['settle_seconds']:.2f} simulated s  |  {support}\n"
               f"TCP TARGET ERROR: {result['webots_error_mm']:.1f} mm  |  FK estimate error: {result['fk_target_error_mm']:.1f} mm\n"
               f"FK/Webots difference: {result['fk_webots_gap_mm']:.2f} mm. The network does not test reachability."
               if result["settled"] else
               f"WEBOTS: {status} after {result['settle_seconds']:.2f} simulated s  |  {support}\n"
               f"LAST MEASURED TCP TARGET ERROR (Webots): {result['webots_error_mm']:.1f} mm")
    ax_flow.text(0.02, 0.015, summary, transform=ax_flow.transAxes, fontsize=8.5, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.45", facecolor="#f2f2f2", edgecolor="#cccccc"))

    fig.suptitle(f"{robot.upper()} end-to-end MLP diagnostic | {model_name} | {case['label']} ({case['kind']})",
                 fontsize=13)
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
    for robot in robots:
        cases = robot_points(robot, args.samples) + extra_points(args.points_csv, robot)
        labels = [case["label"] for case in cases]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate diagnostic labels for {robot}; labels must be unique")
        model, mean, std = load_model(args.model, robot)
        measured_rows = run_webots(robot, args.model, cases, args)
        train_rows = read_rows(DATA / robot / f"train_{args.size}.csv")
        train_xyz = np.array([[float(row[axis]) for axis in "xyz"] for row in train_rows])
        train_q = np.array([[float(row[name]) for name in JOINT_NAMES] for row in train_rows])
        tree_distances, _ = cKDTree(train_xyz).query(train_xyz, k=2)
        support_95 = float(np.percentile(tree_distances[:, 1], 95))
        measured_by_label = {row["label"]: row for row in measured_rows}
        for case in cases:
            result = analyse_point(robot, case["point"], model, mean, std, train_xyz, support_95,
                                  measured_by_label[case["label"]])
            filename = f"{robot}_{case['label']}.png"
            draw_case(robot, args.model, case, result, train_xyz, train_q,
                      args.output / filename)
            summary_rows.append({"robot": robot, "label": case["label"], "kind": case["kind"],
                "x": result["target"][0], "y": result["target"][1], "z": result["target"][2],
                "nearest_training_distance_m": float(np.min(result["distances"])),
                "training_spacing_p95_m": support_95, "weak_training_support": result["weak_support"],
                **{f"raw_{n}": v for n, v in zip(JOINT_NAMES, result["raw"])},
                **{f"command_{n}": v for n, v in zip(JOINT_NAMES, result["command"])},
                **{f"measured_{n}": v for n, v in zip(JOINT_NAMES, result["measured_q"])},
                "fk_target_error_mm": result["fk_target_error_mm"],
                "fk_webots_gap_mm": result["fk_webots_gap_mm"],
                "webots_error_mm": result["webots_error_mm"],
                "settled": result["settled"], "settle_seconds": result["settle_seconds"], "figure": filename})
    summary_path = args.output / "summary.csv"
    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=summary_rows[0].keys())
            writer.writeheader(); writer.writerows(summary_rows)
    print(f"Generated {len(summary_rows)} end-to-end Webots diagnostics in {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
