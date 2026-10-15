"""Generate static MLP IK diagnostics without starting Webots.

Default run:
    python scripts/generate_mlp_diagnostics.py

This creates held-out reachable examples and deliberately unreachable stress
cases for both robots, with one PNG per point plus a CSV summary.
"""

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

sys.path.insert(0, str(ROOT))

from scripts.forward_kinematics_ned import (  # noqa: E402
    SHOULDER_HEIGHT,
    UPPER_ARM_REACH,
    UPPER_ARM_RISE,
    forward_kinematics_ned,
)
from scripts.forward_kinematics_scara import (  # noqa: E402
    BASE_TO_JOINT1_X,
    BASE_TO_JOINT1_Z,
    JOINT1_TO_JOINT2_X,
    JOINT1_TO_JOINT2_Y,
    JOINT1_TO_JOINT2_Z,
    JOINT2_TO_TCP_X,
    JOINT2_TO_TCP_Y,
    TCP_Z_OFFSET,
    forward_kinematics_scara,
)
from scripts.train_mlp import MLP, denormalise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", choices=("ned", "scara", "both"), default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name from models/")
    parser.add_argument("--size", choices=("small", "medium", "large"), default="large",
                        help="training split used to measure support")
    parser.add_argument("--samples", type=int, default=2,
                        help="number of held-out reachable points per robot")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "mlp_diagnostics")
    parser.add_argument("--points-csv", type=Path,
                        help="optional extra targets; columns: robot,label,x,y,z")
    return parser.parse_args()


def load_model(name, robot):
    with (MODELS / f"{name}.json").open(encoding="utf-8") as source:
        config = json.load(source)
    if robot not in config["robots"]:
        raise ValueError(f"Model {name} was not trained on {robot}")
    hidden = config["hidden"] if isinstance(config["hidden"], list) else [config["hidden"]]
    model = MLP((len(INPUTS), *hidden, len(JOINT_NAMES)), np.random.default_rng(0))
    with np.load(MODELS / f"{name}.npz") as stored:
        model.params = {key: stored[key] for key in model.params}
        mean, std = stored["mean"], stored["std"]
    return model, mean, std, config


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def robot_points(robot, count):
    test_rows = read_rows(DATA / robot / "test.csv")
    if not test_rows:
        raise ValueError(f"No held-out test rows found for {robot}")
    indices = np.linspace(0, len(test_rows) - 1, min(count, len(test_rows)), dtype=int)
    cases = []
    for number, index in enumerate(indices, start=1):
        row = test_rows[index]
        point = tuple(float(row[axis]) for axis in ("x", "y", "z"))
        cases.append({"label": f"heldout_{number}", "kind": "held-out reachable", "point": point})

    # These targets deliberately exceed the robot's modeled joint-limited reach.
    if robot == "ned":
        outside = (
            ("above_vertical_reach", (0.0, 0.0, 0.85)),
            ("beyond_horizontal_reach", (0.8, 0.0, 0.30)),
        )
    else:
        outside = (
            ("beyond_planar_reach", (1.0, 0.0, 0.13)),
            ("above_lift_range", (0.06, 0.0, 0.50)),
        )
    cases.extend({"label": label, "kind": "unreachable stress case", "point": point}
                 for label, point in outside)
    return cases


def extra_points(path, robot):
    if path is None:
        return []
    rows = read_rows(path)
    cases = []
    for row in rows:
        if row["robot"].strip().lower() != robot:
            continue
        cases.append({
            "label": row["label"].strip(),
            "kind": "user-specified target",
            "point": tuple(float(row[axis]) for axis in ("x", "y", "z")),
        })
    return cases


def arm_points(robot, q):
    if robot == "ned":
        q1, q2, _q3 = q

        def rotate(radius, height):
            return -radius * np.sin(q1), radius * np.cos(q1), height

        shoulder = (0.0, 0.0, SHOULDER_HEIGHT)
        radius = UPPER_ARM_REACH * np.cos(q2) + UPPER_ARM_RISE * np.sin(q2)
        height = SHOULDER_HEIGHT - UPPER_ARM_REACH * np.sin(q2) + UPPER_ARM_RISE * np.cos(q2)
        return np.asarray([(0.0, 0.0, 0.0), shoulder, rotate(radius, height),
                           forward_kinematics_ned(q)])

    q1, q2, lift = q
    c1, s1 = np.cos(q1), np.sin(q1)
    pivot = (BASE_TO_JOINT1_X, 0.0, BASE_TO_JOINT1_Z)
    elbow = (
        pivot[0] + c1 * JOINT1_TO_JOINT2_X - s1 * JOINT1_TO_JOINT2_Y,
        pivot[1] + s1 * JOINT1_TO_JOINT2_X + c1 * JOINT1_TO_JOINT2_Y,
        pivot[2] + JOINT1_TO_JOINT2_Z,
    )
    c12, s12 = np.cos(q1 + q2), np.sin(q1 + q2)
    tcp = (
        elbow[0] + c12 * JOINT2_TO_TCP_X - s12 * JOINT2_TO_TCP_Y,
        elbow[1] + s12 * JOINT2_TO_TCP_X + c12 * JOINT2_TO_TCP_Y,
        elbow[2] + lift + TCP_Z_OFFSET,
    )
    return np.asarray([(0.0, 0.0, 0.0), pivot, elbow, tcp])


def raw_sensitivity(model, mean, std, robot, target):
    flags = (1.0, 0.0) if robot == "scara" else (0.0, 1.0)
    activation = (np.array([[*flags, *target]]) - mean) / std
    jacobian = np.eye(len(INPUTS))
    for layer in range(1, model.n_layers + 1):
        weights = model.params[f"W{layer}"]
        pre = activation @ weights + model.params[f"b{layer}"]
        jacobian = jacobian @ weights
        if layer < model.n_layers:
            active = pre[0] > 0.0
            jacobian *= active[None, :]
            activation = np.maximum(pre, 0.0)
        else:
            activation = pre
    return jacobian[2:, :].T / std[None, 2:]


def analyse_point(robot, point, model, mean, std, train_xyz, train_q, support_95):
    flags = (1.0, 0.0) if robot == "scara" else (0.0, 1.0)
    target = np.asarray(point, dtype=float)
    input_row = (np.array([[*flags, *target]]) - mean) / std
    raw = model.predict(input_row)[0]
    clipped = np.clip(raw, -1.0, 1.0)
    joints = denormalise(clipped, JOINT_LIMITS[robot])
    fk = forward_kinematics_scara if robot == "scara" else forward_kinematics_ned
    predicted = np.asarray(fk(joints), dtype=float)
    distances = np.linalg.norm(train_xyz - target, axis=1)
    nearest = np.argsort(distances)[:5]
    return {
        "target": target,
        "input_z": input_row[0],
        "raw": raw,
        "clipped": clipped,
        "joints": joints,
        "predicted": predicted,
        "error_mm": float(np.linalg.norm(predicted - target) * 1000.0),
        "distances": distances,
        "nearest": nearest,
        "weak_support": bool(distances[nearest[0]] > support_95),
        "sensitivity": raw_sensitivity(model, mean, std, robot, target),
    }


def draw_case(robot, model_name, case, result, train_xyz, train_q, support_95, output):
    target = result["target"]
    predicted = result["predicted"]
    nearest = result["nearest"]
    nearest_xyz = train_xyz[nearest]
    arm = arm_points(robot, result["joints"])

    rng = np.random.default_rng(42)
    sample_indices = np.arange(len(train_xyz))
    if len(sample_indices) > 4000:
        sample_indices = rng.choice(sample_indices, 4000, replace=False)
    cloud = train_xyz[sample_indices]

    fig = plt.figure(figsize=(14, 9))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.24,
                        wspace=0.27, hspace=0.35)
    ax_cloud = fig.add_subplot(221, projection="3d")
    ax_arm = fig.add_subplot(222, projection="3d")
    ax_output = fig.add_subplot(223)
    ax_sensitivity = fig.add_subplot(224)

    ax_cloud.scatter(*cloud.T, s=5, c="#b7c4ce", alpha=0.35, label="training TCPs")
    ax_cloud.scatter(*nearest_xyz.T, s=38, c="#4e79a7", label="5 nearest examples")
    ax_cloud.scatter(*target, marker="x", s=110, linewidths=3, c="red", label="input target")
    ax_cloud.scatter(*predicted, s=75, c="#f28e2b", label="predicted TCP")
    nearest_point = nearest_xyz[0]
    ax_cloud.plot([target[0], nearest_point[0]], [target[1], nearest_point[1]],
                  [target[2], nearest_point[2]], "--", c="#4e79a7", label="distance to nearest data")
    ax_cloud.plot([target[0], predicted[0]], [target[1], predicted[1]],
                  [target[2], predicted[2]], c="#f28e2b", linewidth=2, label="TCP position error")
    ax_cloud.set_title(
        "Training support: grey=data, blue=nearest, red=target, orange=prediction",
        fontsize=9,
    )
    ax_cloud.set_xlabel("x (m)")
    ax_cloud.set_ylabel("y (m)")
    ax_cloud.set_zlabel("z (m)")

    ax_arm.plot(arm[:, 0], arm[:, 1], arm[:, 2], "o-", linewidth=4,
                markersize=7, color="#4e79a7", label="arm at clipped command")
    ax_arm.scatter(*target, marker="x", s=110, linewidths=3, c="red", label="input target")
    ax_arm.scatter(*predicted, s=75, c="#f28e2b", label="predicted TCP")
    ax_arm.set_title("Pose after clipping: blue=arm, red=target, orange=TCP", fontsize=9)
    ax_arm.set_xlabel("x (m)")
    ax_arm.set_ylabel("y (m)")
    ax_arm.set_zlabel("z (m)")
    bounds = np.vstack((arm, [target, predicted]))
    lows, highs = bounds.min(axis=0), bounds.max(axis=0)
    center = (lows + highs) / 2.0
    radius = max(float(np.max(highs - lows)) * 0.55, 0.15)
    ax_arm.set_xlim(center[0] - radius, center[0] + radius)
    ax_arm.set_ylim(center[1] - radius, center[1] + radius)
    ax_arm.set_zlim(max(-0.15, center[2] - radius), center[2] + radius)
    ax_arm.set_box_aspect((1, 1, 1))

    x = np.arange(3)
    ax_output.bar(x, result["raw"], color="#e15759", alpha=0.8, label="raw MLP output")
    ax_output.scatter(x, result["clipped"], marker="D", s=55, c="#f28e2b",
                      label="after clip: controller command", zorder=3)
    for row in nearest:
        ax_output.scatter(x, train_q[row], s=20, c="#4e79a7", alpha=0.65,
                          label="nearest examples" if row == nearest[0] else None, zorder=2)
    ax_output.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax_output.axhline(-1.0, color="black", linestyle="--", linewidth=0.8,
                      label="joint output limits ±1")
    ax_output.set_xticks(x, JOINT_NAMES)
    ax_output.set_ylabel("normalized joint output")
    ax_output.set_title("Raw network output and controller clipping")
    raw_extent = max(1.2, float(np.max(np.abs(result["raw"]))) * 1.15)
    ax_output.set_ylim(-raw_extent, raw_extent)
    ax_output.legend(fontsize=7, loc="best")
    for index, value in enumerate(result["raw"]):
        offset = 0.03 * raw_extent * (1 if value >= 0 else -1)
        ax_output.text(index, value + offset, f"{value:+.2f}", ha="center",
                       va="bottom" if value >= 0 else "top", fontsize=8)

    sensitivity = result["sensitivity"]
    max_abs = max(float(np.max(np.abs(sensitivity))), 0.1)
    for joint in range(3):
        ax_sensitivity.bar(np.arange(3) + (joint - 1) * 0.24, sensitivity[joint],
                           width=0.22, label=JOINT_NAMES[joint])
    ax_sensitivity.axhline(0.0, color="black", linewidth=0.6)
    ax_sensitivity.set_ylim(-max_abs * 1.2, max_abs * 1.2)
    ax_sensitivity.set_xticks(np.arange(3), ("x", "y", "z"))
    ax_sensitivity.set_xlabel("small change to target coordinate")
    ax_sensitivity.set_ylabel("raw normalized output change / metre")
    ax_sensitivity.set_title("Local response of the raw MLP")
    ax_sensitivity.legend(fontsize=7, loc="best")

    nearest_distance = float(result["distances"][nearest[0]])
    support = "WEAK TRAINING SUPPORT" if result["weak_support"] else "near training examples"
    clipped_joints = [name for name, value in zip(JOINT_NAMES, result["raw"])
                      if value <= -1.0 or value >= 1.0]
    saturation = ", ".join(clipped_joints) if clipped_joints else "none"
    units = ("rad", "rad", "m") if robot == "scara" else ("rad", "rad", "rad")
    physical = ", ".join(
        f"{name}={value:+.3f} {unit}"
        for name, value, unit in zip(JOINT_NAMES, result["joints"], units)
    )
    explanation = (
        f"INPUT: {case['kind']} | target=({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}) m\n"
        f"1. DATA: {support}; nearest training example is {nearest_distance * 1000:.0f} mm away; "
        f"95th-percentile training spacing is {support_95 * 1000:.0f} mm.\n"
        f"2. NETWORK: raw q=[{result['raw'][0]:+.2f}, {result['raw'][1]:+.2f}, {result['raw'][2]:+.2f}] "
        f"→ clipped q=[{result['clipped'][0]:+.2f}, {result['clipped'][1]:+.2f}, {result['clipped'][2]:+.2f}]; "
        f"clipped joints: {saturation}.\n"
        f"3. RESULT: {physical}; FK TCP=({predicted[0]:+.3f}, {predicted[1]:+.3f}, "
        f"{predicted[2]:+.3f}) m; target error={result['error_mm']:.0f} mm.\n"
        "Outside the sampled workspace, the MLP extrapolates its learned mapping; it is not solving constrained IK. "
        "This report uses forward kinematics, not Webots motor physics or settling."
    )
    fig.text(0.06, 0.115, explanation, ha="left", va="center", fontsize=8,
             family="monospace", linespacing=1.3)
    fig.suptitle(f"{robot.upper()} MLP diagnostic | {model_name} | {case['label']}", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    robots = ("ned", "scara") if args.robot == "both" else (args.robot,)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.csv"
    summary_rows = []

    for robot in robots:
        model, mean, std, _config = load_model(args.model, robot)
        train_rows = read_rows(DATA / robot / f"train_{args.size}.csv")
        train_xyz = np.array([[float(row[axis]) for axis in ("x", "y", "z")] for row in train_rows])
        train_q = np.array([[float(row[name]) for name in JOINT_NAMES] for row in train_rows])
        internal_distances, _ = cKDTree(train_xyz).query(train_xyz, k=2)
        support_95 = float(np.percentile(internal_distances[:, 1], 95))

        cases = robot_points(robot, args.samples) + extra_points(args.points_csv, robot)
        for case in cases:
            result = analyse_point(robot, case["point"], model, mean, std,
                                   train_xyz, train_q, support_95)
            filename = f"{robot}_{case['label']}.png"
            draw_case(robot, args.model, case, result, train_xyz, train_q,
                      support_95, args.output / filename)
            summary_rows.append({
                "robot": robot,
                "label": case["label"],
                "kind": case["kind"],
                "x": result["target"][0],
                "y": result["target"][1],
                "z": result["target"][2],
                "nearest_training_distance_m": result["distances"][result["nearest"][0]],
                "training_spacing_p95_m": support_95,
                "weak_training_support": result["weak_support"],
                **{f"raw_{name}": value for name, value in zip(JOINT_NAMES, result["raw"])},
                **{f"clipped_{name}": value for name, value in zip(JOINT_NAMES, result["clipped"])},
                **{f"joint_{name}": value for name, value in zip(JOINT_NAMES, result["joints"])},
                "predicted_x": result["predicted"][0],
                "predicted_y": result["predicted"][1],
                "predicted_z": result["predicted"][2],
                "fk_error_mm": result["error_mm"],
                "figure": filename,
            })

    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Generated {len(summary_rows)} point diagnostics in {args.output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
