"""Task 4.A evaluation of every saved MLP configuration (size, depth, width, activation).

For each model under extras/configuration_sweep/models/ and the sweep entries in models/ and each arm's independent test set:
  1. feed (x, y, z)_true to the model,
  2. de-normalise the predicted joints (theta_pred, d_pred),
  3. apply the Task 1.B forward kinematics,
  4. compare (x, y, z)_pred with (x, y, z)_true.

Metrics follow handout page 12:
  position error  = ||(x, y, z)_pred - (x, y, z)_true||        (reported in mm)
  revolute error  = atan2(sin(t_true - t_pred), cos(t_true - t_pred))   (rad, signed)
  prismatic error = (d_true - d_pred)^2                          (m^2)

The "Task 4.A" models (one per dataset size) are chosen by the lowest validation
MSE on the normalised joints (the training loss, saved as best_val_mse), never by
test error. The validation TCP error is also computed and reported in summary.csv
for reference, but it does not drive the selection.

Usage:
    python extras/configuration_sweep/evaluate_configurations.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.train_mlp import FORWARD_KINEMATICS, INPUTS, JOINT_LIMITS, MLP, OUTPUTS, SPLITS, denormalise

MODELS = ROOT / "models"
SWEEP_MODELS = ROOT / "extras" / "configuration_sweep" / "models"
OUTPUT = ROOT / "reports" / "configuration_evaluation"
ROBOTS = ("ned", "scara")
SIZES = ("small", "medium", "large")
JOINT_NAMES = {
    "ned": ("joint_1", "joint_2", "joint_3"),
    "scara": ("base_arm_motor", "arm_motor", "shaft_linear_motor"),
}
QUANTILES = np.linspace(0.0, 1.0, 41)
CURVE_POINTS = 150


def load_rows(robot, kind):
    with (SPLITS / robot / f"{kind}.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    x = np.array([[float(row[name]) for name in INPUTS] for row in rows])
    y = np.array([[float(row[name]) for name in OUTPUTS] for row in rows])
    return x, y


def load_model(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden = config["hidden"]
    hidden = [hidden] if isinstance(hidden, int) else list(hidden)
    activation = config.get("activation") or "relu"  # pre-activation-flag models were ReLU
    model = MLP((len(INPUTS), *hidden, len(OUTPUTS)), np.random.default_rng(0), activation=activation)
    with np.load(config_path.with_suffix(".npz")) as stored:
        model.params = {key: stored[key].copy() for key in model.params}
        mean, std = stored["mean"], stored["std"]
    return config, hidden, activation, model, mean, std


def predict_tcp(model, mean, std, robot, x, y_true_norm):
    """Return true and predicted joints, and TCP position errors in mm."""

    # Clip like controllers/mlp_ik: a motor cannot be sent beyond its limits.
    y_pred_norm = np.clip(model.predict((x - mean) / std), -1.0, 1.0)
    q_true = denormalise(y_true_norm, JOINT_LIMITS[robot])
    q_pred = denormalise(y_pred_norm, JOINT_LIMITS[robot])
    fk = FORWARD_KINEMATICS[robot]
    tcp_true = np.array([fk(row) for row in q_true], dtype=float)
    tcp_pred = np.array([fk(row) for row in q_pred], dtype=float)
    return q_true, q_pred, np.linalg.norm(tcp_pred - tcp_true, axis=1) * 1000.0


def joint_errors(robot, q_true, q_pred):
    delta = q_true - q_pred
    errors = np.arctan2(np.sin(delta), np.cos(delta))
    if robot == "scara":
        errors[:, 2] = delta[:, 2] ** 2
    return errors


def stats(values):
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}


def learning_curve(history_path):
    if not history_path.is_file():
        return None
    with history_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        return None
    step = max(1, len(rows) // CURVE_POINTS)
    picked = rows[::step] + ([rows[-1]] if (len(rows) - 1) % step else [])
    # Joint-MSE history files use train_mse/val_mse; TCP-loss history files (not
    # part of this activation sweep, but may share the models/ directory) use
    # train_loss_m2/val_loss_m2. Fall back so an unrelated model never crashes this.
    train_key = "train_mse" if "train_mse" in rows[0] else "train_loss_m2"
    val_key = "val_mse" if "val_mse" in rows[0] else "val_loss_m2"
    return {
        "epoch": [float(row["epoch"]) for row in picked],
        "train_mse": [float(row[train_key]) for row in picked],
        "val_mse": [float(row[val_key]) for row in picked],
    }


def evaluate(config_path, tests, validation):
    config, hidden, activation, model, mean, std = load_model(config_path)
    entry = {
        "name": config_path.stem,
        "group": config_path.parent.relative_to(SWEEP_MODELS if SWEEP_MODELS in (config_path.parent, *config_path.parents) else MODELS).as_posix() or ".",
        "size": config["size"],
        "hidden": hidden,
        "depth": len(hidden),
        "parameters": int(sum(value.size for value in model.params.values())),
        "activation": activation,
        "decay": config.get("decay") or "none",
        "best_epoch": config.get("best_epoch"),
        "best_val_mse": config.get("best_val_mse"),
        "val_tcp_mm": {},
        "robots": {},
        "curve": learning_curve(config_path.with_name(f"{config_path.stem}_history.csv")),
    }
    for robot in ROBOTS:
        _, _, val_mm = predict_tcp(model, mean, std, robot, *validation[config["size"]][robot])
        entry["val_tcp_mm"][robot] = float(val_mm.mean())
    entry["val_tcp_mm"]["mean"] = float(np.mean([entry["val_tcp_mm"][robot] for robot in ROBOTS]))
    for robot, (x, y_true_norm) in tests.items():
        q_true, q_pred, position_mm = predict_tcp(model, mean, std, robot, x, y_true_norm)
        errors = joint_errors(robot, q_true, q_pred)
        entry["robots"][robot] = {
            "samples": len(x),
            "position_mm": {
                **stats(position_mm),
                "median": float(np.median(position_mm)),
                "p95": float(np.percentile(position_mm, 95)),
                "max": float(position_mm.max()),
                "within_5mm_pct": float(np.mean(position_mm <= 5.0) * 100),
                "within_10mm_pct": float(np.mean(position_mm <= 10.0) * 100),
                "quantiles": [float(v) for v in np.quantile(position_mm, QUANTILES)],
            },
            "joints": [
                {
                    "joint": name,
                    "unit": "m^2" if (robot == "scara" and index == 2) else "rad",
                    **stats(errors[:, index]),
                    "mean_abs": float(np.abs(errors[:, index]).mean()),
                }
                for index, name in enumerate(JOINT_NAMES[robot])
            ],
        }
    return entry


def in_this_sweep(path):
    """Keep only this activation/architecture sweep's own checkpoints.

    models/ also holds later, unrelated families (TCP loss, directional
    fallback, total16000) trained on different objectives or datasets;
    mixing those into this comparison would corrupt the "lowest validation
    joint MSE per size" selection and the ReLU-vs-tanh comparison. Everything
    under extras/configuration_sweep/models/ is this sweep's own, apart from
    the TCP-loss checkpoints. In models/, keep only names ending "_per-robot".
    """

    name = path.stem
    if SWEEP_MODELS in path.parents:
        return not name.endswith("_tcp") and name not in ("mlp_ned_scara_small_h64", "mlp_ned_scara_small_h64x64")
    return name.endswith("_per-robot")


def main():
    tests = {robot: load_rows(robot, "test") for robot in ROBOTS}
    validation = {size: {robot: load_rows(robot, f"validation_{size}") for robot in ROBOTS} for size in SIZES}
    configs = sorted(
        path for root in (MODELS, SWEEP_MODELS) for path in root.rglob("*.json")
        if path.with_suffix(".npz").is_file() and in_this_sweep(path)
    )
    if not configs:
        raise FileNotFoundError(f"No model .json/.npz pairs under {MODELS} or {SWEEP_MODELS}")
    results = [evaluate(path, tests, validation) for path in configs]

    selected = {}
    for size in SIZES:
        candidates = [r for r in results if r["size"] == size and r["best_val_mse"] is not None]
        if candidates:
            selected[size] = min(candidates, key=lambda r: r["best_val_mse"])["name"]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "test_samples": {robot: len(tests[robot][0]) for robot in ROBOTS},
        "quantile_levels": [float(q) for q in QUANTILES],
        "task4a_selection": selected,
        "models": results,
    }
    (OUTPUT / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        header = ["model", "size", "hidden", "activation", "parameters", "best_val_mse",
                  "val_tcp_ned_mm", "val_tcp_scara_mm", "val_tcp_mean_mm", "task4a_pick"]
        for robot in ROBOTS:
            header += [f"{robot}_pos_mean_mm", f"{robot}_pos_std_mm"]
            for name in JOINT_NAMES[robot]:
                header += [f"{robot}_{name}_mean", f"{robot}_{name}_std"]
        writer.writerow(header)
        for r in results:
            row = [r["name"], r["size"], "x".join(map(str, r["hidden"])), r["activation"], r["parameters"], r["best_val_mse"],
                   r["val_tcp_mm"]["ned"], r["val_tcp_mm"]["scara"], r["val_tcp_mm"]["mean"], r["name"] in selected.values()]
            for robot in ROBOTS:
                block = r["robots"][robot]
                row += [block["position_mm"]["mean"], block["position_mm"]["std"]]
                for joint in block["joints"]:
                    row += [joint["mean"], joint["std"]]
            writer.writerow(row)

    # The Task 4.A deliverable: 6 position-error sets and 18 joint-error sets.
    with (OUTPUT / "task4a_position.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(["size", "model", "robot", "samples", "mean_mm", "std_mm", "median_mm", "p95_mm", "max_mm"])
        for size, name in selected.items():
            r = next(r for r in results if r["name"] == name)
            for robot in ROBOTS:
                p = r["robots"][robot]["position_mm"]
                writer.writerow([size, name, robot, r["robots"][robot]["samples"], p["mean"], p["std"], p["median"], p["p95"], p["max"]])
    with (OUTPUT / "task4a_joints.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(["size", "model", "robot", "joint", "unit", "mean", "std", "mean_abs"])
        for size, name in selected.items():
            r = next(r for r in results if r["name"] == name)
            for robot in ROBOTS:
                for j in r["robots"][robot]["joints"]:
                    writer.writerow([size, name, robot, j["joint"], j["unit"], j["mean"], j["std"], j["mean_abs"]])

    # Architecture x dataset size: shows which combinations were trained.
    grid = {}
    for r in results:
        key = ("x".join(map(str, r["hidden"])), r["activation"], r["decay"])
        grid.setdefault(key, {})[r["size"]] = r
    with (OUTPUT / "architecture_by_size.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(["hidden", "activation", "decay"] + [f"{size}_{robot}_mean_mm" for size in SIZES for robot in ROBOTS])
        for key, by_size in sorted(grid.items()):
            writer.writerow([*key] + [by_size[size]["robots"][robot]["position_mm"]["mean"] if size in by_size else ""
                                      for size in SIZES for robot in ROBOTS])

    print(f"{'model':58s} {'ned mm':>16s} {'scara mm':>16s}")
    for r in sorted(results, key=lambda r: r["robots"]["ned"]["position_mm"]["mean"] + r["robots"]["scara"]["position_mm"]["mean"]):
        cells = [f"{r['robots'][robot]['position_mm']['mean']:7.2f} ± {r['robots'][robot]['position_mm']['std']:6.2f}" for robot in ROBOTS]
        flag = " *" if r["name"] in selected.values() else ""
        print(f"{r['name']:58s} {cells[0]:>16s} {cells[1]:>16s}{flag}")
    print(f"\n* Task 4.A selection (lowest validation joint MSE per size): {selected}")
    print(f"Saved results.json, summary.csv, task4a_position.csv and task4a_joints.csv to {OUTPUT}")


if __name__ == "__main__":
    main()
