"""Shared test loading and the coursework's Task 4 error definitions."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.train_mlp import INPUTS, OUTPUTS, MLP

ROOT = Path(__file__).resolve().parents[1]
SIZES = ("small", "medium", "large")
ROBOTS = ("ned", "scara")


def draw_graphs(function_name, *args):
    """Run a plotting function if matplotlib is installed; return True if drawn.

    The Task 4 numbers (CSV/JSON/Markdown) need only NumPy. Figures are optional,
    so a machine without matplotlib still produces every result.
    """
    try:
        from extras.figures import plot_mlp_evaluation
    except ImportError:
        print("matplotlib not installed: skipping graphs "
              "(run extras/figures/plot_mlp_evaluation.py later to draw them).")
        return False
    getattr(plot_mlp_evaluation, function_name)(*args)
    return True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_name(size, variant):
    if variant == "per_robot":
        return f"mlp_ned_scara_{size}_h64x64x64x64_cosine_per-robot"
    suffix = "_tcp" if variant == "tcp" else ""
    return f"mlp_ned_scara_{size}_h64x64_cosine{suffix}"


def load_test(path, robot, count=None):
    """Read in file order. Task 4.B takes exactly the first 100, unshuffled."""
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not set((*INPUTS, *OUTPUTS)).issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing test columns in {path}")
        rows = list(reader)
    if not rows or (count is not None and len(rows) < count):
        raise ValueError(f"Need {count or 1} test rows in {path}; found {len(rows)}")
    rows = rows if count is None else rows[:count]
    x = np.array([[float(row[key]) for key in INPUTS] for row in rows])
    y = np.array([[float(row[key]) for key in OUTPUTS] for row in rows])
    expected = (1.0, 0.0) if robot == "scara" else (0.0, 1.0)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Nonfinite test values in {path}")
    if not np.all(x[:, :2] == expected) or np.any(np.abs(y) > 1.000000001):
        raise ValueError(f"Wrong robot flags or joint labels outside [-1, 1] in {path}")
    return x, y


def predict_model(directory, size, variant, robot, x, name=None):
    name = name or model_name(size, variant)
    directory = Path(directory)
    config = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    if robot not in config["robots"]:
        raise ValueError(f"{name} was not trained on {robot}")
    hidden = config["hidden"]
    hidden = [hidden] if isinstance(hidden, int) else hidden
    activation = config.get("activation") or "relu"  # pre-activation-flag models were ReLU
    with np.load(directory / f"{name}.npz") as stored:
        model = MLP((len(INPUTS), *hidden, len(OUTPUTS)), np.random.default_rng(0), activation=activation)
        model.params = {key: stored[key].copy() for key in model.params}
        prediction = model.predict((x - stored["mean"]) / stored["std"])
    if prediction.shape != (len(x), 3) or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid predictions from {name}")
    return prediction, config


def physical_joint_errors(robot, true_q, predicted_q):
    """Handout page 12: signed wrapped angles; squared prismatic difference.

    Returns error arrays in rad (revolute) or m² (SCARA lift), not normalised
    joint RMSE. Means of signed angular errors describe bias, not magnitude.
    """
    delta = np.asarray(true_q) - np.asarray(predicted_q)
    errors = np.arctan2(np.sin(delta), np.cos(delta))
    units = ["rad", "rad", "rad"]
    if robot == "scara":
        errors[:, 2] = delta[:, 2] ** 2
        units[2] = "m^2"
    return errors, units


def position_statistics(errors_mm):
    errors = np.asarray(errors_mm, dtype=float)
    if errors.ndim != 1 or not len(errors) or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("Expected finite nonnegative position errors for every sample")
    return {
        "mean_tcp_mm": float(errors.mean()), "std_tcp_mm": float(errors.std(ddof=0)),
        "median_tcp_mm": float(np.median(errors)), "p95_tcp_mm": float(np.percentile(errors, 95)),
        "max_tcp_mm": float(errors.max()), "rmse_tcp_mm": float(np.sqrt(np.mean(errors ** 2))),
        "within_5mm_pct": float(np.mean(errors <= 5.0) * 100),
        "within_10mm_pct": float(np.mean(errors <= 10.0) * 100),
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No evaluation rows for {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
