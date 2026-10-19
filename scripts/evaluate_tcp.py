
"""Evaluate ANN TCP positioning errors on the medium validation sets."""

import json
import sys
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nn" / "splits"
MODEL_PATH = ROOT / "outputs" / "medium_baseline" / "ann_model.joblib"
OUTPUT_PATH = ROOT / "outputs" / "medium_baseline" / "tcp_validation.json"

sys.path.insert(0, str(ROOT))

from scripts.ann_data import load_inputs_targets
from scripts.forward_kinematics_ned import forward_kinematics_ned
from scripts.forward_kinematics_scara import forward_kinematics_scara

INPUTS = ["is_scara", "is_ned", "x", "y", "z"]
TARGETS = ["q1", "q2", "q3"]

JOINT_LIMITS = {
    "ned": [(-2.8, 2.8), (-0.8, 0.8), (0.0, 1.3)],
    "scara": [(-0.73, 0.73), (0.0, 0.83), (-0.2, 0.0)],
}

FK_FUNCTIONS = {
    "ned": forward_kinematics_ned,
    "scara": forward_kinematics_scara,
}


def denormalise_joints(values, limits):
    """Convert normalised joint coordinates back to physical values."""
    values = np.asarray(values)
    lower = np.array([limit[0] for limit in limits])
    upper = np.array([limit[1] for limit in limits])
    return lower + (values + 1.0) * (upper - lower) / 2.0


def calculate_tcp_positions(joints, fk_function):
    """Calculate TCP positions in metres using the robot's existing FK."""
    return np.asarray([fk_function(row) for row in joints], dtype=float)


model = joblib.load(MODEL_PATH)
results = {}

for robot in ["ned", "scara"]:
    X, y_true_normalised = load_inputs_targets(DATA / robot / "validation_medium.csv", INPUTS, TARGETS)
    y_pred_normalised = model.predict(X)

    limits = JOINT_LIMITS[robot]
    fk_function = FK_FUNCTIONS[robot]

    y_true = denormalise_joints(y_true_normalised, limits)
    y_pred = denormalise_joints(y_pred_normalised, limits)

    tcp_true = calculate_tcp_positions(y_true, fk_function)
    tcp_pred = calculate_tcp_positions(y_pred, fk_function)

    # Euclidean distance between predicted and reference TCP positions.
    errors_m = np.linalg.norm(tcp_pred - tcp_true, axis=1)
    errors_mm = errors_m * 1000.0

    results[robot] = {
        "samples": int(len(X)),
        "mean_tcp_error_mm": float(np.mean(errors_mm)),
        "median_tcp_error_mm": float(np.median(errors_mm)),
        "p95_tcp_error_mm": float(np.percentile(errors_mm, 95)),
        "max_tcp_error_mm": float(np.max(errors_mm)),
        "rmse_tcp_error_mm": float(np.sqrt(np.mean(errors_mm**2))),
    }

OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

print("TCP validation results (millimetres):")
print(json.dumps(results, indent=2))
print(f"\nSaved results to: {OUTPUT_PATH}")
