
"""Evaluate the shared ANN separately on Ned and SCARA test data."""

import json
from pathlib import Path
import sys

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ann_data import load_inputs_targets

DATA = ROOT / "data" / "nn" / "splits"
MODEL_PATH = ROOT / "outputs" / "medium_baseline" / "ann_model.joblib"
OUTPUT_PATH = ROOT / "outputs" / "medium_baseline" / "per_robot_test.json"

INPUTS = ["is_scara", "is_ned", "x", "y", "z"]
TARGETS = ["q1", "q2", "q3"]

model = joblib.load(MODEL_PATH)
results = {}

for robot in ["ned", "scara"]:
    path = DATA / robot / "test.csv"
    X, y_true = load_inputs_targets(path, INPUTS, TARGETS)
    y_pred = model.predict(X)

    errors = y_pred - y_true

    results[robot] = {
        "samples": int(len(X)),
        "mse_normalised_joints": float(np.mean(errors**2)),
        "rmse_normalised_joints": float(np.sqrt(np.mean(errors**2))),
        "mae_normalised_joints": float(np.mean(np.abs(errors))),
        "per_joint_mae": {
            joint: float(np.mean(np.abs(errors[:, i])))
            for i, joint in enumerate(TARGETS)
        },
    }

OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

print(json.dumps(results, indent=2))
print(f"\nSaved results to: {OUTPUT_PATH}")
