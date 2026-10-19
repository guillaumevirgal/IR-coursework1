"""Train a shared ANN for Ned and SCARA inverse kinematics."""

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ann_data import load_inputs_targets

DATA = ROOT / "data" / "nn" / "splits"
OUTPUT = ROOT / "outputs" / "medium_baseline"
OUTPUT.mkdir(parents=True, exist_ok=True)

INPUTS = ["is_scara", "is_ned", "x", "y", "z"]
TARGETS = ["q1", "q2", "q3"]


def load_split(split_name):
    arrays = [
        load_inputs_targets(DATA / robot / f"{split_name}_medium.csv", INPUTS, TARGETS)
        for robot in ("ned", "scara")
    ]
    return (
        np.concatenate([inputs for inputs, _ in arrays], axis=0),
        np.concatenate([targets for _, targets in arrays], axis=0),
    )


X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42,
    )),
])

print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print("Training ANN...")

model.fit(X_train, y_train)

predictions = model.predict(X_val)
mse = float(np.mean((y_val - predictions) ** 2))
rmse = float(np.sqrt(mse))
mae = float(np.mean(np.abs(y_val - predictions)))

metrics = {
    "experiment": "medium_baseline",
    "architecture": [64, 64],
    "activation": "relu",
    "optimizer": "adam",
    "learning_rate_init": 0.001,
    "random_state": 42,
    "training_rows": len(X_train),
    "validation_rows": len(X_val),
    "iterations": int(model.named_steps["mlp"].n_iter_),
    "validation_mse_normalised_joints": float(mse),
    "validation_rmse_normalised_joints": rmse,
    "validation_mae_normalised_joints": mae,
}

joblib.dump(model, OUTPUT / "ann_model.joblib")

with open(OUTPUT / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\nValidation results:")
print(json.dumps(metrics, indent=2))
print(f"\nSaved model and metrics to: {OUTPUT}")
