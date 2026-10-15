"""Train a shared ANN for Ned and SCARA inverse kinematics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nn" / "splits"
OUTPUT = ROOT / "outputs" / "medium_baseline"
OUTPUT.mkdir(parents=True, exist_ok=True)

INPUTS = ["is_scara", "is_ned", "x", "y", "z"]
TARGETS = ["q1", "q2", "q3"]


def load_split(split_name):
    ned = pd.read_csv(DATA / "ned" / f"{split_name}_medium.csv")
    scara = pd.read_csv(DATA / "scara" / f"{split_name}_medium.csv")
    return pd.concat([ned, scara], ignore_index=True)


train = load_split("train")
validation = load_split("validation")

X_train = train[INPUTS].to_numpy()
y_train = train[TARGETS].to_numpy()
X_val = validation[INPUTS].to_numpy()
y_val = validation[TARGETS].to_numpy()

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
mse = mean_squared_error(y_val, predictions)
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