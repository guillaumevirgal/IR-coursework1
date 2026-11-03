"""Train the scikit-learn baseline ANN (medium split) shared by Ned and SCARA.

This is the off-the-shelf comparison point for the NumPy MLP in train_mlp.py.
Input  [is_scara, is_ned, x, y, z]  ->  output normalised joints [q1, q2, q3].
The random seed is fixed (RANDOM_STATE) so the run is repeatable.

Usage:  python extras/sklearn_baseline/train_sklearn_baseline.py
Then:   python extras/sklearn_baseline/evaluate_sklearn_baseline.py [--split test]
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.load_dataset import load_inputs_targets
from scripts.robot_config import ROBOTS

DATA = ROOT / "data" / "nn" / "splits"
OUTPUT = ROOT / "outputs" / "medium_baseline"

INPUTS = ["is_scara", "is_ned", "x", "y", "z"]
TARGETS = ["q1", "q2", "q3"]
RANDOM_STATE = 42


def load_split(split_name):
    """Stack both robots' medium CSVs; the is_scara/is_ned flags tell them apart."""
    arrays = [
        load_inputs_targets(DATA / robot / f"{split_name}_medium.csv", INPUTS, TARGETS)
        for robot in ROBOTS
    ]
    return (
        np.concatenate([inputs for inputs, _ in arrays], axis=0),
        np.concatenate([targets for _, targets in arrays], axis=0),
    )


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=30,
            random_state=RANDOM_STATE,
        )),
    ])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("validation")
    print(f"Training rows: {len(x_train)}\nValidation rows: {len(x_val)}\nTraining ANN...")

    model = build_model()
    model.fit(x_train, y_train)

    errors = y_val - model.predict(x_val)
    mse = float(np.mean(errors ** 2))
    metrics = {
        "experiment": "medium_baseline",
        "architecture": [64, 64],
        "activation": "relu",
        "optimizer": "adam",
        "learning_rate_init": 0.001,
        "random_state": RANDOM_STATE,
        "training_rows": len(x_train),
        "validation_rows": len(x_val),
        "iterations": int(model.named_steps["mlp"].n_iter_),
        "validation_mse_normalised_joints": mse,
        "validation_rmse_normalised_joints": float(np.sqrt(mse)),
        "validation_mae_normalised_joints": float(np.mean(np.abs(errors))),
    }

    with (OUTPUT / "ann_model.pkl").open("wb") as file:
        pickle.dump(model, file)
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nValidation results:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model and metrics to: {OUTPUT}")


if __name__ == "__main__":
    main()
