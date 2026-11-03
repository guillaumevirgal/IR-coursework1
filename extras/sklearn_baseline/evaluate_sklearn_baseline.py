"""Evaluate the scikit-learn baseline ANN on the validation or test split.

Reports, per robot, the normalised joint error and the TCP error obtained by
sending the predicted joints through that robot's forward kinematics.

Usage (after extras/sklearn_baseline/train_sklearn_baseline.py):
    python extras/sklearn_baseline/evaluate_sklearn_baseline.py                 # validation split
    python extras/sklearn_baseline/evaluate_sklearn_baseline.py --split test    # independent test split
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.load_dataset import load_inputs_targets
from scripts.robot_config import FORWARD_KINEMATICS, JOINT_LIMITS, ROBOTS, denormalise

DATA = ROOT / "data" / "nn" / "splits"
OUTPUT = ROOT / "outputs" / "medium_baseline"
INPUTS = ["is_scara", "is_ned", "x", "y", "z"]
TARGETS = ["q1", "q2", "q3"]
SPLIT_FILES = {"validation": "validation_medium.csv", "test": "test.csv"}


def joint_metrics(errors):
    """Errors are in normalised joint units, i.e. a fraction of half the joint range."""
    return {
        "mse_normalised_joints": float(np.mean(errors ** 2)),
        "rmse_normalised_joints": float(np.sqrt(np.mean(errors ** 2))),
        "mae_normalised_joints": float(np.mean(np.abs(errors))),
        "per_joint_mae": {joint: float(np.mean(np.abs(errors[:, i])))
                          for i, joint in enumerate(TARGETS)},
    }


def tcp_metrics(robot, true_normalised, predicted_normalised):
    """Euclidean TCP distance (mm) between FK(true joints) and FK(predicted joints)."""
    limits = JOINT_LIMITS[robot]
    forward_kinematics = FORWARD_KINEMATICS[robot]
    tcp_true = np.array([forward_kinematics(q) for q in denormalise(true_normalised, limits)])
    tcp_pred = np.array([forward_kinematics(q) for q in denormalise(predicted_normalised, limits)])
    errors_mm = np.linalg.norm(tcp_pred - tcp_true, axis=1) * 1000.0
    return {
        "mean_tcp_error_mm": float(np.mean(errors_mm)),
        "median_tcp_error_mm": float(np.median(errors_mm)),
        "p95_tcp_error_mm": float(np.percentile(errors_mm, 95)),
        "max_tcp_error_mm": float(np.max(errors_mm)),
        "rmse_tcp_error_mm": float(np.sqrt(np.mean(errors_mm ** 2))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLIT_FILES, default="validation")
    args = parser.parse_args()

    with (OUTPUT / "ann_model.pkl").open("rb") as file:
        model = pickle.load(file)

    results = {}
    for robot in ROBOTS:
        x, y_true = load_inputs_targets(DATA / robot / SPLIT_FILES[args.split], INPUTS, TARGETS)
        y_pred = model.predict(x)
        results[robot] = {"samples": int(len(x)),
                          **joint_metrics(y_pred - y_true),
                          **tcp_metrics(robot, y_true, y_pred)}

    output_path = OUTPUT / f"{args.split}_results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
