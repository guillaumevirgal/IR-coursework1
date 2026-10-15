# Shared ANN Baseline — Medium Dataset

## 1. Overview

This report documents the shared Artificial Neural Network (ANN) baseline
for inverse kinematics prediction on the Ned and SCARA robots.

The model predicts three normalised joint outputs from the robot identifier
and target Cartesian position.

## 2. Dataset

- Training set: 4,000 samples per robot (8,000 combined).
- Validation set: 1,000 samples per robot (2,000 combined).
- Independent test set: 1,000 samples per robot (2,000 combined).
- Inputs: `is_scara`, `is_ned`, `x`, `y`, `z`.
- Outputs: `q1`, `q2`, `q3` (normalised joint coordinates).

The independent test sets were evaluated separately from training and
validation. Test results were not used to tune the model.

## 3. ANN Configuration

- Model: scikit-learn `MLPRegressor`, inside a `StandardScaler` pipeline.
- Hidden layers: `(64, 64)`.
- Activation: ReLU.
- Solver: Adam.
- Initial learning rate: `0.001`.
- Maximum iterations: `1000`.
- Early stopping: enabled.
- No-improvement patience: `30` iterations.
- Random state: `42`.
- Training iterations completed: `408`.

## 4. Overall Validation Results

| Metric | Result |
|---|---:|
| Normalised joint MSE | 0.001088 |
| Normalised joint RMSE | 0.032987 |
| Normalised joint MAE | 0.018354 |

## 5. Per-Robot Joint Validation Results

| Robot | Samples | Normalised RMSE | Normalised MAE |
|---|---:|---:|---:|
| Ned | 1,000 | 0.027081 | 0.016296 |
| SCARA | 1,000 | 0.037985 | 0.020412 |

### Per-joint validation MAE

| Robot | q1 MAE | q2 MAE | q3 MAE |
|---|---:|---:|---:|
| Ned | 0.018703 | 0.017114 | 0.013069 |
| SCARA | 0.011847 | 0.044493 | 0.004896 |

## 6. Independent Test Joint Results

| Robot | Samples | Normalised MSE | Normalised RMSE | Normalised MAE |
|---|---:|---:|---:|---:|
| Ned | 1,000 | 0.000441 | 0.021005 | 0.016049 |
| SCARA | 1,000 | 0.001443 | 0.037988 | 0.020043 |

### Per-joint test MAE

| Robot | q1 MAE | q2 MAE | q3 MAE |
|---|---:|---:|---:|
| Ned | 0.016938 | 0.017904 | 0.013305 |
| SCARA | 0.011902 | 0.043510 | 0.004717 |

Joint errors above are in normalised coordinates, not radians or metres.

## 7. TCP Position Error Using Forward Kinematics

TCP errors are calculated by denormalising the predicted and reference
joint values, applying the existing robot-specific forward-kinematics
functions, and measuring the Euclidean distance between the resulting
TCP positions.

### Validation TCP errors

| Robot | Mean (mm) | Median (mm) | P95 (mm) | Maximum (mm) | RMSE (mm) |
|---|---:|---:|---:|---:|---:|
| Ned | 9.912 | 7.938 | 25.185 | 49.341 | 12.371 |
| SCARA | 2.451 | 2.095 | 5.534 | 14.318 | 2.986 |

### Independent test TCP errors

| Robot | Mean (mm) | Median (mm) | P95 (mm) | Maximum (mm) | RMSE (mm) |
|---|---:|---:|---:|---:|---:|
| Ned | 10.008 | 8.054 | 24.388 | 46.087 | 12.344 |
| SCARA | 2.424 | 2.039 | 5.548 | 12.037 | 2.932 |

These are FK-based estimates of TCP position differences, not measured
physical positioning errors or actual Webots execution errors. The
interpretation depends on the correctness and coordinate conventions of
the existing forward-kinematics implementations.

## 8. Observations

- The independent test results are broadly consistent with the validation
  results for both robots.
- Ned has lower normalised joint RMSE than SCARA on the independent test set.
- SCARA has lower FK-based TCP position error, despite its higher
  normalised joint RMSE. Joint-space errors do not translate equally into
  Cartesian position errors for different robot geometries.
- The SCARA `q2` output has the largest per-joint normalised MAE among the
  reported joints.
- These results establish a baseline for comparison with alternative
  model architectures and training-set sizes.

## 9. Reproducibility

The training and evaluation scripts are in `scripts/`. The trained model
and generated JSON metrics are stored under `outputs/medium_baseline/`
locally and are not committed to Git.

To reproduce the results, install the dependencies in `requirements.txt`,
run `python scripts/train_ann.py`, and then run the validation and test
evaluation scripts. Ensure the expected dataset splits are present.

The independent test results should remain reserved for final evaluation;
model selection and hyperparameter tuning should use training and
validation data.
