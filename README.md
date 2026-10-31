# Intelligent Robotics Coursework 1

Neural inverse kinematics for the Niryo Ned and a SCARA arm in Webots. One shared MLP maps `[is_scara, is_ned, x, y, z]` to the three normalised joint values. Results and analysis are in [reports/](reports/); this file only explains how to run the code.

## Requirements

- Webots R2023b.
- Python 3 with NumPy. Everything that produces results (dataset generation, training, evaluation, controllers) uses only the standard library and NumPy.
- scikit-learn is used only by the baseline in `scripts/train_sklearn_baseline.py`.
- Matplotlib is used only to draw figures. If it is missing, the evaluation scripts skip the graphs and still write all numbers.

```bash
python -m venv .venv
./.venv/bin/python scripts/install_dependencies.py      # Windows: .venv\Scripts\python.exe
```

## Run order

Run everything from the repository root. The datasets (`data/`) and trained models (`models/`) are included, so steps 1 and 2 are optional.

| Step | Command | Needs Webots |
|---|---|---|
| 1. Generate data | `python scripts/run_dataset_generation.py both 16000 42` | yes |
| 2. Train | `python scripts/train_all_models.py` (all 14 models; or one model: `python scripts/train_mlp.py --size large --hidden 64 64 --loss joints`) | no |
| 3. Task 4.A (FK evaluation) | `python scripts/compare_mlp_losses.py` | no |
| 4. Task 4.B (Webots evaluation) | `python scripts/evaluate_mlp_webots.py` | yes |
| 5. Live control | open `worlds/articulated.wbt` or `worlds/scara.wbt` | yes |

**First script to run:** `python scripts/compare_mlp_losses.py`. It works straight after installing, using the supplied models.

If Webots is not found automatically, pass `--webots "<path to webots executable>"` to the scripts that need it, or set `WEBOTS_HOME`.

## Where things are

| Brief task | Files |
|---|---|
| 1.A Joint control, TCP reading | `controllers/common/arm_controller.py` (`ArmController`, used by both arms), `controllers/ned_simple/` |
| 1.B Forward kinematics | `scripts/forward_kinematics_ned.py`, `scripts/forward_kinematics_scara.py` |
| 1.C FK precision in Webots | `controllers/fk_precision/`, `scripts/run_fk_precision.sh`, `scripts/plot_fk_precision.py` |
| 2 Dataset generation | `controllers/generate_dataset/`, `controllers/common/dataset_generator.py`, `scripts/run_dataset_generation.py` |
| 3 Training | `scripts/train_mlp.py`, `scripts/train_all_models.py`, `scripts/tcp_loss.py` |
| 4.A / 4.B Evaluation | `scripts/compare_mlp_losses.py`, `scripts/evaluate_mlp_webots.py`, `scripts/evaluation_common.py`, `controllers/diagnose_mlp/` |
| Live neural IK | `controllers/mlp_ik/mlp_ik.py` |
| Shared joint limits and normalisation | `scripts/robot_config.py` |

```text
worlds/        Webots worlds (articulated.wbt = Ned, scara.wbt)
protos/        Ned robot definition and meshes
controllers/   Run inside Webots
scripts/       Run from a terminal
data/          raw/ (audit rows), nn/ (normalised data and splits), evaluation/
models/        Trained checkpoints: .npz weights, .json config, .csv loss history
outputs/       Written by your own runs (task4a/, task4b/, ...)
reports/       Saved results and experiment write-ups
tests/         python -m unittest discover -s tests
```

## 1. Generate the datasets

```bash
python scripts/run_dataset_generation.py both 16000 42     # robot (ned|scara|both), accepted samples, seed
```

The runner starts Webots headless, runs `controllers/generate_dataset`, then writes:

```text
data/raw/<robot>_master_audit.csv      every attempt, accepted or rejected
data/nn/<robot>_master.csv             accepted rows: is_scara,is_ned,x,y,z,q1,q2,q3
data/nn/splits/<robot>/                train/validation_{small,medium,large}.csv and test.csv
```

Split sizes per robot: 1,000 small, 5,000 medium, 15,000 large (80/20 train/validation) plus 1,000 test. Change them with `--small`, `--medium`, `--large`, `--test`. Joints are normalised to `[-1, 1]` during generation using `scripts/robot_config.py`.

If a run is interrupted, repeat the same command with the same seed to resume from the audit file. To start fresh, remove that robot's audit and derived files.

Inspect an audit: `python data/visualise_dataset.py ned data/raw/ned_master_audit.csv` (use `--save figure.png` to save).

## 2. FK check (Task 1.C)

```bash
scripts/run_fk_precision.sh          # runs both arms in Webots
python scripts/plot_fk_precision.py  # summarise
```

## 3. Train

**Retrain all 14 models with one command** (the exact settings used for the report; seed 2026, 3000 epochs, patience 300, batch 32):

```bash
python scripts/train_all_models.py --dry-run     # list the 14 train commands
python scripts/train_all_models.py               # train all; names get "_repro", shipped models untouched
python scripts/train_all_models.py --only large  # subset, e.g. only large-data models
```

It needs only NumPy and the included `data/`. Training all models takes a while; `--only` runs a subset. Checkpoints are never overwritten, so retrained models are saved with the `_repro` suffix (change it with `--suffix`).

The same commands one by one (`train_mlp.py` trains one model per call). To reproduce a shipped name exactly, move the existing `models/<name>.*` away first or pass `--model-name`. The test set is never loaded during training.

Default names are `mlp_ned_scara_<size>_h<hidden>_<decay>` plus `_tcp` for the TCP loss.

**Main models (reachable data, joint loss and TCP loss)**

| Checkpoint | Command |
|---|---|
| `mlp_ned_scara_small_h64x64_cosine` | `python scripts/train_mlp.py --size small --hidden 64 64 --loss joints` |
| `mlp_ned_scara_medium_h64x64_cosine` | `python scripts/train_mlp.py --size medium --hidden 64 64 --loss joints` |
| `mlp_ned_scara_large_h64x64_cosine` (default live model) | `python scripts/train_mlp.py --size large --hidden 64 64 --loss joints` |
| `mlp_ned_scara_small_h64x64_cosine_tcp` | `python scripts/train_mlp.py --size small --hidden 64 64 --loss tcp` |
| `mlp_ned_scara_medium_h64x64_cosine_tcp` | `python scripts/train_mlp.py --size medium --hidden 64 64 --loss tcp` |
| `mlp_ned_scara_large_h64x64_cosine_tcp` | `python scripts/train_mlp.py --size large --hidden 64 64 --loss tcp` |

**Early architecture and schedule trials (small data, constant learning rate)**

| Checkpoint | Command |
|---|---|
| `mlp_ned_scara_small_h64x64` | `python scripts/train_mlp.py --size small --hidden 64 64 --decay none --model-name mlp_ned_scara_small_h64x64` |
| `mlp_ned_scara_small_h64` | `python scripts/train_mlp.py --size small --hidden 64 --decay none --model-name mlp_ned_scara_small_h64` |

**Unreachable-target experiment** (needs the `data/nn/directional_fallback*` splits, which are included)

| Checkpoint | Command |
|---|---|
| `..._large_h64x64_cosine_directional_fallback` | `python scripts/run_fallback_training.py --name directional_fallback_repeat` |
| `..._medium_h64x64_cosine_directional_fallback_medium` | `python scripts/run_fallback_training.py --size medium --name directional_fallback_medium_repeat --test-requests data/nn/directional_fallback` |
| `mlp_ned_scara_total20000_3to1_joints` | `python scripts/train_mlp.py --size large --hidden 64 64 --loss joints --splits-dir data/nn/directional_fallback_total_20000 --model-name mlp_ned_scara_total20000_3to1_joints` |
| `mlp_ned_scara_total20000_3to1_tcp` | same with `--loss tcp` and `--model-name mlp_ned_scara_total20000_3to1_tcp` |
| `mlp_ned_scara_total16000_3to1_joints` | same as the 20000 joints command with `--splits-dir data/nn/directional_fallback_total_16000` and `--model-name mlp_ned_scara_total16000_3to1_joints` |
| `mlp_ned_scara_total16000_3to1_tcp` | same with `--loss tcp` and `--model-name mlp_ned_scara_total16000_3to1_tcp` |

(`train_all_models.py` trains the two first rows from the included splits. `run_fallback_training.py` without `--name` writes the original checkpoint names; it regenerates the unreachable-target data too, so it needs Webots only if you pass `--webots`.)

Options of `train_mlp.py`: `--size small|medium|large`, `--hidden`, `--loss joints|tcp`, `--decay cosine|none`, `--lr`, `--lr-min`, `--batch`, `--epochs`, `--patience`, `--seed`, `--tcp-limit-weight` (default 0.01), `--splits-dir`, `--model-name`. Outputs per model: `models/<name>.npz` (weights), `.json` (settings and best epoch) and `.csv` (loss history).

## 4. Evaluate

```bash
python scripts/compare_mlp_losses.py       # Task 4.A -> outputs/task4a/
python scripts/evaluate_mlp_webots.py      # Task 4.B -> outputs/task4b/
```

Both evaluate every small/medium/large checkpoint for joint loss and TCP loss, without retraining.

| Option | Effect |
|---|---|
| `--losses joints` / `--losses tcp` | One loss family only |
| `--sizes small medium` | Selected sizes only |
| `--output PATH` | Different output folder |
| `--model-dir PATH` | Checkpoints from another folder |
| `--test-root PATH` | Folder with `ned/test.csv` and `scara/test.csv` |
| `--webots PATH` | Webots executable (4.B) |

Task 4.A scores every test row with forward kinematics. Task 4.B runs the first 100 test rows of each arm in headless Webots and reads the TCP with the Supervisor; temporary worlds are created automatically. Each output folder gets `summary.csv`, `predictions.csv`, `metrics.json` and a Markdown report, plus graphs if Matplotlib is installed.

Redraw graphs from saved results: `python scripts/plot_mlp_evaluation.py` (`--task4a PATH`, `--task4b PATH`).

Saved, verified runs: [reports/coursework_evaluation](reports/coursework_evaluation/README.md).

## 5. Live control in Webots

Open `worlds/articulated.wbt` (Ned) or `worlds/scara.wbt` (SCARA). The world runs `controllers/mlp_ik/mlp_ik.py`. In the simulation: arrow keys move the red target in X/Y, Page Up/Down change Z, Shift gives smaller steps.

The default model is `mlp_ned_scara_large_h64x64_cosine`. To use another, select the `ARM` node, edit `controllerArgs` (robot name first), save and reset:

```text
controllerArgs [
  "ned"                                                  # or "scara"
  "--model=mlp_ned_scara_large_h64x64_cosine_tcp"
]
```

## 6. Diagnostics

```bash
python scripts/generate_mlp_diagnostics.py                                  # both arms, default model
python scripts/generate_mlp_diagnostics.py --robot scara --samples 4
python scripts/generate_mlp_diagnostics.py --model mlp_ned_scara_large_h64x64_cosine_tcp
python scripts/generate_mlp_diagnostics.py --points-csv points.csv          # columns: robot,label,x,y,z
```

Runs targets through the network in headless Webots and writes one PNG per target and `summary.csv` to `outputs/mlp_diagnostics/`. Options: `--webots`, `--settle-time`, `--timeout`, `--output`.

## 7. Other experiments

- Unreachable-target experiment (data and checkpoints named `directional_fallback`): [reports/README_unreachable_targets.md](reports/README_unreachable_targets.md).
- Extra commands: [reports/README_extra.md](reports/README_extra.md).
- scikit-learn baseline: `python scripts/train_sklearn_baseline.py`, then `python scripts/evaluate_sklearn_baseline.py [--split test]`.

## Reproducibility

- All paths are relative to the repository root.
- Datasets are in `data/`, trained models in `models/`.
- Seeds: dataset generation `42`, MLP training `2026` (`--seed`), scikit-learn baseline `42`, unreachable-target experiment `2026`.
- Tests: `python -m unittest discover -s tests`.
- Submission ZIP: `python scripts/make_submission_zip.py`.
