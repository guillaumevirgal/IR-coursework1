# Intelligent Robotics Coursework 1

Neural inverse kinematics for the Niryo Ned and a SCARA arm in Webots. One
shared MLP maps `[is_scara, is_ned, x, y, z]` to the three normalised joint
values. Datasets and trained models are already included, so you can run
evaluation and live control immediately without retraining anything.

## Requirements

- Webots R2023b.
- Python 3 with NumPy (standard library + NumPy is all that's needed to
  generate data, train and evaluate).
- Matplotlib, only for figures — evaluation still writes all numbers without it.


## Fastest way to see results

```bash
python scripts/evaluate_task4a.py       # Task 4.A, FK-based, no Webots needed -> outputs/task4a/
```

Then, to see it move: open `worlds/articulated.wbt` (Ned) or `worlds/scara.wbt`
(SCARA) in Webots — arrow keys move the target, Page Up/Down change Z.

## Run order

Everything runs from the repository root. Steps 1–2 are optional — data and
checkpoints are already in `data/` and `models/`.

| Step | Command | Needs Webots |
|---|---|---|
| 1. Generate data | `python scripts/generate_dataset.py both` (settings in `configs/dataset_generation.json`) | yes |
| 2. Train | `python scripts/train_all_models.py` (all models; or one: `python scripts/train_mlp.py --size large --hidden 64 64 --loss joints`) | no |
| 3. Task 4.A (FK evaluation) | `python scripts/evaluate_task4a.py` | no |
| 4. Task 4.B (Webots evaluation) | `python scripts/evaluate_task4b.py` | yes |
| 5. Live control | open `worlds/articulated.wbt` or `worlds/scara.wbt` | yes |

If Webots isn't found automatically, pass `--webots "<path to webots executable>"`
to the scripts that need it, or set `WEBOTS_HOME`.

## Where things are

| Brief task | Files |
|---|---|
| 1.A Joint control, TCP reading | `controllers/common/arm_controller.py`, `controllers/ned_simple/` |
| 1.B Forward kinematics | `scripts/forward_kinematics_ned.py`, `scripts/forward_kinematics_scara.py` |
| 1.C FK precision in Webots | `controllers/fk_precision/`, `scripts/run_fk_precision.sh`, `scripts/plot_fk_precision.py` |
| 2 Dataset generation | `controllers/generate_dataset/`, `controllers/common/dataset_generator.py`, `scripts/generate_dataset.py` |
| 3 Training | `scripts/train_mlp.py`, `scripts/train_all_models.py`, `scripts/tcp_loss.py` |
| 4.A / 4.B Evaluation | `scripts/evaluate_task4a.py`, `scripts/evaluate_task4b.py`, `scripts/task4_common.py`, `controllers/diagnose_mlp/` |
| Architecture / activation sweep (Task 3 justification) | `extras/configuration_sweep/evaluate_configurations.py`, `extras/configuration_sweep/evaluate_configurations_webots.py` — write-up and ranking in [reports/configuration_evaluation/](reports/configuration_evaluation/README.md) |
| Live neural IK | `controllers/mlp_ik/mlp_ik.py` |
| Shared joint limits and normalisation | `scripts/robot_config.py` |

```text
worlds/        Webots worlds (articulated.wbt = Ned, scara.wbt)
protos/        Ned robot definition and meshes
controllers/   Run inside Webots
scripts/       Run from a terminal
data/          raw/ (audit rows), nn/ (normalised data and splits), evaluation/
figures/       Dataset analysis plots
models/        Trained checkpoints: .npz weights, .json config, .csv loss history
outputs/       Written by your own runs (task4a/, task4b/, ...)
reports/       Saved results and experiment write-ups
tests/         python -m unittest discover -s tests
```

## Training and evaluation, in brief

Default live model: `mlp_ned_scara_large_h64x64_cosine`. `train_mlp.py`
options: `--size small|medium|large`, `--hidden`, `--loss joints|tcp`,
`--activation relu|tanh|per-robot`, `--decay cosine|none`, `--lr`, `--seed`,
`--splits-dir`, `--model-name`. The full checkpoint-by-checkpoint command
list (every trial model, not just the main ones) is in
[reports/README_extra.md](reports/README_extra.md).

`evaluate_task4a.py` (Task 4.A, FK-based) and `evaluate_task4b.py`
(Task 4.B, first 100 points per arm in headless Webots) both score every
small/medium/large checkpoint without retraining; see `--help` for
`--losses`, `--sizes`, `--output`, `--webots`. Saved, verified runs:
[reports/coursework_evaluation](reports/coursework_evaluation/README.md).

To change the live model in `worlds/articulated.wbt` / `scara.wbt`: select
the `ARM` node, edit `controllerArgs` (robot name first, then `--model=...`),
save and reset.

Diagnostics (per-target Webots plots): `python extras/diagnostics/generate_mlp_diagnostics.py`.

## Further reading

- Architecture / activation sweep (depth, width, dataset size, relu vs tanh
  vs per-robot), 36 configurations: [reports/configuration_evaluation/README.md](reports/configuration_evaluation/README.md),
  ranked by architecture in [ranking_by_architecture.md](reports/configuration_evaluation/ranking_by_architecture.md).
- Saved coursework evaluation runs: [reports/coursework_evaluation/README.md](reports/coursework_evaluation/README.md).
- Unreachable-target experiment: [extras/unreachable_targets/reports/README.md](extras/unreachable_targets/reports/README.md).
- Full per-checkpoint training commands and other condensed detail: [reports/README_extra.md](reports/README_extra.md).
- scikit-learn baseline: `python extras/sklearn_baseline/train_sklearn_baseline.py`, then `python extras/sklearn_baseline/evaluate_sklearn_baseline.py [--split test]`.

## Reproducibility

- All paths are relative to the repository root.
- Datasets are in `data/`, trained models in `models/`.
- Seeds: dataset generation `42`, MLP training `2026` (`--seed`), scikit-learn baseline `42`, unreachable-target experiment `2026`.
- Tests: `python -m unittest discover -s tests`.
