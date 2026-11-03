# Unreachable-target experiment (extra)

Not part of the coursework deliverable. The coursework model, its data
(`data/nn/splits`, `data/raw`) and its scripts are outside this folder.

The experiment asks what a network should do when the requested point is outside
the workspace. Each dataset adds certified-unreachable requests to the reachable
rows. The label for an unreachable request is the closest recorded reachable pose
that points within 2 degrees of the best available direction.

## Layout

| Path | Contents |
|---|---|
| `data/nn/<name>/` | Datasets: `<robot>/train_*.csv`, `validation_*.csv`, `test.csv`, `manifest.json`, `README.md` |
| `data/raw/<name>/` | Per-request audit, one CSV per robot |
| `configs/<name>.json` | The settings that produced each dataset |
| `fallback_dataset.py` | Dataset building only (certificate, direction labelling, sampling) |
| `build_dataset.py` | Builds one dataset from a config |
| `run_fallback_training.py` | Trains a model on a dataset and writes the report in `reports/<name>/` |

Trained models stay in `models/` and reports in `reports/`, so existing model
configs and report hashes remain valid.

## Datasets

| Name | Reachable : unreachable | Test requests from |
|---|---|---|
| `directional_fallback` | 12,000 : 6,000 (2:1), large | generated here |
| `directional_fallback_medium` | 4,000 : 2,000 (2:1), medium | `directional_fallback` |
| `directional_fallback_total_16000` | 12,000 : 4,000 (3:1), large | `directional_fallback` |

All three use seed 2026 and 1,000 reachable + 1,000 unreachable test rows per robot.

## Rebuild a dataset

```
python extras/unreachable_targets/build_dataset.py extras/unreachable_targets/configs/directional_fallback.json
```

Build `directional_fallback` first: the other two take their test points from it.
Existing outputs are refused. Use `--output-root DIR` to build elsewhere.
No Webots is needed; the inputs are the reachable splits and audits named in the config.

## Train and report

```
python extras/unreachable_targets/run_fallback_training.py --name directional_fallback_repeat
```

The earlier 20,000 experiments (datasets, reports and models) were deleted. They
remain in git history.
