# Unreachable-target experiment (files named `directional_fallback`)

How the network behaves when asked for a TCP position the arm cannot reach. The data folders, checkpoints and scripts use the older working name "directional fallback"; it is the same thing as "unreachable-target data". Run every command from the repository root.

The separate [directional fallback experiment](directional_fallback/README.md)
adds unreachable requests labelled with accepted joint configurations. The rule
is **point toward the request, then approach it**: match the best available
recorded direction, allow a 2° angular tolerance, then select the closest accepted
TCP in that direction. Ned uses the 3D direction from its shoulder; SCARA uses
horizontal bearing because its links cannot tilt upward. A target exactly on
SCARA's yaw axis has no horizontal bearing, so distance decides the label.

The unreachable requests are certified outside the FK workspace using conservative
geometric bounds. Every label comes from the original **training** pose bank and
links to its earlier Webots audit row. These are virtual requests paired with
recorded poses; there is no physical reading at an unreachable requested point.
The rule approximates the best recorded direction and position. The MLP learns
that rule using joint MSE, so exact alignment and collision-free motion are not
guaranteed. The report compares TCP endpoint RMSE to the requested point, error
to the chosen fallback endpoint and the effect on reachable accuracy.

Each arm retains its 12,000/3,000/1,000 reachable train/validation/test rows and
adds 6,000/1,500/1,000 unreachable rows. New queries have independent seeds for
each split; validation/test poses never supply training labels. The original
datasets and model checkpoints retain their contents, checked by saved SHA-256
hashes. Files for this experiment are:

| File/folder | Purpose |
|---|---|
| `extras/unreachable_targets/data/nn/directional_fallback/` | New train/validation/test CSVs; `kind` and `desired_x/desired_y/desired_z` are audit/metric columns, not extra network inputs |
| `extras/unreachable_targets/data/raw/directional_fallback/` | Requested point, chosen pose, direction error, reachability bound and source Webots audit row |
| `models/mlp_ned_scara_large_h64x64_cosine_directional_fallback.*` | Separate joint-loss checkpoint, config and history |
| `reports/directional_fallback/` | Held-out comparisons, graphs, simulator measurements and source hashes |

The master script generates data, calls the existing MLP trainer and evaluates
the saved checkpoint. Reproduce it under a fresh name to protect saved results:

```bash
python extras/unreachable_targets/run_fallback_training.py --name directional_fallback_repeat
```

Add `--webots PATH` to measure representative held-out requests in headless
Webots as well. For live control, keep controller `mlp_ik` and add
`"--model=mlp_ned_scara_large_h64x64_cosine_directional_fallback"` after the
robot-name argument in the world's `controllerArgs`.

The saved [per-target Webots diagnostics](../../diagnostics/saved_results_directional_fallback/README.md)
sit beside the earlier MLP outputs. To run them again:

```bash
python extras/diagnostics/generate_mlp_diagnostics.py --model mlp_ned_scara_large_h64x64_cosine_directional_fallback --output extras/diagnostics/saved_results_directional_fallback
```

Diagnostics read the training directory from this checkpoint's configuration,
so nearby examples include its synthetic unreachable requests.

The [medium experiment](directional_fallback_medium/README.md) uses
4,000 reachable + 2,000 unreachable training rows and 1,000 + 500 validation
rows per arm. Both sizes use **66.7% reachable / 33.3% unreachable** rows for
training and validation. Test rows are **50% / 50%** (1,000 of each per arm).
The [size comparison](directional_fallback_medium/size_comparison.md)
uses identical requested test points for medium and large. Medium labels come
only from its own 4,000-pose training bank; no large-bank joint labels are copied.

```bash
python extras/unreachable_targets/run_fallback_training.py --size medium --name directional_fallback_medium_repeat --test-requests extras/unreachable_targets/data/nn/directional_fallback
```

Add `--webots PATH` for the simulator comparison. The saved medium model is
`mlp_ned_scara_medium_h64x64_cosine_directional_fallback_medium`; select it with
`--model` in the same `mlp_ik` controller.

The [16,000-total 3:1 experiment](directional_fallback_total_16000/README.md)
uses 12,000 reachable + 4,000 unreachable training rows per arm and includes
joint-loss and TCP-loss checkpoints.

Both new models use the same architecture, seed and optimiser. Joint loss learns
normalised joint labels. TCP loss uses the robot's FK to compare its predicted
TCP with `desired_x,desired_y,desired_z` when those columns exist; otherwise it
uses requested XYZ. Requested XYZ always remains the model input. This allows
the unreachable examples to teach the same chosen fallback endpoints under both
losses. Test errors to requested points are reported separately from fallback
imitation error.

```bash
python scripts/train_mlp.py --size large --hidden 64 64 --loss joints --epochs 3000 --seed 2026 --splits-dir extras/unreachable_targets/data/nn/directional_fallback_total_16000 --model-name mlp_ned_scara_total16000_3to1_joints
python scripts/train_mlp.py --size large --hidden 64 64 --loss tcp --epochs 3000 --seed 2026 --splits-dir extras/unreachable_targets/data/nn/directional_fallback_total_16000 --model-name mlp_ned_scara_total16000_3to1_tcp
```

Checkpoint names are protected against replacement; choose a new `--model-name` when repeating training. For either arm in Webots, keep `mlp_ik` and append `"--model=mlp_ned_scara_total16000_3to1_joints"` or `"--model=mlp_ned_scara_total16000_3to1_tcp"` after the robot-name argument.

Add `--webots PATH` for measured endpoint errors. All saved models receive the
same test inputs. Comparisons with earlier models change data size and ratio
together; they do not isolate the effect of choosing 3:1 instead of 2:1.
The `large` filenames retain compatibility with the existing trainer; manifests
record actual counts. `--dataset-only` creates a dataset without training.
