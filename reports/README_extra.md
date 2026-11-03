# README extra notes

Passages from the original README that were cut or condensed in the rewrite. Text is copied from the original. Each heading says which section of `README.md` it belonged to.

## Full checkpoint-by-checkpoint training commands (section 3)

`train_all_models.py` trains the two main-model rows below from the included
splits; the rest are one-off trials, run individually with `train_mlp.py` (or
`run_fallback_training.py` for the unreachable-target rows). Default names are
`mlp_ned_scara_<size>_h<hidden>_<decay>` plus `_tcp` for the TCP loss.

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

**Unreachable-target experiment** (data and scripts live in `extras/unreachable_targets/`; see its README)

| Checkpoint | Command |
|---|---|
| `..._large_h64x64_cosine_directional_fallback` | `python extras/unreachable_targets/run_fallback_training.py --name directional_fallback_repeat` |
| `..._medium_h64x64_cosine_directional_fallback_medium` | `python extras/unreachable_targets/run_fallback_training.py --size medium --name directional_fallback_medium_repeat --test-requests extras/unreachable_targets/data/nn/directional_fallback` |
| `mlp_ned_scara_total16000_3to1_joints` | `python scripts/train_mlp.py --size large --hidden 64 64 --loss joints --splits-dir extras/unreachable_targets/data/nn/directional_fallback_total_16000` and `--model-name mlp_ned_scara_total16000_3to1_joints` |
| `mlp_ned_scara_total16000_3to1_tcp` | same with `--loss tcp` and `--model-name mlp_ned_scara_total16000_3to1_tcp` |

`extras/unreachable_targets/run_fallback_training.py` without `--name` writes the original checkpoint
names; it regenerates the unreachable-target data too, so it needs Webots
only if you pass `--webots`. The test set is never loaded during training.
To reproduce a shipped name exactly, move the existing `models/<name>.*`
away first or pass `--model-name`.

## Dataset generation (section 2, opening)

The generator writes the audit; when it finishes, it derives the ready-to-train master CSV. The runner then creates nested train/validation sets and an independent test set. The visualiser reads the raw audit so measurements and rejected attempts remain inspectable.

## Resuming (section 2)

It prints the existing count, remaining count, and a completion message when the target is reached.

## File formats (section 2)

The complete audit makes the sampling, FK agreement, and filtering reviewable.

## Training one size at a time (section 4)

Choose the dataset size when training. These commands create the corresponding TCP-loss checkpoint; the `_tcp` suffix distinguishes it from the joint-loss model trained on the same split:

```bash
python scripts/train_mlp.py --size small --hidden 64 64 --loss tcp
python scripts/train_mlp.py --size medium --hidden 64 64 --loss tcp
python scripts/train_mlp.py --size large --hidden 64 64 --loss tcp
```

## Evaluating TCP-loss checkpoints only (section 5)

To evaluate only TCP-loss checkpoints, use the same `--losses tcp` selection in either coursework evaluation script:

```bash
python scripts/compare_mlp_losses.py --losses tcp
python scripts/evaluate_mlp_webots.py --losses tcp
```

Both evaluate small, medium and large by default. Add `--sizes large` (or `--sizes small medium`) to select sizes. The first command evaluates all test rows using FK; the second evaluates the first 100 test rows per arm in headless Webots.

## Large baseline option (section 5)

Add `--include-large-baseline` to also score the supplied existing large joint-loss checkpoint (the controller's default model) as a reference.

Note: the original README later says this argument is still accepted and that the large joint-loss model is included by default, so the option has no effect now.

## Windows launcher example (section 5)

On Windows, Webots is located automatically when possible. Otherwise pass its launcher explicitly, for example:

```powershell
python scripts/evaluate_mlp_webots.py --webots "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
```

## Graph notes (section 5)

Test data do not choose the checkpoint.

## Live control (section 6)

For live Webots control, keep the controller set to `mlp_ik` in both `worlds/articulated.wbt` (Ned) and `worlds/scara.wbt` (SCARA).

## Diagnostics (section 7)

If the weights are not present, train the default model first:

```bash
python scripts/train_mlp.py --size large --hidden 64 64
```

It writes one PNG per target and `outputs/mlp_diagnostics/summary.csv`. Figures separate the sampled FK workspace boundary and target/result markers from a focused comparison of the five closest training points and the robot's final position, measured in millimetres from the requested point.

Increase `--samples` to include more reachable targets; each gets its own diagnostic graph and a clear `REACHABLE TARGET` banner, while the unreachable cases are labelled separately.

The side-by-side overview images identify known reachable and unreachable stress targets and show the Webots TCP error.

The nearest-data warning measures training-data support, not physical reachability.

A short caption explains whether nearby training examples exist, using the training set's usual spacing as context. Distance shows familiarity with the target; it does not prove physical reachability or explain the model's internal reasoning.