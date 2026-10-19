# Intelligent Robotics Coursework 1

Neural inverse kinematics for two simulated arms in Webots: the Niryo Ned (articulated) and a SCARA. One shared MLP takes a target TCP position plus a robot flag and predicts the three joint values. The repo covers dataset generation, a forward-kinematics check, training with two loss functions, and evaluation both by FK and by measurement in Webots.

## Requirements

- Webots, installed separately (it supplies the controller API)
- Python 3 with NumPy and Matplotlib (installed by the script below)

## Quick start

Run from the repository root. Details for each step are in the sections below.

```bash
python -m venv .venv
./.venv/bin/python scripts/install_dependencies.py          # Windows: .venv\Scripts\python.exe

python scripts/run_dataset_generation.py both 16000 42      # 1. data
python scripts/train_mlp.py --size large --hidden 64 64 --loss joints   # 2. train
python scripts/train_mlp.py --size large --hidden 64 64 --loss tcp
python scripts/compare_mlp_losses.py                        # 3. Task 4.A
python scripts/evaluate_mlp_webots.py                       # 4. Task 4.B
```

Then open `worlds/articulated.wbt` or `worlds/scara.wbt` in Webots to drive an arm with the trained network.

Saved small and medium checkpoints for both losses are supplied in `models/`, so evaluation works without retraining.

## Repository structure

```text
IR-coursework1/
├── worlds/
│   ├── articulated.wbt                 # Ned Webots world
│   └── scara.wbt                       # SCARA Webots world
├── controllers/
│   ├── mlp_ik/mlp_ik.py                # Shared neural IK controller for both arms
│   ├── generate_dataset/generate_dataset.py   # Webots dataset-generation entry point
│   ├── common/
│   │   ├── arm_controller.py           # Motor movement and TCP measurement
│   │   └── dataset_generator.py        # Sample, audit, reject, export NN rows
│   └── fk_precision/fk_precision.py    # Task 1.C FK check in Webots
├── scripts/
│   ├── install_dependencies.py         # Install packages from requirements.txt
│   ├── run_dataset_generation.py       # Master generation and splitting command
│   ├── forward_kinematics_ned.py       # Task 1.B Ned FK
│   ├── forward_kinematics_scara.py     # Task 1.B SCARA FK
│   ├── run_fk_precision.sh             # Run Task 1.C for both arms
│   ├── plot_fk_precision.py            # Summarise Task 1.C measurements
│   ├── train_mlp.py                    # Shared MLP trainer (joint loss or TCP loss)
│   ├── run_fallback_training.py        # Separate directional fallback dataset/model experiment
│   ├── tcp_loss.py                     # Batched FK and analytical TCP-loss gradients
│   ├── compare_mlp_losses.py           # Task 4.A: full independent-test evaluation
│   ├── evaluate_mlp_webots.py          # Task 4.B: first 100 test points in Webots
│   ├── plot_mlp_evaluation.py          # Graph both evaluations from saved results
│   ├── evaluation_common.py            # Shared test loading and coursework metrics
│   ├── webots_mlp_runner.py            # Shared temporary worlds and Webots launch
│   └── generate_mlp_diagnostics.py     # Headless MLP prediction diagnostics
├── data/
│   ├── raw/                            # Full audit rows, accepted and rejected
│   ├── nn/                             # Normalised master data and splits, including fallback experiments
│   ├── evaluation/                     # Task 1.C measurements and summary
│   ├── visualise_dataset.py            # Shared Ned/SCARA audit visualiser
│   └── visualisation/common.py         # CSV parsing and arm drawing helpers
├── models/                             # Trained checkpoints (.npz), configs (.json), loss histories (.csv)
├── outputs/                            # Results from your own runs (task4a/, task4b/, mlp_diagnostics/)
└── reports/                            # Saved, verified results and model comparisons
```

## 1. Install

Create or activate a virtual environment, then run the installer with that environment's Python. The executable coursework workflows use Python's standard library and NumPy; the requirements also include scikit-learn for the separate legacy ANN comparison and Matplotlib for figures. The coursework brief allows any plotting libraries for scripts used only to produce figures.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\python.exe scripts\install_dependencies.py
```

Linux or macOS:

```bash
./.venv/bin/python scripts/install_dependencies.py
```

## 2. Generate the datasets

```bash
python scripts/run_dataset_generation.py both 16000 42
```

Arguments: robot (`ned`, `scara` or `both`), accepted sample count, random seed. The master runner starts Webots headlessly and temporarily selects `controllers/generate_dataset/generate_dataset.py`, which supplies the robot-specific joint names, limits, FK function and TCP slot to the shared generator.

The runner finds Webots through `PATH`, `WEBOTS_HOME` or a standard install location. To set it yourself:

```bash
python scripts/run_dataset_generation.py ned 16000 42 --webots "C:\\Program Files\\Webots\\msys64\\mingw64\\bin\\webots.exe"
```

### Splits

Default sizes per robot are 1,000 small, 5,000 medium and 15,000 large development samples, plus 1,000 independent test samples. Each development size is split 80/20 into train and validation. Small sets are nested inside medium, and medium inside large. Change sizes with `--small`, `--medium`, `--large` and `--test`. The sample count must be at least `large + test`.

```text
data/raw/ned_master_audit.csv
data/nn/ned_master.csv
data/nn/splits/ned/
├── train_small.csv   validation_small.csv
├── train_medium.csv  validation_medium.csv
├── train_large.csv   validation_large.csv
└── test.csv
```

SCARA files are created alongside the Ned files. Ned and SCARA train and validation files are combined when fitting the shared model. Test files stay separate for per-robot evaluation.

### Resuming

The raw audit is the resume checkpoint. If a run is interrupted, run the same command again with the same robot and seed. The controller counts accepted rows already in the audit, skips past the recorded random attempts, and generates only what is still needed. It also trims an incomplete final CSV line left by an abrupt stop. NN and split files are rebuilt from the checkpoint, and the test set and nested prefixes stay stable as the master grows.

- A different seed is rejected, because mixing random sequences makes the checkpoint ambiguous.
- Audit files from the previous format are upgraded using the seed you supply. They did not store their original seed, so use the seed from that run.
- If derived NN or split files exist without their raw audit, the runner stops. Restore the audit or move those files.
- To start fresh, move or remove that robot's raw audit and derived NN outputs.
- With `both`, each robot resumes independently.

### File formats

Raw audit, one row per attempted pose:

```text
robot,seed,attempt,accepted,rejection_reason,joint1,joint2,joint3,ex,ey,ez,ax,ay,az,error
```

`joint1` to `joint3` are physical motor values. `ex,ey,ez` are the FK TCP coordinates and `ax,ay,az` are the Webots Supervisor measurement. `error` is their Euclidean distance. Rejected rows stay in the file with a reason, and fields that could not be measured are blank.

NN master, accepted samples only:

```text
is_scara,is_ned,x,y,z,q1,q2,q3
```

The model input is `[is_scara, is_ned, x, y, z]` and the target is `[q1, q2, q3]`. `x,y,z` come from FK in metres. `q1,q2,q3` are physical joints mapped to `[-1, 1]` using that arm's limits. Normalisation happens during generation, so there is no separate step. Webots positions and errors are kept for audit only, not as model inputs.

### Sampling and rejection rules

Only accepted rows count toward the requested total. Every attempt is recorded.

| Rule | Action | Reason |
|---|---|---|
| Ned `joint_3` or SCARA `arm_motor` is negative | Not sampled | Restricts data to the elbow-up configuration. |
| Ned FK gives `z < 0.14 m` | Reject before moving | Avoids known Ned floor contact. |
| SCARA FK gives `z < 0.02 m` | Reject before moving | Avoids known SCARA floor contact. |
| Motors fail to settle within 10 simulated seconds | Reject as `timeout` | No reliable settled pose. |
| FK and Webots TCP differ by more than `0.001 m` | Reject as `fk_error` | Keeps inaccurate pairs out of training. |
| All checks pass | Accept | Written to raw audit and NN master. |

The height checks cover known floor contact only. They are not a self-collision detector, and samples are not deduplicated.

### Visualise the raw audit

```bash
python data/visualise_dataset.py ned data/raw/ned_master_audit.csv
python data/visualise_dataset.py scara data/raw/scara_master_audit.csv
```

Shows accepted and rejected FK points, one accepted arm pose, the FK/Webots error distribution and rejection counts. Options: `--row` selects an accepted pose, `--tcp-source actual` plots measured positions where available, `--save figure.png` saves instead of opening a window.

## 3. Forward-kinematics check (Task 1.C)

Run `scripts/run_fk_precision.sh`, then summarise with `scripts/plot_fk_precision.py`. This checks the FK model against Webots for both arms. It is separate from the Task 4 network evaluation.

## 4. Train

Both losses use the same NumPy MLP (ReLU hidden layers, linear outputs), Adam, input standardisation, dataset splits, seed, batches and learning-rate schedule. Only the objective differs.

```bash
python scripts/train_mlp.py --size large --hidden 64 64 --loss joints
python scripts/train_mlp.py --size large --hidden 64 64 --loss tcp
```

Use `--size small`, `medium` or `large`.

**Joint loss** (baseline): MSE against normalised joint labels.

**TCP loss**: predictions are clipped to joint limits, converted to physical joints, and passed through the robot FK equations. The loss is the mean squared distance in m² between the predicted TCP and the requested position. Analytical Jacobians carry the error back through FK and joint denormalisation into the same backpropagation. It trains on TCP targets, not joint labels.

Linear outputs can exceed joint limits, and clipping then gives zero TCP gradient. A small penalty pulls them back:

```text
loss = mean(||FK(denormalise(clip(q))) - target_xyz||^2) + tcp_limit_weight * mean(||q - clip(q)||^2)
```

`--tcp-limit-weight` defaults to `0.01` m² per squared normalised joint unit. It is a constraint penalty, not a joint-label loss, and can be tuned on validation data. Zero removes the restoring gradient.

**Checkpoints.** Each version picks its checkpoint and early-stops on its own validation objective. The test set is never loaded during training. Baseline files keep their names, and TCP-loss files get a `_tcp` suffix, for example `models/mlp_ned_scara_large_h64x64_cosine_tcp.npz`. A JSON config and CSV loss history are saved too. The JSON records per-robot validation FK error in millimetres (mean, median, P95, maximum, RMSE).

Joint MSE and TCP loss have different units and cannot be compared numerically. Compare models using the same TCP metrics, then confirm with Webots measurements. FK-based training does not guarantee collision-free poses or accuracy outside the training workspace.

## 5. Evaluate (Tasks 4.A and 4.B)

```bash
python scripts/compare_mlp_losses.py       # Task 4.A
python scripts/evaluate_mlp_webots.py      # Task 4.B
```

Both evaluate all six checkpoints (small, medium, large, for joint loss and TCP loss) without retraining. Options:

| Option | Effect |
|---|---|
| `--losses joints` or `--losses tcp` | Evaluate one loss family |
| `--sizes small medium` | Evaluate selected sizes |
| `--output PATH` | Write to a separate run folder |
| `--model-dir PATH` | Read checkpoints from another folder |
| `--test-root PATH` | Folder containing `ned/test.csv` and `scara/test.csv` |
| `--webots PATH` | Webots launcher (4.B), needed if it is not found automatically |
| `--include-large-baseline` | Accepted for compatibility, no effect |

Only evaluate on samples that were not used to train the models.

### Task 4.A: full test set, using FK

Reads every row of each arm's `test.csv` and writes to `outputs/task4a/`:

- `summary.csv`: position error mean, standard deviation and RMSE in mm, plus extra comparison metrics.
- `joint_errors.csv`: mean and standard deviation per active joint in physical units. Angles use the handout's signed wrapped difference (rad). The SCARA lift uses squared prismatic displacement error (m²).
- `predictions.csv`: per test row, the true joints, raw prediction, clipped command, FK endpoint and errors.
- `metrics.json`, `task4a.md`, `comparison.png`: provenance, coverage and summaries.

This measures FK agreement only. It does not measure physical execution.

### Task 4.B: first 100 test rows, measured in Webots

Launches headless Webots with exactly the first 100 rows of each original test file, in file order. The `diagnose_mlp` controller commands the predicted joints, waits until sensors stay within tolerance for 0.25 simulated seconds, and the Supervisor reads the TCP. Temporary worlds are created for you, so you do not need to edit any world. Writes to `outputs/task4b/`:

- `test_sets/ned_test_first100.csv`, `scara_test_first100.csv`: the reduced tests.
- `summary.csv`: measured TCP error mean and standard deviation per model and arm, settling and timeout counts, and FK statistics on the same 100 points.
- `predictions.csv`: every command, joint sensor reading, measured TCP and error.
- `metrics.json`, `task4b.md`: model and test hashes, run status, coverage.

Statistics use population standard deviation (`ddof=0`). All 100 samples count even if some do not settle. Timeouts are reported, not discarded. Incomplete or wrong-model measurements are rejected, and FK coordinates never stand in for a missing Supervisor measurement.

A complete family needs six position summaries per task and eighteen per-joint summaries in 4.A (three sizes, two arms, three joints).

A verified run of both tasks is saved in [reports/coursework_evaluation](reports/coursework_evaluation/README.md). The earlier small/medium comparison is in [reports/tcp_loss_comparison/report.md](reports/tcp_loss_comparison/report.md) and is kept as a snapshot.

### Graphs

Both commands draw graphs automatically, with a `graphs.md` guide in each output folder. To redraw from saved results without retraining or rerunning Webots:

```bash
python scripts/plot_mlp_evaluation.py
```

Use `--task4a PATH` and `--task4b PATH` for other run folders, and `--models PATH` if the training histories are elsewhere.

| Task 4.A (all test points, FK) | Task 4.B (first 100, Webots) |
|---|---|
| Endpoint mean and standard deviation | Measured endpoint mean and standard deviation |
| Endpoint RMSE | Measured endpoint RMSE |
| Each joint's signed mean (squared lift error for SCARA) | |
| Each joint's error standard deviation | |
| Training and validation loss, plus one test score per checkpoint | Same loss graph |

In the test graphs, blue is the joint-label model and orange is the TCP-loss model. In the loss graph, rows are dataset sizes and columns are objectives. Blue is smoothed training loss, orange is smoothed validation loss (25-epoch moving average), and green diamonds mark the test loss at the checkpoint chosen by validation. Compare loss values only within one objective column. Read joint means together with standard deviations, since signed angular errors can cancel. SCARA lift errors are converted from m² to mm² and labelled as such.

The [graph gallery](reports/coursework_evaluation/graphs/README.md) shows figures from the verified run.

## 6. Live control in Webots

Opening either world runs `controllers/mlp_ik/mlp_ik.py`. The world passes `ned` or `scara` so the controller uses the matching input flag, joint limits, motors and FK check. Each world has a reachable red target sphere. While the simulation runs:

- Arrow keys move the sphere in X/Y.
- Page Up and Page Down change Z.
- Hold Shift for smaller steps.

The default model is `mlp_ned_scara_large_h64x64_cosine` (large, joint loss). To use another, select the `ARM` node in the Scene Tree and edit `controllerArgs`, keeping the robot name first. For the large TCP-loss model:

```text
# worlds/articulated.wbt (Ned)
controllerArgs [
  "ned"
  "--model=mlp_ned_scara_large_h64x64_cosine_tcp"
]

# worlds/scara.wbt (SCARA)
controllerArgs [
  "scara"
  "--model=mlp_ned_scara_large_h64x64_cosine_tcp"
]
```

Save the world and reset the simulation. For small or medium checkpoints, replace `large` with `small` or `medium`. Accuracy is not guaranteed outside the trained workspace.

## 7. Diagnostics

```bash
python scripts/generate_mlp_diagnostics.py --robot both --model mlp_ned_scara_large_h64x64_cosine_tcp
```

The small/medium joint-loss versus TCP-loss comparison is in
[reports/tcp_loss_comparison/report.md](reports/tcp_loss_comparison/report.md).
Saved small and medium checkpoints for both objectives are supplied in `models/`.
To evaluate these fixed checkpoints on the same independent test sets:

```bash
python scripts/compare_mlp_losses.py
```

Add `--include-large-baseline` to also score the supplied existing large
joint-loss checkpoint (the controller's default model) as a reference.

It writes the current Task 4.A results to `outputs/task4a/`. The older comparison
in `reports/tcp_loss_comparison/` remains a saved experiment snapshot.

### Teaching a directional response to unreachable requests

The separate [directional fallback experiment](reports/directional_fallback/README.md)
adds unreachable requests labelled with accepted joint configurations. The rule
is **point toward the request, then approach it**: match the best available
recorded direction, allow a 2° angular tolerance, then select the closest accepted
TCP in that direction. Ned uses the 3D direction from its shoulder; SCARA uses
horizontal bearing because its links cannot tilt upward. A target exactly on
SCARA's yaw axis has no horizontal bearing, so distance decides the label.

The new requests are certified outside the FK workspace using conservative
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
| `data/nn/directional_fallback/` | New train/validation/test CSVs; `kind` and `desired_x/desired_y/desired_z` are audit/metric columns, not extra network inputs |
| `data/raw/directional_fallback/` | Requested point, chosen pose, direction error, reachability bound and source Webots audit row |
| `models/mlp_ned_scara_large_h64x64_cosine_directional_fallback.*` | Separate joint-loss checkpoint, config and history |
| `reports/directional_fallback/` | Held-out comparisons, graphs, simulator measurements and source hashes |

The master script generates data, calls the existing MLP trainer and evaluates
the saved checkpoint. Reproduce it under a fresh name to protect saved results:

```bash
python scripts/run_fallback_training.py --name directional_fallback_repeat
```

Add `--webots PATH` to measure representative held-out requests in headless
Webots as well. For live control, keep controller `mlp_ik` and add
`"--model=mlp_ned_scara_large_h64x64_cosine_directional_fallback"` after the
robot-name argument in the world's `controllerArgs`.

The saved [per-target Webots diagnostics](outputs/mlp_diagnostics_directional_fallback/README.md)
sit beside the earlier MLP outputs. To run them again:

```bash
python scripts/generate_mlp_diagnostics.py --model mlp_ned_scara_large_h64x64_cosine_directional_fallback --output outputs/mlp_diagnostics_directional_fallback
```

Diagnostics read the training directory from this checkpoint's configuration,
so nearby examples include its synthetic unreachable requests.

The [medium experiment](reports/directional_fallback_medium/README.md) uses
4,000 reachable + 2,000 unreachable training rows and 1,000 + 500 validation
rows per arm. Both sizes use **66.7% reachable / 33.3% unreachable** rows for
training and validation. Test rows are **50% / 50%** (1,000 of each per arm).
The [size comparison](reports/directional_fallback_medium/size_comparison.md)
uses identical requested test points for medium and large. Medium labels come
only from its own 4,000-pose training bank; no large-bank joint labels are copied.

```bash
python scripts/run_fallback_training.py --size medium --name directional_fallback_medium_repeat --test-requests data/nn/directional_fallback
```

Add `--webots PATH` for the simulator comparison. The saved medium model is
`mlp_ned_scara_medium_h64x64_cosine_directional_fallback_medium`; select it with
`--model` in the same `mlp_ik` controller.

The [20,000-total dataset](data/nn/directional_fallback_total_20000/README.md)
contains **15,000 reachable + 5,000 unreachable training rows per robot (3:1)**.
Validation has 3,750 + 1,250 rows. The 1,000 reachable + 1,000 unreachable test
requests match previous experiments. Reachable rows come from the larger Webots
bank, with separate train/validation/test memberships; fallback labels use only
this experiment's 15,000-pose training bank. The earlier
[20,000-reachable preparation](data/nn/directional_fallback_20000/README.md)
remains a separate dataset containing 26,667 total training rows per arm.

The [16,000-total 3:1 experiment](reports/directional_fallback_total_16000/README.md)
uses 12,000 reachable + 4,000 unreachable training rows per arm and includes
joint-loss and TCP-loss checkpoints. Its results are compared with the 20,000-total
models on the same held-out requests.

Both new models use the same architecture, seed and optimiser. Joint loss learns
normalised joint labels. TCP loss uses the robot's FK to compare its predicted
TCP with `desired_x,desired_y,desired_z` when those columns exist; otherwise it
uses requested XYZ. Requested XYZ always remains the model input. This allows
the unreachable examples to teach the same chosen fallback endpoints under both
losses. Test errors to requested points are reported separately from fallback
imitation error.

```bash
python scripts/train_mlp.py --size large --hidden 64 64 --loss joints --epochs 3000 --seed 2026 --splits-dir data/nn/directional_fallback_total_20000 --model-name mlp_ned_scara_total20000_3to1_joints
python scripts/train_mlp.py --size large --hidden 64 64 --loss tcp --epochs 3000 --seed 2026 --splits-dir data/nn/directional_fallback_total_20000 --model-name mlp_ned_scara_total20000_3to1_tcp
```

Checkpoint names are protected against replacement; choose a new `--model-name`
when repeating training. For either arm in Webots, keep `mlp_ik` and append
`"--model=mlp_ned_scara_total20000_3to1_joints"` or
`"--model=mlp_ned_scara_total20000_3to1_tcp"` after the robot-name argument.

[Saved-checkpoint comparison and graphs](reports/directional_fallback_total_20000/README.md):

```bash
python scripts/run_fallback_training.py --name directional_fallback_total_20000 --evaluate-models mlp_ned_scara_large_h64x64_cosine mlp_ned_scara_large_h64x64_cosine_tcp mlp_ned_scara_large_h64x64_cosine_directional_fallback mlp_ned_scara_total20000_3to1_joints mlp_ned_scara_total20000_3to1_tcp
```

Add `--webots PATH` for measured endpoint errors. All saved models receive the
same test inputs. Comparisons with earlier models change data size and ratio
together; they do not isolate the effect of choosing 3:1 instead of 2:1.
The `large` filenames retain compatibility with the existing trainer; manifests
record actual counts. `--dataset-only` creates a dataset without training.

### Coursework evaluation: Tasks 4.A and 4.B

Run from the repository root after installing dependencies:

```bash
python scripts/compare_mlp_losses.py
python scripts/evaluate_mlp_webots.py
```

Both commands evaluate every available small/medium/large joint-loss and TCP-loss
checkpoint, without retraining. All six checkpoints are now available, including
the large TCP-loss model. Use `--losses joints`
or `--losses tcp` to evaluate one family, and `--sizes small medium` for selected
sizes. The existing `--include-large-baseline` argument remains accepted; the
large joint-loss model is now included by default.

**Task 4.A** reads every row of each arm's independent `test.csv`. It saves:

- `outputs/task4a/summary.csv`: position-error mean, standard deviation and RMSE
  in millimetres, plus additional comparison metrics.
- `joint_errors.csv`: mean and standard deviation for each active joint in
  physical units, using the handout's signed wrapped angular difference (rad)
  and squared prismatic displacement error (m² for SCARA lift).
- `predictions.csv`: every requested point, true joints, raw network prediction,
  clipped physical command, FK endpoint and errors, identified by test row.
- `metrics.json`, `task4a.md`, `comparison.png`: provenance, coverage and summaries.

**Task 4.B** launches headless Webots and uses exactly the **first 100 rows** of
each original test file, in file order. It uses the existing `diagnose_mlp`
controller to command predicted joints, wait until sensors remain within tolerance
for 0.25 simulated seconds, and read the TCP with the Supervisor. It creates
temporary worlds; you do not need to change a world's controller manually.
Outputs under `outputs/task4b/` include:

- `test_sets/ned_test_first100.csv` and `scara_test_first100.csv`: the reduced tests.
- `summary.csv`: measured TCP-error mean and standard deviation per model/arm,
  settling/timeout counts and FK statistics on the same 100 points for comparison.
- `predictions.csv`: every command, joint sensor reading, measured TCP and error.
- `metrics.json` and `task4b.md`: model/test hashes, run status and coverage.

Statistics use population standard deviation (`ddof=0`). Task 4.B includes all
100 measured samples even when some do not settle; timeouts are reported and
not silently discarded. Incomplete or wrong-model measurements are rejected.
FK coordinates never replace a missing Supervisor measurement. Each complete
family requires six position summaries in each task and eighteen per-joint
summaries in Task 4.A (three sizes × two arms × three joints).

On Windows, Webots is located automatically when possible. Otherwise pass its
launcher explicitly, for example:

```powershell
python scripts/evaluate_mlp_webots.py --webots "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
```

Use `--output PATH` for a separate run folder. Both scripts also accept
`--model-dir PATH` and `--test-root PATH`; the latter contains `ned/test.csv`
and `scara/test.csv`. Only evaluate on samples not used to train those models.
Task 4.A FK results do not measure physical execution or establish accuracy
outside the training workspace.

A verified run of both tasks, including the measured Webots statistics, is
saved in [reports/coursework_evaluation](reports/coursework_evaluation/README.md).

Both evaluation commands now create graphs automatically. Each output folder
contains a `graphs.md` guide explaining them:

| Task 4.A: all test points, FK | Task 4.B: first 100, measured in Webots |
|---|---|
| Endpoint mean and standard deviation | Measured endpoint mean and standard deviation |
| Endpoint RMSE | Measured endpoint RMSE |
| Each joint's signed mean/squared lift error | |
| Each joint's error standard deviation | |
| Training and validation loss over epochs, plus one test score per saved checkpoint | Same loss graph |

To redraw graphs from results already saved, without retraining or running
Webots again:

```bash
python scripts/plot_mlp_evaluation.py
```

For another run folder, pass `--task4a PATH` and/or `--task4b PATH`; use
`--models PATH` if its training histories are in a different folder. In test
graphs, blue is the joint-label model and orange is the TCP-error model. In the
loss graph, rows show dataset size and columns show the objective. Blue lines
are smoothed training loss, orange lines are smoothed validation loss, and
green diamonds show the test loss once at the checkpoint chosen by validation.
The lines use a 25-epoch moving average. Test data do not
choose the checkpoint. Compare loss values only within an objective column.
Joint means and standard deviations should be read together:
signed angular errors can cancel. SCARA lift errors are squared, with plot
units converted from m² to mm² and explicitly labelled.

The saved [graph gallery](reports/coursework_evaluation/graphs/README.md)
includes figures generated from the verified evaluation run.

### Generate end-to-end MLP diagnostics

The diagnostic runner starts each selected world in headless Webots fast mode.
It sends targets through the saved MLP, applies joint-limit clipping, commands
the motors, waits for joint sensors to settle, and records the TCP measured by
Webots. Webots and the trained model weights (`models/<model>.npz`) are
required. If the weights are not present, train the default model first:

```bash
python scripts/train_mlp.py --size large --hidden 64 64
```

Then run diagnostics:

```bash
python scripts/generate_mlp_diagnostics.py
python scripts/generate_mlp_diagnostics.py --robot scara --samples 4
python scripts/generate_mlp_diagnostics.py --robot both --model mlp_ned_scara_large_h64x64_cosine_tcp
python scripts/generate_mlp_diagnostics.py --points-csv points.csv   # columns: robot,label,x,y,z
```

By default each robot gets two reachable held-out test targets and two deliberately unreachable stress points. `--samples` adds more reachable targets. Use `--webots` if Webots is not found, and `--settle-time` and `--timeout` to adjust settling.

Output is one PNG per target plus `outputs/mlp_diagnostics/summary.csv`. Each robot also gets `ned_reachable_vs_unreachable.png` or `scara_reachable_vs_unreachable.png`. These compare how far each target is from its nearest training point, on one shared scale. The dashed line marks typical training-point spacing, and targets beyond it have fewer nearby examples.

| Colour | Meaning |
|---|---|
| Red | Requested TCP |
| Blue | Webots TCP (final position) |
| Grey | Estimated workspace or arm pose |
| Green dots | Five closest training points |

In the training comparison, the red line at zero is the requested point, and dots closer to zero are closer to the request. Each dot states its distance in mm.

Notes on what this does and does not show:

- Distance to training data shows how familiar a target is. It does not prove physical reachability.
- The workspace estimate has no collision checking, and user-supplied targets are not classified automatically.
- The script reports what the arm did with the network's command. It does not claim a target is reachable or collision-free.
- The network predicts from learned weights. It does not look up a training point.

## Extra notes

Extra commands, examples and clarifications that did not fit above are in [README_extra.md](reports/README_extra.md).

## Limitations

- Data covers the elbow-up configuration only.
- The floor checks are not a general self-collision detector.
- Task 4.A FK results do not establish physical execution accuracy or accuracy outside the training workspace.
