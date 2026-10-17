# Intelligent Robotics Coursework 1

## Repository structure

```text
IR-coursework1/
├── worlds/
│   ├── articulated.wbt                 # Ned Webots world
│   └── scara.wbt                       # SCARA Webots world
├── controllers/
│   ├── mlp_ik/
│   │   └── mlp_ik.py                   # Shared neural IK controller for both arms
│   ├── generate_dataset/
│   │   └── generate_dataset.py         # Webots dataset-generation entry point
│   ├── common/
│   │   ├── arm_controller.py           # Motor movement and TCP measurement
│   │   ├── dataset_generator.py        # Sample, audit, reject, export NN rows
│   └── fk_precision/fk_precision.py    # Task 1.C FK check in Webots
├── scripts/
│   ├── run_dataset_generation.py       # Master generation and splitting command
│   ├── install_dependencies.py          # Install Python packages from requirements.txt
│   ├── forward_kinematics_ned.py       # Task 1.B Ned FK
│   ├── forward_kinematics_scara.py     # Task 1.B SCARA FK
│   ├── train_mlp.py                    # Shared MLP trainer: joint loss or TCP loss
│   ├── tcp_loss.py                     # Batched FK and analytical TCP-loss gradients
│   ├── compare_mlp_losses.py           # Task 4.A: full independent-test evaluation
│   ├── evaluate_mlp_webots.py          # Task 4.B: first 100 test points in Webots
│   ├── plot_mlp_evaluation.py          # Graph both evaluations from saved results
│   ├── evaluation_common.py            # Shared test loading and coursework metrics
│   ├── webots_mlp_runner.py            # Shared temporary worlds and Webots launch
│   ├── generate_mlp_diagnostics.py     # Headless MLP prediction diagnostics
│   ├── run_fk_precision.sh             # Run Task 1.C for both arms
│   └── plot_fk_precision.py             # Summarise Task 1.C measurements
└── data/
    ├── raw/                             # Full audit rows, accepted and rejected
    ├── nn/                              # Normalised master data and splits
    ├── evaluation/                      # Task 1.C measurements and summary
    ├── visualise_dataset.py             # Shared Ned/SCARA audit visualiser
    └── visualisation/common.py          # CSV parsing and arm drawing helpers
```

## Install dependencies

Create or activate a Python virtual environment, then run the installer with
that environment's Python:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\python.exe scripts\install_dependencies.py
```

On Linux or macOS:

```bash
./.venv/bin/python scripts/install_dependencies.py
```

The installer reads `requirements.txt` and installs NumPy and Matplotlib into
the interpreter used to run it. Webots is a separate application and must be
installed separately; its Python controller API is supplied by Webots.

Opening either world runs `controllers/mlp_ik/mlp_ik.py`, the shared trained
network controller. The world passes `ned` or `scara` so the controller uses
the matching input flag, joint limits, motors, and forward-kinematics check.
Both worlds include a reachable target sphere. The controller loads
`mlp_ned_scara_large_h64x64_cosine` by default; to select another model, add
`"--model=MODEL_NAME"` to that world's `controllerArgs` list.

While the simulation is running, use the arrow keys to move the red target
sphere in X/Y and Page Up/Page Down to change Z. Hold Shift for smaller
adjustments. Network accuracy is not guaranteed outside its trained workspace.

### Train with joint loss or TCP loss

Both versions share the same NumPy MLP, ReLU layers, linear outputs, Adam,
input standardisation, dataset splits, seed, batches and learning-rate schedule.
Use matching arguments to compare the objectives:

```bash
python scripts/train_mlp.py --size large --hidden 64 64 --loss joints
python scripts/train_mlp.py --size large --hidden 64 64 --loss tcp
```

The baseline minimises MSE against normalised joint labels. The TCP version
clips predictions to joint limits, converts them to physical joints and uses
the existing robot FK equations to minimise mean squared distance from the
predicted TCP to the requested dataset position, in m². Analytical Jacobians
pass that error through FK and joint denormalisation into the same backpropagation.
It uses TCP targets rather than joint labels for its training objective.

The linear outputs can exceed joint limits. A small penalty pulls them back
where clipping would otherwise give zero TCP gradient:
`loss = mean(||FK(denormalise(clip(q))) - target_xyz||²)
+ tcp_limit_weight * mean(||q - clip(q)||²)`.
`--tcp-limit-weight` defaults to `0.01` m² per squared normalised joint unit;
it is a constraint penalty, not a joint-label loss. This weight can be tuned
using validation data. A zero weight removes that restoring gradient.

Each version selects its checkpoint and early stopping using its own validation
objective; the independent test set is not loaded. The TCP version saves
`models/mlp_ned_scara_large_h64x64_cosine_tcp.npz`, plus JSON configuration and
CSV loss history; baseline filenames stay unchanged. The JSON also records
per-robot validation FK error in millimetres (mean, median, P95, maximum, RMSE).
Joint MSE and TCP loss have different units and cannot be compared numerically.
Compare model quality with the same TCP metrics, then verify endpoint accuracy
using Webots measurements. FK-based training does not guarantee collision-free
poses or reliable predictions outside the training workspace.

Load the TCP model in either world with
`"--model=mlp_ned_scara_large_h64x64_cosine_tcp"` in `controllerArgs`, or run
diagnostics with:

```bash
python scripts/generate_mlp_diagnostics.py --model mlp_ned_scara_large_h64x64_cosine_tcp
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
| Each joint's signed mean/squared lift error | FK versus measured error on matching targets |
| Each joint's error standard deviation | Percentage of targets within each error threshold |
| Percentage of targets within each error threshold | |

To redraw graphs from results already saved, without retraining or running
Webots again:

```bash
python scripts/plot_mlp_evaluation.py
```

For another run folder, pass `--task4a PATH` and/or `--task4b PATH`. Blue always
means joint-loss training and orange means TCP-loss training. Unavailable
models are labelled, not plotted as zero. The accuracy curves use a log error
axis; at a chosen error threshold, a higher percentage means more accurate
predictions. Joint means and standard deviations should be read together:
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
```

By default it compares two reachable held-out test targets and two deliberately
unreachable stress points for each robot. Increase `--samples` to include more
reachable targets; each gets its own diagnostic graph and a clear `REACHABLE TARGET`
banner, while the unreachable cases are labelled separately. It writes one PNG per target and
`outputs/mlp_diagnostics/summary.csv`. Figures separate the sampled FK
workspace boundary and target/result markers from a focused comparison of the
five closest training points and the robot's final position, measured in millimetres
from the requested point.
It also writes a side-by-side overview per robot comparing how close each
reachable and unreachable target is to its nearest training point:
`ned_reachable_vs_unreachable.png` and `scara_reachable_vs_unreachable.png`.
Both panels share one distance scale; the dashed line marks typical spacing
between training points. Points beyond it have fewer nearby examples.
They identify known reachable and unreachable stress targets and show the
Webots TCP error. The workspace estimate does not include collision checking;
user-specified targets are not classified automatically.
The nearest-data warning measures training-data support, not physical reachability. The script
does not claim that a target is reachable or collision-free; it reports what
the simulated arm did with the network's command. Use `--webots` if Webots is
not found through `PATH` or `WEBOTS_HOME`. Adjust settling with `--settle-time`
and `--timeout`.

In the workspace and pose plots, red marks the requested TCP, blue marks the
Webots TCP, and grey marks the estimated workspace or arm. The training comparison
uses green dots for the five closest training points and a blue dot for where the
robot ended up. The red line at zero is the requested point: dots closer to zero
are closer to the request. Each dot states its distance in millimetres. A short
caption explains whether nearby training examples exist, using the training
set's usual spacing as context. Distance shows familiarity with the target;
it does not prove physical reachability or explain the model's internal reasoning.
The model predicts from learned weights rather than selecting a training point.

To add arbitrary target points, pass a CSV with columns `robot,label,x,y,z`:

```bash
python scripts/generate_mlp_diagnostics.py --points-csv points.csv
```

Dataset generation is separate: the master runner starts Webots headlessly and
temporarily selects `controllers/generate_dataset/generate_dataset.py`. That
entry point supplies the robot-specific joint names, limits, FK function, and
TCP slot to the shared generator. The generator writes the audit; when it
finishes, it derives the ready-to-train master CSV. The runner then creates
nested train/validation sets and an independent test set. The visualiser reads
the raw audit so measurements and rejected attempts remain inspectable.

## Generate the datasets

The master command can generate both arms in sequence:

```bash
python scripts/run_dataset_generation.py both 16000 42
```

The arguments are robot (`ned`, `scara`, or `both`), accepted sample count,
and random seed. The default coursework split sizes are 1,000 small, 5,000
medium, and 15,000 large
development samples, plus 1,000 independent test samples per robot. Each
development size is split 80/20 into training and validation. The small sets
are nested inside medium, which are nested inside large. Matching Ned and
SCARA train/validation files are combined when fitting each shared model; the
test files remain separate for per-robot evaluation.

The command creates:

```text
data/raw/ned_master_audit.csv
data/nn/ned_master.csv
data/nn/splits/ned/
├── train_small.csv
├── validation_small.csv
├── train_medium.csv
├── validation_medium.csv
├── train_large.csv
├── validation_large.csv
└── test.csv
```

The equivalent SCARA files are created alongside the Ned files. The raw audit
is retained as a checkpoint; NN and split files are rebuilt from it on each
completed run. To use different split sizes, pass `--small`,
`--medium`, `--large`, and `--test`;
the generation count must be at least `large + test`.

### Running it again

The raw audit is the resume checkpoint. If a run is interrupted, run the same
command again with the same robot and seed. The controller counts accepted
rows already in the audit, skips past the recorded random attempts, and
generates only the remaining rows needed to reach the requested accepted
sample count. It prints the existing count, remaining count, and a completion
message when the target is reached. It trims an incomplete final CSV line left
by an abrupt stop. The NN master and split files are then rebuilt from the
checkpoint; the test and nested split prefixes stay stable as the master grows.

Use the same seed when resuming. A different seed is rejected because mixing
random sequences would make the checkpoint ambiguous. Existing audit files
from the immediately preceding format are upgraded using the supplied seed;
those files did not store their original seed, so resume with the seed used for
that run. If derived NN or split files exist without their raw audit, the
runner stops and asks you to restore the audit or move those files. To start a
completely new dataset, move or remove that robot's raw audit and derived NN
outputs first. With `both`, each robot resumes independently.

The runner finds Webots through `PATH`, `WEBOTS_HOME`, or a standard install
location on Windows, Linux, or macOS. If needed, specify it explicitly:

```bash
python scripts/run_dataset_generation.py ned 16000 42 --webots "C:\\Program Files\\Webots\\msys64\\mingw64\\bin\\webots.exe"
```

## What goes into the files

The raw audit CSV records every attempted pose, along with the robot and
generation seed:

```text
robot,seed,attempt,accepted,rejection_reason,
joint1,joint2,joint3,
ex,ey,ez,ax,ay,az,error
```

`joint1` to `joint3` are physical motor values. `ex,ey,ez` are the FK TCP
coordinates and `ax,ay,az` are the Webots Supervisor measurement. `error` is
their Euclidean distance. Rejected samples stay in this file with a reason;
fields that could not be measured are blank. The complete audit makes the
sampling, FK agreement, and filtering reviewable.

The NN master contains only accepted samples, in the format:

```text
is_scara,is_ned,x,y,z,q1,q2,q3
```

`x,y,z` come from FK in metres, as Task 2 specifies. The two robot flags
identify the arm. `q1,q2,q3` are physical joint values mapped to `[-1,1]` using
that arm's joint limits. Therefore the model input is `[is_scara, is_ned, x,
y, z]` and its target is `[q1, q2, q3]`. Webots positions and errors are kept
for audit and analysis; they are not model inputs. Normalisation is part of
generation, so there is no separate normalisation command.

## Sampling and rejection rules

Only accepted rows count toward the requested sample count. The generator
continues sampling until it has that many accepted rows. Every attempt is
recorded.

| Rule | Action | Reason |
|---|---|---|
| Ned `joint_3` or SCARA `arm_motor` is negative | Not sampled | Restricts data to the coursework's elbow-up configuration. |
| Ned FK gives `z < 0.14 m` | Reject before moving | Avoids known Ned floor contact. |
| SCARA FK gives `z < 0.02 m` | Reject before moving | Avoids known SCARA floor contact. |
| Motors fail to settle within 10 simulated seconds | Reject as `timeout` | No reliable settled pose is available. |
| FK and Webots TCP differ by more than `0.001 m` | Reject as `fk_error` | Keeps inaccurate pairs out of training. |
| All checks pass | Accept | Write to raw audit and NN master. |

The height checks address known floor contact and are not a general
self-collision detector. The generator does not deduplicate samples.

## Visualise the raw audit

The shared visualiser shows accepted and rejected FK points, a selected
accepted arm pose, the FK/Webots error distribution, and rejection counts:

```bash
python data/visualise_dataset.py ned data/raw/ned_master_audit.csv
python data/visualise_dataset.py scara data/raw/scara_master_audit.csv
```

Use `--row` to select an accepted pose, `--tcp-source actual` to plot measured
positions where available, or `--save figure.png` to save a figure without
opening a window.

## Task 1.C forward-kinematics check

The existing Task 1.C controller compares FK with Webots for both arms. Run
`scripts/run_fk_precision.sh` and summarise its measurements with
`scripts/plot_fk_precision.py`. These measurements check the FK model; they are
separate from the Task 4 ANN evaluation.
