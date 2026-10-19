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
│   ├── generate_mlp_diagnostics.py     # Headless MLP prediction diagnostics
│   ├── train_mlp.py                    # Task 3: shared NumPy MLP trainer
│   ├── evaluate_configurations.py      # Task 4.A: test every saved model, write CSVs
│   ├── evaluate_configurations_webots.py  # Task 4.B: first 100 test points measured in Webots
│   ├── webots_mlp_runner.py            # Headless Webots launcher used by Task 4.B
│   ├── plot_configuration_results.py   # Graphs from the Task 4.A and 4.B results
│   ├── run_fk_precision.sh             # Run Task 1.C for both arms
│   └── plot_fk_precision.py             # Summarise Task 1.C measurements
├── models/                              # Checkpoints (.npz), configs (.json), loss histories (.csv)
├── reports/configuration_evaluation/    # Task 4.A results, figures and findings
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
`mlp_ned_scara_large_h64x64x64x64_cosine_tanh` (the Task 4.A large model) by default; to select another model, add
`"--model=MODEL_NAME"` to that world's `controllerArgs` list.

While the simulation is running, use the arrow keys to move the red target
sphere in X/Y and Page Up/Page Down to change Z. Hold Shift for smaller
adjustments. Network accuracy is not guaranteed outside its trained workspace.

### Generate end-to-end MLP diagnostics

The diagnostic runner starts each selected world in headless Webots fast mode.
It sends targets through the saved MLP, applies joint-limit clipping, commands
the motors, waits for joint sensors to settle, and records the TCP measured by
Webots. Webots and the trained model weights (`models/<model>.npz`) are
required. If the weights are not present, train the default model first:

```bash
python scripts/train_mlp.py --size large --hidden 64 64 64 64 --activation tanh
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

## Task 3: train the MLP

`scripts/train_mlp.py` trains one shared NumPy network for both arms. The input
is `[is_scara, is_ned, x, y, z]` (x, y, z standardised with training-set
statistics) and the output is the three normalised joints `q1, q2, q3`. For a
given size it combines the Ned and SCARA training sets, and keeps the epoch with
the lowest MSE on the combined validation set. The test sets are never read.

```bash
python scripts/train_mlp.py --size large --hidden 64 64 64 64 --activation tanh
python scripts/train_mlp.py --size large --hidden 64 64 64 64 --activation per-robot
```

| Option | Default | Meaning |
|---|---|---|
| `--size` | `small` | `small`, `medium` or `large` training/validation sets |
| `--hidden` | `64 64` | Units per hidden layer, one number per layer |
| `--activation` | `relu` | `relu`, `tanh`, or `per-robot` (tanh for Ned rows, ReLU for SCARA rows) |
| `--lr`, `--lr-min`, `--decay` | `1e-3`, `1e-5`, `cosine` | Adam learning rate and its cosine decay (`--decay none` keeps it fixed) |
| `--epochs`, `--patience`, `--batch`, `--seed` | `3000`, `300`, `32`, `2026` | Training length, early-stopping patience, batch size, seed |

`per-robot` is still a single network with shared weights. Each hidden unit
computes `is_ned * tanh(z) + is_scara * relu(z)`, so the arm flag already in the
input chooses the non-linearity.

Each run writes `models/<name>.npz` (weights and input standardisation),
`<name>.json` (settings and best validation MSE) and `<name>_history.csv`
(train and validation MSE per epoch). The name encodes the settings, for example
`mlp_ned_scara_large_h64x64x64x64_cosine_tanh`. The Webots controller and the
evaluation scripts read the activation from the `.json`.

## Task 4.A: evaluate every configuration

```bash
python scripts/evaluate_configurations.py
python scripts/plot_configuration_results.py   # figures only; needs matplotlib
```

The evaluation script finds every model under `models/` and runs the handout's
Task 4.A procedure on the full test set of each arm. It feeds
`(x, y, z)_true` to the model, clips the predicted joints to [-1, 1] as the
controller does, de-normalises them and applies the Task 1.B FK, then compares
`(x, y, z)_pred` with `(x, y, z)_true`:

- position error: Euclidean distance in mm;
- revolute joints: `atan2(sin(θ_true − θ_pred), cos(θ_true − θ_pred))` in rad;
- prismatic joint (SCARA shaft): `(d_true − d_pred)²` in m².

For each dataset size it selects the model with the lowest validation MSE on
the normalised joints (the training loss, `best_val_mse` in each model's `.json`).
Those three models are the Task 4.A models. The validation TCP error is reported
in `summary.csv` for reference only. Outputs
go to `reports/configuration_evaluation/`:

| File | Contents |
|---|---|
| `task4a_position.csv` | The 6 position-error sets (3 models × 2 arms): mean, std, median, P95, max |
| `task4a_joints.csv` | The 18 joint-error sets (3 models × 2 arms × 3 joints): mean, std, mean absolute |
| `summary.csv` | Every model: settings, validation scores, test position and joint errors |
| `architecture_by_size.csv` | Mean test position error for each architecture and dataset size |
| `results.json` | Everything above, plus error quantiles and downsampled learning curves |
| `figures/*.png` | Graphs made by `plot_configuration_results.py` |

## Task 4.B: measure the same models in Webots

```bash
python scripts/evaluate_configurations_webots.py
python scripts/plot_configuration_results.py   # adds the 4.A vs 4.B graph
```

Run it after `evaluate_configurations.py`, which chooses the three models. The
script takes the first 100 rows of each arm's `test.csv`, in file order, and runs
each model in headless Webots. It uses temporary copies of the worlds with the
`diagnose_mlp` controller, so the original worlds are not modified. The controller
commands the predicted joints (clipped and de-normalised as in `mlp_ik`), waits for
the joints to settle, and reads the TCP from the Supervisor. Outputs in
`reports/configuration_evaluation/`:

| File | Contents |
|---|---|
| `task4b_position.csv` | The 6 position-error sets: Webots mean, std, median, P95, max and settled count, with the FK error on the same 100 points and the 4.A full-test result |
| `task4b_measurements.csv` | Every target, raw and commanded joints, sensed joints, FK TCP and Webots TCP |
| `test_sets/<robot>_test_first100.csv` | The two reduced test sets |

| Option | Effect |
|---|---|
| `--models NAME ...` | Also measure other models (file stems), e.g. the per-robot ones |
| `--only` | Measure only the `--models` list |
| `--webots PATH` | Webots launcher, if it is not found on `PATH` or `WEBOTS_HOME` |
| `--settle-time`, `--timeout` | Settling window and per-target limit, in simulated seconds |

Webots starts controllers with the first `python3` on `PATH`. The script puts the
interpreter that runs it first, so run it with the Python that has NumPy (your
`.venv`). The snap version of Webots cannot use that interpreter, so use a
standard Webots install.

`reports/configuration_evaluation/README.md` summarises the findings.
