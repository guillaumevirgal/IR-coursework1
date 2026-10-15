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

By default it uses two held-out reachable targets and two deliberately
unreachable stress points for each robot. It writes one PNG per target and
`outputs/mlp_diagnostics/summary.csv`. Figures separate the sampled FK
workspace boundary and target/result markers from a focused comparison of the
five nearest training joint labels with the raw and clipped model outputs.
They identify known reachable and unreachable stress targets and show the
Webots TCP error. The workspace estimate does not include collision checking;
user-specified targets are not classified automatically.
The nearest-data warning measures training-data support, not physical reachability. The script
does not claim that a target is reachable or collision-free; it reports what
the simulated arm did with the network's command. Use `--webots` if Webots is
not found through `PATH` or `WEBOTS_HOME`. Adjust settling with `--settle-time`
and `--timeout`.

In the workspace and pose plots, red marks the requested TCP, blue marks the
Webots TCP, and grey marks the estimated workspace or arm. In the joint-value
plot only, blue means a joint value near its low limit, white is the midpoint,
and red is near its high limit. This heatmap color means joint position, not
confidence; its colorbar labels the scale. The five training rows are reference
examples; the MLP predicts from learned weights, not by selecting one row.
Each normalized value maps to a fraction of that joint's allowed range as
`(value + 1) / 2`; for example, `+0.02` is 51% through the range and `-0.98`
is 1% through it.

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
