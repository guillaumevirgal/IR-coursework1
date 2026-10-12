# Intelligent Robotics Coursework 1

## Repository structure

```text
IR-coursework1/
├── worlds/
│   ├── articulated.wbt                 # Ned Webots world
│   └── scara.wbt                       # SCARA Webots world
├── controllers/
│   ├── main/main.py                    # Robot config and Webots entry point
│   ├── common/
│   │   ├── arm_controller.py           # Motor movement and TCP measurement
│   │   └── dataset_generator.py        # Sample, audit, reject, export NN rows
│   └── fk_precision/fk_precision.py    # Task 1.C FK check in Webots
├── scripts/
│   ├── run_dataset_generation.py       # Master generation and splitting command
│   ├── forward_kinematics_ned.py       # Task 1.B Ned FK
│   ├── forward_kinematics_scara.py     # Task 1.B SCARA FK
│   ├── run_fk_precision.sh             # Run Task 1.C for both arms
│   └── plot_fk_precision.py             # Summarise Task 1.C measurements
└── data/
    ├── raw/                             # Full audit rows, accepted and rejected
    ├── nn/                              # Normalised master data and splits
    ├── evaluation/                      # Task 1.C measurements and summary
    ├── visualise_dataset.py             # Shared Ned/SCARA audit visualiser
    └── visualisation/common.py          # CSV parsing and arm drawing helpers
```

The master runner starts Webots headlessly. The selected world calls
`controllers/main/main.py`, which supplies the robot-specific joint names,
limits, FK function, and TCP slot to the shared generator. The generator writes
the audit. When generation finishes, it derives the ready-to-train master CSV;
the runner then creates nested train/validation sets and an independent test
set. The visualiser reads the raw audit so measurements and rejected attempts
remain inspectable.

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

The equivalent SCARA files are created alongside the Ned files. Existing
outputs are not overwritten. To use different split sizes, pass `--small`,
`--medium`, `--large`, and `--test`;
the generation count must be at least `large + test`.

### Running it again

The runner does not append to or resume an earlier run. Before starting, it
checks that the raw audit file, NN master file, and split directory do not
already exist. If any one exists, it stops before launching Webots. A run that
was interrupted may leave a partial raw audit file; that file also blocks a
restart.

To start a fresh run, first move or remove the previous outputs for each robot:

```text
data/raw/ned_master_audit.csv
data/nn/ned_master.csv
data/nn/splits/ned/
data/raw/scara_master_audit.csv
data/nn/scara_master.csv
data/nn/splits/scara/
```

When using `both`, clear the outputs for both arms before rerunning. The runner
checks both sets of destinations before it starts, so an existing Ned output
also prevents a `both` run from starting SCARA. Keep a copy of the audit files
if you need to preserve the previous dataset.

The runner finds Webots through `PATH`, `WEBOTS_HOME`, or a standard install
location on Windows, Linux, or macOS. If needed, specify it explicitly:

```bash
python scripts/run_dataset_generation.py ned 16000 42 --webots "C:\\Program Files\\Webots\\msys64\\mingw64\\bin\\webots.exe"
```

## What goes into the files

The raw audit CSV records every attempted pose:

```text
attempt,accepted,rejection_reason,
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
