# Intelligent Robotics Coursework 1

Neural inverse kinematics for the Niryo Ned and SCARA arms in Webots. A shared
NumPy MLP takes the target TCP position and robot identifier, then predicts
normalised joint values.

## Quick start

Use Python 3 and Webots 2023b. Run commands from the repository root.

```bash
python -m venv .venv
./.venv/bin/python scripts/install_dependencies.py

# Generate a physics-measured dataset in headless Webots
python scripts/run_dataset_generation.py both 16000 42

# Train with joint-label loss or TCP-position loss
python scripts/train_mlp.py --size large --hidden 64 64 --loss joints
python scripts/train_mlp.py --size large --hidden 64 64 --loss tcp

# Coursework evaluation: FK test-set evaluation, then measured Webots evaluation
python scripts/compare_mlp_losses.py
python scripts/evaluate_mlp_webots.py
```

On Windows, use `.venv\Scripts\python.exe` in place of `./.venv/bin/python`.
Pass `--webots "C:\\path\\to\\webots.exe"` if Webots is not found automatically.

## How the files fit together

```text
worlds/                         Ned and SCARA Webots scenes
controllers/generate_dataset/   Webots entry point for physics-based data
controllers/common/              Shared robot control and dataset audit
controllers/mlp_ik/              Live neural IK controller for both arms
scripts/forward_kinematics_*.py  Robot-specific FK used for training/evaluation
scripts/run_dataset_generation.py Generate, resume and split datasets
scripts/train_mlp.py             NumPy MLP training (joint or TCP loss)
scripts/tcp_loss.py              FK and gradient calculation for TCP loss
scripts/compare_mlp_losses.py    Task 4.A evaluation on held-out test sets
scripts/evaluate_mlp_webots.py   Task 4.B measured evaluation in Webots
scripts/plot_mlp_evaluation.py   Graph saved evaluation results
data/raw/                        Full audit rows, including rejected attempts
data/nn/                         Normalised samples and train/validation/test splits
models/                          Saved network weights, configuration and history
outputs/                         Results from local evaluation and diagnostics
reports/                         Saved experiment comparisons and graph guides
```

## Dataset generation

```bash
python scripts/run_dataset_generation.py both 16000 42
```

The arguments are robot (`ned`, `scara`, or `both`), accepted-sample target, and
random seed. Webots runs headlessly so the dataset uses simulator physics. The
default nested training sizes are 1,000, 5,000, and 15,000 rows per arm, plus
1,000 independent test rows; development data are split into training and
validation sets. Change these with `--small`, `--medium`, `--large`, and `--test`.

If interrupted, rerun with the same robot and seed. Generation resumes from the
raw audit and stops when the requested target is reached. The audit records each
attempt, its FK and measured Webots TCP values, and any rejection reason. Only
accepted samples enter the normalised neural-network dataset. The floor-height
checks do not detect all self-collisions.

To inspect an audit:

```bash
python data/visualise_dataset.py ned data/raw/ned_master_audit.csv
python data/visualise_dataset.py scara data/raw/scara_master_audit.csv
```

## Training and evaluation

The model input is `[is_scara, is_ned, x, y, z]`; outputs are the three joint
values normalised to `[-1, 1]`. Joint loss minimises error against dataset joint
labels. TCP loss passes predicted joints through the appropriate FK equations
and minimises squared endpoint distance. Both losses use the same NumPy MLP.

```bash
python scripts/train_mlp.py --size medium --hidden 64 64 --loss joints
python scripts/train_mlp.py --size medium --hidden 64 64 --loss tcp
```

Task 4.A evaluates every test row using FK. Task 4.B runs the first 100 test
rows through Webots and measures the resulting TCP position. Results and graphs
are written under `outputs/task4a/` and `outputs/task4b/`.

```bash
python scripts/compare_mlp_losses.py --sizes small medium large
python scripts/evaluate_mlp_webots.py --sizes small medium large
python scripts/plot_mlp_evaluation.py  # redraw graphs from saved results
```

The verified coursework evaluation and graph guide are in
[reports/coursework_evaluation](reports/coursework_evaluation/README.md).

## Webots control and diagnostics

Open `worlds/articulated.wbt` for Ned or `worlds/scara.wbt` for SCARA. The
`mlp_ik` controller accepts the robot name and an optional model name, for
example:

```text
ned --model=mlp_ned_scara_large_h64x64_cosine_tcp
```

The target sphere moves with arrow keys; Page Up and Page Down change height.
To inspect how a model responds to targets, including targets beyond the
training workspace, run the headless diagnostic:

```bash
python scripts/generate_mlp_diagnostics.py --robot both --model mlp_ned_scara_large_h64x64_cosine_tcp
```

Diagnostic figures and summaries go to `outputs/mlp_diagnostics/`. They show
the requested target, the settled Webots TCP, the commanded pose, and nearby
training examples. Nearby examples provide context; they are not how the MLP
generates its prediction.

## Separate unreachable-target experiments

Additional datasets train the arm to choose a recorded pose that points toward
an unreachable request. They are separate from the standard datasets and
checkpoints. The fallback policy, dataset ratios, commands, and model
comparisons are documented with each experiment:

- [16,000-total 3:1 experiment](reports/directional_fallback_total_16000/README.md)
- [20,000-total 3:1 experiment](reports/directional_fallback_total_20000/README.md)
- [Earlier directional fallback experiment](reports/directional_fallback/README.md)

## Dependencies and limitations

The coursework brief allows Python's standard library, NumPy, scikit-learn,
and Webots libraries for executable coursework code. Matplotlib is used for
figures; the brief exempts plotting scripts from its library restriction.
Install listed packages with `scripts/install_dependencies.py`.

The dataset uses the elbow-up configuration. Floor checks are not full collision
detection. FK test scores do not measure simulator execution, and the network's
accuracy is not guaranteed outside its training workspace.
