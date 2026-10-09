See the wiki for a basic tutorial.

## Structure

- `controllers/main`: shared Ned/SCARA dataset entry point.
- `controllers/common`: reusable arm control and dataset generation.
- `controllers/fk_precision`: Task 1.C FK validation for both robots.
- `scripts/forward_kinematics_*`: Task 1.B FK implementations.
- `scripts/normalise_dataset.py`: Task 2 joint normalisation.
- `data/visualise_dataset.py`: shared dataset inspection.
- `worlds` and `protos`: Webots environments and Ned model assets.

## Dataset generation

The shared generation logic is in `controllers/common/dataset_generator.py`.
Both robots use `controllers/main/main.py`; their world files pass either
`ned` or `scara` as the first controller argument. Starting either world
continuously generates random elbow-up samples under `data/csv/`.

This separation keeps sampling and CSV behaviour identical while leaving motor
names, limits, FK and fixed joints in one small robot configuration. Elbow-up
sampling makes the position-to-joint mapping single-valued for model training.

The main controller accepts optional Webots arguments. For example,
`scara --samples 1000 --seed 42` generates exactly 1,000 valid, reproducible
SCARA rows and then closes the simulation. Edit `POSES` in `main.py` to check
fixed poses.

All generated files use the generic columns `joint1`, `joint2`, and `joint3` so
both robots can train one network. For SCARA, `joint3` is the third active model
output, `shaft_linear_motor` (physical q4). Values remain raw here so the FK and
Webots measurements are auditable; normalisation is a separate processing step.

Before commanding Webots, the generator rejects poses whose FK endpoint is
below the floor-clearance threshold: 0.14 m for Ned and 0.02 m for SCARA.
These limits avoid known floor contacts. It also rejects poses that fail to
settle or exceed 1 mm FK/TCP error, preventing obstructed or inaccurate rows
from entering the dataset. A fixed seed makes comparisons reproducible.

## Dataset visualisation

`data/visualise_dataset.py` uses the shared geometry and CSV helpers in
`data/visualisation/common.py` for both robots. It displays the TCP distribution,
one robot pose, and the FK/Webots error histogram. Sharing the plots makes the
robots directly comparable while retaining their different geometry. For example:

```bash
python3 data/visualise_dataset.py ned data/csv/ned_robot_data_1e-3.csv
python3 data/visualise_dataset.py scara data/csv/scara_robot_data_1e-3.csv
```

Use `--row` to select the pose, `--tcp-source expected` to plot FK positions,
or `--save figure.png` to save without opening an interactive window.
