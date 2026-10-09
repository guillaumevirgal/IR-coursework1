See the wiki for a basic tutorial.

## Dataset generation

The shared generation logic is in `controllers/common/dataset_generator.py`.
Both robots use `controllers/main/main.py`; their world files pass either
`ned` or `scara` as the first controller argument. Starting either world
continuously generates random elbow-up samples under `data/csv/`.

The main controller accepts optional Webots arguments. For example,
`scara --samples 1000 --seed 42` generates exactly 1,000 valid, reproducible
SCARA rows and then closes the simulation. Edit `POSES` in `main.py` to check
fixed poses.

All generated files use the generic columns `joint1`, `joint2`, and `joint3`.
For SCARA, `joint3` represents the third active model output,
`shaft_linear_motor` (the physical q4 joint).

Before commanding Webots, the generator rejects poses whose FK endpoint is
below the floor-clearance threshold: 0.14 m for Ned and 0.02 m for SCARA.
It also rejects poses that fail to settle or exceed 1 mm FK/TCP error.
