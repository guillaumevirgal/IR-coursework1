# TCP-loss MLP diagnostics

Generated with the saved model `mlp_ned_scara_large_h64x64_cosine_tcp` using
headless Webots. These figures sit beside the joint-loss results in
`../mlp_diagnostics/`.

## Out-of-workspace cases

- [Ned: above vertical reach](ned_above_vertical_reach.png)
- [Ned: beyond horizontal reach](ned_beyond_horizontal_reach.png)
- [SCARA: beyond planar reach](scara_beyond_planar_reach.png)
- [SCARA: above lift range](scara_above_lift_range.png)

Each target graph shows the requested point, the robot's settled Webots pose,
the model's raw output and clipped motor command, and the distance to nearby
training examples. The [Ned comparison](ned_reachable_vs_unreachable.png) and
[SCARA comparison](scara_reachable_vs_unreachable.png) compare each group's
distance to the closest training example. `summary.csv` contains the measured
TCP errors and settling results for all targets.

All 8 targets settled. For the four out-of-workspace targets, measured TCP
error was 448.5 mm (Ned above vertical reach), 828.0 mm (Ned beyond horizontal
reach), 642.5 mm (SCARA beyond planar reach), and 642.9 mm (SCARA above lift
range). The script also generated two reachable held-out test point graphs per
arm.
