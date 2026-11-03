# Directional fallback model: measured Webots diagnostics

Model: `mlp_ned_scara_large_h64x64_cosine_directional_fallback`, trained with
joint loss on the separate reachable-plus-unreachable dataset.

All eight diagnostic targets settled in Webots. `summary.csv` records the
measurements. The main [held-out evaluation report](../../reports/directional_fallback/README.md)
compares endpoint RMSE for this model and the original joint-loss model.

| Unreachable target | Measured endpoint error | Graph |
|---|---:|---|
| Ned above vertical reach | 329.9 mm | [Pose and prediction](ned_above_vertical_reach.png) |
| Ned beyond horizontal reach | 486.2 mm | [Pose and prediction](ned_beyond_horizontal_reach.png) |
| SCARA beyond planar reach | 375.5 mm | [Pose and prediction](scara_beyond_planar_reach.png) |
| SCARA above lift range | 653.5 mm | [Pose and prediction](scara_above_lift_range.png) |

These are individual endpoint errors to impossible requested points, not RMSE
across the test set. Four reachable target graphs are also included.

The nearby-example panels use this model's actual augmented training requests.
Some are virtual unreachable inputs paired with fallback joint labels; their
presence does not establish that the requested TCP position is physically
reachable. The grey workspace boundary still comes from joint-limited FK.

The SCARA above-lift target is exactly on its yaw axis, so horizontal direction
is undefined. Its nearest accepted reference is 610.8 mm away, beyond the
600 mm maximum displacement used to generate augmentation queries. This case
shows that the learned response can fail outside the generated distribution.
