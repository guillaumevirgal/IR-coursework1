# Evaluation graphs

Blue = joint-loss model; orange = TCP-loss model. Large TCP has not been trained and is marked not evaluated. These graphs use the saved, verified predictions; they do not require a new simulation.

## Task 4.A — all independent test rows, calculated using FK

- [Endpoint mean and standard deviation](../task4a_comparison.png): average distance from the target and variation across points.
- [Endpoint RMSE](task4a/tcp_rmse.png): larger misses receive greater weight.
- [Per-joint mean errors](task4a/joint_mean_errors.png): angular bias, and squared SCARA lift error.
- [Per-joint standard deviations](task4a/joint_error_std.png): read alongside the means; a near-zero signed mean can hide large variation.
- [Accuracy curves](task4a/tcp_accuracy_curves.png): percentage within each allowed endpoint error.

## Task 4.B — first 100 test rows per arm, measured using Webots

- [Measured endpoint mean and standard deviation](task4b/comparison.png).
- [Measured endpoint RMSE](task4b/tcp_rmse.png).
- [Measured accuracy curves](task4b/tcp_accuracy_curves.png).
- [FK versus measured error](task4b/fk_vs_webots.png): compares the same target on both axes; above the diagonal means the measured error is larger. Equal error magnitudes do not imply identical TCP positions.

For accuracy curves, choose an allowed error (mm) on the log horizontal axis and read the share of targets within it; higher is better at that threshold. Each robot/size has its own panel. All recorded samples count, including any motor timeouts (none occurred in this saved run).

Joint angular axes use radians. SCARA lift uses the handout's squared error; plots convert m² to mm². Different joints have separate axes and units, so compare models within each panel.

The evaluation scripts regenerate these figures automatically. To redraw existing exports only, run `python scripts/plot_mlp_evaluation.py`. Working outputs and full predictions are under `outputs/task4a/` and `outputs/task4b/`.
