# Evaluation graphs

For test-result graphs, blue is the model trained against joint labels and orange is the model trained to reduce TCP endpoint error. Per-joint plots use dot size for the small, medium, and large datasets. Training-loss plots use a different key: line colour is dataset size, and solid/dashed lines are training/validation.

## Training progress

- [Training and validation MSE/loss](task4a/training_loss.png): conventional loss-versus-epoch curves. The joint-label MSE and TCP objective have different units, so compare curves only within each panel. These curves show training progress; the test RMSE bars below show held-out model performance.

## Task 4.A — all independent test rows, calculated using FK

- [Endpoint mean and standard deviation](../task4a_comparison.png): average distance from the target and variation across points.
- [Endpoint RMSE](task4a/tcp_rmse.png): larger misses receive greater weight.
- [Per-joint mean errors](task4a/joint_mean_errors.png): angular bias, and squared SCARA lift error.
- [Per-joint standard deviations](task4a/joint_error_std.png): read alongside the means; a near-zero signed mean can hide large variation.
## Task 4.B — first 100 test rows per arm, measured using Webots

- [Measured endpoint mean and standard deviation](task4b/comparison.png).
- [Measured endpoint RMSE](task4b/tcp_rmse.png).

All 1,200 measured points in this saved run settled. Mean and standard deviation are in millimetres.

Joint angular axes use radians. SCARA lift uses the handout's squared error; plots convert m² to mm². Different joints have separate axes and units, so compare models within each panel.

The evaluation scripts regenerate these figures automatically. To redraw existing exports only, run `python scripts/plot_mlp_evaluation.py`. Working outputs and full predictions are under `outputs/task4a/` and `outputs/task4b/`.
