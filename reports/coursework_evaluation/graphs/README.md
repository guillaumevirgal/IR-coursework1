# Evaluation graphs

For test-result graphs, blue is the model trained against joint labels and orange is the model trained to reduce TCP endpoint error. Per-joint plots use dot size for the small, medium, and large datasets. The loss graph has one row per dataset size and one column per objective. Blue lines show training loss, orange lines show validation loss, and green diamonds show independent test loss once at the checkpoint selected by validation. The curves are 25-epoch moving averages to make their trends readable. The test set does not select epochs. Compare values only within the same objective column.

## Training progress

- [Training, validation and test MSE/loss](task4a/training_loss.png): conventional loss-versus-epoch curves with one held-out test marker for every saved model.

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

The evaluation scripts regenerate these figures automatically. To redraw existing exports only, run `python extras/figures/plot_mlp_evaluation.py`. Working outputs and full predictions are under `outputs/task4a/` and `outputs/task4b/`.
