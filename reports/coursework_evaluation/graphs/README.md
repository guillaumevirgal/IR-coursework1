# Evaluation graphs

For test-result graphs, blue is the model trained against joint labels and orange is the model trained to reduce TCP endpoint error. Per-joint plots use dot size for the small, medium, and large datasets. In the loss graph, line colour shows dataset size; solid lines show training loss each epoch, dashed lines show validation loss each epoch, and a diamond shows the independent test loss once at the checkpoint selected by validation. The test set does not select epochs. The two panels use different objectives and units, so compare curves only within the same panel.

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

The evaluation scripts regenerate these figures automatically. To redraw existing exports only, run `python scripts/plot_mlp_evaluation.py`. Working outputs and full predictions are under `outputs/task4a/` and `outputs/task4b/`.
