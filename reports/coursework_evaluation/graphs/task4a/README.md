# Task 4.A graphs

Blue = joint-loss model; orange = TCP-loss model. Missing checkpoints are not zero errors.

- [Endpoint mean and standard deviation](../../task4a_comparison.png): closer to the target and variation across test points.
- [Endpoint RMSE](tcp_rmse.png): gives larger misses more weight.
- [Per-joint mean errors](joint_mean_errors.png): signed angular bias; SCARA lift uses squared error.
- [Per-joint standard deviations](joint_error_std.png): read with mean bias; do not compare different units.
- [Accuracy curves](tcp_accuracy_curves.png): percentage of test points below each allowed error.

All test rows are included. FK positions are calculated, not measured in Webots.
