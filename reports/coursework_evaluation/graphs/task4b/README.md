# Task 4.B graphs

Blue = joint-loss model; orange = TCP-loss model. All measured points, including any motor timeouts, count. Check summary.csv for settling status.

- [Measured endpoint mean and standard deviation](comparison.png).
- [Measured endpoint RMSE](tcp_rmse.png).
- [Accuracy curves](tcp_accuracy_curves.png): percentage below each allowed measured error.
- [FK versus measured error](fk_vs_webots.png): paired errors on exactly the same first 100 test points per arm.

The diagonal means equal error magnitudes, not identical positions. Do not compare first-100 Webots results directly with an FK full-test average as if they used the same samples.
