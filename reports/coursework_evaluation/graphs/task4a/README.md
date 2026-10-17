# Task 4.A graphs

Blue = model trained against joint labels; orange = model trained to reduce endpoint error.

- [Endpoint mean and standard deviation](../../task4a_comparison.png): closer to the target and variation across test points.
- [Endpoint RMSE](tcp_rmse.png): gives larger misses more weight.
- [Per-joint mean errors](joint_mean_errors.png): dot size shows dataset size; signed angular bias; SCARA lift uses squared error.
- [Per-joint standard deviations](joint_error_std.png): dot size shows dataset size; read with mean bias and do not compare different units.
- [Training, validation and test loss](training_loss.png): solid/dashed lines show training/validation loss each epoch; diamonds show independent test loss once at the checkpoint selected by validation. Test data do not select the checkpoint.

All test rows are included. FK positions are calculated, not measured in Webots.
