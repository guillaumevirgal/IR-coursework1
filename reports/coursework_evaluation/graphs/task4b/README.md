# Task 4.B graphs

Blue = model trained against joint labels; orange = model trained to reduce endpoint error. All 100 measured points per robot are included.

- [Measured endpoint mean and standard deviation](comparison.png).
- [Measured endpoint RMSE](tcp_rmse.png).
- [Training, validation and test loss](../task4a/training_loss.png): blue/orange lines show smoothed training/validation loss; green diamonds show independent test loss once at the checkpoint selected by validation. Rows identify dataset size. Test data do not select the checkpoint.

These are Webots measurements on the first 100 test points per arm. Mean and standard deviation are in millimetres.
