See the wiki for a basic tutorial.

## SCARA Task 1.A data

Set the SCARA robot's Webots controller to `scara_task1a` to continuously
sample random elbow-up poses and compare the forward-kinematics result with
the TCP measured by Webots. Valid rows are written to
`controllers/scara_task1a/scara_robot_data_1e-3.csv`.

Edit `POSES` in the controller to test fixed poses, or set `CSV_FILE = False`
to print the comparisons without recording them.
