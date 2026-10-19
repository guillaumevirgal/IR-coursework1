# Directional fallback dataset

This experiment has **16,000 training rows per arm**: 12,000 reachable + 4,000 certified unreachable (3:1). Validation has 3,000 + 1,000 rows; test has 1,000 + 1,000. Its saved joint-loss and TCP-loss models are `mlp_ned_scara_total16000_3to1_joints` and `mlp_ned_scara_total16000_3to1_tcp`. The shared test comparison is in `reports/directional_fallback_total_16000/README.md`.

The network inputs are `is_scara,is_ned,x,y,z`; XYZ is the requested point in metres. Outputs `q1,q2,q3` are normalised joint labels in [-1,1]. The trainer standardises XYZ using this training split only.

`kind` identifies reachable or certified unreachable requests. `desired_x,desired_y,desired_z` give the desired achieved TCP in metres. These four columns are for auditing and metrics, not additional network inputs. For reachable rows the desired and requested points agree. For unreachable rows the desired point is an accepted training pose chosen to point toward the request.

Direction is 3D from Ned's shoulder, and horizontal bearing from SCARA's yaw axis. Among accepted poses within 2 degrees of the best recorded direction, select the closest TCP to the request. All labels use the original training bank.

The per-request policy audit is in `data/raw/directional_fallback_total_16000`; it links labels to the original training CSV row and Webots audit row. These are synthetic requests with recorded reference poses, not Webots readings at impossible requested points. `manifest.json` records counts, seeds and hashes.
