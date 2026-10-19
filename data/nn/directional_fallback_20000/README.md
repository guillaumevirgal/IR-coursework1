# Directional fallback dataset

The network inputs are `is_scara,is_ned,x,y,z`; XYZ is the requested point in metres. Outputs `q1,q2,q3` are normalised joint labels in [-1,1]. The trainer standardises XYZ using this training split only.

`kind` identifies reachable or certified unreachable requests. `desired_x,desired_y,desired_z` give the desired achieved TCP in metres. These four columns are for auditing and metrics, not additional network inputs. For reachable rows the desired and requested points agree. For unreachable rows the desired point is an accepted training pose chosen to point toward the request.

Direction is 3D from Ned's shoulder, and horizontal bearing from SCARA's yaw axis. Among accepted poses within 2 degrees of the best recorded direction, select the closest TCP to the request. All labels use the original training bank.

The per-request policy audit is in `data/raw/directional_fallback_20000`; it links labels to the original training CSV row and Webots audit row. These are synthetic requests with recorded reference poses, not Webots readings at impossible requested points. `manifest.json` records counts, seeds and hashes.

Per robot, training has **20,000 reachable + 6,667 unreachable** rows (approximately **75% / 25%**, or **3:1**). Validation has 5,000 + 1,667 rows. Testing has 1,000 + 1,000 rows and uses the previous experiment's requested points. The `large` filenames identify the existing trainer slot; the counts here exceed the previous large dataset. Reachable source data are in `data/nn/reachable_20000`, with physics audits in `data/raw/reachable_20000`.

This dataset has not trained a model yet. Changing the ratio may improve reachable accuracy but must be evaluated against 2:1 using the same test inputs. Increasing the reachable bank and changing the ratio together does not isolate the effect of the ratio.
