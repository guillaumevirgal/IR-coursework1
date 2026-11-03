# Joint-loss directional fallback experiment

The new dataset teaches the arm to point toward an unreachable request, then approach it. Ned uses the 3D direction from its shoulder; SCARA uses horizontal bearing because its links cannot tilt upward. Find the best recorded direction; among accepted poses within 2 degrees of that direction error, choose the TCP closest to the request. If SCARA's requested point is exactly on its yaw axis, bearing is undefined and distance alone chooses the label. This selects the best available recorded pose; a finite pose bank does not prove a globally optimal result.

New requests are certified outside the full joint-limited FK workspace using a conservative grid-cover bound or an analytic enclosing volume, with a minimum 5 mm clearance. Half the generated requests are within 100 mm of the accepted pose bank and half farther away. Query displacements span 15–600 mm from bank positions. Every label uses an original training pose with a Webots audit row. The requests are synthetic; no simulated TCP reading is invented at an impossible requested location.

The medium experiment reuses only unreachable test requested XYZ from `extras/unreachable_targets/data/nn/directional_fallback` for comparison on identical inputs. Its fallback joint labels are recomputed using the medium training bank only. The shared test group need not retain the medium bank's near/far proportions.

Original reachable rows retain their labels. New query splits have independent seeds, and validation/test poses never supply training labels. This tests new requests against a fixed training reference bank. Accepted static poses do not guarantee collision-free motion, and the MLP can predict an unrecorded configuration. Its response beyond the sampled query distribution is unverified.

Training uses the medium split and reuses the 5→64→64→3 NumPy ReLU MLP, Adam and joint MSE, from scratch with seed 2026 and up to 3000 epochs. Validation joint MSE selected epoch 1070; training ended at epoch 1370. Best validation joint MSE: 0.00848313. Test data were evaluated afterward. This comparison changes data quantity and distribution together.

Split counts per arm: {"train_medium": {"reachable": 4000, "unreachable": 2000}, "validation_medium": {"reachable": 1000, "unreachable": 500}, "test": {"reachable": 1000, "unreachable": 1000}}

## Requested-point endpoint error

For each sample, e = Euclidean distance from final TCP to requested XYZ, in mm. MSE = mean(e²), in mm²; RMSE = sqrt(mean(e²)), in mm. Mean/std are statistics of e, using population std. Unreachable requests include an unavoidable geometric gap, so their RMSE cannot be interpreted as ordinary reachable IK accuracy.

![Held-out requested-point TCP RMSE](test_comparison.png)

| Arm | Group | Training data | MSE (mm²) | RMSE (mm) | Mean (mm) | Std (mm) |
|---|---|---|---:|---:|---:|---:|
| ned | reachable | Original | 21.88 | 4.68 | 3.63 | 2.96 |
| ned | unreachable | Original | 137963.58 | 371.43 | 300.61 | 218.17 |
| ned | reachable | With fallback examples | 218.28 | 14.77 | 12.00 | 8.62 |
| ned | unreachable | With fallback examples | 48338.67 | 219.86 | 184.74 | 119.20 |
| scara | reachable | Original | 1.79 | 1.34 | 1.07 | 0.80 |
| scara | unreachable | Original | 150425.12 | 387.85 | 314.12 | 227.49 |
| scara | reachable | With fallback examples | 10.63 | 3.26 | 2.73 | 1.78 |
| scara | unreachable | With fallback examples | 45082.66 | 212.33 | 158.79 | 140.95 |

## Error to the chosen fallback endpoint

For unreachable inputs, this measures final TCP distance to the accepted fallback TCP used to create the joint label. It measures how accurately the new response was learned. The requested-point error above remains the main endpoint comparison.

![Fallback endpoint RMSE](fallback_comparison.png)

| Arm | Training data | Fallback RMSE (mm) | Mean (mm) | Std (mm) | P95 (mm) | Maximum (mm) |
|---|---|---:|---:|---:|---:|---:|
| ned | Original | 246.73 | 205.31 | 136.84 | 457.84 | 627.74 |
| ned | With fallback examples | 52.74 | 39.20 | 35.28 | 90.77 | 449.97 |
| scara | Original | 300.91 | 229.41 | 194.72 | 632.92 | 757.96 |
| scara | With fallback examples | 15.19 | 12.46 | 8.68 | 29.51 | 80.85 |

Individual failures remain. A joint-MSE regressor can average distinct joint solutions where the selected label switches branch, particularly near Ned's rear yaw limits. `failure_examples.csv` records the worst fallback endpoint error for each arm. `test_predictions.csv` retains all requested/desired/predicted points, normalised labels, raw outputs and physical joint commands. Direction determines the labelling rule; evaluation uses TCP endpoint errors.

## Measured in Webots

The first 10 reachable and 20 unreachable test requests per arm were executed for each saved model. Settling required joint sensors to remain within tolerance for 0.25 simulated seconds, with a 10-second timeout. Every result is included, including any timeouts.

![Measured requested-point TCP RMSE](webots_comparison.png)

| Arm | Group | Training data | Samples | Settled | Requested TCP RMSE (mm) | Fallback TCP RMSE (mm) |
|---|---|---|---:|---:|---:|---:|
| ned | reachable | Original | 10 | 10 | 3.20 | 3.20 |
| ned | reachable | With fallback examples | 10 | 10 | 12.20 | 12.20 |
| ned | unreachable | Original | 20 | 20 | 328.95 | 216.11 |
| ned | unreachable | With fallback examples | 20 | 20 | 223.50 | 85.01 |
| scara | reachable | Original | 10 | 10 | 1.52 | 1.52 |
| scara | reachable | With fallback examples | 10 | 10 | 2.86 | 2.86 |
| scara | unreachable | Original | 20 | 20 | 371.37 | 298.48 |
| scara | unreachable | With fallback examples | 20 | 20 | 184.31 | 12.86 |

## Training loss

![Training, validation and test joint MSE](training_loss.png)

The test marker is reweighted to validation group proportions (2 reachable to 1 unreachable in this run); the test files have an equal group mix. Per-group endpoint metrics above are unweighted. Curves are 25-epoch moving averages.

## Run and use

Run `python extras/unreachable_targets/run_fallback_training.py --size medium --name directional_fallback_medium_repeat --test-requests extras/unreachable_targets/data/nn/directional_fallback` from the repository root; add `--webots PATH` for physics measurements. Existing paths/checkpoints are refused. Data: `extras/unreachable_targets/data/nn/directional_fallback_medium`; policy audit: `extras/unreachable_targets/data/raw/directional_fallback_medium`. `manifest.json` records dataset hashes and `source_integrity.json` confirms original data and checkpoint hashes.

In Webots keep controller `mlp_ik` and append `"--model=mlp_ned_scara_medium_h64x64_cosine_directional_fallback_medium"` after `"ned"` or `"scara"` in `controllerArgs`. The network supplies the learned fallback response.

[Medium versus large comparison on the same test inputs](size_comparison.md).
