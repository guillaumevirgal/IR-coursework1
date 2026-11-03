# 16,000-row fallback models (3:1 reachable to unreachable)

Each arm has 12,000 reachable and 4,000 certified-unreachable training rows, plus 3,000 + 1,000 validation rows. Both models use seed 2026, a 5→64→64→3 architecture. The joint-loss checkpoint selected epoch 652 (stopped at 952); the TCP-loss checkpoint selected epoch 520 (stopped at 820). Validation selected each checkpoint; test data did not.

## Requested-point TCP RMSE on the shared held-out requests

The comparison uses 1,000 reachable and 1,000 unreachable test requests per arm. Lower is better. Unreachable RMSE includes the geometric gap from the impossible request.

| Arm | Loss | Reachable (mm) | Unreachable (mm) |
|---|---|---:|---:|
| Ned | Joint | 8.82 | 204.74 |
| Ned | TCP | 5.64 | 199.76 |
| SCARA | Joint | 2.89 | 211.85 |
| SCARA | TCP | 2.09 | 211.22 |

These are FK-based test evaluations; the 16k models have not been measured in Webots. (A comparison against a 20,000-row experiment was removed when that experiment was deleted; it remains in git history.)

Training data and audit are in `extras/unreachable_targets/data/nn/directional_fallback_total_16000/` and `extras/unreachable_targets/data/raw/directional_fallback_total_16000/`. Checkpoints are:

- `mlp_ned_scara_total16000_3to1_joints`
- `mlp_ned_scara_total16000_3to1_tcp`
