# Dataset generation only

No model was trained. See `data/nn/directional_fallback_20000/manifest.json` for counts and provenance.

Split counts per robot:

```json
{
  "train_large": {
    "reachable": 20000,
    "unreachable": 6667
  },
  "validation_large": {
    "reachable": 5000,
    "unreachable": 1667
  },
  "test": {
    "reachable": 1000,
    "unreachable": 1000
  }
}
```

Reproduce with a fresh experiment name:

```bash
python scripts/run_fallback_training.py --size large --name directional_fallback_20000_repeat --seed 2026 --train-extra 6667 --test-extra 1000 --epochs 3000 --reachable-splits "data/nn/reachable_20000" --reachable-audits "data/raw/reachable_20000" --test-requests "data/nn/directional_fallback" --dataset-only
```

[Recorded data integrity checks](data_checks.json): counts, split separation, normalised limits, accepted Webots provenance and unchanged previous data/models.
