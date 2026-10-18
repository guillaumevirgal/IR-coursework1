# Dataset generation only

No model was trained. See `data/nn/directional_fallback_total_20000/manifest.json` for counts and provenance.

Split counts per robot:

```json
{
  "train_large": {
    "reachable": 15000,
    "unreachable": 5000
  },
  "validation_large": {
    "reachable": 3750,
    "unreachable": 1250
  },
  "test": {
    "reachable": 1000,
    "unreachable": 1000
  }
}
```

Reproduce with a fresh experiment name:

```bash
python scripts/run_fallback_training.py --size large --name directional_fallback_total_20000_repeat --seed 2026 --train-extra 5000 --test-extra 1000 --epochs 3000 --reachable-splits "data/nn/reachable_total_20000" --reachable-audits "data/raw/reachable_20000" --test-requests "data/nn/directional_fallback" --dataset-only
```
