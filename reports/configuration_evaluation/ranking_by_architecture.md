# Ranking by architecture, best dataset size

The main [README.md](README.md) picks one model per **dataset size** (lowest
validation joint MSE). This table instead groups the 33 evaluated
configurations by **architecture + activation**, dataset size collapsed: each
row is whichever of small/medium/large scored lowest mean Task 4.A position
error for that design. This answers "what's the best network shape" rather
than "what's the best checkpoint of each size", and it is not the same
ranking as the README's three picks — e.g. 64-64-64-64 tanh's best result
comes from the medium set, not large.

Source: `reports/configuration_evaluation/results.json`, produced by
`python scripts/evaluate_configurations.py`. All 33 configurations are
listed individually, with the same metrics, in [summary.csv](summary.csv)
and [task4a_position.csv](task4a_position.csv).

| # | Architecture · Activation | Best size | Val MSE | Ned: mean/std/med/p95/max (mm) | Ned ≤5/≤10mm | SCARA: mean/std/med/p95/max (mm) | SCARA ≤5/≤10mm |
|---|---|---|---|---|---|---|---|
| 1 | **64-64-64-64 · per-robot** | large | 0.000093 | 1.29/1.69/0.91/3.10/19.4 | 98.1% / 99.1% | 0.60/0.64/0.44/1.66/7.7 | 99.6% / 100% |
| 2 | 64-64-32 · tanh | large | 0.000090 | 1.53/1.61/1.09/3.98/16.6 | 96.9% / 99.2% | 0.78/0.77/0.58/2.08/6.9 | 99.5% / 100% |
| 3 | 128-64-32 · tanh | large | 0.000109 | 1.68/1.65/1.23/4.22/22.0 | 96.3% / 99.6% | 0.74/0.70/0.55/1.89/6.4 | 99.6% / 100% |
| 4 | 128-128 · tanh | large | 0.000183 | 1.89/2.35/1.33/5.33/53.3 | 94.6% / 99.3% | 0.62/0.53/0.48/1.54/6.3 | 99.9% / 100% |
| 5 | 64-64-64-64 · tanh | medium | 0.000137 | 1.65/2.20/1.13/4.12/31.9 | 96.6% / 99.2% | 0.95/1.32/0.64/2.64/29.2 | 99.1% / 99.8% |
| 6 | 64-32-16 · tanh | large | 0.000129 | 1.81/1.48/1.36/4.73/13.6 | 96.2% / 99.8% | 0.85/0.69/0.65/2.09/4.8 | 100% / 100% |
| 7 | 64-64 · tanh | large | 0.000218 | 2.07/1.93/1.56/5.20/26.6 | 93.5% / 99.4% | 0.90/0.67/0.72/2.15/5.6 | 99.9% / 100% |
| 8 | 64-64 · relu (controller default family) | large | 0.000172 | 2.74/1.91/2.29/6.51/14.1 | 89.9% / 99.1% | 0.80/0.65/0.63/1.94/6.3 | 99.9% / 100% |
| 9 | 64-64-64 · tanh | medium | 0.000189 | 2.62/2.44/1.83/7.47/20.5 | 88.2% / 97.7% | 1.07/1.51/0.74/2.77/33.6 | 98.8% / 99.7% |
| 10 | 64-64-64-64-64 · tanh | small | 0.000257 | 2.41/4.16/1.42/6.72/79.6 | 90.7% / 97.4% | 1.51/2.79/0.98/3.76/68.7 | 97.3% / 99.2% |
| 11 | 256-256-256 · tanh | medium | 0.000270 | 3.08/3.08/2.22/8.14/47.0 | 84.8% / 97.2% | 1.58/1.43/1.20/4.09/14.5 | 97.3% / 99.6% |
| 12 | 128-128-128 · tanh | large | 0.000216 | 3.27/2.29/2.81/6.57/34.0 | 83.2% / 98.6% | 1.64/1.12/1.35/3.80/5.7 | 99.6% / 100% |
| 13 | 1024-1024-1024 · relu | small | 0.000389 | 3.71/5.47/2.32/10.16/69.5 | 82.6% / 94.7% | 1.59/1.09/1.35/3.69/9.8 | 98.7% / 100% |
| 14 | 128-128 · relu | small | 0.000483 | 5.26/6.85/3.61/14.51/109.4 | 67.7% / 90.8% | 1.22/0.83/1.01/2.76/6.6 | 99.4% / 100% |
| 15 | 256-128-64 · tanh | large | 0.000237 | 3.74/2.64/3.12/8.04/29.3 | 78.6% / 97.6% | 2.92/1.79/2.47/6.44/10.2 | 86.2% / 99.9% |
| 16 | 64-64-64 · relu | small | 0.000656 | 6.09/7.07/4.24/16.09/72.5 | 58.9% / 86.3% | 1.73/1.37/1.37/4.25/17.6 | 97.4% / 99.8% |
| 17 | 1024-1024 · tanh | medium | 0.000279 | 6.11/6.81/4.40/16.27/99.5 | 57.5% / 85.7% | 2.18/1.90/1.65/5.83/17.2 | 91.7% / 99.2% |

17 distinct architecture/activation designs (33 size-variants collapsed to
whichever size won).

## Reading this table

- **`64-64-64-64 · per-robot` stays #1** even against every architecture's
  best dataset size — it isn't winning just because "large" has the most
  data; it's winning on design.
- **64-64-64-64 tanh's best result comes from medium, not large** (#5) —
  more data didn't help that shape past a point, consistent with the "medium
  to large barely changes" finding in the main [README.md](README.md).
- Small, 4-5 layer tanh nets (#10) punch above their weight on a tiny
  dataset but have fat tails (SCARA max 68.7mm) — good mean, bad worst case.
