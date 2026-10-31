"""Retrain every checkpoint in models/ with the exact settings used for the report.

Run from the repository root:
    python scripts/train_all_models.py --dry-run        # list the commands only
    python scripts/train_all_models.py --suffix _repro  # train all, names get "_repro"
    python scripts/train_all_models.py --only large     # only commands containing "large"

Checkpoints are never overwritten, so a suffix is required to retrain over the
shipped models (default "_repro"). Needs only NumPy and the included data/.
All models use seed 2026, 3000 epochs, patience 300, batch 32 (train_mlp.py defaults).
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ["scripts/train_mlp.py"]

# (checkpoint name, extra train_mlp.py arguments)
REACHABLE = [
    ("mlp_ned_scara_small_h64x64_cosine", ["--size", "small", "--hidden", "64", "64", "--loss", "joints"]),
    ("mlp_ned_scara_medium_h64x64_cosine", ["--size", "medium", "--hidden", "64", "64", "--loss", "joints"]),
    ("mlp_ned_scara_large_h64x64_cosine", ["--size", "large", "--hidden", "64", "64", "--loss", "joints"]),
    ("mlp_ned_scara_small_h64x64_cosine_tcp", ["--size", "small", "--hidden", "64", "64", "--loss", "tcp"]),
    ("mlp_ned_scara_medium_h64x64_cosine_tcp", ["--size", "medium", "--hidden", "64", "64", "--loss", "tcp"]),
    ("mlp_ned_scara_large_h64x64_cosine_tcp", ["--size", "large", "--hidden", "64", "64", "--loss", "tcp"]),
    # early trials: constant learning rate
    ("mlp_ned_scara_small_h64x64", ["--size", "small", "--hidden", "64", "64", "--decay", "none"]),
    ("mlp_ned_scara_small_h64", ["--size", "small", "--hidden", "64", "--decay", "none"]),
]
# unreachable-target experiment: trained on the included directional_fallback splits
UNREACHABLE = [
    (f"mlp_ned_scara_total{n}_3to1_{loss}",
     ["--size", "large", "--hidden", "64", "64", "--loss", loss,
      "--splits-dir", f"data/nn/directional_fallback_total_{n}"])
    for n in (20000, 16000) for loss in ("joints", "tcp")
] + [
    ("mlp_ned_scara_large_h64x64_cosine_directional_fallback",
     ["--size", "large", "--hidden", "64", "64", "--loss", "joints",
      "--splits-dir", "data/nn/directional_fallback"]),
    ("mlp_ned_scara_medium_h64x64_cosine_directional_fallback_medium",
     ["--size", "medium", "--hidden", "64", "64", "--loss", "joints",
      "--splits-dir", "data/nn/directional_fallback_medium"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suffix", default="_repro", help="appended to every checkpoint name")
    parser.add_argument("--only", help="train only checkpoints whose name contains this text")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    for name, extra in REACHABLE + UNREACHABLE:
        if args.only and args.only not in name:
            continue
        command = [sys.executable, *TRAIN, *extra, "--model-name", name + args.suffix]
        print(" ".join(command[1:]), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
