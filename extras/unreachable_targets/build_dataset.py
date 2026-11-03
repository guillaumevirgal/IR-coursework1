"""Build one unreachable-target dataset from its config file.

    python extras/unreachable_targets/build_dataset.py extras/unreachable_targets/configs/NAME.json

Writes extras/unreachable_targets/data/nn/NAME/ (train, validation and test CSVs
per robot, manifest.json, README.md) and data/raw/NAME/ beside it (one audit row
per unreachable request). Needs only
NumPy and the reachable data named in the config; it does not use Webots.

Existing outputs are never overwritten. To check a rebuild without touching the
repository, pass --output-root DIR: files then go to DIR/data/nn and DIR/data/raw.
Build order matters when a config reuses another dataset's test requests; see the
README next to this file.
"""

import argparse
from pathlib import Path

from fallback_dataset import EXTRAS, build_dataset, load_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="dataset config JSON")
    parser.add_argument("--output-root", type=Path, default=EXTRAS,
                        help="parent of data/nn and data/raw (default: extras/unreachable_targets)")
    args = parser.parse_args()

    experiment = load_experiment(args.config)
    root = args.output_root.resolve()
    dataset_dir = root / "data/nn" / experiment.name
    audit_dir = root / "data/raw" / experiment.name
    for path in (dataset_dir, audit_dir):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    build_dataset(experiment, dataset_dir, audit_dir, root)
    print(f"Dataset created: {dataset_dir}")


if __name__ == "__main__":
    main()
