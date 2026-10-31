"""Build the coursework source ZIP: code, worlds, data, trained models, reports.

Usage:  python scripts/make_submission_zip.py [--output IR-coursework1-submission.zip]

Paths inside the ZIP are relative to a single top-level folder, so it unpacks
anywhere. Excluded: git history, virtual environments, caches, generated
outputs/ that can be recreated, the coursework brief, and videos.
Files over the size limit are listed so you can decide what to do with them.
"""

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".idea", ".vscode"}
EXCLUDED_SUFFIXES = {".pyc", ".mp4", ".mkv", ".avi", ".mov", ".bak", ".log"}
EXCLUDED_NAMES = {"Coursework_1_2627.pdf"}
# outputs/ holds results that the scripts regenerate; the saved copies live in
# reports/. The two saved diagnostics folders are kept.
KEPT_OUTPUT_DIRS = {"mlp_diagnostics_directional_fallback", "mlp_diagnostics_tcp"}


def included(path):
    relative = path.relative_to(ROOT)
    parts = relative.parts
    if any(part in EXCLUDED_DIRS for part in parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES or path.name in EXCLUDED_NAMES:
        return False
    if parts[0] == "outputs" and (len(parts) < 2 or parts[1] not in KEPT_OUTPUT_DIRS):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "IR-coursework1-submission.zip")
    args = parser.parse_args()

    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and included(path))
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path(ROOT.name) / path.relative_to(ROOT))

    size_mb = args.output.stat().st_size / 1e6
    print(f"Wrote {len(files)} files ({size_mb:.1f} MB) to {args.output}")
    missing = [name for name in ("README.md", "requirements.txt", "worlds/articulated.wbt",
                                 "worlds/scara.wbt", "models/mlp_ned_scara_large_h64x64_cosine.npz")
               if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"Required files missing from the repository: {missing}")


if __name__ == "__main__":
    main()
