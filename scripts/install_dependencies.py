"""Install this repository's Python dependencies into the current interpreter."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"


def main():
    command = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    print(f"Installing Python packages with {sys.executable}", flush=True)
    subprocess.run(command, check=True)
    print("Python dependencies are installed.")


if __name__ == "__main__":
    main()
