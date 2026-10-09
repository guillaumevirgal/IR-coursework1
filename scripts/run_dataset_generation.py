"""Run physics-based dataset generation in headless Webots fast mode."""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORLDS = {"ned": "articulated.wbt", "scara": "scara.wbt"}


def find_webots(explicit=None):
    """Find the Webots launcher on Windows, Linux or macOS."""

    candidates = []
    if explicit:
        candidates.append(Path(explicit))

    executable = shutil.which("webots") or shutil.which("webots.exe")
    if executable:
        candidates.append(Path(executable))

    webots_home = os.environ.get("WEBOTS_HOME")
    if webots_home:
        home = Path(webots_home)
        candidates.extend((
            home / "webots.exe",
            home / "msys64" / "mingw64" / "bin" / "webots.exe",
            home / "webots",
        ))

    program_files = os.environ.get("ProgramFiles")
    if program_files:
        install = Path(program_files) / "Webots"
        candidates.extend((
            install / "webots.exe",
            install / "msys64" / "mingw64" / "bin" / "webots.exe",
        ))

    candidates.extend((
        Path("/snap/bin/webots"),
        Path("/Applications/Webots.app/Contents/MacOS/webots"),
    ))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find Webots. Add it to PATH, set WEBOTS_HOME, or pass --webots."
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=WORLDS)
    parser.add_argument("samples", nargs="?", type=int, default=1000)
    parser.add_argument("seed", nargs="?", type=int, default=2026)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--webots", type=Path, help="path to webots or webots.exe")
    return parser.parse_args()


def temporary_world(robot, samples, seed, output):
    """Create a world beside the originals so relative PROTO paths still work."""

    source = ROOT / "worlds" / WORLDS[robot]
    text = source.read_text(encoding="utf-8")
    marker = f'    "{robot}"'
    if text.count(marker) != 1:
        raise RuntimeError(f"Expected one {robot!r} controller argument in {source}")

    extra_arguments = (
        f'{marker}\n'
        f'    "--samples={samples}"\n'
        f'    "--seed={seed}"\n'
        f'    "--output={output.as_posix()}"'
    )
    text = text.replace(marker, extra_arguments, 1)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".wbt",
        prefix=f"dataset_{robot}_",
        dir=source.parent,
        delete=False,
    )
    with handle:
        handle.write(text)
    return Path(handle.name)


def remove_temporary_world(path):
    """Remove the temporary world and Webots project metadata it may create."""

    project_files = (
        path,
        path.with_suffix(".wbproj"),
        path.with_name(f".{path.stem}.wbproj"),
    )
    for project_file in project_files:
        try:
            project_file.unlink()
        except FileNotFoundError:
            pass


def main():
    args = parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")

    output = args.output or ROOT / "data" / "csv" / (
        f"{args.robot}_{args.samples}_seed{args.seed}.csv"
    )
    output = output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to append to existing dataset: {output}. Choose another path."
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    webots = find_webots(args.webots)
    world = temporary_world(args.robot, args.samples, args.seed, output)
    command = [
        str(webots),
        "--batch",
        "--mode=fast",
        "--no-rendering",
        "--stdout",
        "--stderr",
        str(world),
    ]
    print(
        f"Generating {args.samples} {args.robot} rows in Webots fast mode...",
        flush=True,
    )
    try:
        subprocess.run(command, check=True)
    finally:
        remove_temporary_world(world)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
