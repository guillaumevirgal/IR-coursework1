"""Launch saved MLP targets in Webots; shared by diagnostics and Task 4.B."""

import csv
from pathlib import Path
import shutil
import subprocess
import tempfile

from scripts.generate_dataset import ROOT, find_webots

WORLDS = {"ned": "articulated.wbt", "scara": "scara.wbt"}


def write_cases(path, cases):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=("label", "kind", "x", "y", "z"))
        writer.writeheader()
        for case in cases:
            writer.writerow(dict(zip(("label", "kind", "x", "y", "z"),
                                     (case["label"], case["kind"], *case["point"]))))


def make_diagnostic_world(robot, model, points, output, settle_time, timeout, model_dir=None):
    source = ROOT / "worlds" / WORLDS[robot]
    content = source.read_text(encoding="utf-8")
    if content.count('  controller "mlp_ik"') != 1:
        raise RuntimeError(f"Expected one MLP controller in {source}")
    content = content.replace('  controller "mlp_ik"', '  controller "diagnose_mlp"', 1)
    args_block = (f'  controllerArgs [\n    "{robot}"\n    "--model={model}"\n'
                  f'    "--points={points.as_posix()}"\n    "--output={output.as_posix()}"\n'
                  f'    "--settle-time={settle_time}"\n    "--timeout={timeout}"\n  ]')
    if model_dir is not None:
        args_block = args_block.replace("\n  ]", f'\n    "--model-dir={Path(model_dir).resolve().as_posix()}"\n  ]')
    start = content.index("  controllerArgs [", content.index('controller "diagnose_mlp"'))
    end = content.index("  ]", start) + 3
    content = content[:start] + args_block + content[end:]
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".wbt",
                                         prefix=f"diagnostic_{robot}_", dir=source.parent,
                                         delete=False)
    with handle:
        handle.write(content)
    return Path(handle.name)


def remove_world(path):
    for candidate in (path, path.with_suffix(".wbproj"), path.with_name(f".{path.stem}.wbproj")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def run_webots(robot, model, cases, args):
    webots = find_webots(args.webots)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{robot}_", dir=args.output))
    points_path, measured_path = work / "points.csv", work / "webots.csv"
    write_cases(points_path, cases)
    world = make_diagnostic_world(robot, model, points_path, measured_path,
                                  args.settle_time, args.timeout, getattr(args, "model_dir", None))
    command = [str(webots), "--batch", "--mode=fast", "--no-rendering", "--stdout", "--stderr", str(world)]
    print(f"Running {robot} diagnostics in headless Webots ({len(cases)} targets)...", flush=True)
    try:
        subprocess.run(command, check=True, timeout=getattr(args, "wall_timeout", None))
    finally:
        remove_world(world)
    if not measured_path.is_file():
        raise RuntimeError(f"Webots did not produce {measured_path}; check its controller console")
    with measured_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    shutil.rmtree(work, ignore_errors=True)
    return rows
