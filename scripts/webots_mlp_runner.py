"""Run a saved MLP on a list of targets in headless Webots and return the measured TCPs.

Uses only the standard library. A temporary copy of the robot's world swaps the
mlp_ik controller for controllers/diagnose_mlp, which commands the predicted
joints, waits for the joint sensors to settle and reads the TCP through the
Supervisor. The original world files are never modified.
"""

import csv
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

from scripts.run_dataset_generation import ROOT, WORLDS, find_webots


def write_cases(path, cases):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=("label", "kind", "x", "y", "z"))
        writer.writeheader()
        for case in cases:
            writer.writerow(dict(zip(("label", "kind", "x", "y", "z"), (case["label"], case["kind"], *case["point"]))))


def make_diagnostic_world(robot, model, points, output, settle_time, timeout):
    source = ROOT / "worlds" / WORLDS[robot]
    content = source.read_text(encoding="utf-8")
    if content.count('  controller "mlp_ik"') != 1:
        raise RuntimeError(f"Expected one MLP controller in {source}")
    content = content.replace('  controller "mlp_ik"', '  controller "diagnose_mlp"', 1)
    args_block = (f'  controllerArgs [\n    "{robot}"\n    "--model={model}"\n'
                  f'    "--points={points.as_posix()}"\n    "--output={output.as_posix()}"\n'
                  f'    "--settle-time={settle_time}"\n    "--timeout={timeout}"\n  ]')
    start = content.index("  controllerArgs [", content.index('controller "diagnose_mlp"'))
    end = content.index("  ]", start) + 3
    content = content[:start] + args_block + content[end:]
    # The world must sit next to the originals so its relative PROTO paths resolve.
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".wbt",
                                         prefix=f"diagnostic_{robot}_", dir=source.parent, delete=False)
    with handle:
        handle.write(content)
    return Path(handle.name)


def remove_world(path):
    for candidate in (path, path.with_suffix(".wbproj"), path.with_name(f".{path.stem}.wbproj")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def run_webots(robot, model, cases, work_root, webots=None, settle_time=0.25, timeout=10.0, wall_timeout=900.0):
    """Measure every case for one model and robot; return the controller's rows in order."""

    # Use an explicit launcher as given: resolving a snap link would lose the 'webots' name.
    launcher = webots if webots else find_webots()
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{robot}_", dir=work_root))
    points_path, measured_path = work / "points.csv", work / "webots.csv"
    write_cases(points_path, cases)
    world = make_diagnostic_world(robot, model, points_path, measured_path, settle_time, timeout)
    command = [str(launcher), "--batch", "--mode=fast", "--no-rendering", "--stdout", "--stderr", str(world)]
    # Webots starts controllers with the first python3 on PATH. Put this script's
    # interpreter first so the controller sees the same packages (NumPy).
    env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
    log_path = work / "webots.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            # Own process group, so a timeout also stops webots-bin, not just its launcher script.
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=wall_timeout)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg"):
                    os.killpg(process.pid, signal.SIGKILL)
                else:  # Windows
                    process.kill()
                process.wait()
                raise RuntimeError(f"Webots did not finish within {wall_timeout:.0f} s; see {log_path}")
        if code != 0:
            raise RuntimeError(f"Webots exited with status {code}; see {log_path}")
    finally:
        remove_world(world)
    if not measured_path.is_file():
        raise RuntimeError(f"Webots did not produce {measured_path}; see {log_path}")
    with measured_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    shutil.rmtree(work, ignore_errors=True)
    return rows
