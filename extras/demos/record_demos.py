"""Open Webots once per demo run, with the right robot and model already set.

    python extras/demos/record_demos.py               # all clips, in sheet order (1a, 1b, 3, 4)
    python extras/demos/record_demos.py list          # show clip ids
    python extras/demos/record_demos.py layout        # one-off: arrange the console, close Webots
    python extras/demos/record_demos.py 3             # clip 3 (Run 1 model, then Run 2 model)
    python extras/demos/record_demos.py 1a 4          # only these, in the order given

Webots opens paused. Press Play when ready: the target then moves by itself
(slowly through reachable points, then out of reach, then back); close the window to move on to
the next run. The console layout you saved with `layout` is copied to every
run, so the console sits where you put it.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORLDS = ROOT / "worlds"
WORLD = {"ned": "articulated", "scara": "scara"}

PR = "mlp_ned_scara_large_h64x64x64x64_cosine_per-robot"
SMALL = "mlp_ned_scara_small_h64x64x64x64_cosine_per-robot"
REACH = "mlp_ned_scara_large_h64x64_cosine"
JOINT = "mlp_ned_scara_small_h64x64_cosine"
TCPL = "mlp_ned_scara_small_h64x64_cosine_tcp"
FALL = "mlp_ned_scara_large_h64x64_cosine_directional_fallback"

# id: (robot, description, [models in recording order])
CLIPS = {
    "1a": ("ned", "How it works (keys)", [PR]),
    "1b": ("scara", "How it works (keys)", [PR]),
    # Data size (clip 2) is not recorded: it reuses run 1 of clip 3 (small, JOINT)
    # and run 1 of clip 4 (large, REACH), same 2x64 architecture, only the data differs.
    "3":("ned", "Loss function (joint error vs TCP error)", [JOINT, TCPL]),
    "4": ("ned", "Out of reach", [REACH, FALL]),
}

# Cut for time; add back to CLIPS above if the video has room.
# "2b": ("scara", "Data size", [SMALL, PR]),
# clip 2 only records SMALL: the large model on Ned is already clip 1a.
# "act": ("ned", "Activation", [REACH, PR]),
# "3b": ("scara", "Out of reach", [REACH, FALL]),
# "4a"/"4b": reachable target with the REACH/FALL pair (Ned/SCARA)


def find_webots():
    exe = shutil.which("webots")
    if exe:
        return exe
    for base in (r"C:\Program Files\Webots", r"C:\Program Files (x86)\Webots",
                 str(Path.home() / "AppData/Local/Programs/Webots")):
        for rel in ("msys64/mingw64/bin/webots.exe", "webots.exe"):
            if (Path(base) / rel).exists():
                return str(Path(base) / rel)
    sys.exit("Webots not found: add it to PATH.")


def make_world(robot, model, tag):
    """Copy the world with controllerArgs set; keep it beside the original so
    relative controller/proto paths still resolve."""
    src = WORLDS / f"{WORLD[robot]}.wbt"
    dst = WORLDS / f"_demo_{tag}.wbt"
    text = src.read_text(encoding="utf-8")
    old = f'controllerArgs [\n    "{robot}"\n  ]'
    new = f'controllerArgs [\n    "{robot}"\n    "--model={model}"\n    "--sweep"\n    "--show-gap"\n  ]'
    if old not in text:
        sys.exit(f"controllerArgs block not found in {src.name}")
    dst.write_text(text.replace(old, new), encoding="utf-8")
    layout = WORLDS / f".{WORLD[robot]}.wbproj"
    if layout.exists():
        # Webots marks its .wbproj Hidden; Windows refuses to overwrite a hidden file.
        copy = WORLDS / f"._demo_{tag}.wbproj"
        copy.unlink(missing_ok=True)
        shutil.copy(layout, copy)
    return dst


def run_clip(webots, cid):
    robot, desc, models = CLIPS[cid]
    for i, model in enumerate(models, 1):
        world = make_world(robot, model, f"{cid}_run{i}")
        print(f"\n=== {cid} {robot.upper()} | {desc} | Run {i}/{len(models)} ===")
        print(f"model: {model}")
        input("Press Enter to open Webots (close the window when the run is done)... ")
        subprocess.run([webots, "--mode=pause", str(world)])


def main():
    arg = sys.argv[1:] or ["all"]
    if arg[0] == "list":
        for cid, (robot, desc, models) in CLIPS.items():
            print(f"{cid:3} {robot:6} {desc}")
            for i, m in enumerate(models, 1):
                print(f"      Run {i}: {m}")
        return
    webots = find_webots()
    if arg[0] == "layout":
        print("Drag the console to the side (and size the 3D view), then close Webots.")
        for name in WORLD.values():
            subprocess.run([webots, str(WORLDS / f"{name}.wbt")])
        return
    ids = list(CLIPS) if arg[0] == "all" else arg  # CLIPS is in sheet order
    for cid in ids:
        if cid not in CLIPS:
            sys.exit(f"Unknown clip {cid}. Try: python extras/demos/record_demos.py list")
        run_clip(webots, cid)


if __name__ == "__main__":
    main()
