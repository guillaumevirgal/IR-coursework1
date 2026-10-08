"""Task 1.C: measures how precise the forward kinematics scripts are against the simulator.

Controller args: <robot> [samples] [seed], where robot is "ned" or "scara".
Normally started by scripts/run_fk_precision.sh, which sets these for you.

For every sample it commands random joints, waits for the arm to settle, and
compares the TCP Webots measures with the TCP forward kinematics predicts.
Results go to data/evaluation/<robot>_fk.csv; the simulation quits at the end.
"""

import csv
import math
import random
import sys
from pathlib import Path

# Webots starts each controller from its own folder.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "controllers"))
sys.path.insert(0, str(ROOT / "scripts"))
from common.arm_controller import ArmController, Joint
from forward_kinematics_ned import forward_kinematics_ned
from forward_kinematics_scara import forward_kinematics_scara

# Tight tolerances so the joints settle well inside the precision being measured.
ROBOTS = {
    "ned": {
        "joints": {
            "joint_1": Joint("joint_1", speed=2.0, tolerance=1e-4),
            "joint_2": Joint("joint_2", speed=2.0, tolerance=1e-4),
            "joint_3": Joint("joint_3", speed=2.0, tolerance=1e-4),
        },
        "tcp_slot": "tcpSlot",
        "fk": forward_kinematics_ned,
        "limits": [(-2.8, 2.8), (-0.8, 0.8), (-1.5, 1.3)],  # protos/TeachingNed.proto
        "min_z": 0.14,  # below ~0.13 m the wrist/gripper touches the floor and pushes the arm
    },
    "scara": {
        "joints": {
            "shoulder": Joint("base_arm_motor", speed=0.7, tolerance=1e-4),
            "elbow": Joint("arm_motor", speed=0.7, tolerance=1e-4),
            "lift": Joint("shaft_linear_motor", speed=0.05, tolerance=1e-5),
        },
        "tcp_slot": "handSlot",
        "fk": forward_kinematics_scara,
        "limits": [(-0.73, 0.73), (-0.83, 0.83), (-0.200, 0.0)],  # ScaraT6 proto
        "min_z": 0.02,  # below ~0.012 m the shaft hits the floor before the lift limit
    },
}

FIELDS = [
    "robot", "sample",
    "q1", "q2", "q3",                    # commanded joints
    "q1_meas", "q2_meas", "q3_meas",     # joints read back after settling
    "fk_x", "fk_y", "fk_z",              # TCP predicted by FK from the commanded joints
    "meas_x", "meas_y", "meas_z",        # TCP measured by Webots, robot base frame
    "error",                             # |measured - FK(commanded joints)| in metres
    "model_error",                       # |measured - FK(measured joints)| in metres
    "status",
]


def random_pose(config, rng):
    """Random joints inside the limits whose TCP stays above min_z."""
    while True:
        pose = [rng.uniform(low, high) for low, high in config["limits"]]
        if config["fk"](pose)[2] >= config["min_z"]:
            return pose


def measure(arm, config, robot_name, sample, pose):
    predicted = config["fk"](pose)
    row = {"robot": robot_name, "sample": sample}
    row.update(zip(("q1", "q2", "q3"), pose))
    row.update(zip(("fk_x", "fk_y", "fk_z"), predicted))
    try:
        measured = arm.move_to(pose)["arm"]
        row["status"] = "ok"
    except TimeoutError as error:
        # Still measure: model_error uses the joints actually reached, so it stays valid.
        print(f"sample {sample}: {error}")
        measured = arm.get_tcp_position()["arm"]
        row["status"] = "not_settled"

    measured_joints = list(arm.read_joints().values())
    row.update(zip(("q1_meas", "q2_meas", "q3_meas"), measured_joints))
    row.update(zip(("meas_x", "meas_y", "meas_z"), measured))
    row["error"] = math.dist(measured, predicted)
    row["model_error"] = math.dist(measured, config["fk"](measured_joints))
    return row


def main():
    robot_name = sys.argv[1] if len(sys.argv) > 1 else "ned"
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    config = ROBOTS[robot_name]
    rng = random.Random(seed)

    arm = ArmController(config["joints"], tcp_slot=config["tcp_slot"])
    out_dir = ROOT / "data" / "evaluation"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{robot_name}_fk.csv"

    with open(out_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for sample in range(samples):
            writer.writerow(measure(arm, config, robot_name, sample, random_pose(config, rng)))
            file.flush()

    print(f"Saved {out_path}")
    arm.robot.simulationQuit(0)


if __name__ == "__main__":
    main()
