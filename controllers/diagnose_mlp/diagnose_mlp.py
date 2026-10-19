"""Run saved MLP targets through the actual Webots arm and record settled TCPs."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from controller import Supervisor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.forward_kinematics_ned import forward_kinematics_ned
from scripts.forward_kinematics_scara import forward_kinematics_scara
from scripts.train_mlp import INPUTS, JOINT_LIMITS, MLP, OUTPUTS, denormalise
from controllers.mlp_ik.mlp_ik import ROBOTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", choices=tuple(ROBOTS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--settle-time", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def load_model(name, robot):
    weights_path = ROOT / "models" / f"{name}.npz"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Missing model weights {weights_path}; run scripts/train_mlp.py first"
        )
    with (ROOT / "models" / f"{name}.json").open(encoding="utf-8") as source:
        config = json.load(source)
    if robot not in config["robots"]:
        raise ValueError(f"Model {name} was not trained on {robot}")
    hidden = config["hidden"] if isinstance(config["hidden"], list) else [config["hidden"]]
    model = MLP((len(INPUTS), *hidden, len(OUTPUTS)), np.random.default_rng(0), activation=config.get("activation") or "relu")
    with np.load(weights_path) as stored:
        model.params = {key: stored[key] for key in model.params}
        mean, std = stored["mean"], stored["std"]
    return model, mean, std


def main():
    args = parse_args()
    robot_cfg = ROBOTS[args.robot]
    fk = forward_kinematics_scara if args.robot == "scara" else forward_kinematics_ned
    model, mean, std = load_model(args.model, args.robot)
    sim = Supervisor()
    timestep = int(sim.getBasicTimeStep())
    arm = sim.getSelf()
    tcp = sim.getFromDef("TCP")
    if tcp is None:
        raise RuntimeError("The diagnostic world must define DEF TCP")
    sensors, motors = [], []
    for name, speed in zip(robot_cfg["joints"], robot_cfg["speeds"]):
        motor = sim.getDevice(name)
        sensor = motor.getPositionSensor()
        sensor.enable(timestep)
        motor.setVelocity(min(speed, motor.getMaxVelocity()))
        motors.append(motor)
        sensors.append(sensor)
    for name, value, speed in robot_cfg.get("fixed_joints", ()):
        motor = sim.getDevice(name)
        motor.setVelocity(min(speed, motor.getMaxVelocity()))
        motor.setPosition(value)
    sim.step(timestep)

    fields = ["robot", "label", "kind", "x", "y", "z", "raw_q1", "raw_q2", "raw_q3",
              "command_q1", "command_q2", "command_q3", "measured_q1", "measured_q2",
              "measured_q3", "fk_x", "fk_y", "fk_z", "webots_x", "webots_y", "webots_z",
              "fk_target_error_mm", "fk_webots_gap_mm", "webots_error_mm", "settled", "settle_seconds"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.points.open(newline="", encoding="utf-8") as source, args.output.open(
            "w", newline="", encoding="utf-8") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            target = np.array([float(row[axis]) for axis in ("x", "y", "z")])
            flags = (1.0, 0.0) if args.robot == "scara" else (0.0, 1.0)
            normalized_input = (np.array([[*flags, *target]]) - mean) / std
            raw = model.predict(normalized_input)[0]
            clipped = np.clip(raw, -1.0, 1.0)
            command = denormalise(clipped, JOINT_LIMITS[args.robot])
            for motor, value in zip(motors, command):
                motor.setPosition(float(value))

            start = sim.getTime()
            stable = 0.0
            settled = False
            while sim.step(timestep) != -1:
                errors = [abs(sensor.getValue() - value) for sensor, value in zip(sensors, command)]
                if all(error <= 1e-3 for error in errors):
                    stable += timestep / 1000.0
                else:
                    stable = 0.0
                if stable >= args.settle_time:
                    settled = True
                    break
                if sim.getTime() - start >= args.timeout:
                    break

            measured_q = [sensor.getValue() for sensor in sensors]
            fk_tcp = np.asarray(fk(command), dtype=float)
            pose = tcp.getPose(arm)
            webots_tcp = np.asarray((pose[3], pose[7], pose[11]), dtype=float)
            writer.writerow({
                "robot": args.robot, "label": row["label"], "kind": row["kind"],
                **dict(zip(("x", "y", "z"), target)),
                **dict(zip(("raw_q1", "raw_q2", "raw_q3"), raw)),
                **dict(zip(("command_q1", "command_q2", "command_q3"), command)),
                **dict(zip(("measured_q1", "measured_q2", "measured_q3"), measured_q)),
                **dict(zip(("fk_x", "fk_y", "fk_z"), fk_tcp)),
                **dict(zip(("webots_x", "webots_y", "webots_z"), webots_tcp)),
                "fk_target_error_mm": float(np.linalg.norm(fk_tcp - target) * 1000.0),
                "fk_webots_gap_mm": float(np.linalg.norm(fk_tcp - webots_tcp) * 1000.0),
                "webots_error_mm": float(np.linalg.norm(webots_tcp - target) * 1000.0),
                "settled": settled, "settle_seconds": sim.getTime() - start,
            })
            destination.flush()
    print(f"Webots diagnostics saved to {args.output}")
    sim.simulationQuit(0)


if __name__ == "__main__":
    main()
