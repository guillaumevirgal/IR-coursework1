"""Run the shared trained MLP inverse-kinematics controller in Webots.

Both worlds pass their robot name as a controller argument. The MLP input is
[is_scara, is_ned, x, y, z], and its three outputs are mapped to the active
joint ranges for that robot.
"""

import argparse
import json
import math
import shlex
import sys
from pathlib import Path

import numpy as np
from controller import Keyboard, Supervisor

# Webots starts each controller from its own folder.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.forward_kinematics_ned import forward_kinematics_ned
from scripts.forward_kinematics_scara import forward_kinematics_scara
from scripts.robot_config import JOINT_LIMITS, denormalise
from scripts.train_mlp import INPUTS, MLP, MODELS, OUTPUTS


DEFAULT_MODEL = "mlp_ned_scara_large_h64x64_cosine"
COMMAND_PERIOD = 0.05  # s
PRINT_PERIOD = 0.5  # s
TARGET_STEP = 0.002  # m per simulation step; hold a key to keep moving
TARGET_FINE_STEP = 0.0005  # m per step while Shift is held

ROBOTS = {
    "ned": {
        "flags": (0.0, 1.0),
        "joints": ("joint_1", "joint_2", "joint_3"),
        "start": (0.0, 0.3, 0.3),
        "speeds": (0.7, 0.7, 0.7),
        "forward_kinematics": forward_kinematics_ned,
    },
    "scara": {
        "flags": (1.0, 0.0),
        # Coursework outputs are base angle, arm angle, and vertical lift.
        "joints": ("base_arm_motor", "arm_motor", "shaft_linear_motor"),
        "start": (0.0, 0.3, -0.05),
        "speeds": (0.7, 0.7, 0.05),
        "forward_kinematics": forward_kinematics_scara,
        "fixed_joints": (("shaft_rotation_motor", 0.0, 1.0),),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=ROBOTS)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name from models/")
    arguments = sys.argv[1:]
    # Webots' GUI controller-argument field can pass the complete text as one
    # argv item, unlike the world's controllerArgs MFString, which passes one
    # item per entry. Accept both forms so a pasted line such as
    # "ned --model=..." works in either place.
    if len(arguments) == 1 and any(char.isspace() for char in arguments[0]):
        arguments = shlex.split(arguments[0])
    return parser.parse_args(arguments)


class SharedMLPIK:
    """Load a shared model and predict joints for either robot."""

    def __init__(self, robot_name, model_name):
        with (MODELS / f"{model_name}.json").open(encoding="utf-8") as source:
            config = json.load(source)
        if robot_name not in config["robots"]:
            raise ValueError(f"Model {model_name} was not trained on {robot_name}")
        hidden = config["hidden"] if isinstance(config["hidden"], list) else [config["hidden"]]
        activation = config.get("activation") or "relu"  # pre-activation-flag models were ReLU
        self.model = MLP((len(INPUTS), *hidden, len(OUTPUTS)), np.random.default_rng(0), activation=activation)
        with np.load(MODELS / f"{model_name}.npz") as stored:
            self.model.params = {key: stored[key] for key in self.model.params}
            self.mean = stored["mean"]
            self.std = stored["std"]
        self.robot_name = robot_name
        self.description = f"{model_name} ({'x'.join(map(str, hidden))} hidden, {config['size']} set)"

    def solve(self, target):
        """Return the robot's three active joints for a base-frame TCP target."""

        flags = ROBOTS[self.robot_name]["flags"]
        x = (np.array([[*flags, *target]]) - self.mean) / self.std
        q_norm = np.clip(self.model.predict(x)[0], -1.0, 1.0)
        return tuple(
            float(q) for q in denormalise(q_norm, JOINT_LIMITS[self.robot_name])
        )


def position_in_arm_frame(node, arm_node):
    """Translation part of the node's 4x4 pose relative to the arm."""

    pose = node.getPose(arm_node)
    return pose[3], pose[7], pose[11]


def move_target_from_keyboard(keyboard, translation_field):
    """Move the target sphere in world coordinates using the arrow keys."""

    key = keyboard.getKey()
    if key < 0:
        return False

    step = TARGET_FINE_STEP if key & Keyboard.SHIFT else TARGET_STEP
    directions = {
        Keyboard.LEFT: (0, -step),
        Keyboard.RIGHT: (0, step),
        Keyboard.DOWN: (1, -step),
        Keyboard.UP: (1, step),
        Keyboard.PAGEDOWN: (2, -step),
        Keyboard.PAGEUP: (2, step),
    }
    movement = directions.get(key & Keyboard.KEY)
    if movement is None:
        return False

    axis, amount = movement
    position = list(translation_field.getSFVec3f())
    position[axis] += amount
    translation_field.setSFVec3f(position)
    return True


def main():
    args = parse_args()
    config = ROBOTS[args.robot]
    ik = SharedMLPIK(args.robot, args.model)
    print(f"Loaded {args.robot} MLP IK model {ik.description}")

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    arm_node = robot.getFromDef("ARM")
    target_node = robot.getFromDef("TARGET")
    tcp_node = robot.getFromDef("TCP")
    if arm_node is None or target_node is None:
        print("ERROR: the world needs DEF ARM and DEF TARGET nodes")
        return
    if tcp_node is None:
        print("WARNING: no DEF TCP node, the measured TCP error will not be printed")

    keyboard = Keyboard()
    keyboard.enable(timestep)
    print("Move target: arrows change X/Y, Page Up/Down changes Z; hold Shift for finer steps")

    target_translation = target_node.getField("translation")
    motors = [robot.getDevice(name) for name in config["joints"]]
    for motor, value, speed in zip(motors, config["start"], config["speeds"]):
        motor.setVelocity(speed)
        motor.setPosition(value)
    for name, value, speed in config.get("fixed_joints", ()):
        motor = robot.getDevice(name)
        motor.setVelocity(speed)
        motor.setPosition(value)

    last_command = last_print = -math.inf
    joints = config["start"]
    target = None

    while robot.step(timestep) != -1:
        now = robot.getTime()
        if move_target_from_keyboard(keyboard, target_translation):
            print(f"Target moved to world position: {target_translation.getSFVec3f()}")

        if now - last_command >= COMMAND_PERIOD:
            target = position_in_arm_frame(target_node, arm_node)
            joints = ik.solve(target)
            for motor, value in zip(motors, joints):
                motor.setPosition(value)
            last_command = now

        if now - last_print >= PRINT_PERIOD:
            predicted_tcp = config["forward_kinematics"](joints)
            print("\n--------------------------------")
            print(f"Target (ARM frame):   x={target[0]: .4f}  y={target[1]: .4f}  z={target[2]: .4f}")
            print("MLP joints:           " + "  ".join(f"q{i + 1}={q: .4f}" for i, q in enumerate(joints)))
            print(f"FK error of MLP pose: {math.dist(predicted_tcp, target) * 1000:.1f} mm")
            if tcp_node is not None:
                tcp = position_in_arm_frame(tcp_node, arm_node)
                print(f"Measured TCP:         x={tcp[0]: .4f}  y={tcp[1]: .4f}  z={tcp[2]: .4f}")
                print(f"Measured TCP error:   {math.dist(tcp, target) * 1000:.1f} mm")
            last_print = now


if __name__ == "__main__":
    main()
