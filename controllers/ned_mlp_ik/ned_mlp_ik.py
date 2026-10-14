"""Make the Ned TCP follow the TARGET sphere, using the trained MLP as the IK.

Every command period the sphere position is read in the ARM frame, fed to the
MLP as [is_scara=0, is_ned=1, x, y, z], and the predicted normalised joints are
mapped back to radians and sent to joint_1, joint_2 and joint_3.

MODEL_NAME picks the model: the stem of a file in models/ trained with

The MLP only knows the workspace it was trained on (joint ranges in
scripts/train_mlp.py, TCP above ~0.14 m). Outside it, the prediction is the
network's best guess, so the printed FK error is the thing to watch.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from controller import Supervisor

# Webots starts each controller from its own folder.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.forward_kinematics_ned import forward_kinematics_ned
from scripts.train_mlp import INPUTS, JOINT_LIMITS, MLP, MODELS, OUTPUTS, denormalise


MODEL_NAME = "mlp_ned_scara_large_h64x64_cosine"
NED_FLAGS = (0.0, 1.0)  # (is_scara, is_ned)

JOINTS = ("joint_1", "joint_2", "joint_3")
START_POSE = (0.0, 0.3, 0.3)
SPEED = 0.7  # rad/s, same as ned_sphere_ik

COMMAND_PERIOD = 0.05  # s
PRINT_PERIOD = 0.5  # s


class MLP_IK:
    """Load a trained model and turn Ned TCP targets into joint angles."""

    def __init__(self, name):
        with (MODELS / f"{name}.json").open(encoding="utf-8") as source:
            config = json.load(source)
        if "ned" not in config["robots"]:
            raise ValueError(f"Model {name} was not trained on the Ned")
        hidden = config["hidden"] if isinstance(config["hidden"], list) else [config["hidden"]]
        self.model = MLP((len(INPUTS), *hidden, len(OUTPUTS)), np.random.default_rng(0))
        stored = np.load(MODELS / f"{name}.npz")
        self.model.params = {key: stored[key] for key in self.model.params}
        self.mean = stored["mean"]
        self.std = stored["std"]
        self.description = f"{name} ({'x'.join(map(str, hidden))} hidden, {config['size']} set)"

    def solve(self, target):
        """Return (q1, q2, q3) in radians for a TCP target in the ARM frame."""

        x = (np.array([[*NED_FLAGS, *target]]) - self.mean) / self.std
        q_norm = np.clip(self.model.predict(x)[0], -1.0, 1.0)
        return tuple(float(q) for q in denormalise(q_norm, JOINT_LIMITS["ned"]))


def position_in_arm_frame(node, arm_node):
    """Translation part of the node's 4x4 pose relative to the arm."""

    pose = node.getPose(arm_node)
    return pose[3], pose[7], pose[11]


def main():
    ik = MLP_IK(MODEL_NAME)
    print(f"Loaded MLP IK model {ik.description}")

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

    motors = [robot.getDevice(name) for name in JOINTS]
    for motor, value in zip(motors, START_POSE):
        motor.setVelocity(SPEED)
        motor.setPosition(value)

    last_command = last_print = -math.inf
    joints = START_POSE
    target = None

    while robot.step(timestep) != -1:
        now = robot.getTime()

        if now - last_command >= COMMAND_PERIOD:
            target = position_in_arm_frame(target_node, arm_node)
            joints = ik.solve(target)
            for motor, value in zip(motors, joints):
                motor.setPosition(value)
            last_command = now

        if now - last_print >= PRINT_PERIOD:
            predicted_tcp = forward_kinematics_ned(joints)
            print("\n--------------------------------")
            print(f"Target (ARM frame):   x={target[0]: .4f}  y={target[1]: .4f}  z={target[2]: .4f}")
            print(f"MLP joints (rad):     q1={joints[0]: .4f}  q2={joints[1]: .4f}  q3={joints[2]: .4f}")
            print(f"FK error of MLP pose: {math.dist(predicted_tcp, target) * 1000:.1f} mm")
            if tcp_node is not None:
                tcp = position_in_arm_frame(tcp_node, arm_node)
                print(f"Measured TCP:         x={tcp[0]: .4f}  y={tcp[1]: .4f}  z={tcp[2]: .4f}")
                print(f"Measured TCP error:   {math.dist(tcp, target) * 1000:.1f} mm")
            last_print = now


if __name__ == "__main__":
    main()
