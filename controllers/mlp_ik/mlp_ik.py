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


DEFAULT_MODEL = "mlp_ned_scara_large_h64x64x64x64_cosine_per-robot_tcp_p1000"
COMMAND_PERIOD = 0.05  # s
PRINT_PERIOD = 0.5  # s
TARGET_STEP = 0.002  # m per simulation step; hold a key to keep moving
TARGET_FINE_STEP = 0.0005  # m per step while Shift is held
SWEEP_SPEED = 0.10  # m/s, target speed in --sweep mode
SWEEP_PAUSE = 0.8  # s held at each sweep waypoint
SWEEP_DELAY = 2.0  # s before the sweep starts, so the arm settles first
OUT_OF_REACH_MARGIN = 0.12  # m beyond the arm's horizontal reach
# Sweep stop labels, printed in the console.
WITHIN_REACH = "within reach"
OUT_OF_REACH = "OUT OF REACH"

ROBOTS = {
    "ned": {
        "flags": (0.0, 1.0),
        "joints": ("joint_1", "joint_2", "joint_3"),
        "start": (0.0, 0.3, 0.3),
        "speeds": (0.7, 0.7, 0.7),
        "forward_kinematics": forward_kinematics_ned,
        "reach": 0.35,  # m, largest horizontal distance from the base
        # Sweep points (ARM frame): left, right, near and far.
        "sweep_points": (
            (-0.181, 0.176, 0.262),
            (0.196, 0.190, 0.255),
            (0.0, 0.154, 0.283),
            (0.0, 0.318, 0.204),
        ),
    },
    "scara": {
        "flags": (1.0, 0.0),
        # Coursework outputs are base angle, arm angle, and vertical lift.
        "joints": ("base_arm_motor", "arm_motor", "shaft_linear_motor"),
        "start": (0.0, 0.3, -0.05),
        "speeds": (0.7, 0.7, 0.05),
        "forward_kinematics": forward_kinematics_scara,
        "fixed_joints": (("shaft_rotation_motor", 0.0, 1.0),),
        "reach": 0.625,
        "sweep_points": (
            (0.471, 0.364, 0.082),
            (0.587, -0.175, 0.032),
            (0.568, 0.165, 0.182),
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=ROBOTS)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name from models/")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="move the target slowly through points within reach and out of reach",
    )
    parser.add_argument(
        "--show-gap",
        action="store_true",
        help="draw a bar from the hand to the target and label the gap in mm (for video)",
    )
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


def sweep_waypoints(config):
    """Start pose, all the points within reach (swept across by angle), then out of
    reach past the last point and past the first point (two different directions),
    then back to the start. Each waypoint is (position, label)."""

    start = config["start_tcp"]
    ordered = sorted(config["sweep_points"], key=lambda p: (math.atan2(p[1], p[0]), math.hypot(p[0], p[1])))
    points = [(start, WITHIN_REACH)] + [(p, WITHIN_REACH) for p in ordered]
    for p in (ordered[-1], ordered[0]):
        scale = (config["reach"] + OUT_OF_REACH_MARGIN) / math.hypot(p[0], p[1])
        points.append(((p[0] * scale, p[1] * scale, p[2]), OUT_OF_REACH))
    points.append((start, WITHIN_REACH))
    return points


class TargetSweep:
    """Move the target in a straight line between waypoints at a fixed slow speed."""

    def __init__(self, waypoints, begin_time):
        self.waypoints = waypoints
        self.index = 0  # waypoint being moved towards is index + 1
        self.position = waypoints[0][0]
        self.resume_at = begin_time
        self.last_time = begin_time
        self.done = False

    def update(self, now):
        """Return (position, message) where message is set on reaching a waypoint."""

        if self.done or now < self.resume_at:
            self.last_time = now
            return self.position, None
        dt, self.last_time = now - self.last_time, now
        goal, label = self.waypoints[self.index + 1]
        distance = math.dist(self.position, goal)
        travel = SWEEP_SPEED * dt
        if distance <= travel:
            self.position = goal
            self.index += 1
            self.resume_at = now + SWEEP_PAUSE
            self.done = self.index + 1 >= len(self.waypoints)
            message = f"Sweep waypoint {self.index}/{len(self.waypoints) - 1} ({label})"
            return self.position, message + (" - sweep finished" if self.done else "")
        fraction = travel / distance
        self.position = tuple(a + (b - a) * fraction for a, b in zip(self.position, goal))
        return self.position, None


def x_axis_of(rotation):
    """World direction of the local +X axis for an axis-angle rotation (Rodrigues)."""

    ax, ay, az, angle = rotation
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    kx, ky, kz = ax / norm, ay / norm, az / norm
    c, s = math.cos(angle), math.sin(angle)
    cross = (0.0, kz, -ky)  # k x (1, 0, 0)
    dot = kx
    return tuple((1, 0, 0)[i] * c + cross[i] * s + (kx, ky, kz)[i] * dot * (1 - c) for i in range(3))


class GapMarker:
    """Yellow bar from the hand to the target's centre, plus the gap in mm as an
    on-screen label. The target is made see-through so the bar shows inside it."""

    TARGET_TRANSPARENCY = 0.6
    CLOSE_DISTANCE = 0.25  # m, camera distance from the hand when zoomed in
    ZOOM_IN_GAP = 0.015  # m, fully zoomed in below this gap
    ZOOM_OUT_GAP = 0.040  # m, fully zoomed out above this gap
    ZOOM_TIME = 0.6  # s, easing time constant

    def __init__(self, robot, target_node):
        self.robot = robot
        # A line between two world points: both ends are set directly, so there is
        # no rotation to get wrong.
        line = (
            "Shape { appearance Appearance { material Material { diffuseColor 1 1 0 "
            "emissiveColor 1 1 0 } } geometry IndexedLineSet { coord Coordinate { "
            "point [ 0 0 0, 0 0 0 ] } coordIndex [ 0 1 -1 ] } }"
        )
        children = robot.getRoot().getField("children")
        children.importMFNodeFromString(-1, line)
        shape = children.getMFNode(children.getCount() - 1)
        self.points = shape.getField("geometry").getSFNode().getField("coord").getSFNode().getField("point")
        appearance = target_node.getField("children").getMFNode(0).getField("appearance").getSFNode()
        appearance.getField("transparency").setSFFloat(self.TARGET_TRANSPARENCY)
        # The camera starts at the world's own viewpoint and zooms in on the hand
        # whenever the gap is small, so a tiny gap is still readable on screen.
        root_children = robot.getRoot().getField("children")
        viewpoint = next(
            root_children.getMFNode(i)
            for i in range(root_children.getCount())
            if root_children.getMFNode(i).getTypeName() == "Viewpoint"
        )
        self.camera_position = viewpoint.getField("position")
        self.camera_start = self.camera_position.getSFVec3f()
        # Webots cameras look along their local +X axis.
        self.look = x_axis_of(viewpoint.getField("orientation").getSFRotation())
        self.zoom = 0.0  # 0 = the world's viewpoint, 1 = close on the hand
        self.last_time = robot.getTime()

    def move_camera(self, hand, gap, now):
        """Ease the viewpoint towards the hand while the gap is small."""

        wanted = min(1.0, max(0.0, (self.ZOOM_OUT_GAP - gap) / (self.ZOOM_OUT_GAP - self.ZOOM_IN_GAP)))
        self.zoom += (wanted - self.zoom) * (1 - math.exp(-(now - self.last_time) / self.ZOOM_TIME))
        self.last_time = now
        close = [h - f * self.CLOSE_DISTANCE for h, f in zip(hand, self.look)]
        self.camera_position.setSFVec3f([s + (c - s) * self.zoom for s, c in zip(self.camera_start, close)])

    def update(self, hand, target, now):
        """hand and target are world positions."""

        gap = math.dist(hand, target)
        self.move_camera(hand, gap, now)
        self.points.setMFVec3f(0, list(hand))
        self.points.setMFVec3f(1, list(target))
        self.robot.setLabel(0, f"Gap: {gap * 1000:.1f} mm", 0.02, 0.02, 0.08, 0xFFFF00, 0.0, "Arial")


def arm_to_world(arm_node, point):
    """Convert an ARM-frame point to world coordinates."""

    pose = arm_node.getPose()
    return [sum(pose[4 * row + col] * point[col] for col in range(3)) + pose[4 * row + 3] for row in range(3)]


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

    gap_marker = None
    if args.show_gap:
        if tcp_node is None:
            print("WARNING: --show-gap needs a DEF TCP node, so no gap marker is drawn")
        else:
            gap_marker = GapMarker(robot, target_node)

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

    sweep = None
    if args.sweep:
        # The sweep starts from wherever the world places the target.
        config = {**config, "start_tcp": position_in_arm_frame(target_node, arm_node)}
        sweep = TargetSweep(sweep_waypoints(config), robot.getTime() + SWEEP_DELAY)
        print(f"Sweep mode: target moves by itself at {SWEEP_SPEED * 100:.0f} cm/s after {SWEEP_DELAY:.0f} s")

    while robot.step(timestep) != -1:
        now = robot.getTime()
        if move_target_from_keyboard(keyboard, target_translation):
            print(f"Target moved to world position: {target_translation.getSFVec3f()}")

        if sweep is not None:
            position, message = sweep.update(now)
            target_translation.setSFVec3f(arm_to_world(arm_node, position))
            if message:
                print(f"\n*** {message} ***")

        if gap_marker is not None:
            gap_marker.update(tcp_node.getPosition(), target_node.getPosition(), now)

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
