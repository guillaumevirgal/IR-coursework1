"""Run the shared MLP IK model and show a live explanation dashboard in Webots."""

import math
import sys
from pathlib import Path

from controller import Keyboard, Supervisor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "controllers" / "mlp_ik"))

from mlp_ik import (  # noqa: E402
    COMMAND_PERIOD,
    DEFAULT_MODEL,
    PRINT_PERIOD,
    ROBOTS,
    SharedMLPIK,
    move_target_from_keyboard,
    parse_args,
    position_in_arm_frame,
)


def main():
    args = parse_args()
    config = ROBOTS[args.robot]
    ik = SharedMLPIK(args.robot, args.model or DEFAULT_MODEL)
    print(f"Loaded {args.robot} MLP IK model {ik.description}")
    try:
        from controllers.common.mlp_prediction_dashboard import MLPPredictionDashboard

        dashboard = MLPPredictionDashboard(ik)
    except Exception as error:
        print(f"ERROR: could not start the live MLP dashboard: {error}")
        print("Install the repository requirements into the Python used by Webots.")
        return

    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    arm_node = robot.getFromDef("ARM")
    target_node = robot.getFromDef("TARGET")
    tcp_node = robot.getFromDef("TCP")
    if arm_node is None or target_node is None:
        print("ERROR: the world needs DEF ARM and DEF TARGET nodes")
        return
    if tcp_node is None:
        print("WARNING: no DEF TCP node; the dashboard will omit measured TCP")

    keyboard = Keyboard()
    keyboard.enable(timestep)
    print("Move target: arrows change X/Y, Page Up/Down changes Z; hold Shift for finer steps")
    print("The live MLP explanation is in the separate Matplotlib window.")

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
    last_dashboard_update = -math.inf
    joints = config["start"]
    target = None
    last_visual_target = None

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

            # Also notices target edits made directly in the Webots scene tree.
            target_changed = last_visual_target is None or any(
                abs(current - previous) > 1e-9
                for current, previous in zip(target, last_visual_target)
            )
            # Refresh measured TCP while the robot moves, and immediately when
            # the target changes. This keeps the network and physical result distinct.
            if target_changed or now - last_dashboard_update >= 0.2:
                measured_tcp = (
                    position_in_arm_frame(tcp_node, arm_node) if tcp_node is not None else None
                )
                dashboard.update(target, joints, measured_tcp)
                last_visual_target = tuple(target)
                last_dashboard_update = now

        if now - last_print >= PRINT_PERIOD:
            predicted_tcp = config["forward_kinematics"](joints)
            print("\n--------------------------------")
            print(f"Target (ARM frame):   x={target[0]: .4f}  y={target[1]: .4f}  z={target[2]: .4f}")
            print("MLP joints:           " + "  ".join(
                f"q{i + 1}={q: .4f}" for i, q in enumerate(joints)
            ))
            print(f"FK error of MLP pose: {math.dist(predicted_tcp, target) * 1000:.1f} mm")
            if tcp_node is not None:
                tcp = position_in_arm_frame(tcp_node, arm_node)
                print(f"Measured TCP:         x={tcp[0]: .4f}  y={tcp[1]: .4f}  z={tcp[2]: .4f}")
                print(f"Measured TCP error:   {math.dist(tcp, target) * 1000:.1f} mm")
            last_print = now

        dashboard.process_events()


if __name__ == "__main__":
    main()
