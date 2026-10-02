"""Prints the Ned's TCP position (via forward_kinematics_ned) every few seconds.

Never commands any motor, so the arm is free to be moved by hand while this
runs. To use it, set the Ned robot's "controller" field to "ned_fk_print" in
the Webots Scene Tree (it defaults to "ned_keyboard").
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from forward_kinematics_ned import forward_kinematics_ned

from controller import Robot

PRINT_EVERY_SECONDS = 2.0

robot = Robot()
timestep = int(robot.getBasicTimeStep())

sensors = {}
for name in ("joint_1", "joint_2", "joint_3"):
    sensor = robot.getDevice(name).getPositionSensor()
    sensor.enable(timestep)
    sensors[name] = sensor

last_print = -PRINT_EVERY_SECONDS
while robot.step(timestep) != -1:
    now = robot.getTime()
    if now - last_print < PRINT_EVERY_SECONDS:
        continue
    last_print = now

    q1 = sensors["joint_1"].getValue()
    q2 = sensors["joint_2"].getValue()
    q3 = sensors["joint_3"].getValue()
    x, y, z = forward_kinematics_ned([q1, q2, q3])
    print(f"q1={q1:+.3f} q2={q2:+.3f} q3={q3:+.3f} -> TCP=({x:+.4f}, {y:+.4f}, {z:+.4f}) m")
