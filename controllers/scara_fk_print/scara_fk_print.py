"""Prints the SCARA's TCP position via forward kinematics every few seconds.

Never commands any motor, so the arm is free to be moved by hand while this
runs. To use it, set the SCARA robot's "controller" field to "scara_fk_print".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from forward_kinematics_scara import forward_kinematics_scara

from controller import Robot

PRINT_EVERY_SECONDS = 2.0

robot = Robot()
timestep = int(robot.getBasicTimeStep())

sensors = {}
for name in (
    "base_arm_motor",
    "arm_motor",
    "shaft_linear_motor",
):
    sensor = robot.getDevice(name).getPositionSensor()
    sensor.enable(timestep)
    sensors[name] = sensor

last_print = -PRINT_EVERY_SECONDS

while robot.step(timestep) != -1:
    now = robot.getTime()

    if now - last_print < PRINT_EVERY_SECONDS:
        continue

    last_print = now

    q1 = sensors["base_arm_motor"].getValue()
    q2 = sensors["arm_motor"].getValue()
    q4 = sensors["shaft_linear_motor"].getValue()

    x, y, z = forward_kinematics_scara([q1, q2, q4])

    print(
        f"q1={q1:+.3f} q2={q2:+.3f} q4={q4:+.3f} "
        f"-> TCP=({x:+.4f}, {y:+.4f}, {z:+.4f}) m"
    )