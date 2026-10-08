"""Task 1.C validation for SCARA forward kinematics."""

import sys
import math
from pathlib import Path

# Allow access to the shared controller and FK scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from common.arm_controller import ArmController, Joint
from forward_kinematics_scara import forward_kinematics_scara


JOINTS = {
    "shoulder": Joint("base_arm_motor", speed=0.7, tolerance=0.001),
    "elbow": Joint("arm_motor", speed=0.7, tolerance=0.001),
    "rotation": Joint("shaft_rotation_motor", speed=1.0, tolerance=0.001),
    "lift": Joint("shaft_linear_motor", speed=0.05, tolerance=0.0001),
}


# [q1, q2, shaft_rotation, q4]
# q2 is positive to keep the SCARA in the required elbow-up configuration.
POSES = [
    [0.00, 0.20, 0.0, -0.10],
    [0.30, 0.40, 0.0, -0.05],
    [-0.30, 0.60, 0.0, -0.15],
    [0.50, 0.70, 0.0, -0.20],
    [-0.50, 0.30, 0.0, -0.08],
]


def position_error(fk_position, webots_position):
    """Return x, y, z errors and total Euclidean position error."""

    dx = fk_position[0] - webots_position[0]
    dy = fk_position[1] - webots_position[1]
    dz = fk_position[2] - webots_position[2]

    total = math.sqrt(dx**2 + dy**2 + dz**2)

    return dx, dy, dz, total


if __name__ == "__main__":

    arm = ArmController(JOINTS, tcp_slot="handSlot")

    total_errors = []

    for pose_number, pose in enumerate(POSES, start=1):

        # Move the robot to the test pose and get Webots TCP.
        tcp = arm.move_to(pose)

        webots_position = tcp["arm"]

        # Only the three active SCARA joints are used by FK.
        q1 = arm.sensors["shoulder"].getValue()
        q2 = arm.sensors["elbow"].getValue()
        q4 = arm.sensors["lift"].getValue()

        # Calculate TCP using our FK.
        fk_position = forward_kinematics_scara([q1, q2, q4])

        # Calculate the difference.
        dx, dy, dz, error = position_error(
            fk_position,
            webots_position,
        )

        total_errors.append(error)

        print()
        print(f"========== Pose {pose_number} ==========")

        print(
            f"Joint values: "
            f"q1={q1:+.6f}, "
            f"q2={q2:+.6f}, "
            f"q4={q4:+.6f}"
        )

        print(
            "FK TCP:      "
            f"x={fk_position[0]:+.6f}, "
            f"y={fk_position[1]:+.6f}, "
            f"z={fk_position[2]:+.6f}"
        )

        print(
            "Webots TCP:  "
            f"x={webots_position[0]:+.6f}, "
            f"y={webots_position[1]:+.6f}, "
            f"z={webots_position[2]:+.6f}"
        )

        print(
            "Error:       "
            f"dx={dx * 1000:+.3f} mm, "
            f"dy={dy * 1000:+.3f} mm, "
            f"dz={dz * 1000:+.3f} mm"
        )

        print(
            f"Position error: {error * 1000:.3f} mm"
        )

    print()
    print("========== Summary ==========")

    print(
        f"Maximum position error: "
        f"{max(total_errors) * 1000:.3f} mm"
    )

    print(
        f"Mean position error: "
        f"{sum(total_errors) / len(total_errors) * 1000:.3f} mm"
    )

    arm.hold_position()