"""Reusable motor control and TCP measurement for Webots arms."""

import math
from dataclasses import dataclass
from controller import Supervisor


@dataclass(frozen=True)
class Joint:
    motor_name: str    # Device name in the Webots robot model
    speed: float       # Maximum joint speed
    tolerance: float   # How close the sensor must be to the target (rad or m)

    def __post_init__(self):
        # "not x > 0" also rejects NaN, which "x <= 0" would let through.
        if not self.speed > 0:
            raise ValueError(f"{self.motor_name}: speed must be positive.")
        if not self.tolerance > 0:
            raise ValueError(f"{self.motor_name}: tolerance must be positive.")


class ArmController:
    def __init__(self, joints, tcp_slot, settle_time=0.25, timeout=10.0):
        """
        joints:      dict mapping a friendly name to a Joint.
        tcp_slot:    name of the robot field holding the TCP marker, e.g. "handSlot".
        settle_time: seconds all joints must stay on target.
        timeout:     seconds to wait for the arm before raising TimeoutError.
        """
        # Supervisor lets this controller inspect the robot and the TCP marker.
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.joints = joints
        self.settle_time = settle_time
        self.timeout = timeout

        self.motors = {}
        self.sensors = {}
        self._setup_joints()

        # Sensors return NaN until the first step, so take one before anyone reads them.
        if self.robot.step(self.timestep) == -1:
            raise RuntimeError("Simulation stopped during startup.")

        self.arm = self.robot.getSelf()
        self.tcp = self._find_tcp(tcp_slot)

    # ---- Public API ----

    def move_to(self, joint_values):
        """Move to targets, wait for the arm, and return the measured TCP position.

        Targets can be a list in the same order as the joints were configured,
        or a dictionary with one value for each joint name.
        """
        targets = self._check_targets(joint_values)
        for name, target in targets.items():
            self.motors[name].setPosition(target)
        self.wait_until_settled(targets)
        return self.get_tcp_position()

    def read_joints(self):
        """Return the current sensor reading for each joint, keyed by joint name."""
        return {name: sensor.getValue() for name, sensor in self.sensors.items()}

    def wait_until_settled(self, targets):
        """Block until every joint stays within tolerance for settle_time seconds."""
        start_time = self.robot.getTime()
        settled_time = 0.0

        while self.robot.step(self.timestep) != -1:
            errors = self._joint_errors(targets)

            if self._off_target(errors):
                settled_time = 0.0  # A joint drifted away, so restart the clock.
            else:
                settled_time += self.timestep / 1000.0
            if settled_time >= self.settle_time:
                return

            if self.robot.getTime() - start_time >= self.timeout:
                still_off = {n: round(e, 4) for n, e in self._off_target(errors).items()}
                raise TimeoutError(
                    f"The arm did not settle within {self.timeout} seconds. "
                    f"Joints still off target (error): {still_off}"
                )

        raise RuntimeError("Simulation stopped before the arm settled.")

    def get_tcp_position(self):
        """Return TCP coordinates in metres, in world and arm frames."""
        # getPose returns a 4x4 matrix as a flat list; entries 3, 7, 11 are x, y, z.
        pose = self.tcp.getPose(self.arm)
        return {
            "world": tuple(self.tcp.getPosition()),
            "arm": (pose[3], pose[7], pose[11]),
        }

    def hold_position(self):
        """Keep stepping so Webots stays open and the motors hold their targets."""
        while self.robot.step(self.timestep) != -1:
            pass

    # ---- Setup helpers/Private functions ----

    def _setup_joints(self):
        """Find each motor, enable its position sensor, and cap its speed."""
        for name, joint in self.joints.items():
            motor = self.robot.getDevice(joint.motor_name)
            if motor is None:
                raise RuntimeError(f"Missing motor: {joint.motor_name}")
            sensor = motor.getPositionSensor()
            if sensor is None:
                raise RuntimeError(f"Missing position sensor: {joint.motor_name}")

            sensor.enable(self.timestep)
            motor.setVelocity(min(joint.speed, motor.getMaxVelocity()))
            self.motors[name] = motor
            self.sensors[name] = sensor

    def _find_tcp(self, slot_name):
        """Return the TCP marker node, which must be the first node in the slot."""
        slot = self.arm.getField(slot_name)
        if slot is None or slot.getCount() == 0:
            raise RuntimeError(f"Add a TCP marker as the first node in {slot_name}.")
        return slot.getMFNode(0)

    # ---- Movement helpers ----

    def _check_targets(self, joint_values):
        """Check the targets and match each value with its joint name."""
        if isinstance(joint_values, dict):
            # Dictionaries label each value, so check that all joint names are present.
            if joint_values.keys() != self.joints.keys():
                raise ValueError(f"Provide exactly these joints: {list(self.joints)}")
            targets = {name: float(value) for name, value in joint_values.items()}
        else:
            # Lists have no labels, so their values must follow the configured joint order.
            values = list(joint_values)
            names = list(self.joints)
            if len(values) != len(names):
                raise ValueError(f"Provide {len(names)} values in this order: {names}")
            targets = {name: float(value) for name, value in zip(names, values)}

        for name, target in targets.items():
            if not math.isfinite(target):
                raise ValueError(f"{name}: target must be finite.")
            motor = self.motors[name]
            lower, upper = motor.getMinPosition(), motor.getMaxPosition()
            # Equal limits mean the Webots motor is unlimited.
            if lower != upper and not lower <= target <= upper:
                raise ValueError(f"{name}: target must be between {lower} and {upper}.")
        return targets

    def _joint_errors(self, targets):
        """Return how far each joint is from its target."""
        return {
            name: abs(measured - targets[name])
            for name, measured in self.read_joints().items()
        }

    def _off_target(self, errors):
        """Return the joints outside their tolerance."""
        # "not error <= tolerance" so a NaN reading counts as off target.
        return {
            name: error
            for name, error in errors.items()
            if not error <= self.joints[name].tolerance
        }
