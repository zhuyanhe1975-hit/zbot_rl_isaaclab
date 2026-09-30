"""Hardware-facing observation and action contract for the periodic student."""

from dataclasses import dataclass, field

import numpy as np

DEFAULT_JOINT_POS = np.array([0.312, 0.837, -2.02, 2.02, -0.837, -0.312], dtype=np.float32)
CONTROL_DT = 0.02
JOINT_SPEED_LIMIT = 2.0 * np.pi
POSITION_OFFSET_LIMIT = 0.5 * np.pi


def _vector(values: np.ndarray | list[float] | tuple[float, ...], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values.")
    return result


@dataclass(frozen=True)
class StudentSensors:
    """All values are expressed in the robot base frame or motor joint convention."""

    angular_velocity_b: np.ndarray | list[float]
    projected_gravity_b: np.ndarray | list[float]
    yaw_rad: float
    linear_acceleration_b: np.ndarray | list[float]
    joint_position: np.ndarray | list[float]
    joint_velocity: np.ndarray | list[float]
    motor_torque_nm: np.ndarray | list[float]


@dataclass
class StudentControllerState:
    """Maintain the previous raw action, phase, and integrated position target."""

    frequency_hz: float = 1.0
    phase: float = 0.0
    heading_reference_rad: float = 0.0
    previous_raw_action: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    position_offset: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))

    def __post_init__(self) -> None:
        self.set_frequency(self.frequency_hz)

    def set_frequency(self, value: float) -> None:
        if not np.isfinite(value) or not 0.5 <= value <= 2.0:
            raise ValueError("Frequency must be within 0.5–2.0 Hz.")
        self.frequency_hz = float(value)

    def reset(self, yaw_rad: float = 0.0) -> None:
        if not np.isfinite(yaw_rad):
            raise ValueError("Initial yaw must be finite.")
        self.heading_reference_rad = float(yaw_rad)
        self.phase = 0.0
        self.previous_raw_action = np.zeros(6, dtype=np.float32)
        self.position_offset = np.zeros(6, dtype=np.float32)

    def observation(self, sensors: StudentSensors) -> np.ndarray:
        """Return the 38 policy inputs in the exact training order."""
        angular = _vector(sensors.angular_velocity_b, 3, "angular_velocity_b")
        gravity = _vector(sensors.projected_gravity_b, 3, "projected_gravity_b")
        if not np.isfinite(sensors.yaw_rad):
            raise ValueError("yaw_rad must be finite.")
        heading_error = self.heading_reference_rad - sensors.yaw_rad
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        acceleration = _vector(sensors.linear_acceleration_b, 3, "linear_acceleration_b")
        position = _vector(sensors.joint_position, 6, "joint_position")
        velocity = _vector(sensors.joint_velocity, 6, "joint_velocity")
        torque = _vector(sensors.motor_torque_nm, 6, "motor_torque_nm")
        previous = _vector(self.previous_raw_action, 6, "previous_raw_action")
        side = 1.0 if self.phase < 0.5 else -1.0
        angle = 2.0 * np.pi * self.phase
        return np.concatenate(
            (
                angular,
                gravity,
                np.array([heading_error], dtype=np.float32),
                acceleration * 0.1,
                position - DEFAULT_JOINT_POS,
                velocity,
                torque * 0.01,
                previous,
                np.array([self.frequency_hz, side, np.sin(angle), np.cos(angle)], dtype=np.float32),
            )
        ).astype(np.float32)

    def advance(self, raw_action: np.ndarray | list[float]) -> np.ndarray:
        """Integrate six raw outputs into bounded joint-position targets."""
        raw = _vector(raw_action, 6, "raw_action")
        self.position_offset = np.clip(
            self.position_offset + np.tanh(raw) * JOINT_SPEED_LIMIT * CONTROL_DT,
            -POSITION_OFFSET_LIMIT,
            POSITION_OFFSET_LIMIT,
        ).astype(np.float32)
        self.previous_raw_action = raw.copy()
        self.phase = (self.phase + self.frequency_hz * CONTROL_DT) % 1.0
        return DEFAULT_JOINT_POS + self.position_offset
