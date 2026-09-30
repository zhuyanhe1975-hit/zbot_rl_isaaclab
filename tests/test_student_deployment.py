import numpy as np
import torch

from zbot_rl_isaaclab.deployment import DEFAULT_JOINT_POS, StudentControllerState, StudentSensors
from zbot_rl_isaaclab.tasks.velocity.mdp.actions import integrate_velocity_to_position_offset


def test_hardware_observation_matches_the_38_input_student_contract():
    controller = StudentControllerState(frequency_hz=1.5)
    controller.reset(yaw_rad=0.4)
    sensors = StudentSensors(
        angular_velocity_b=[0.1, 0.2, 0.3],
        projected_gravity_b=[0, 0, -1],
        yaw_rad=0.6,
        linear_acceleration_b=[0, 0, 9.81],
        joint_position=DEFAULT_JOINT_POS + 0.1,
        joint_velocity=[1, 2, 3, 4, 5, 6],
        motor_torque_nm=[10, 20, 30, 40, 50, 60],
    )
    observation = controller.observation(sensors)

    assert observation.shape == (38,)
    np.testing.assert_allclose(observation[:6], [0.1, 0.2, 0.3, 0, 0, -1])
    np.testing.assert_allclose(observation[6], -0.2, atol=1.0e-6)
    np.testing.assert_allclose(observation[7:10], [0, 0, 0.981], atol=1.0e-6)
    np.testing.assert_allclose(observation[10:16], 0.1, atol=1.0e-6)
    np.testing.assert_allclose(observation[16:22], [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(observation[22:28], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(observation[28:34], 0)
    np.testing.assert_allclose(observation[34:38], [1.5, 1, 0, 1])


def test_hardware_action_integration_matches_training_and_keeps_raw_action_history():
    controller = StudentControllerState(frequency_hz=2.0)
    action = np.array([0.5, -0.5, 1.0, -1.0, 0.0, 2.0], dtype=np.float32)
    expected_offset = integrate_velocity_to_position_offset(
        torch.zeros((1, 6)), torch.from_numpy(action).unsqueeze(0), torch.tensor([[2.0 * np.pi]]), 0.02, 0.5 * np.pi
    )[0].numpy()

    target = controller.advance(action)
    np.testing.assert_allclose(target, DEFAULT_JOINT_POS + expected_offset, atol=1.0e-6)
    np.testing.assert_array_equal(controller.previous_raw_action, action)
    assert controller.phase == 0.04

    controller.reset()
    assert controller.phase == 0.0
    np.testing.assert_array_equal(controller.previous_raw_action, np.zeros(6))


def test_relative_imu_yaw_wraps_across_pi():
    controller = StudentControllerState()
    controller.reset(yaw_rad=np.pi - 0.1)
    sensors = StudentSensors(
        angular_velocity_b=[0, 0, 0],
        projected_gravity_b=[0, 0, -1],
        yaw_rad=-np.pi + 0.1,
        linear_acceleration_b=[0, 0, 9.81],
        joint_position=DEFAULT_JOINT_POS,
        joint_velocity=[0] * 6,
        motor_torque_nm=[0] * 6,
    )

    np.testing.assert_allclose(controller.observation(sensors)[6], -0.2, atol=1.0e-6)
