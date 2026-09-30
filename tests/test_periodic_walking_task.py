from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof.env_cfg import VelocityEnvCfg
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_base.env_cfg import BaseEnvCfg
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking.agents.rsl_rl_ppo_cfg import (
    PPORunnerCfg as PeriodicPPORunnerCfg,
)
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking.env_cfg import PeriodicWalkingEnvCfg
from zbot_rl_isaaclab.tasks.velocity.mdp.observations import heading_error
from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import StepLengthReward, heading_error_l2, world_forward_velocity


def test_periodic_walking_preserves_base_phase_and_adds_walking_rewards():
    cfg = PeriodicWalkingEnvCfg()

    assert cfg.commands.weight_shift.include_phase_features
    assert cfg.commands.weight_shift.frequency_is_full_cycle
    assert cfg.observations.policy.ideal_support_foot.params["command_name"] == "weight_shift"
    assert cfg.rewards.support_distance.params["phase_sin_index"] == 2
    assert cfg.rewards.support_force.params["phase_sin_index"] == 2
    assert cfg.rewards.body_forward_speed.func is world_forward_velocity
    assert cfg.rewards.body_forward_speed.weight == 5.0
    assert cfg.rewards.step_length.func is StepLengthReward
    assert cfg.rewards.step_length.weight == 100.0
    assert cfg.rewards.step_length.params["minimum_air_time"] == 0.05
    assert cfg.rewards.step_length_asymmetry.weight == -100.0
    assert cfg.rewards.heading_error_l2.func is heading_error_l2
    assert cfg.rewards.heading_error_l2.weight == -1.0
    assert cfg.rewards.yaw_rate_l2.weight == -0.2
    assert cfg.rewards.body_lateral_vel_l2.weight == -1.0
    assert cfg.rewards.single_support_foot_height_l2.weight == -1.0
    assert cfg.rewards.single_support_foot_height_l2.params["maximum_height_difference"] == 0.05
    assert cfg.rewards.feet_slide.weight == -0.5
    assert cfg.rewards.joint_deviation.weight == -0.1
    assert cfg.rewards.alternating_touchdown.weight == 1.0e-6
    assert cfg.rewards.stage_one_step_frequency.weight == 0.0
    assert cfg.rewards.step_frequency_error_l2.weight == 0.0
    assert cfg.rewards.step_frequency_excess_l2.weight == 0.0


def test_periodic_walking_exposes_rewarded_heading_error_to_policy():
    cfg = PeriodicWalkingEnvCfg()

    assert not hasattr(BaseEnvCfg().observations.policy, "heading_error")
    assert cfg.observations.policy.heading_error.func is heading_error
    assert cfg.observations.policy.heading_error.params == cfg.rewards.heading_error_l2.params


def test_periodic_walking_early_fall_is_costly_after_step_scaling():
    cfg = PeriodicWalkingEnvCfg()

    assert cfg.rewards.alive.weight == 2.0
    assert cfg.rewards.termination_penalty.weight * cfg.sim.dt * cfg.decimation == -30.0


def test_periodic_walking_penalizes_physical_joint_acceleration_more_than_walking_base():
    assert PeriodicWalkingEnvCfg().rewards.joint_acc_l2.weight == -5.0e-7
    assert VelocityEnvCfg().rewards.joint_acc_l2.weight == -2.5e-7


def test_periodic_walking_reduces_entropy_pressure_without_removing_initial_exploration():
    runner = PeriodicPPORunnerCfg()

    assert runner.algorithm.entropy_coef == 0.002
    assert runner.actor.distribution_cfg.init_std == 0.7


def test_periodic_walking_play_mode_orders_the_full_frequency_range():
    cfg = PeriodicWalkingEnvCfg()

    cfg.play_mode()

    assert cfg.commands.weight_shift.minimum_frequency == 0.5
    assert cfg.commands.weight_shift.maximum_frequency == 2.0
    assert cfg.commands.weight_shift.curriculum_initial_maximum_frequency is None
    assert cfg.commands.weight_shift.ordered
    assert cfg.commands.weight_shift.resampling_time_range == (1.0e9, 1.0e9)
