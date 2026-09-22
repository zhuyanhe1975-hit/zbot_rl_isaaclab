# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from pathlib import Path

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from zbot_rl_isaaclab.assets import ZBOT_6DOF_USD_PATH
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof.agents.rsl_rl_ppo_cfg import PPORunnerCfg
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof.env_cfg import VelocityEnvCfg
from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import world_forward_velocity
from zbot_rl_isaaclab.tasks.velocity.mdp.terminations import heading_deviation_above_limit


def test_zbot_task_contract():
    """The baseline task must retain its six-action walking contract."""
    cfg = VelocityEnvCfg()

    assert Path(ZBOT_6DOF_USD_PATH).is_file()
    assert cfg.scene.robot.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert cfg.actions.joint_pos.joint_names == [f"joint{index}" for index in range(1, 7)]
    assert cfg.actions.joint_pos.preserve_order
    assert cfg.actions.joint_pos.velocity_limit_range == (0.2 * math.pi, 2.0 * math.pi)
    assert cfg.actions.joint_pos.position_offset_limit == math.pi
    assert cfg.scene.contact_forces.track_air_time
    assert cfg.commands is None
    assert cfg.observations.policy.base_height.params["asset_cfg"].body_names == "base"
    assert cfg.observations.policy.foot_contacts.params["sensor_cfg"].body_names == "foot_.*"
    assert cfg.observations.policy.foot_contacts.params["force_threshold"] == 10.0
    assert cfg.rewards.body_forward_speed.func is world_forward_velocity
    assert cfg.rewards.body_forward_speed.weight == 0.0
    assert cfg.rewards.heading_error_l2.weight == -5.0
    assert cfg.rewards.yaw_rate_l2.weight == -0.2
    assert cfg.terminations.heading_deviation.func is heading_deviation_above_limit
    assert cfg.terminations.heading_deviation.params["maximum_deviation"] == math.pi / 4.0
    assert cfg.rewards.stage_one_horizontal_velocity_l2.weight == -2.0
    assert cfg.rewards.single_support_foot_height_l2.weight == -20.0
    assert cfg.rewards.single_support_foot_height_l2.params["force_threshold"] == 10.0
    assert cfg.rewards.termination_penalty.weight == -50.0
    assert cfg.rewards.flat_orientation_l2.weight == -2.0
    assert cfg.rewards.ang_vel_xy_l2.weight == -0.05
    assert cfg.rewards.alternating_touchdown.weight == 1.0e-6
    assert cfg.rewards.alternating_touchdown.params["minimum_air_time"] == 0.05
    assert cfg.rewards.alternating_touchdown.params["force_threshold"] == 10.0
    assert "crossing_margin" not in cfg.rewards.alternating_touchdown.params
    assert "asset_cfg" not in cfg.rewards.alternating_touchdown.params
    assert cfg.rewards.alternating_touchdown.params["minimum_frequency"] == 1.0
    assert cfg.rewards.alternating_touchdown.params["maximum_frequency"] == 2.0
    assert cfg.rewards.stage_one_step_frequency.weight == 5.0
    assert cfg.rewards.step_frequency_error_l2.weight == 0.0
    assert cfg.rewards.step_frequency_excess_l2.weight == 0.0
    assert cfg.curriculum.walking_stages.params["touchdown_state_weight"] == 1.0e-6
    assert "stage_one_touchdown_weight" not in cfg.curriculum.walking_stages.params
    assert "stage_two_touchdown_weight" not in cfg.curriculum.walking_stages.params
    assert cfg.curriculum.walking_stages.params["stage_one_frequency_initial_weight"] == 5.0
    assert cfg.curriculum.walking_stages.params["stage_one_frequency_weight"] == 30.0
    assert cfg.curriculum.walking_stages.params["stage_two_frequency_weight"] == 30.0
    assert cfg.curriculum.walking_stages.params["stage_one_frequency_error_weight"] == 0.0
    assert cfg.curriculum.walking_stages.params["stage_two_frequency_error_weight"] == -1.0
    assert cfg.curriculum.walking_stages.params["frequency_control_survival_start"] == 0.30
    assert cfg.curriculum.walking_stages.params["stage_one_frequency_excess_weight"] == -0.25
    assert cfg.curriculum.walking_stages.params["stage_two_frequency_excess_weight"] == 0.0
    assert "minimum_training_steps" not in cfg.curriculum.walking_stages.params
    assert "transition_steps" not in cfg.curriculum.walking_stages.params
    assert not hasattr(cfg.rewards, "feet_air_time")
    assert cfg.rewards.step_length.weight == 0.0
    assert cfg.rewards.step_length.params["return_symmetry_error"] is False
    assert cfg.rewards.step_length.params["crossing_margin"] == 0.0
    assert cfg.rewards.step_length_asymmetry.weight == 0.0
    assert cfg.rewards.step_length_asymmetry.params["return_symmetry_error"] is True
    assert not hasattr(cfg.rewards, "lin_vel_z_l2")
    assert cfg.curriculum.walking_stages.params["survival_ratio_threshold"] == 0.60
    assert cfg.curriculum.walking_stages.params["minimum_alternation_frequency"] == 1.0
    assert cfg.curriculum.walking_stages.params["maximum_alternation_frequency"] == 2.0
    assert cfg.decimation == 4
    assert cfg.sim.dt == 0.005


def test_ppo_prioritizes_long_horizon_stability_with_bounded_initial_exploration():
    cfg = PPORunnerCfg()

    assert cfg.num_steps_per_env == 48
    assert isinstance(cfg.actor.distribution_cfg, RslRlMLPModelCfg.GaussianDistributionCfg)
    assert cfg.actor.distribution_cfg.init_std == 0.7
    assert cfg.algorithm.learning_rate == 5.0e-4
    assert cfg.algorithm.schedule == "fixed"
    assert cfg.algorithm.entropy_coef == 0.01
    assert cfg.algorithm.gamma == 0.995
    assert cfg.algorithm.lam == 0.97
