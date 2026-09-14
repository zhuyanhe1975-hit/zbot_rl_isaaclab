# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from pathlib import Path

from zbot_rl_isaaclab.assets import ZBOT_6DOF_USD_PATH
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof.env_cfg import VelocityEnvCfg
from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import body_forward_velocity


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
    assert cfg.rewards.body_forward_speed.func is body_forward_velocity
    assert cfg.rewards.body_forward_speed.weight == 0.0
    assert cfg.rewards.stage_one_horizontal_velocity_l2.weight == -2.0
    assert cfg.rewards.single_support_foot_height_l2.weight == -20.0
    assert cfg.rewards.single_support_foot_height_l2.params["force_threshold"] == 10.0
    assert cfg.rewards.alternating_touchdown.weight == 10.0
    assert cfg.rewards.alternating_touchdown.params["minimum_air_time"] == 0.05
    assert cfg.rewards.alternating_touchdown.params["force_threshold"] == 10.0
    assert cfg.rewards.alternating_touchdown.params["crossing_margin"] == 0.0
    assert cfg.rewards.alternating_touchdown.params["asset_cfg"].body_names == "foot_.*"
    assert cfg.rewards.alternating_touchdown.params["minimum_frequency"] == 1.0
    assert cfg.rewards.alternating_touchdown.params["maximum_frequency"] == 2.0
    assert cfg.rewards.stage_one_step_frequency.weight == 5.0
    assert not hasattr(cfg.rewards, "feet_air_time")
    assert cfg.rewards.step_length.weight == 0.0
    assert cfg.rewards.step_length.params["return_symmetry_error"] is False
    assert cfg.rewards.step_length.params["crossing_margin"] == 0.0
    assert cfg.rewards.step_length_asymmetry.weight == 0.0
    assert cfg.rewards.step_length_asymmetry.params["return_symmetry_error"] is True
    assert not hasattr(cfg.rewards, "lin_vel_z_l2")
    assert not hasattr(cfg.rewards, "ang_vel_xy_l2")
    assert cfg.curriculum.walking_stages.params["survival_ratio_threshold"] == 0.60
    assert cfg.curriculum.walking_stages.params["minimum_alternation_frequency"] == 1.0
    assert cfg.curriculum.walking_stages.params["maximum_alternation_frequency"] == 2.0
    assert cfg.decimation == 4
    assert cfg.sim.dt == 0.005
