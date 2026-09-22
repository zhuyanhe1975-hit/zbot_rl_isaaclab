# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...zbot_6dof.agents.rsl_rl_ppo_cfg import PPORunnerCfg as WalkingPPORunnerCfg


@configclass
class PPORunnerCfg(WalkingPPORunnerCfg):
    """PPO defaults for the compact double-support balance task."""

    experiment_name = "zbot_6dof_base"
    max_iterations = 1500
