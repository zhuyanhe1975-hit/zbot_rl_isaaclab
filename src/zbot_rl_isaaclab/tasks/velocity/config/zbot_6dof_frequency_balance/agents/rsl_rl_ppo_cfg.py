# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...zbot_6dof_base.agents.rsl_rl_ppo_cfg import PPORunnerCfg as BasePPORunnerCfg


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "zbot_6dof_frequency_balance"
    max_iterations = 2000
