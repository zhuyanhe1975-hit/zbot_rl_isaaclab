# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import copy

from isaaclab.utils import configclass

from ...zbot_6dof_base.agents.rsl_rl_ppo_cfg import PPORunnerCfg as BasePPORunnerCfg


def _periodic_algorithm_cfg():
    algorithm = copy.deepcopy(BasePPORunnerCfg().algorithm)
    # Keep exploration at initialization, but reduce the pressure to grow action noise.
    algorithm.entropy_coef = 0.002
    return algorithm


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    """PPO settings for periodic walking trained with its own observation contract."""

    experiment_name = "zbot_6dof_periodic_walking"
    algorithm = _periodic_algorithm_cfg()
