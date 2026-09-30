# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from typing import cast

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from ...zbot_6dof.agents.rsl_rl_ppo_cfg import PPORunnerCfg as WalkingPPORunnerCfg


def _base_actor_cfg():
    actor = copy.deepcopy(WalkingPPORunnerCfg().actor)
    distribution = cast(RslRlMLPModelCfg.GaussianDistributionCfg, actor.distribution_cfg)
    distribution.init_std = 0.7
    return actor


def _base_algorithm_cfg():
    algorithm = copy.deepcopy(WalkingPPORunnerCfg().algorithm)
    algorithm.learning_rate = 5.0e-4
    algorithm.entropy_coef = 0.01
    return algorithm


@configclass
class PPORunnerCfg(WalkingPPORunnerCfg):
    """Base-specific PPO settings that preserve broad weight-transfer exploration."""

    experiment_name = "zbot_6dof_base"
    max_iterations = 1500
    actor = _base_actor_cfg()
    algorithm = _base_algorithm_cfg()
