# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""PPO continuation of the 45-observation teacher on the quality objective."""

import copy

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import PPORunnerCfg as PeriodicPPORunnerCfg


@configclass
class QualityPPORunnerCfg(PeriodicPPORunnerCfg):
    experiment_name = "zbot_6dof_periodic_walking_teacher_quality"
    max_iterations = 300
    save_interval = 25
    algorithm = copy.deepcopy(PeriodicPPORunnerCfg().algorithm)
    algorithm.learning_rate = 3.0e-5
    algorithm.entropy_coef = 0.001
