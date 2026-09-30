# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""PPO fine-tuning with the distilled recurrent student as actor prior."""

import copy

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from ...zbot_6dof_periodic_walking.agents.rsl_rl_ppo_cfg import PPORunnerCfg as TeacherPPORunnerCfg
from ...zbot_6dof_periodic_walking_student.agents.rsl_rl_distillation_cfg import DistillationRunnerCfg

_teacher_ppo = TeacherPPORunnerCfg()


@configclass
class AnchoredPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "zbot_rl_isaaclab.student_ppo:AnchoredPPO"
    max_relative_drift: float = 0.02


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 100
    save_interval = 25
    experiment_name = "zbot_6dof_periodic_walking_student_finetune"
    obs_groups = {"actor": ["policy"], "critic": ["teacher"]}
    actor = copy.deepcopy(DistillationRunnerCfg().student)
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128], activation="elu", obs_normalization=False)
    algorithm = AnchoredPPOAlgorithmCfg(
        **{
            name: value
            for name, value in _teacher_ppo.algorithm.to_dict().items()  # pyright: ignore[reportAttributeAccessIssue]
            if name != "class_name"
        }
    )
    algorithm.learning_rate = 1.0e-5
    algorithm.num_learning_epochs = 2
    algorithm.clip_param = 0.1
    algorithm.entropy_coef = 0.0


@configclass
class RobustRunnerCfg(PPORunnerCfg):
    max_iterations = 100
    experiment_name = "zbot_6dof_periodic_walking_student_robust"
