# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL student/teacher distillation with the frozen 45-input teacher."""

import copy

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
)

from ...zbot_6dof_periodic_walking.agents.rsl_rl_ppo_cfg import PPORunnerCfg as TeacherPPORunnerCfg


@configclass
class GuidedDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    class_name = "zbot_rl_isaaclab.student_distillation:GuidedDistillation"
    teacher_fraction_start: float = 1.0
    teacher_fraction_end: float = 0.25
    teacher_fraction_decay_updates: int = 500


@configclass
class DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Train a recurrent 38-input student from the saved 45-input teacher."""

    num_steps_per_env = 48
    max_iterations = 600
    save_interval = 50
    experiment_name = "zbot_6dof_periodic_walking_student"
    obs_groups = {"student": ["policy"], "teacher": ["teacher"]}
    student = RslRlRNNModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
    teacher = copy.deepcopy(TeacherPPORunnerCfg().actor)
    algorithm = GuidedDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=5.0e-4,
        gradient_length=16,
        max_grad_norm=1.0,
        loss_type="huber",
    )
