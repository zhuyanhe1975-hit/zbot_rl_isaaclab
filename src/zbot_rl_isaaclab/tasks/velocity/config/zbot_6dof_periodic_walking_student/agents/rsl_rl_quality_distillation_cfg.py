# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Distill the approved quality teacher into the 38-input recurrent student."""

import copy
from typing import cast

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .rsl_rl_distillation_cfg import DistillationRunnerCfg, GuidedDistillationAlgorithmCfg


@configclass
class QualityDistillationRunnerCfg(DistillationRunnerCfg):
    experiment_name = "zbot_6dof_periodic_walking_student_quality"
    save_interval = 25
    student = copy.deepcopy(DistillationRunnerCfg().student)
    student.distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.03)
    algorithm = copy.deepcopy(cast(GuidedDistillationAlgorithmCfg, DistillationRunnerCfg().algorithm))
    algorithm.learning_rate = 1.0e-4
    algorithm.teacher_fraction_end = 0.7
    algorithm.teacher_fraction_decay_updates = 600
