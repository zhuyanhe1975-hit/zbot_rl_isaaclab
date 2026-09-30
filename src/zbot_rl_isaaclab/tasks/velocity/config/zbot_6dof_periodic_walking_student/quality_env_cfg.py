# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Deployable student distillation on the quality teacher's objective."""

from isaaclab.utils import configclass

from ..zbot_6dof_periodic_walking.quality_rewards_cfg import QualityRewardsCfg
from .env_cfg import StudentEnvCfg


@configclass
class QualityStudentEnvCfg(StudentEnvCfg):
    rewards: QualityRewardsCfg = QualityRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.weight_shift.frequency_reward_term_name = "alternating_touchdown"
