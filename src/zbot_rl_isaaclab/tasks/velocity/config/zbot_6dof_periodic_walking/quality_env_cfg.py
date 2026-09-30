# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged teacher environment for the shared quality walking objective."""

from isaaclab.utils import configclass

from .env_cfg import PeriodicWalkingEnvCfg
from .quality_rewards_cfg import QualityRewardsCfg


@configclass
class QualityTeacherEnvCfg(PeriodicWalkingEnvCfg):
    rewards: QualityRewardsCfg = QualityRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.weight_shift.curriculum_initial_maximum_frequency = None
        self.commands.weight_shift.endpoint_probability = 0.25
        self.commands.weight_shift.frequency_reward_term_name = "alternating_touchdown"
