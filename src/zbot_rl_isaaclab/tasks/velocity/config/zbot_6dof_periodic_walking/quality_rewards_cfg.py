# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared walking objective for the quality teacher and deployable student."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ...mdp.rewards import (
    commanded_step_frequency_score,
    touchdown_impact_force_l2,
)
from .env_cfg import RewardsCfg as PeriodicRewardsCfg


@configclass
class QualityRewardsCfg(PeriodicRewardsCfg):
    """Keep cadence and support while discouraging oversized, hard landings."""

    commanded_step_frequency = RewTerm(
        func=commanded_step_frequency_score,
        weight=50.0,
        params={
            "reward_term_name": "alternating_touchdown",
            "command_name": "weight_shift",
            # Alternating touchdowns occur twice per commanded full gait cycle.
            "target_multiplier": 2.0,
            "tolerance": 0.35,
        },
    )
    touchdown_impact = RewTerm(
        func=touchdown_impact_force_l2,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot_0", "foot_1"], preserve_order=True),
            "minimum_air_time": 0.05,
            "force_threshold": 10.0,
            "maximum_unpenalized_force": 130.0,
        },
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.body_forward_speed.weight = 0.0
        self.single_support_foot_height_l2.params["maximum_height_difference"] = 0.04
        self.single_support_foot_height_l2.weight = -40.0
        self.step_length.params["maximum_rewarded_step_length"] = 0.35
        self.joint_torques_l2.weight = -1.0e-3
        self.heading_error_l2.weight = -2.5
        self.yaw_rate_l2.weight = -0.4
        self.alternating_touchdown.params["frequency_command_name"] = "weight_shift"
        self.alternating_touchdown.params["minimum_alternation_interval"] = 0.15
