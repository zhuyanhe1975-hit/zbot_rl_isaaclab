# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ...mdp.commands_cfg import UniformStepFrequencyCommandCfg
from ...mdp.rewards import (
    commanded_step_frequency_relative_error_l2,
    commanded_step_frequency_score,
    commanded_support_force_contrast,
    commanded_swing_foot_forward_progress,
    commanded_swing_foot_height,
    world_forward_velocity_score,
)
from ...mdp.terminations import heading_deviation_above_limit
from ..zbot_6dof.env_cfg import ObservationsCfg as WalkingObservationsCfg
from ..zbot_6dof.env_cfg import RewardsCfg as WalkingRewardsCfg
from ..zbot_6dof.env_cfg import TerminationsCfg as WalkingTerminationsCfg
from ..zbot_6dof.env_cfg import VelocityEnvCfg

FREQUENCY_COMMAND_OBSERVATION_DIM = 46


@configclass
class CommandsCfg:
    """Target cadence sampled independently for every environment."""

    step_frequency = UniformStepFrequencyCommandCfg(
        minimum_frequency=0.5,
        maximum_frequency=2.0,
        resampling_time_range=(8.0, 12.0),
    )


@configclass
class ObservationsCfg(WalkingObservationsCfg):
    """Walking observations plus frequency and scheduled support-foot commands."""

    @configclass
    class PolicyCfg(WalkingObservationsCfg.PolicyCfg):
        step_frequency_command = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "step_frequency"},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg(WalkingRewardsCfg):
    """Replace fixed-band cadence objectives with command-conditioned tracking."""

    body_forward_speed = RewTerm(
        func=world_forward_velocity_score,
        weight=5.0,
        params={
            "target_speed": 0.15,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )

    stage_one_step_frequency = RewTerm(
        func=commanded_step_frequency_score,
        weight=0.0,
        params={
            "reward_term_name": "alternating_touchdown",
            "command_name": "step_frequency",
            "tolerance": 0.15,
        },
    )
    step_frequency_error_l2 = RewTerm(
        func=commanded_step_frequency_relative_error_l2,
        weight=0.0,
        params={
            "reward_term_name": "alternating_touchdown",
            "command_name": "step_frequency",
        },
    )
    step_frequency_excess_l2 = RewTerm(
        func=commanded_step_frequency_relative_error_l2,
        weight=0.0,
        params={
            "reward_term_name": "alternating_touchdown",
            "command_name": "step_frequency",
        },
    )
    commanded_step_frequency = RewTerm(
        func=commanded_step_frequency_score,
        weight=200.0,
        params={
            "reward_term_name": "alternating_touchdown",
            "command_name": "step_frequency",
            "tolerance": 0.15,
        },
    )
    commanded_step_frequency_error_l2 = RewTerm(
        func=commanded_step_frequency_relative_error_l2,
        weight=-100.0,
        params={
            "reward_term_name": "alternating_touchdown",
            "command_name": "step_frequency",
        },
    )
    commanded_support_force = RewTerm(
        func=commanded_support_force_contrast,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "command_name": "step_frequency",
            "minimum_total_force": 10.0,
            "command_index": 1,
        },
    )
    commanded_swing_height = RewTerm(
        func=commanded_swing_foot_height,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
            "command_name": "step_frequency",
            "target_height": 0.03,
            "command_index": 1,
        },
    )
    commanded_swing_forward = RewTerm(
        func=commanded_swing_foot_forward_progress,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
            "body_cfg": SceneEntityCfg("robot", body_names="base"),
            "command_name": "step_frequency",
            "target_distance": 0.05,
            "command_index": 1,
        },
    )


@configclass
class TerminationsCfg(WalkingTerminationsCfg):
    """Walking terminations including a hard 45-degree heading boundary."""

    heading_deviation = DoneTerm(
        func=heading_deviation_above_limit,
        params={
            "maximum_deviation": math.pi / 4.0,
            "target_heading": 0.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )


@configclass
class FrequencyCommandEnvCfg(VelocityEnvCfg):
    """Flat-ground walking conditioned on a 0.5–2.0 Hz stepping-frequency command."""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    commands: CommandsCfg = CommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Prioritize command tracking while retaining stable, forward walking."""
        super().__post_init__()
        # The prior already walks quickly with long steps. Those objectives remain small
        # tie-breakers while cadence tracking and survival drive adaptation.
        self.rewards.step_length.weight = 100.0
        self.rewards.alive.weight = 1.0
        self.rewards.yaw_rate_l2.weight = -0.05
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.step_length_asymmetry.weight = -50.0
        self.rewards.single_support_foot_height_l2.weight = -1.0
        touchdown_params = self.rewards.alternating_touchdown.params
        touchdown_params["minimum_air_time"] = 0.15
        touchdown_params["minimum_frequency"] = 0.5
        touchdown_params["maximum_frequency"] = 2.0
        touchdown_params["frequency_command_name"] = "step_frequency"
        self.rewards.step_length.params["minimum_air_time"] = 0.15
        self.rewards.step_length.params["crossing_margin"] = 0.01
        self.rewards.step_length_asymmetry.params["minimum_air_time"] = 0.15
        self.rewards.step_length_asymmetry.params["crossing_margin"] = 0.01

    def play_mode(self) -> None:
        """Assign fixed ascending frequencies across the displayed environment array."""
        self.commands.step_frequency.ordered = True
        self.commands.step_frequency.resampling_time_range = (1.0e9, 1.0e9)
