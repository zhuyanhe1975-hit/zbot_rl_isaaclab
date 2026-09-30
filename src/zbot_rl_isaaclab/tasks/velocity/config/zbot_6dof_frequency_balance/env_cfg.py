# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ...mdp.commands_cfg import UniformStepFrequencyCommandCfg
from ...mdp.observations import ideal_support_foot_normalized_distance_error
from ...mdp.rewards import commanded_support_joint_force_contrast, ideal_support_foot_distance_reward
from ..zbot_6dof_base.env_cfg import BaseEnvCfg
from ..zbot_6dof_base.env_cfg import ObservationsCfg as BaseObservationsCfg
from ..zbot_6dof_base.env_cfg import RewardsCfg as BaseRewardsCfg

FREQUENCY_BALANCE_OBSERVATION_DIM = 42


@configclass
class CommandsCfg:
    """Frequency-conditioned support-foot schedule without stepping."""

    weight_shift = UniformStepFrequencyCommandCfg(
        minimum_frequency=0.5,
        maximum_frequency=2.0,
        resampling_time_range=(8.0, 12.0),
        frequency_reward_term_name=None,
    )


@configclass
class ObservationsCfg(BaseObservationsCfg):
    """Balance observations with frequency and scheduled support side."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        ideal_support_foot = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "weight_shift"})
        support_distance_error = ObsTerm(
            func=ideal_support_foot_normalized_distance_error,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
                "command_name": "weight_shift",
                "command_index": 1,
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg(BaseRewardsCfg):
    """Dense COM and ankle-force tracking of the frequency-driven support schedule."""

    support_distance = RewTerm(
        func=ideal_support_foot_distance_reward,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
            "command_name": "weight_shift",
            "command_index": 1,
        },
    )
    support_force = RewTerm(
        func=commanded_support_joint_force_contrast,
        weight=4.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "ankle_wrenches",
                body_names=["b1", "foot_1"],
                preserve_order=True,
            ),
            "command_name": "weight_shift",
            "minimum_total_force": 1.0,
            "command_index": 1,
        },
    )


@configclass
class FrequencyBalanceEnvCfg(BaseEnvCfg):
    """Stationary weight transfer conditioned on a 0.5–2.0 Hz command."""

    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def play_mode(self) -> None:
        self.commands.weight_shift.ordered = True
        self.commands.weight_shift.resampling_time_range = (1.0e9, 1.0e9)
