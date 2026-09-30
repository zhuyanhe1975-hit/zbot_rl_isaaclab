# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import copy

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ...mdp.observations import heading_error
from ..zbot_6dof.env_cfg import RewardsCfg as WalkingRewardsCfg
from ..zbot_6dof_base.env_cfg import BaseEnvCfg
from ..zbot_6dof_base.env_cfg import ObservationsCfg as BaseObservationsCfg
from ..zbot_6dof_base.env_cfg import RewardsCfg as BaseRewardsCfg

_base_rewards = BaseRewardsCfg()
_walking_rewards = WalkingRewardsCfg()


@configclass
class ObservationsCfg(BaseObservationsCfg):
    """Base phase observations plus world-heading feedback for walking."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        heading_error = ObsTerm(
            func=heading_error,
            params={"target_heading": 0.0, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg(WalkingRewardsCfg):
    """Full walking objective augmented with Base periodic tracking."""

    alive = copy.deepcopy(_base_rewards.alive)
    # Isaac Lab multiplies every reward weight by the 0.02 s policy step. A
    # -200 termination weight costs only -4 per fall, allowing early resets to
    # avoid the accumulated heading and action penalties.
    termination_penalty = copy.deepcopy(_walking_rewards.termination_penalty)
    termination_penalty.weight = -1500.0
    joint_acc_l2 = copy.deepcopy(_walking_rewards.joint_acc_l2)
    joint_acc_l2.weight = -5.0e-7
    support_distance = copy.deepcopy(_base_rewards.support_distance)
    support_force = copy.deepcopy(_base_rewards.support_force)
    feet_collision_penalty = copy.deepcopy(_base_rewards.feet_collision_penalty)

    def __post_init__(self) -> None:
        # The Base phase terms define cadence. The old fixed touchdown-frequency band
        # uses a different event-rate convention and must not compete with that command.
        self.stage_one_step_frequency.weight = 0.0
        self.step_frequency_error_l2.weight = 0.0
        self.step_frequency_excess_l2.weight = 0.0


@configclass
class PeriodicWalkingEnvCfg(BaseEnvCfg):
    """Periodic walking with heading feedback and a fresh actor observation contract."""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def play_mode(self) -> None:
        """Assign ascending fixed frequencies to ascending environment indices."""
        self.commands.weight_shift.minimum_frequency = 0.5
        self.commands.weight_shift.maximum_frequency = 2.0
        self.commands.weight_shift.curriculum_initial_maximum_frequency = None
        self.commands.weight_shift.ordered = True
        self.commands.weight_shift.resampling_time_range = (1.0e9, 1.0e9)
