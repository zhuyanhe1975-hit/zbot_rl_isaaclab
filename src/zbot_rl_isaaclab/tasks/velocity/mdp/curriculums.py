# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .rewards import AlternatingFeetTouchdownReward


def compute_stepping_promotion_metrics(
    episode_length_steps: torch.Tensor,
    maximum_episode_steps: int,
    alternating_steps: torch.Tensor,
    step_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean survival ratio and alternating-step rate for completed episodes."""
    valid = episode_length_steps > 0
    if not torch.any(valid):
        zero = torch.tensor(0.0, device=episode_length_steps.device)
        return zero, zero
    episode_length = episode_length_steps[valid].float()
    survival_ratio = torch.mean(episode_length / maximum_episode_steps)
    episode_duration = episode_length * step_dt
    alternation_rate = torch.mean(alternating_steps[valid] / episode_duration.clamp_min(step_dt))
    return survival_ratio, alternation_rate


def stage_two_blend(common_step_counter: int, promotion_step: int, transition_steps: int) -> float:
    """Return the zero-to-one blend for stage-two reward weights."""
    if promotion_step < 0:
        return 0.0
    return min(1.0, max(0.0, (common_step_counter - promotion_step) / max(1, transition_steps)))


def meets_stepping_promotion_gate(
    common_step_counter: int,
    minimum_training_steps: int,
    survival_ratio: float,
    survival_ratio_threshold: float,
    alternation_frequency: float,
    minimum_alternation_frequency: float,
    maximum_alternation_frequency: float,
) -> bool:
    """Return whether stable stepping metrics satisfy every stage-two promotion gate."""
    return (
        common_step_counter >= minimum_training_steps
        and survival_ratio >= survival_ratio_threshold
        and minimum_alternation_frequency <= alternation_frequency <= maximum_alternation_frequency
    )


class TwoStageWalkingCurriculum(ManagerTermBase):
    """Promote stable in-place stepping into forward walking with a smooth reward transition."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        alternating_cfg = env.reward_manager.get_term_cfg("alternating_touchdown")
        self._alternating_reward = cast("AlternatingFeetTouchdownReward", alternating_cfg.func)
        self._stage = 1
        self._promotion_step = -1
        self._survival_ratio_ema = 0.0
        self._alternation_rate_ema = 0.0
        self._has_metrics = False
        self._set_reward_weight("body_forward_speed", 0.0)
        self._set_reward_weight("step_length", 0.0)
        self._set_reward_weight("step_length_asymmetry", 0.0)
        stage_one_velocity_weight = cast(float, cfg.params["stage_one_velocity_weight"])
        self._set_reward_weight("stage_one_horizontal_velocity_l2", stage_one_velocity_weight)
        stage_one_frequency_weight = cast(float, cfg.params["stage_one_frequency_weight"])
        self._set_reward_weight("stage_one_step_frequency", stage_one_frequency_weight)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | slice,
        minimum_training_steps: int,
        survival_ratio_threshold: float,
        minimum_alternation_frequency: float,
        maximum_alternation_frequency: float,
        ema_alpha: float,
        transition_steps: int,
        stage_one_velocity_weight: float,
        stage_one_frequency_weight: float,
        forward_speed_weight: float,
        step_length_weight: float,
        step_length_asymmetry_weight: float,
    ) -> dict[str, float]:
        """Update promotion metrics and apply stage-dependent reward weights."""
        episode_length = env.episode_length_buf[env_ids]
        alternating_steps = self._alternating_reward.episode_alternations[env_ids]
        valid_episodes = episode_length > 0
        if torch.any(valid_episodes):
            survival_ratio, alternation_rate = compute_stepping_promotion_metrics(
                episode_length,
                env.max_episode_length,
                alternating_steps,
                env.step_dt,
            )
            survival_value = float(survival_ratio.item())
            alternation_value = float(alternation_rate.item())
            if not self._has_metrics:
                self._survival_ratio_ema = survival_value
                self._alternation_rate_ema = alternation_value
                self._has_metrics = True
            else:
                self._survival_ratio_ema += ema_alpha * (survival_value - self._survival_ratio_ema)
                self._alternation_rate_ema += ema_alpha * (alternation_value - self._alternation_rate_ema)

        if self._stage == 1 and meets_stepping_promotion_gate(
            env.common_step_counter,
            minimum_training_steps,
            self._survival_ratio_ema,
            survival_ratio_threshold,
            self._alternation_rate_ema,
            minimum_alternation_frequency,
            maximum_alternation_frequency,
        ):
            self._stage = 2
            self._promotion_step = env.common_step_counter

        blend = stage_two_blend(env.common_step_counter, self._promotion_step, transition_steps)
        self._set_reward_weight("body_forward_speed", forward_speed_weight * blend)
        self._set_reward_weight("step_length", step_length_weight * blend)
        self._set_reward_weight("step_length_asymmetry", step_length_asymmetry_weight * blend)
        self._set_reward_weight("stage_one_horizontal_velocity_l2", stage_one_velocity_weight * (1.0 - blend))
        self._set_reward_weight("stage_one_step_frequency", stage_one_frequency_weight * (1.0 - blend))

        return {
            "stage": float(self._stage),
            "stage_two_blend": blend,
            "survival_ratio_ema": self._survival_ratio_ema,
            "alternation_rate_ema": self._alternation_rate_ema,
        }

    def _set_reward_weight(self, term_name: str, weight: float) -> None:
        env = cast("ManagerBasedRLEnv", self._env)
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        if term_cfg.weight != weight:
            term_cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, term_cfg)
