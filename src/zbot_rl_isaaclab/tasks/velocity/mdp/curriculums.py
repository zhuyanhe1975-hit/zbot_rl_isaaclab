# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
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


def meets_stepping_performance_gate(
    survival_ratio: float,
    survival_ratio_threshold: float,
    alternation_frequency: float,
    minimum_alternation_frequency: float,
    maximum_alternation_frequency: float,
) -> bool:
    """Return whether stepping is stable and inside the target frequency band."""
    return (
        survival_ratio >= survival_ratio_threshold
        and minimum_alternation_frequency <= alternation_frequency <= maximum_alternation_frequency
    )


def stability_progress(survival_ratio: float, survival_ratio_threshold: float) -> float:
    """Return normalized stability progress toward the survival gate."""
    if survival_ratio_threshold <= 0.0:
        raise ValueError("Survival threshold must be positive.")
    return min(1.0, max(0.0, survival_ratio / survival_ratio_threshold))


def frequency_control_progress(
    survival_ratio: float,
    control_start: float,
    survival_ratio_threshold: float,
) -> float:
    """Ramp cadence control from zero to full strength over a stability interval."""
    if not 0.0 <= control_start < survival_ratio_threshold:
        raise ValueError("Frequency-control start must be below the positive survival threshold.")
    return min(1.0, max(0.0, (survival_ratio - control_start) / (survival_ratio_threshold - control_start)))


def stepping_performance_blend(
    survival_ratio: float,
    survival_ratio_threshold: float,
    alternation_frequency: float,
    minimum_alternation_frequency: float,
    maximum_alternation_frequency: float,
    frequency_tolerance: float,
) -> float:
    """Return a performance-driven stage-two blend without elapsed-step gates."""
    if frequency_tolerance <= 0.0:
        raise ValueError("Frequency tolerance must be positive.")
    progress = stability_progress(survival_ratio, survival_ratio_threshold)
    if progress < 1.0:
        return 0.0
    distance_below = max(0.0, minimum_alternation_frequency - alternation_frequency)
    distance_above = max(0.0, alternation_frequency - maximum_alternation_frequency)
    frequency_score = math.exp(-(((distance_below + distance_above) / frequency_tolerance) ** 2))
    return frequency_score


class TwoStageWalkingCurriculum(ManagerTermBase):
    """Promote stable in-place stepping into forward walking with a smooth reward transition."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        alternating_cfg = env.reward_manager.get_term_cfg("alternating_touchdown")
        self._alternating_reward = cast("AlternatingFeetTouchdownReward", alternating_cfg.func)
        self._stage = 1
        self._survival_ratio_ema = 0.0
        self._alternation_rate_ema = 0.0
        self._set_reward_weight("body_forward_speed", 0.0)
        self._set_reward_weight("step_length", 0.0)
        self._set_reward_weight("step_length_asymmetry", 0.0)
        stage_one_velocity_weight = cast(float, cfg.params["stage_one_velocity_weight"])
        self._set_reward_weight("stage_one_horizontal_velocity_l2", stage_one_velocity_weight)
        touchdown_state_weight = cast(float, cfg.params["touchdown_state_weight"])
        self._set_reward_weight("alternating_touchdown", touchdown_state_weight)
        initial_frequency_weight = cast(float, cfg.params["stage_one_frequency_initial_weight"])
        self._set_reward_weight("stage_one_step_frequency", initial_frequency_weight)
        stage_one_frequency_error_weight = cast(float, cfg.params["stage_one_frequency_error_weight"])
        self._set_reward_weight("step_frequency_error_l2", stage_one_frequency_error_weight)
        self._set_reward_weight("step_frequency_excess_l2", 0.0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | slice,
        survival_ratio_threshold: float,
        minimum_alternation_frequency: float,
        maximum_alternation_frequency: float,
        frequency_tolerance: float,
        frequency_control_survival_start: float,
        ema_alpha: float,
        stage_one_velocity_weight: float,
        touchdown_state_weight: float,
        stage_one_frequency_initial_weight: float,
        stage_one_frequency_weight: float,
        stage_two_frequency_weight: float,
        stage_one_frequency_error_weight: float,
        stage_two_frequency_error_weight: float,
        stage_one_frequency_excess_weight: float,
        stage_two_frequency_excess_weight: float,
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
            self._survival_ratio_ema += ema_alpha * (survival_value - self._survival_ratio_ema)
            self._alternation_rate_ema += ema_alpha * (alternation_value - self._alternation_rate_ema)

        promotion_ready = meets_stepping_performance_gate(
            self._survival_ratio_ema,
            survival_ratio_threshold,
            self._alternation_rate_ema,
            minimum_alternation_frequency,
            maximum_alternation_frequency,
        )

        if self._stage == 1 and promotion_ready:
            self._stage = 2

        blend = stepping_performance_blend(
            self._survival_ratio_ema,
            survival_ratio_threshold,
            self._alternation_rate_ema,
            minimum_alternation_frequency,
            maximum_alternation_frequency,
            frequency_tolerance,
        )
        progress = stability_progress(self._survival_ratio_ema, survival_ratio_threshold)
        self._set_reward_weight("body_forward_speed", forward_speed_weight * blend)
        self._set_reward_weight("step_length", step_length_weight * blend)
        self._set_reward_weight("step_length_asymmetry", step_length_asymmetry_weight * blend)
        self._set_reward_weight("alternating_touchdown", touchdown_state_weight)
        self._set_reward_weight("stage_one_horizontal_velocity_l2", stage_one_velocity_weight * (1.0 - blend))
        stage_one_frequency = stage_one_frequency_initial_weight + progress * (
            stage_one_frequency_weight - stage_one_frequency_initial_weight
        )
        frequency_weight = stage_one_frequency * (1.0 - blend) + stage_two_frequency_weight * blend
        self._set_reward_weight("stage_one_step_frequency", frequency_weight)
        frequency_error_weight = (
            stage_one_frequency_error_weight * (1.0 - blend) + stage_two_frequency_error_weight * blend
        )
        self._set_reward_weight("step_frequency_error_l2", frequency_error_weight)
        cadence_control = frequency_control_progress(
            self._survival_ratio_ema,
            frequency_control_survival_start,
            survival_ratio_threshold,
        )
        frequency_excess_weight = cadence_control * (
            stage_one_frequency_excess_weight * (1.0 - blend) + stage_two_frequency_excess_weight * blend
        )
        self._set_reward_weight("step_frequency_excess_l2", frequency_excess_weight)

        return {
            "stage": float(self._stage),
            "stage_two_blend": blend,
            "promotion_ready": float(promotion_ready),
            "survival_ratio_ema": self._survival_ratio_ema,
            "alternation_rate_ema": self._alternation_rate_ema,
        }

    def _set_reward_weight(self, term_name: str, weight: float) -> None:
        env = cast("ManagerBasedRLEnv", self._env)
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        if term_cfg.weight != weight:
            term_cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, term_cfg)
