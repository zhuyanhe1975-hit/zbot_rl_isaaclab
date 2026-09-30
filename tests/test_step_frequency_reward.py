# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from typing import Any, cast

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import (
    commanded_step_frequency_score,
    commanded_swing_foot_score,
    frequency_above_band_error_l2,
    frequency_band_error_l2,
    frequency_band_score,
    frequency_command_error_l2,
    frequency_command_relative_error_l2,
    frequency_command_score,
    time_normalize_frequency_event_metric,
    world_forward_velocity_score,
)


def test_step_frequency_scores_one_to_two_hertz_band():
    """The requested in-place stepping band must receive the maximum score."""
    frequency = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    score = frequency_band_score(
        frequency,
        minimum_frequency=1.0,
        maximum_frequency=2.0,
        tolerance=0.5,
    )

    expected = torch.tensor(
        [torch.exp(torch.tensor(-1.0)), 1.0, 1.0, 1.0, torch.exp(torch.tensor(-1.0)), torch.exp(torch.tensor(-4.0))]
    )
    torch.testing.assert_close(score, expected)


def test_step_frequency_penalizes_distance_outside_target_band():
    frequency = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 5.0])

    error = frequency_band_error_l2(frequency, minimum_frequency=1.0, maximum_frequency=2.0)

    torch.testing.assert_close(error, torch.tensor([0.25, 0.0, 0.0, 0.0, 0.25, 9.0]))


def test_stage_one_frequency_penalty_only_rejects_excessive_cadence():
    frequency = torch.tensor([0.5, 1.0, 2.0, 2.5, 4.0])

    error = frequency_above_band_error_l2(frequency, maximum_frequency=2.0)

    torch.testing.assert_close(error, torch.tensor([0.0, 0.0, 0.0, 0.25, 4.0]))


def test_frequency_reward_is_independent_of_event_count_inside_band():
    one_hz_reward_per_second = time_normalize_frequency_event_metric(torch.tensor(1.0), torch.tensor(1.0))
    two_hz_reward_per_second = 2.0 * time_normalize_frequency_event_metric(torch.tensor(1.0), torch.tensor(0.5))

    torch.testing.assert_close(one_hz_reward_per_second, two_hz_reward_per_second)


def test_frequency_command_reward_distinguishes_targets_inside_the_old_band():
    measured = torch.tensor([0.5, 1.0, 1.5, 2.0])
    commanded = torch.full_like(measured, 1.5)

    score = frequency_command_score(measured, commanded, tolerance=0.25)
    error = frequency_command_error_l2(measured, commanded)

    assert score.argmax().item() == 2
    assert torch.all(score > 0.0)
    torch.testing.assert_close(error, torch.tensor([1.0, 0.25, 0.0, 0.25]))


def test_frequency_command_reward_retains_signal_across_full_command_range():
    measured = torch.tensor([0.5, 2.0])
    commanded = torch.tensor([2.0, 0.5])

    score = frequency_command_score(measured, commanded, tolerance=0.4)

    torch.testing.assert_close(score, torch.full((2,), 1.0 / (1.0 + (1.5 / 0.4) ** 2)))


def test_narrow_command_tolerance_rejects_a_fixed_midpoint_frequency_policy():
    commanded = torch.linspace(0.5, 2.0, 1001)
    fixed_midpoint = torch.full_like(commanded, 1.25)

    score = frequency_command_score(fixed_midpoint, commanded, tolerance=0.15)

    assert score.mean() < 0.3
    assert score[500] == 1.0


def test_relative_frequency_error_uses_equal_scale_across_commands():
    measured = torch.tensor([0.75, 1.5, 3.0])
    commanded = torch.tensor([0.5, 1.0, 2.0])

    error = frequency_command_relative_error_l2(measured, commanded)

    torch.testing.assert_close(error, torch.full((3,), 0.25))


def test_full_cycle_commands_match_twice_as_many_alternating_touchdowns():
    command = torch.tensor([[0.5], [1.5], [2.0]])
    event = SimpleNamespace(
        measured_frequency=torch.tensor([1.0, 3.0, 4.0]),
        frequency_event_interval=torch.ones(3),
        has_frequency_event=torch.ones(3, dtype=torch.bool),
    )
    env = SimpleNamespace(
        reward_manager=SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(func=event)),
        command_manager=SimpleNamespace(get_command=lambda _: command),
    )

    score = commanded_step_frequency_score(
        cast(Any, env), "alternating_touchdown", "weight_shift", tolerance=0.35, target_multiplier=2.0
    )

    torch.testing.assert_close(score, torch.ones(3))


def test_commanded_swing_score_selects_the_non_support_foot():
    per_foot_progress = torch.tensor([[0.00, 0.03], [0.06, 0.00], [0.01, 0.00]])
    target_support_side = torch.tensor([1.0, -1.0, -1.0])

    score = commanded_swing_foot_score(per_foot_progress, target_support_side, target_value=0.03)

    torch.testing.assert_close(score, torch.tensor([1.0, 1.0, 1.0 / 3.0]))


def test_bounded_forward_score_has_strong_low_speed_signal_and_saturates(monkeypatch):
    monkeypatch.setattr(
        "zbot_rl_isaaclab.tasks.velocity.mdp.rewards.world_forward_velocity",
        lambda env, asset_cfg: torch.tensor([-0.15, 0.0, 0.15, 0.45]),
    )

    score = world_forward_velocity_score(cast(Any, None), target_speed=0.15)

    torch.testing.assert_close(score, torch.tanh(torch.tensor([-1.0, 0.0, 1.0, 3.0])))
