# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import (
    frequency_above_band_error_l2,
    frequency_band_error_l2,
    frequency_band_score,
    time_normalize_frequency_event_metric,
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
