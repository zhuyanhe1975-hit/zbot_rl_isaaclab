# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import (
    filter_short_touchdown_intervals,
    update_alternating_touchdown_state,
)


def test_rewards_only_exclusive_alternating_touchdowns():
    """First, repeated, and simultaneous landings must not receive alternation reward."""
    valid_touchdown = torch.tensor(
        [
            [True, False],
            [False, True],
            [False, True],
            [True, True],
        ]
    )
    last_landing_foot = torch.tensor([-1, 0, 1, 0])

    reward, updated_history = update_alternating_touchdown_state(valid_touchdown, last_landing_foot)

    torch.testing.assert_close(reward, torch.tensor([0.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(updated_history, torch.tensor([0, 1, 1, 0]))


def test_alternation_state_supports_batched_environments():
    """Left-to-right and right-to-left transitions must both be rewarded."""
    valid_touchdown = torch.tensor([[False, True], [True, False]])
    last_landing_foot = torch.tensor([0, 1])

    reward, updated_history = update_alternating_touchdown_state(valid_touchdown, last_landing_foot)

    torch.testing.assert_close(reward, torch.ones(2))
    torch.testing.assert_close(updated_history, torch.tensor([1, 0]))


def test_contact_bounce_does_not_replace_last_accepted_landing():
    landings = torch.tensor([[False, True], [False, True], [False, True]])
    last_foot = torch.tensor([0, 0, -1])
    elapsed = torch.tensor([0.08, 0.96, 0.02])

    accepted = filter_short_touchdown_intervals(landings, last_foot, elapsed, minimum_interval=0.15)
    reward, updated_history = update_alternating_touchdown_state(accepted, last_foot)

    torch.testing.assert_close(accepted, torch.tensor([[False, False], [False, True], [False, True]]))
    torch.testing.assert_close(reward, torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(updated_history, torch.tensor([0, 1, 1]))


def test_low_frequency_contact_bounce_preserves_one_second_alternating_intervals():
    landings = {48: 1, 98: 0, 102: 1, 146: 1, 196: 0}
    last_foot = torch.tensor([0])
    elapsed = torch.zeros(1)
    measured_intervals = []
    for step in range(197):
        elapsed += 0.02
        touchdown = torch.zeros((1, 2), dtype=torch.bool)
        if step in landings:
            touchdown[0, landings[step]] = True
        accepted = filter_short_touchdown_intervals(touchdown, last_foot, elapsed, minimum_interval=0.15)
        reward, last_foot = update_alternating_touchdown_state(accepted, last_foot)
        if reward.item():
            measured_intervals.append(round(elapsed.item(), 2))
            elapsed.zero_()

    assert measured_intervals == [0.98, 1.0, 0.96, 1.0]
