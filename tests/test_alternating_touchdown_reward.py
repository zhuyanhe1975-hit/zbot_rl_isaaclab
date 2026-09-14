# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import update_alternating_touchdown_state


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
