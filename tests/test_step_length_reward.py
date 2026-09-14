# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import update_step_length_state


def test_step_length_reward_is_unbounded_and_preserves_metric_units():
    """Every positive landing length must pass through without a target or upper cap."""
    forward_step_length = torch.tensor([[0.01, 0.0], [0.0, 0.50]])
    valid_touchdown = torch.tensor([[True, False], [False, True]])

    reward, _, _, _ = update_step_length_state(
        forward_step_length,
        valid_touchdown,
        last_step_length=torch.zeros(2, 2),
        has_step_length=torch.zeros(2, 2, dtype=torch.bool),
    )

    torch.testing.assert_close(reward, torch.tensor([0.01, 0.50]))


def test_airborne_motion_without_touchdown_has_zero_step_length():
    """A foot displacement cannot earn step-length reward before a valid landing."""
    reward, symmetry_error, updated_length, updated_seen = update_step_length_state(
        forward_step_length=torch.tensor([[10.0, 8.0]]),
        valid_touchdown=torch.zeros(1, 2, dtype=torch.bool),
        last_step_length=torch.tensor([[0.20, 0.25]]),
        has_step_length=torch.tensor([[True, True]]),
    )

    torch.testing.assert_close(reward, torch.zeros(1))
    torch.testing.assert_close(symmetry_error, torch.zeros(1))
    torch.testing.assert_close(updated_length, torch.tensor([[0.20, 0.25]]))
    assert torch.equal(updated_seen, torch.tensor([[True, True]]))


def test_step_length_symmetry_compares_latest_left_and_right_steps():
    """Symmetry error must be emitted only after both feet have measured steps."""
    forward_step_length = torch.tensor([[0.30, 0.0], [0.0, 0.50]])
    valid_touchdown = torch.tensor([[True, False], [False, True]])
    last_step_length = torch.tensor([[0.0, 0.0], [0.20, 0.0]])
    has_step_length = torch.tensor([[False, False], [True, False]])

    _, symmetry_error, updated_length, updated_seen = update_step_length_state(
        forward_step_length,
        valid_touchdown,
        last_step_length,
        has_step_length,
    )

    torch.testing.assert_close(symmetry_error, torch.tensor([0.0, 0.30]))
    torch.testing.assert_close(updated_length, torch.tensor([[0.30, 0.0], [0.20, 0.50]]))
    assert torch.equal(updated_seen, torch.tensor([[True, False], [True, True]]))
