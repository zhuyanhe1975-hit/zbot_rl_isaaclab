# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import update_crossing_touchdown_state


def test_counts_touchdown_after_foot_crosses_from_behind_to_ahead():
    """A foot must be observed behind during swing before an ahead touchdown counts."""
    was_behind = torch.zeros(1, 2, dtype=torch.bool)
    no_contact = torch.zeros(1, 2, dtype=torch.bool)

    crossing, was_behind = update_crossing_touchdown_state(
        relative_position_x=torch.tensor([[-0.10, 0.10]]),
        airborne=torch.tensor([[True, False]]),
        first_contact=no_contact,
        valid_touchdown=no_contact,
        was_behind=was_behind,
        crossing_margin=0.0,
    )
    assert not torch.any(crossing)
    assert torch.equal(was_behind, torch.tensor([[True, False]]))

    crossing, was_behind = update_crossing_touchdown_state(
        relative_position_x=torch.tensor([[0.12, -0.12]]),
        airborne=no_contact,
        first_contact=torch.tensor([[True, False]]),
        valid_touchdown=torch.tensor([[True, False]]),
        was_behind=was_behind,
        crossing_margin=0.0,
    )
    assert torch.equal(crossing, torch.tensor([[True, False]]))
    assert not torch.any(was_behind)


def test_rejects_touchdown_without_crossing_other_foot():
    """Landing behind or landing without a prior behind state must not count."""
    crossing, _ = update_crossing_touchdown_state(
        relative_position_x=torch.tensor([[-0.05, 0.05], [0.10, -0.10]]),
        airborne=torch.zeros(2, 2, dtype=torch.bool),
        first_contact=torch.tensor([[True, False], [True, False]]),
        valid_touchdown=torch.tensor([[True, False], [True, False]]),
        was_behind=torch.tensor([[True, False], [False, False]]),
        crossing_margin=0.0,
    )

    assert not torch.any(crossing)
