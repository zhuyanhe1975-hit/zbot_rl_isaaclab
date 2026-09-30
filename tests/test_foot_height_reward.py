# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import (
    compute_single_support_foot_height_difference_l2,
    touchdown_impact_excess_l2,
)


def test_touchdown_impact_penalizes_only_high_force_landings():
    peak_forces = torch.tensor([[100.0, 200.0], [260.0, 160.0]])
    valid_touchdowns = torch.tensor([[True, True], [False, True]])

    penalty = touchdown_impact_excess_l2(peak_forces, valid_touchdowns, maximum_unpenalized_force=130.0)

    torch.testing.assert_close(penalty, torch.tensor([(200.0 / 130.0 - 1.0) ** 2, (160.0 / 130.0 - 1.0) ** 2]))


def test_penalizes_foot_height_difference_only_during_single_support():
    """High swing feet must be penalized without penalizing double support."""
    foot_positions_w = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.10]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.30]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.30]],
        ]
    )
    foot_contacts = torch.tensor([[True, False], [False, True], [True, True]])

    penalty = compute_single_support_foot_height_difference_l2(
        foot_positions_w,
        foot_contacts,
        maximum_height_difference=0.05,
    )

    torch.testing.assert_close(penalty, torch.tensor([0.0025, 0.0625, 0.0]))


def test_height_penalty_is_symmetric_between_feet():
    """Raising either foot by the same height must have the same cost."""
    foot_positions_w = torch.tensor(
        [
            [[0.0, 0.0, 0.20], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.20]],
        ]
    )
    foot_contacts = torch.tensor([[False, True], [True, False]])

    penalty = compute_single_support_foot_height_difference_l2(
        foot_positions_w,
        foot_contacts,
        maximum_height_difference=0.05,
    )

    torch.testing.assert_close(penalty, torch.tensor([0.0225, 0.0225]))


def test_normal_swing_clearance_is_unpenalized():
    foot_positions_w = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.03]]])
    foot_contacts = torch.tensor([[True, False]])

    penalty = compute_single_support_foot_height_difference_l2(
        foot_positions_w,
        foot_contacts,
        maximum_height_difference=0.05,
    )

    torch.testing.assert_close(penalty, torch.zeros(1))
