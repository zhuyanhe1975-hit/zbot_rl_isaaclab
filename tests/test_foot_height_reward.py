# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import compute_single_support_foot_height_difference_l2


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

    penalty = compute_single_support_foot_height_difference_l2(foot_positions_w, foot_contacts)

    torch.testing.assert_close(penalty, torch.tensor([0.01, 0.09, 0.0]))


def test_height_penalty_is_symmetric_between_feet():
    """Raising either foot by the same height must have the same cost."""
    foot_positions_w = torch.tensor(
        [
            [[0.0, 0.0, 0.20], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.20]],
        ]
    )
    foot_contacts = torch.tensor([[False, True], [True, False]])

    penalty = compute_single_support_foot_height_difference_l2(foot_positions_w, foot_contacts)

    torch.testing.assert_close(penalty, torch.tensor([0.04, 0.04]))
