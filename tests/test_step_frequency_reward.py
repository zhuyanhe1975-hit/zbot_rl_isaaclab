# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import frequency_band_score


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
