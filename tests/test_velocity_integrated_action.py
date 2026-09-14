# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.actions import integrate_velocity_to_position_offset


def test_integrates_bounded_action_with_velocity_limit():
    """The action increment must equal bounded velocity times the environment step."""
    position_offset = torch.tensor([[0.1, -0.2]])
    raw_action = torch.atanh(torch.tensor([[0.5, -0.25]]))
    velocity_limit = torch.tensor([[2.0]])

    result = integrate_velocity_to_position_offset(
        position_offset,
        raw_action,
        velocity_limit,
        step_dt=0.02,
        position_offset_limit=1.0,
    )

    torch.testing.assert_close(result, torch.tensor([[0.12, -0.21]]))


def test_clamps_accumulated_position_offset():
    """Integrated targets must remain inside the configured offset interval."""
    result = integrate_velocity_to_position_offset(
        torch.tensor([[0.99, -0.99]]),
        torch.tensor([[100.0, -100.0]]),
        torch.tensor([[2.0]]),
        step_dt=0.02,
        position_offset_limit=1.0,
    )

    torch.testing.assert_close(result, torch.tensor([[1.0, -1.0]]))
