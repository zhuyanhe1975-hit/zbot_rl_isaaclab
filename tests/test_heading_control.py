# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from types import SimpleNamespace
from typing import Any

import torch

from isaaclab.managers import SceneEntityCfg

from zbot_rl_isaaclab.tasks.velocity.mdp.observations import heading_error, wrapped_heading_error
from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import heading_error_l2, world_forward_velocity, yaw_rate_l2
from zbot_rl_isaaclab.tasks.velocity.mdp.terminations import heading_deviation_above_limit


def _env(heading: list[float], lin_vel_x: list[float] | None = None, yaw_rate: list[float] | None = None) -> Any:
    count = len(heading)
    data = SimpleNamespace(
        heading_w=SimpleNamespace(torch=torch.tensor(heading)),
        root_lin_vel_w=SimpleNamespace(torch=torch.tensor(lin_vel_x or [0.0] * count).unsqueeze(-1).repeat(1, 3)),
        root_ang_vel_w=SimpleNamespace(
            torch=torch.stack((torch.zeros(count), torch.zeros(count), torch.tensor(yaw_rate or [0.0] * count)), dim=-1)
        ),
    )
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})


def test_wrapped_heading_error_uses_shortest_signed_angle():
    heading = torch.tensor([0.0, math.pi - 0.1, -math.pi + 0.1])

    error = wrapped_heading_error(heading)

    torch.testing.assert_close(error, torch.tensor([0.0, -math.pi + 0.1, math.pi - 0.1]))


def test_direction_terms_expose_and_penalize_world_heading():
    env = _env([0.0, 0.5], lin_vel_x=[1.2, -0.3], yaw_rate=[0.1, -0.4])

    torch.testing.assert_close(heading_error(env), torch.tensor([[0.0], [-0.5]]))
    torch.testing.assert_close(world_forward_velocity(env), torch.tensor([1.2, -0.3]))
    torch.testing.assert_close(heading_error_l2(env), torch.tensor([0.0, 0.25]))
    torch.testing.assert_close(yaw_rate_l2(env), torch.tensor([0.01, 0.16]))


def test_heading_termination_resets_only_beyond_45_degrees():
    limit = math.pi / 4.0
    env = _env([limit - 1.0e-3, limit + 1.0e-3, -limit - 1.0e-3])

    terminated = heading_deviation_above_limit(
        env,
        maximum_deviation=limit,
        asset_cfg=SceneEntityCfg("robot"),
    )

    assert terminated.tolist() == [False, True, True]
