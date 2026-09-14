# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from typing import Any

import torch

from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import (
    body_forward_velocity,
    body_lateral_velocity_l2,
)


def _make_env(root_velocity_body: torch.Tensor) -> Any:
    robot = SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=SimpleNamespace(torch=root_velocity_body)))
    return SimpleNamespace(scene={"robot": robot})


def test_forward_reward_uses_body_x_velocity_directly():
    """Lateral motion must not change the body-forward tracking reward."""
    env = _make_env(root_velocity_body=torch.tensor([[1.0, 5.0, 0.0], [0.5, 0.0, 0.0]]))

    reward = body_forward_velocity(env)

    torch.testing.assert_close(reward, torch.tensor([1.0, 0.5]))


def test_lateral_penalty_uses_body_y_velocity():
    """The lateral penalty must measure the body's local Y velocity."""
    env = _make_env(root_velocity_body=torch.tensor([[1.0, 2.0, 0.0], [1.0, -3.0, 0.0]]))

    torch.testing.assert_close(body_lateral_velocity_l2(env), torch.tensor([4.0, 9.0]))
