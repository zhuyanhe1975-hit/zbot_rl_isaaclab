# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from typing import Any

import torch

from isaaclab.managers import SceneEntityCfg

from zbot_rl_isaaclab.tasks.velocity.mdp.observations import (
    binary_contact_state,
    body_height_above_ground,
    foot_contacts,
)


class _Scene:
    def __init__(self, robot: Any, sensor: Any) -> None:
        self._robot = robot
        self.env_origins = torch.tensor([[1.0, 2.0, 0.04], [3.0, 4.0, 0.01]])
        self.sensors = {"contact_forces": sensor}

    def __getitem__(self, name: str) -> Any:
        assert name == "robot"
        return self._robot


def test_binary_contact_state_uses_per_foot_force_magnitude():
    forces = torch.tensor([[[0.0, 0.0, 9.9], [6.0, 8.0, 0.0]], [[0.0, 0.0, 10.1], [0.0, 0.0, 0.0]]])

    contacts = binary_contact_state(forces, force_threshold=10.0)

    torch.testing.assert_close(contacts, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_stability_observations_return_base_height_and_two_contacts():
    robot = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=SimpleNamespace(torch=torch.tensor([[[1.0, 2.0, 0.24]], [[3.0, 4.0, 0.31]]])))
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            net_normal_forces_w=SimpleNamespace(
                torch=torch.tensor(
                    [
                        [[0.0, 0.0, 12.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 11.0], [0.0, 0.0, 13.0]],
                    ]
                )
            )
        )
    )
    env: Any = SimpleNamespace(scene=_Scene(robot, sensor))
    asset_cfg = SceneEntityCfg("robot", body_ids=[0])
    sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[0, 1])

    torch.testing.assert_close(body_height_above_ground(env, asset_cfg), torch.tensor([[0.20], [0.30]]))
    torch.testing.assert_close(foot_contacts(env, sensor_cfg, 10.0), torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
