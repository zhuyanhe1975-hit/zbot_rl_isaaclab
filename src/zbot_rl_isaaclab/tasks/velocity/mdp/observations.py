# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions import VelocityIntegratedJointPositionAction


def joint_velocity_limit(env: ManagerBasedEnv, action_name: str) -> torch.Tensor:
    """Return the sampled joint angular-velocity limit [rad/s]."""
    action_term = cast("VelocityIntegratedJointPositionAction", env.action_manager.get_term(action_name))
    return action_term.velocity_limit
