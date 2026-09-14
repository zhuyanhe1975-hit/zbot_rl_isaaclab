# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

from isaaclab.envs.mdp.actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import VelocityIntegratedJointPositionActionCfg


def integrate_velocity_to_position_offset(
    position_offset: torch.Tensor,
    raw_action: torch.Tensor,
    velocity_limit: torch.Tensor,
    step_dt: float,
    position_offset_limit: float,
) -> torch.Tensor:
    """Integrate bounded joint velocities into a position offset [rad].

    Args:
        position_offset: Accumulated joint-position offsets [rad], shape ``(N, J)``.
        raw_action: Unbounded policy actions, shape ``(N, J)``.
        velocity_limit: Per-environment or per-joint angular-velocity limits [rad/s].
        step_dt: Environment control period [s].
        position_offset_limit: Symmetric accumulated offset limit [rad].

    Returns:
        Updated joint-position offsets [rad], shape ``(N, J)``.
    """
    finite_action = torch.nan_to_num(raw_action, nan=0.0, posinf=1.0, neginf=-1.0)
    increment = torch.tanh(finite_action) * velocity_limit * step_dt
    return torch.clamp(position_offset + increment, -position_offset_limit, position_offset_limit)


class VelocityIntegratedJointPositionAction(JointAction):
    """Integrate normalized velocity actions and command the resulting joint positions."""

    cfg: VelocityIntegratedJointPositionActionCfg

    def __init__(self, cfg: VelocityIntegratedJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._controlled_joint_ids = cast(torch.Tensor, self._joint_ids)
        velocity_min, velocity_max = cfg.velocity_limit_range
        if not math.isfinite(velocity_min) or not math.isfinite(velocity_max):
            raise ValueError("velocity_limit_range values must be finite.")
        if velocity_min <= 0.0 or velocity_max < velocity_min:
            raise ValueError("velocity_limit_range must satisfy 0 < minimum <= maximum.")
        if not math.isfinite(cfg.position_offset_limit) or cfg.position_offset_limit <= 0.0:
            raise ValueError("position_offset_limit must be finite and positive.")

        self._default_joint_pos = self._asset.data.default_joint_pos.torch[:, self._controlled_joint_ids].clone()
        self._position_offset = torch.zeros_like(self._raw_actions)
        self._velocity_limit = torch.empty(self.num_envs, 1, device=self.device)
        self._resample_velocity_limit(slice(None))
        self._processed_actions.copy_(self._default_joint_pos)

    @property
    def velocity_limit(self) -> torch.Tensor:
        """Sampled angular-velocity limit [rad/s], shape ``(num_envs, 1)``."""
        return self._velocity_limit

    @property
    def position_offset(self) -> torch.Tensor:
        """Accumulated joint-position offset [rad], shape ``(num_envs, action_dim)``."""
        return self._position_offset

    def process_actions(self, actions: torch.Tensor) -> None:
        """Integrate one policy action over the environment control period."""
        self._raw_actions.copy_(torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0))
        self._position_offset.copy_(
            integrate_velocity_to_position_offset(
                self._position_offset,
                self._raw_actions,
                self._velocity_limit,
                self._env.step_dt,
                self.cfg.position_offset_limit,
            )
        )
        self._processed_actions.copy_(self._default_joint_pos + self._position_offset)

        joint_limits = self._asset.data.soft_joint_pos_limits.torch[:, self._controlled_joint_ids]
        self._processed_actions.clamp_(joint_limits[..., 0], joint_limits[..., 1])

    def apply_actions(self) -> None:
        """Apply the held joint-position targets at each physics step."""
        self._asset.actuators.target_command.set_position_index(
            value=self._processed_actions,
            joint_ids=self._controlled_joint_ids,
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear integrated offsets and resample limits for selected environments."""
        selected = slice(None) if env_ids is None else env_ids
        self._raw_actions[selected] = 0.0
        self._position_offset[selected] = 0.0
        self._processed_actions[selected] = self._default_joint_pos[selected]
        self._resample_velocity_limit(selected)

    def _resample_velocity_limit(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        velocity_min, velocity_max = self.cfg.velocity_limit_range
        sampled_limits = torch.empty_like(self._velocity_limit[env_ids]).uniform_(velocity_min, velocity_max)
        self._velocity_limit[env_ids] = sampled_limits
