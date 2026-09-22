# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import PeriodicSupportFootCommandCfg


def periodic_support_side(
    initial_side: torch.Tensor,
    episode_length_buf: torch.Tensor,
    step_dt: float,
    period: float,
) -> torch.Tensor:
    """Return the ideal support-foot side for a fixed full-cycle period."""
    if period <= 0.0 or step_dt <= 0.0:
        raise ValueError("Support-foot period and environment step must be positive.")
    if initial_side.shape != episode_length_buf.shape:
        raise ValueError("Initial side and episode length must have matching shapes.")
    half_cycle = torch.floor(episode_length_buf.float() * step_dt / (0.5 * period)).long()
    phase_sign = torch.where(half_cycle.remainder(2) == 0, 1.0, -1.0)
    return initial_side * phase_sign


class PeriodicSupportFootCommand(CommandTerm):
    """Alternate the ideal support foot on a fixed time schedule."""

    cfg: PeriodicSupportFootCommandCfg

    def __init__(self, cfg: PeriodicSupportFootCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        if cfg.period <= 0.0:
            raise ValueError("Support-foot command period must be positive.")
        self._rl_env = env
        self._asset = cast("Articulation", env.scene[cfg.asset_name])
        self._target_side = torch.ones(env.num_envs, 1, device=env.device)
        self._initial_side = torch.ones_like(self._target_side)
        self.metrics["switches"] = torch.zeros(env.num_envs, device=env.device)

    @property
    def command(self) -> torch.Tensor:
        """Return ``+1`` for foot 0 and ``-1`` for foot 1."""
        return self._target_side

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        count = self._target_side[env_ids].shape[0]
        random_side = torch.randint(0, 2, (count, 1), device=self.device, dtype=torch.long)
        sampled_side = random_side.float() * 2.0 - 1.0
        self._initial_side[env_ids] = sampled_side
        self._target_side[env_ids] = sampled_side

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        updated_target = periodic_support_side(
            self._initial_side[:, 0],
            self._rl_env.episode_length_buf,
            self._rl_env.step_dt,
            self.cfg.period,
        ).unsqueeze(-1)
        switched = updated_target[:, 0] != self._target_side[:, 0]
        self._target_side.copy_(updated_target)
        self.metrics["switches"] += switched.float()

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "com_projection_visualizer"):
                self.com_projection_visualizer = VisualizationMarkers(self.cfg.com_projection_visualizer_cfg)
            self.com_projection_visualizer.set_visibility(True)
        elif hasattr(self, "com_projection_visualizer"):
            self.com_projection_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        del event
        if not self._asset.is_initialized:
            return
        body_com_pos_w = self._asset.data.body_com_pos_w.torch
        body_mass = self._asset.data.body_mass.torch
        center_of_mass_w = (body_com_pos_w * body_mass.unsqueeze(-1)).sum(dim=1) / body_mass.sum(
            dim=1, keepdim=True
        )
        projected_position = center_of_mass_w.clone()
        projected_position[:, 2] = self._env.scene.env_origins[:, 2] + 0.035
        self.com_projection_visualizer.visualize(
            translations=projected_position,
            environment_ids=self._env.scene._ALL_INDICES,
        )
