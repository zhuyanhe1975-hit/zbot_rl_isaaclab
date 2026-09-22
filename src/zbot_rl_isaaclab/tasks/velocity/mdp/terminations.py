# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import wrapped_heading_error

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def body_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when a selected robot body falls below ``minimum_height`` [m]."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_heights = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    return torch.any(body_heights < minimum_height, dim=-1)


def heading_deviation_above_limit(
    env: ManagerBasedRLEnv,
    maximum_deviation: float,
    target_heading: float = 0.0,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Terminate when absolute world-frame heading error exceeds ``maximum_deviation`` [rad]."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(wrapped_heading_error(asset.data.heading_w.torch, target_heading)) > maximum_deviation


def filtered_contact_above_threshold(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Terminate when a contact sensor reports force against its configured partner [N]."""
    if threshold <= 0.0:
        raise ValueError("Filtered-contact threshold must be positive.")
    sensor = cast("ContactSensor", env.scene.sensors[sensor_cfg.name])
    force_matrix_w = sensor.data.normal_force_matrix_w
    if force_matrix_w is None:
        raise RuntimeError("Contact sensor must define filter_prim_paths_expr for filtered-contact termination.")
    force_magnitudes = torch.linalg.vector_norm(force_matrix_w.torch, dim=-1)
    return torch.any(force_magnitudes >= threshold, dim=(1, 2))
