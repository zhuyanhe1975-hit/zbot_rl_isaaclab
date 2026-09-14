# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def body_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when a selected robot body falls below ``minimum_height`` [m]."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_heights = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    return torch.any(body_heights < minimum_height, dim=-1)
