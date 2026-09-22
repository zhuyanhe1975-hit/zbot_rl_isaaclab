# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass


@configclass
class PeriodicSupportFootCommandCfg(CommandTermCfg):
    """Time-driven alternating support-foot target."""

    class_type: type | str = "{DIR}.commands:PeriodicSupportFootCommand"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    asset_name: str = "robot"
    period: float = 1.0
    com_projection_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ZBot/center_of_mass_projection",
        markers={
            "com_projection": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.2),
                    emissive_color=(0.0, 0.3, 0.05),
                ),
            )
        },
    )
