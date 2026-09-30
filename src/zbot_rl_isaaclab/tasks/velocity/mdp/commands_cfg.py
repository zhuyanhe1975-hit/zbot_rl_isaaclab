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


@configclass
class UniformStepFrequencyCommandCfg(CommandTermCfg):
    """Uniform scalar stepping-frequency command in hertz."""

    class_type: type | str = "{DIR}.commands:UniformStepFrequencyCommand"
    resampling_time_range: tuple[float, float] = (8.0, 12.0)
    asset_name: str = "robot"
    minimum_frequency: float = 0.5
    maximum_frequency: float = 2.0
    endpoint_probability: float = 0.0
    ordered: bool = False
    include_phase_features: bool = False
    frequency_is_full_cycle: bool = False
    randomize_phase: bool = True
    curriculum_initial_maximum_frequency: float | None = None
    curriculum_steps: int = 0
    evaluation_frequencies: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    frequency_reward_term_name: str | None = "alternating_touchdown"
    com_projection_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ZBot/frequency_center_of_mass_projection",
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
