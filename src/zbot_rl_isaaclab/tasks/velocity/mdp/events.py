"""Short physical force pulses for student disturbance training."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.sensors.contact_sensor.contact_force_marker import ContactForceVisualizer
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg


def planar_force_for_velocity_change(
    total_mass: torch.Tensor,
    delta_velocity: torch.Tensor,
    direction_rad: torch.Tensor,
    duration_s: float,
) -> torch.Tensor:
    """Convert a desired horizontal velocity change into a finite-duration force."""
    if duration_s <= 0.0:
        raise ValueError("Force-pulse duration must be positive.")
    magnitude = total_mass * delta_velocity / duration_s
    return torch.stack(
        (magnitude * direction_rad.cos(), magnitude * direction_rad.sin(), torch.zeros_like(magnitude)), dim=-1
    )


def force_pulse_velocity_limit(
    initial: float, maximum: float, step: int, offset: int, ramp_steps: int
) -> float:
    """Continue the force curriculum across resumed training runs."""
    if not 0.0 <= initial <= maximum or step < 0 or offset < 0 or ramp_steps < 0:
        raise ValueError("Invalid force-pulse velocity limits or ramp position.")
    progress = min((step + offset) / ramp_steps, 1.0) if ramp_steps else 1.0
    return initial + progress * (maximum - initial)


def pulse_start_probability(mean_interval_s: float, active_duration_s: float, step_dt: float) -> float:
    """Keep the expected start-to-start interval when a pulse blocks new starts."""
    if step_dt <= 0.0 or active_duration_s <= 0.0 or mean_interval_s <= active_duration_s:
        raise ValueError("Mean pulse interval must exceed its duration, with a positive policy step.")
    return min(step_dt / (mean_interval_s - active_duration_s), 1.0)


class ExternalForcePulse(ManagerTermBase):
    """Apply brief base forces and clear them when each pulse ends."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 1:
            raise ValueError("External force pulse requires exactly one base body.")
        self._asset = cast("Articulation", env.scene[asset_cfg.name])
        self._body_ids = asset_cfg.body_ids
        self._remaining = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.pulse_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._forces = torch.zeros((env.num_envs, 1, 3), device=env.device)
        self._torques = torch.zeros_like(self._forces)
        self._duration_s = cast(float, cfg.params["duration_s"])
        self._force_visualizer: ContactForceVisualizer | None = None
        self._pulse_indicator: VisualizationMarkers | None = None
        if cfg.params.get("visualize_force", False):
            marker_cfg = copy.deepcopy(RED_ARROW_X_MARKER_CFG)
            marker_cfg.prim_path = "/Visuals/ZBot/ExternalForcePulse"
            self._force_visualizer = ContactForceVisualizer(marker_cfg, force_scale=2.0)
            self._pulse_indicator = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/ZBot/ExternalForceIndicator",
                    markers={
                        "active": sim_utils.SphereCfg(
                            radius=0.07,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.1, 0.1), emissive_color=(0.4, 0.0, 0.0)
                            ),
                        )
                    },
                )
            )

    def _env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        all_ids = torch.arange(self._remaining.shape[0], device=self._remaining.device)
        if env_ids is None:
            return all_ids
        if isinstance(env_ids, slice):
            return all_ids[env_ids]
        return torch.as_tensor(env_ids, device=self._remaining.device, dtype=torch.long)

    def _write(self, ids: torch.Tensor) -> None:
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._forces[ids],
            torques=self._torques[ids],
            body_ids=self._body_ids,
            env_ids=ids.to(dtype=torch.int32),
        )

    def _visualize(self) -> None:
        if self._force_visualizer is None:
            return
        body_index = self._body_ids[0]
        body_positions = self._asset.data.body_pos_w.torch[:, body_index].clone()
        body_positions[:, 2] += 0.25
        body_orientations = self._asset.data.body_quat_w.torch[:, body_index]
        world_forces = quat_apply(body_orientations, self._forces[:, 0])
        total_mass = self._asset.data.body_mass.torch.sum(dim=1).unsqueeze(-1).clamp_min(1.0e-6)
        # Arrow length represents the equivalent velocity kick, so it is visible
        # without depending on the robot's total mass.
        equivalent_velocity = world_forces * (self._duration_s / total_mass)
        self._force_visualizer.visualize(body_positions, equivalent_velocity, force_threshold=1.0e-4)
        if self._pulse_indicator is not None:
            active = torch.linalg.norm(equivalent_velocity, dim=-1) > 1.0e-4
            scales = active[:, None].expand(-1, 3).float()
            self._pulse_indicator.visualize(translations=body_positions, scales=scales)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        ids = self._env_ids(env_ids)
        self._remaining[ids] = 0
        self._forces[ids] = 0.0
        self._write(ids)
        self._visualize()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mean_interval_s: float,
        duration_s: float,
        minimum_delta_velocity: float,
        initial_delta_velocity: float,
        maximum_delta_velocity: float,
        ramp_steps: int,
        ramp_offset_steps: int,
        visualize_force: bool,
    ) -> None:
        del asset_cfg, visualize_force
        if mean_interval_s <= 0.0 or duration_s <= 0.0 or ramp_steps < 0 or ramp_offset_steps < 0:
            raise ValueError("Invalid force-pulse interval, duration, ramp steps, or offset.")
        ids = self._env_ids(env_ids)
        if ids.numel() == 0:
            return
        duration_steps = max(1, round(duration_s / env.step_dt))
        start_probability = pulse_start_probability(mean_interval_s, duration_steps * env.step_dt, env.step_dt)
        start = (self._remaining[ids] == 0) & (
            torch.rand(ids.numel(), device=env.device) < start_probability
        )
        self.pulse_count[ids] += start.long()
        maximum = force_pulse_velocity_limit(
            initial_delta_velocity, maximum_delta_velocity, env.common_step_counter, ramp_offset_steps, ramp_steps
        )
        if not 0.0 <= minimum_delta_velocity <= maximum:
            raise ValueError("Force-pulse minimum velocity change must lie inside the current range.")
        delta_velocity = minimum_delta_velocity + torch.rand(ids.numel(), device=env.device) * (
            maximum - minimum_delta_velocity
        )
        direction = torch.rand(ids.numel(), device=env.device) * (2.0 * torch.pi)
        total_mass = self._asset.data.body_mass.torch[ids].sum(dim=1)
        sampled_force = planar_force_for_velocity_change(total_mass, delta_velocity, direction, duration_s)
        self._forces[ids] = torch.where(start[:, None, None], sampled_force[:, None, :], self._forces[ids])
        remaining = torch.where(start, duration_steps, self._remaining[ids])
        self._forces[ids] = torch.where((remaining > 0)[:, None, None], self._forces[ids], 0.0)
        self._write(ids)
        self._remaining[ids] = (remaining - 1).clamp_min(0)
        self._visualize()
