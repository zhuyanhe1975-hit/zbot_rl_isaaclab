# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def body_forward_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward linear velocity along the body's forward ``+X`` axis [m/s]."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b.torch[:, 0]


def body_lateral_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize lateral velocity along the body's ``Y`` axis [m/s]."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b.torch[:, 1])


def body_horizontal_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize planar body velocity for in-place stepping [m^2/s^2]."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b.torch[:, :2]), dim=1)


def compute_single_support_foot_height_difference_l2(
    foot_positions_w: torch.Tensor,
    foot_contacts: torch.Tensor,
) -> torch.Tensor:
    """Compute squared foot-height difference during single support [m^2]."""
    single_support = torch.sum(foot_contacts.int(), dim=1) == 1
    height_difference = foot_positions_w[:, 0, 2] - foot_positions_w[:, 1, 2]
    return torch.square(height_difference) * single_support.float()


def single_support_foot_height_difference_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    """Penalize excessive swing-foot height relative to the supporting foot."""
    if isinstance(sensor_cfg.body_ids, slice) or len(sensor_cfg.body_ids) != 2:
        raise ValueError("Foot-height penalty requires exactly two resolved contact-sensor foot bodies.")
    if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 2:
        raise ValueError("Foot-height penalty requires exactly two resolved articulation foot bodies.")

    contact_sensor = cast("ContactSensor", env.scene.sensors[sensor_cfg.name])
    asset = cast("Articulation", env.scene[asset_cfg.name])
    normal_force_history = contact_sensor.data.net_normal_forces_w_history
    if normal_force_history is None:
        raise RuntimeError("Contact sensor must retain force history for the foot-height penalty.")
    contact_force = normal_force_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return compute_single_support_foot_height_difference_l2(
        asset.data.body_pos_w.torch[:, asset_cfg.body_ids],
        contact_force >= force_threshold,
    )


def update_alternating_touchdown_state(
    valid_touchdown: torch.Tensor,
    last_landing_foot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score exclusive left/right touchdown alternation and update landing history."""
    exclusive_touchdown = torch.sum(valid_touchdown.int(), dim=1) == 1
    landing_foot = torch.full_like(last_landing_foot, -1)
    landing_foot = torch.where(exclusive_touchdown & valid_touchdown[:, 0], 0, landing_foot)
    landing_foot = torch.where(exclusive_touchdown & valid_touchdown[:, 1], 1, landing_foot)
    alternated = (landing_foot >= 0) & (last_landing_foot >= 0) & (landing_foot != last_landing_foot)
    updated_history = torch.where(landing_foot >= 0, landing_foot, last_landing_foot)
    return alternated.float(), updated_history


def frequency_band_score(
    frequency: torch.Tensor,
    minimum_frequency: float,
    maximum_frequency: float,
    tolerance: float,
) -> torch.Tensor:
    """Score frequencies inside a target band and smoothly decay outside it."""
    if minimum_frequency <= 0.0 or maximum_frequency < minimum_frequency or tolerance <= 0.0:
        raise ValueError("Frequency bounds must satisfy 0 < minimum <= maximum and tolerance > 0.")
    distance_below = (minimum_frequency - frequency).clamp_min(0.0)
    distance_above = (frequency - maximum_frequency).clamp_min(0.0)
    distance_to_band = distance_below + distance_above
    return torch.exp(-torch.square(distance_to_band / tolerance))


def foot_relative_position_x(
    foot_positions_w: torch.Tensor,
    root_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Return each foot's sagittal position relative to the other foot [m]."""
    foot_0_from_1_w = foot_positions_w[:, 0] - foot_positions_w[:, 1]
    foot_0_from_1_b = quat_apply_inverse(root_quat_w, foot_0_from_1_w)
    return torch.stack((foot_0_from_1_b[:, 0], -foot_0_from_1_b[:, 0]), dim=1)


def update_crossing_touchdown_state(
    relative_position_x: torch.Tensor,
    airborne: torch.Tensor,
    first_contact: torch.Tensor,
    valid_touchdown: torch.Tensor,
    was_behind: torch.Tensor,
    crossing_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept touchdowns only after the landing foot moves from behind to ahead."""
    tracked_behind = was_behind | (airborne & (relative_position_x < -crossing_margin))
    crossing_touchdown = valid_touchdown & tracked_behind & (relative_position_x > crossing_margin)
    updated_was_behind = torch.where(first_contact, False, tracked_behind)
    return crossing_touchdown, updated_was_behind


def update_step_length_state(
    forward_step_length: torch.Tensor,
    valid_touchdown: torch.Tensor,
    last_step_length: torch.Tensor,
    has_step_length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update per-foot step lengths and return length reward plus symmetry error [m]."""
    landing_step_length = torch.sum(forward_step_length * valid_touchdown.float(), dim=1)
    updated_step_length = last_step_length.clone()
    updated_step_length[valid_touchdown] = forward_step_length[valid_touchdown]
    updated_has_step_length = has_step_length | valid_touchdown
    has_both_steps = torch.all(updated_has_step_length, dim=1)
    landed = torch.any(valid_touchdown, dim=1)
    symmetry_error = torch.abs(updated_step_length[:, 0] - updated_step_length[:, 1])
    symmetry_error = torch.where(has_both_steps & landed, symmetry_error, 0.0)
    return landing_step_length, symmetry_error, updated_step_length, updated_has_step_length


class AlternatingFeetTouchdownReward(ManagerTermBase):
    """Reward alternating valid touchdowns between exactly two feet."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        if isinstance(sensor_cfg.body_ids, slice) or len(sensor_cfg.body_ids) != 2:
            raise ValueError("Alternating touchdown reward requires exactly two resolved foot bodies.")
        if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 2:
            raise ValueError("Alternating touchdown reward requires exactly two articulation foot bodies.")
        self._foot_body_ids = sensor_cfg.body_ids
        self._asset_foot_ids = asset_cfg.body_ids
        self._contact_sensor = cast("ContactSensor", env.scene.sensors[sensor_cfg.name])
        self._asset = cast("Articulation", env.scene[asset_cfg.name])
        self._last_landing_foot = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self._was_behind = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self._episode_alternations = torch.zeros(env.num_envs, device=env.device)
        self._time_since_alternation = torch.zeros(env.num_envs, device=env.device)
        self._has_previous_alternation = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._frequency_score = torch.zeros(env.num_envs, device=env.device)

    @property
    def last_landing_foot(self) -> torch.Tensor:
        """Index of the last exclusively landed foot per environment, or ``-1``."""
        return self._last_landing_foot

    @property
    def episode_alternations(self) -> torch.Tensor:
        """Number of rewarded alternating touchdowns in the current episode."""
        return self._episode_alternations

    @property
    def frequency_score(self) -> torch.Tensor:
        """Score for the latest valid alternating-step frequency event."""
        return self._frequency_score

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear touchdown history for selected environments."""
        selected = slice(None) if env_ids is None else env_ids
        self._last_landing_foot[selected] = -1
        self._was_behind[selected] = False
        self._episode_alternations[selected] = 0.0
        self._time_since_alternation[selected] = 0.0
        self._has_previous_alternation[selected] = False
        self._frequency_score[selected] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        minimum_air_time: float,
        force_threshold: float,
        crossing_margin: float,
        minimum_frequency: float,
        maximum_frequency: float,
        frequency_tolerance: float,
    ) -> torch.Tensor:
        """Return one for a valid alternating touchdown and zero otherwise."""
        last_air_time = self._contact_sensor.data.last_air_time
        current_air_time = self._contact_sensor.data.current_air_time
        normal_force_history = self._contact_sensor.data.net_normal_forces_w_history
        if last_air_time is None or current_air_time is None or normal_force_history is None:
            raise RuntimeError("Contact sensor must enable air-time tracking for alternating touchdowns.")

        first_contact = self._contact_sensor.compute_first_contact(env.step_dt).torch[:, self._foot_body_ids].bool()
        air_time = last_air_time.torch[:, self._foot_body_ids]
        contact_force = normal_force_history.torch[:, :, self._foot_body_ids, :].norm(dim=-1).max(dim=1)[0]
        valid_touchdown = first_contact & (air_time >= minimum_air_time) & (contact_force >= force_threshold)
        relative_position_x = foot_relative_position_x(
            self._asset.data.body_pos_w.torch[:, self._asset_foot_ids],
            self._asset.data.root_quat_w.torch,
        )
        valid_touchdown, updated_was_behind = update_crossing_touchdown_state(
            relative_position_x,
            current_air_time.torch[:, self._foot_body_ids] > 0.0,
            first_contact,
            valid_touchdown,
            self._was_behind,
            crossing_margin,
        )
        self._was_behind.copy_(updated_was_behind)
        reward, updated_history = update_alternating_touchdown_state(
            valid_touchdown,
            self._last_landing_foot,
        )
        self._last_landing_foot.copy_(updated_history)
        self._episode_alternations += reward
        self._time_since_alternation += env.step_dt
        alternating_event = reward > 0.0
        has_interval = alternating_event & self._has_previous_alternation
        frequency = torch.reciprocal(self._time_since_alternation.clamp_min(env.step_dt))
        score = frequency_band_score(
            frequency,
            minimum_frequency,
            maximum_frequency,
            frequency_tolerance,
        )
        self._frequency_score.copy_(torch.where(has_interval, score, 0.0))
        self._has_previous_alternation |= alternating_event
        self._time_since_alternation[alternating_event] = 0.0
        return reward


def alternating_step_frequency_score(
    env: ManagerBasedRLEnv,
    reward_term_name: str,
) -> torch.Tensor:
    """Return the current event score from an alternating-touchdown reward term."""
    term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    reward_term = cast(AlternatingFeetTouchdownReward, term_cfg.func)
    return reward_term.frequency_score


class StepLengthReward(ManagerTermBase):
    """Reward forward foot displacement at exclusive valid touchdowns."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        if isinstance(sensor_cfg.body_ids, slice) or len(sensor_cfg.body_ids) != 2:
            raise ValueError("Step-length reward requires exactly two resolved contact-sensor foot bodies.")
        if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 2:
            raise ValueError("Step-length reward requires exactly two resolved articulation foot bodies.")
        self._sensor_foot_ids = sensor_cfg.body_ids
        self._asset_foot_ids = asset_cfg.body_ids
        self._contact_sensor = cast("ContactSensor", env.scene.sensors[sensor_cfg.name])
        self._asset = cast("Articulation", env.scene[asset_cfg.name])
        self._previous_touchdown_pos_w = self._asset.data.body_pos_w.torch[:, self._asset_foot_ids].clone()
        self._last_step_length = torch.zeros(env.num_envs, 2, device=env.device)
        self._has_step_length = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self._was_behind = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Reset stored foot touchdown positions for selected environments."""
        selected = slice(None) if env_ids is None else env_ids
        selected_foot_positions = self._asset.data.body_pos_w.torch[selected][:, self._asset_foot_ids]
        self._previous_touchdown_pos_w[selected] = selected_foot_positions
        self._last_step_length[selected] = 0.0
        self._has_step_length[selected] = False
        self._was_behind[selected] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        minimum_air_time: float,
        force_threshold: float,
        crossing_margin: float,
        return_symmetry_error: bool,
    ) -> torch.Tensor:
        """Return positive step length or the latest left-right asymmetry."""
        last_air_time = self._contact_sensor.data.last_air_time
        current_air_time = self._contact_sensor.data.current_air_time
        normal_force_history = self._contact_sensor.data.net_normal_forces_w_history
        if last_air_time is None or current_air_time is None or normal_force_history is None:
            raise RuntimeError("Contact sensor must enable air-time tracking for step length.")

        first_contact = self._contact_sensor.compute_first_contact(env.step_dt).torch[:, self._sensor_foot_ids].bool()
        air_time = last_air_time.torch[:, self._sensor_foot_ids]
        contact_force = normal_force_history.torch[:, :, self._sensor_foot_ids, :].norm(dim=-1).max(dim=1)[0]
        valid_touchdown = first_contact & (air_time >= minimum_air_time) & (contact_force >= force_threshold)
        relative_position_x = foot_relative_position_x(
            self._asset.data.body_pos_w.torch[:, self._asset_foot_ids],
            self._asset.data.root_quat_w.torch,
        )
        valid_touchdown, updated_was_behind = update_crossing_touchdown_state(
            relative_position_x,
            current_air_time.torch[:, self._sensor_foot_ids] > 0.0,
            first_contact,
            valid_touchdown,
            self._was_behind,
            crossing_margin,
        )
        self._was_behind.copy_(updated_was_behind)
        exclusive_touchdown = torch.sum(valid_touchdown.int(), dim=1) == 1
        valid_touchdown &= exclusive_touchdown.unsqueeze(-1)

        feet_pos_w = self._asset.data.body_pos_w.torch[:, self._asset_foot_ids]
        displacement_w = feet_pos_w - self._previous_touchdown_pos_w
        root_quat_w = self._asset.data.root_quat_w.torch.unsqueeze(1).expand(-1, 2, -1)
        forward_step_length = quat_apply_inverse(root_quat_w, displacement_w)[..., 0].clamp_min(0.0)
        self._previous_touchdown_pos_w[valid_touchdown] = feet_pos_w[valid_touchdown]
        landing_step_length, symmetry_error, updated_step_length, updated_has_step_length = update_step_length_state(
            forward_step_length,
            valid_touchdown,
            self._last_step_length,
            self._has_step_length,
        )
        self._last_step_length.copy_(updated_step_length)
        self._has_step_length.copy_(updated_has_step_length)

        if return_symmetry_error:
            return symmetry_error
        return landing_step_length
