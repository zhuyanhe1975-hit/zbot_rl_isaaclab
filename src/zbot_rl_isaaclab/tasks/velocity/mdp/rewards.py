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

from .observations import (
    heading_error,
    selected_body_ang_vel_b,
    selected_body_lin_vel_b,
    selected_body_projected_gravity,
    support_foot_planar_distances,
    whole_body_center_of_mass,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, JointWrenchSensor


def body_forward_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward linear velocity along the body's forward ``+X`` axis [m/s]."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b.torch[:, 0]


def world_forward_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward linear velocity along the fixed world ``+X`` direction [m/s]."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: RigidObject = env.scene[asset_cfg.name]
    if isinstance(asset_cfg.body_ids, slice):
        return asset.data.root_lin_vel_w.torch[:, 0]
    if len(asset_cfg.body_ids) != 1:
        raise ValueError("World-forward velocity requires exactly one selected rigid body.")
    return asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids[0], 0]


def world_forward_velocity_score(
    env: ManagerBasedRLEnv,
    target_speed: float,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Return a bounded dense score for forward velocity, with saturation at useful walking speed."""
    if target_speed <= 0.0:
        raise ValueError("Target forward speed must be positive.")
    return torch.tanh(world_forward_velocity(env, asset_cfg) / target_speed)


def heading_error_l2(
    env: ManagerBasedRLEnv,
    target_heading: float = 0.0,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize squared deviation from the target world-frame heading [rad^2]."""
    error = heading_error(env, target_heading=target_heading, asset_cfg=asset_cfg)
    return torch.square(error[:, 0])


def yaw_rate_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize squared world-frame yaw rate [rad^2/s^2]."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: RigidObject = env.scene[asset_cfg.name]
    if isinstance(asset_cfg.body_ids, slice):
        return torch.square(asset.data.root_ang_vel_w.torch[:, 2])
    if len(asset_cfg.body_ids) != 1:
        raise ValueError("Yaw-rate penalty requires exactly one selected rigid body.")
    return torch.square(asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids[0], 2])


def body_lateral_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize lateral velocity along the body's ``Y`` axis [m/s]."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: RigidObject = env.scene[asset_cfg.name]
    if not isinstance(asset_cfg.body_ids, slice):
        return torch.square(selected_body_lin_vel_b(env, asset_cfg)[:, 1])
    return torch.square(asset.data.root_lin_vel_b.torch[:, 1])


def body_horizontal_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize planar body velocity for in-place stepping [m^2/s^2]."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: RigidObject = env.scene[asset_cfg.name]
    if not isinstance(asset_cfg.body_ids, slice):
        return torch.sum(torch.square(selected_body_lin_vel_b(env, asset_cfg)[:, :2]), dim=1)
    return torch.sum(torch.square(asset.data.root_lin_vel_b.torch[:, :2]), dim=1)


def selected_body_lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize one selected body's vertical velocity in its own frame."""
    return torch.square(selected_body_lin_vel_b(env, asset_cfg)[:, 2])


def selected_body_ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize one selected body's roll and pitch angular velocities in its own frame."""
    return torch.sum(torch.square(selected_body_ang_vel_b(env, asset_cfg)[:, :2]), dim=1)


def selected_body_flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize one selected body's deviation from an upright gravity projection."""
    return torch.sum(torch.square(selected_body_projected_gravity(env, asset_cfg)[:, :2]), dim=1)


def ideal_support_foot_distance_score(
    ideal_distance: torch.Tensor,
    other_distance: torch.Tensor,
) -> torch.Tensor:
    """Return normalized other-minus-ideal distance reward in ``[-1, 1]``."""
    if ideal_distance.shape != other_distance.shape:
        raise ValueError("Ideal and non-ideal support-foot distances must have matching shapes.")
    return (other_distance - ideal_distance) / (ideal_distance + other_distance).clamp_min(1.0e-6)


def ideal_support_foot_distance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_index: int = 0,
) -> torch.Tensor:
    """Reward moving the COM ground projection toward the ideal support foot."""
    distances = support_foot_planar_distances(env, asset_cfg, command_name, command_index)
    return ideal_support_foot_distance_score(distances[:, 0], distances[:, 1])


def periodic_tracking_score(
    actual: torch.Tensor,
    target: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Return a smooth signed score in ``(-1, 1]`` for periodic tracking."""
    if actual.shape != target.shape:
        raise ValueError("Actual and target periodic signals must have matching shapes.")
    if tolerance <= 0.0:
        raise ValueError("Periodic tracking tolerance must be positive.")
    normalized_error_l2 = torch.square((actual - target) / tolerance)
    return 2.0 / (1.0 + normalized_error_l2) - 1.0


def signed_support_distance_contrast(foot_distances: torch.Tensor) -> torch.Tensor:
    """Return positive values near foot 0 and negative values near foot 1."""
    if foot_distances.shape[1:] != (2,):
        raise ValueError("Foot distances must have shape (num_envs, 2).")
    return (foot_distances[:, 1] - foot_distances[:, 0]) / foot_distances.sum(dim=1).clamp_min(1.0e-6)


def signed_support_force_contrast(
    foot_force_magnitudes: torch.Tensor,
    minimum_total_force: float,
) -> torch.Tensor:
    """Return positive values when foot 0 carries more load and negative values for foot 1."""
    if foot_force_magnitudes.shape[1:] != (2,):
        raise ValueError("Foot forces must have shape (num_envs, 2).")
    if minimum_total_force <= 0.0:
        raise ValueError("Minimum total foot force must be positive.")
    total_force = foot_force_magnitudes.sum(dim=1)
    contrast = (foot_force_magnitudes[:, 0] - foot_force_magnitudes[:, 1]) / total_force.clamp_min(minimum_total_force)
    return torch.where(total_force >= minimum_total_force, contrast, 0.0)


def periodic_support_distance_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    phase_sin_index: int,
    target_amplitude: float,
    tolerance: float,
) -> torch.Tensor:
    """Track a sinusoidal COM transfer between the two planted feet."""
    if not 0.0 < target_amplitude <= 1.0:
        raise ValueError("COM target amplitude must be in (0, 1].")
    command = env.command_manager.get_command(command_name)
    if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 2:
        raise ValueError("Periodic COM tracking requires exactly two ordered foot bodies.")
    asset = cast("Articulation", env.scene[asset_cfg.name])
    center_of_mass_w = whole_body_center_of_mass(asset.data.body_com_pos_w.torch, asset.data.body_mass.torch)
    feet_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    distances = torch.linalg.vector_norm(center_of_mass_w[:, None, :2] - feet_w[:, :, :2], dim=2)
    actual = signed_support_distance_contrast(distances)
    target = target_amplitude * command[:, phase_sin_index]
    return periodic_tracking_score(actual, target, tolerance)


def commanded_support_force_contrast_score(
    foot_force_magnitudes: torch.Tensor,
    target_side: torch.Tensor,
    minimum_total_force: float,
) -> torch.Tensor:
    """Return normalized ideal-minus-swing foot force in ``[-1, 1]``."""
    if foot_force_magnitudes.shape[1:] != (2,) or target_side.shape != foot_force_magnitudes.shape[:1]:
        raise ValueError("Foot forces must have shape (num_envs, 2) and target side shape (num_envs,).")
    if minimum_total_force <= 0.0:
        raise ValueError("Minimum total foot force must be positive.")
    target_index = torch.where(target_side > 0.0, 0, 1)
    swing_index = 1 - target_index
    environment_index = torch.arange(target_side.shape[0], device=target_side.device)
    target_force = foot_force_magnitudes[environment_index, target_index]
    swing_force = foot_force_magnitudes[environment_index, swing_index]
    total_force = target_force + swing_force
    contrast = (target_force - swing_force) / total_force.clamp_min(minimum_total_force)
    return torch.where(total_force >= minimum_total_force, contrast, -1.0)


def commanded_support_force_contrast(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    minimum_total_force: float,
    command_index: int = 0,
) -> torch.Tensor:
    """Densely reward loading the ideal support foot over the swing foot."""
    if isinstance(sensor_cfg.body_ids, slice) or len(sensor_cfg.body_ids) != 2:
        raise ValueError("Support-foot contact reward requires exactly two foot contact sensors.")
    sensor = cast("ContactSensor", env.scene.sensors[sensor_cfg.name])
    normal_forces_w = sensor.data.net_normal_forces_w
    if normal_forces_w is None:
        raise RuntimeError("Contact sensor must provide normal forces for support-foot contact reward.")
    forces = normal_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=-1)
    target_side = env.command_manager.get_command(command_name)[:, command_index]
    return commanded_support_force_contrast_score(forces, target_side, minimum_total_force)


def commanded_support_joint_force_contrast(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    minimum_total_force: float,
    command_index: int = 0,
) -> torch.Tensor:
    """Reward commanded-foot loading using the two ankle joint reaction forces.

    Joint reaction wrenches avoid backend-specific rigid-contact manifold aggregation. The configured
    body order must be the foot-0-side ankle child followed by the foot-1-side ankle child.
    """
    if isinstance(sensor_cfg.body_ids, slice) or len(sensor_cfg.body_ids) != 2:
        raise ValueError("Support-foot joint-force reward requires exactly two ankle wrench bodies.")
    sensor = cast("JointWrenchSensor", env.scene.sensors[sensor_cfg.name])
    joint_forces = sensor.data.force
    if joint_forces is None:
        raise RuntimeError("Joint wrench sensor must provide forces for support-foot loading reward.")
    forces = joint_forces.torch[:, sensor_cfg.body_ids].norm(dim=-1)
    target_side = env.command_manager.get_command(command_name)[:, command_index]
    return commanded_support_force_contrast_score(forces, target_side, minimum_total_force)


def periodic_support_force_tracking(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    minimum_total_force: float,
    phase_sin_index: int,
    target_amplitude: float,
    tolerance: float,
) -> torch.Tensor:
    """Track signed sinusoidal load transfer using ordered foot contact forces."""
    if not 0.0 < target_amplitude <= 1.0:
        raise ValueError("Force target amplitude must be in (0, 1].")
    if isinstance(sensor_cfg.body_ids, slice) or len(sensor_cfg.body_ids) != 2:
        raise ValueError("Periodic support tracking requires exactly two ordered foot sensors.")
    sensor = cast("ContactSensor", env.scene.sensors[sensor_cfg.name])
    normal_forces_w = sensor.data.net_normal_forces_w
    if normal_forces_w is None:
        raise RuntimeError("Foot contact sensor must provide normal forces for periodic support tracking.")
    command = env.command_manager.get_command(command_name)
    forces = normal_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=-1)
    actual = signed_support_force_contrast(forces, minimum_total_force)
    target = target_amplitude * command[:, phase_sin_index]
    return periodic_tracking_score(actual, target, tolerance)


def periodic_support_joint_force_tracking(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    minimum_total_force: float,
    phase_sin_index: int,
    target_amplitude: float,
    tolerance: float,
) -> torch.Tensor:
    """Track signed sinusoidal load transfer using the proven ankle-reaction signal."""
    if not 0.0 < target_amplitude <= 1.0:
        raise ValueError("Force target amplitude must be in (0, 1].")
    if isinstance(sensor_cfg.body_ids, slice) or len(sensor_cfg.body_ids) != 2:
        raise ValueError("Periodic support tracking requires exactly two ordered ankle wrench bodies.")
    sensor = cast("JointWrenchSensor", env.scene.sensors[sensor_cfg.name])
    joint_forces = sensor.data.force
    if joint_forces is None:
        raise RuntimeError("Joint wrench sensor must provide forces for periodic support tracking.")
    command = env.command_manager.get_command(command_name)
    forces = joint_forces.torch[:, sensor_cfg.body_ids].norm(dim=-1)
    actual = signed_support_force_contrast(forces, minimum_total_force)
    target = target_amplitude * command[:, phase_sin_index]
    return periodic_tracking_score(actual, target, tolerance)


def compute_single_support_foot_height_difference_l2(
    foot_positions_w: torch.Tensor,
    foot_contacts: torch.Tensor,
    maximum_height_difference: float,
) -> torch.Tensor:
    """Penalize only the foot-height difference above a normal swing clearance."""
    if maximum_height_difference < 0.0:
        raise ValueError("Maximum unpenalized foot-height difference must be non-negative.")
    single_support = torch.sum(foot_contacts.int(), dim=1) == 1
    height_difference = torch.abs(foot_positions_w[:, 0, 2] - foot_positions_w[:, 1, 2])
    excessive_height = (height_difference - maximum_height_difference).clamp_min(0.0)
    return torch.square(excessive_height) * single_support.float()


def single_support_foot_height_difference_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float,
    maximum_height_difference: float,
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
        maximum_height_difference,
    )


def touchdown_impact_excess_l2(
    peak_forces: torch.Tensor,
    valid_touchdowns: torch.Tensor,
    maximum_unpenalized_force: float,
) -> torch.Tensor:
    """Sum squared relative force excess for newly landed feet."""
    if peak_forces.shape != valid_touchdowns.shape or peak_forces.ndim != 2:
        raise ValueError("Peak forces and touchdown flags must have matching (num_envs, num_feet) shapes.")
    if maximum_unpenalized_force <= 0.0:
        raise ValueError("Maximum unpenalized touchdown force must be positive.")
    excess = (peak_forces / maximum_unpenalized_force - 1.0).clamp_min(0.0)
    return (torch.square(excess) * valid_touchdowns).sum(dim=1)


def touchdown_impact_force_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    minimum_air_time: float,
    force_threshold: float,
    maximum_unpenalized_force: float,
) -> torch.Tensor:
    """Penalize high peak contact forces only at valid foot touchdowns."""
    if isinstance(sensor_cfg.body_ids, slice) or len(sensor_cfg.body_ids) != 2:
        raise ValueError("Touchdown impact penalty requires exactly two resolved foot sensors.")
    sensor = cast("ContactSensor", env.scene.sensors[sensor_cfg.name])
    history = sensor.data.net_normal_forces_w_history
    last_air_time = sensor.data.last_air_time
    if history is None or last_air_time is None:
        raise RuntimeError("Touchdown impact penalty requires force history and air-time tracking.")
    peak_forces = history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    touchdowns = sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids].bool()
    valid = touchdowns & (last_air_time.torch[:, sensor_cfg.body_ids] >= minimum_air_time)
    valid &= peak_forces >= force_threshold
    return touchdown_impact_excess_l2(peak_forces, valid, maximum_unpenalized_force)


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


def filter_short_touchdown_intervals(
    valid_touchdown: torch.Tensor,
    last_landing_foot: torch.Tensor,
    time_since_alternation: torch.Tensor,
    minimum_interval: float,
) -> torch.Tensor:
    """Ignore brief contact bounces without changing the last accepted landing foot."""
    if minimum_interval < 0.0:
        raise ValueError("Minimum touchdown interval must be non-negative.")
    eligible = (last_landing_foot < 0) | (time_since_alternation >= minimum_interval)
    return valid_touchdown & eligible.unsqueeze(1)


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


def frequency_band_error_l2(
    frequency: torch.Tensor,
    minimum_frequency: float,
    maximum_frequency: float,
) -> torch.Tensor:
    """Return squared distance from a target frequency band in Hz squared."""
    if minimum_frequency <= 0.0 or maximum_frequency < minimum_frequency:
        raise ValueError("Frequency bounds must satisfy 0 < minimum <= maximum.")
    distance_below = (minimum_frequency - frequency).clamp_min(0.0)
    distance_above = (frequency - maximum_frequency).clamp_min(0.0)
    return torch.square(distance_below + distance_above)


def frequency_above_band_error_l2(frequency: torch.Tensor, maximum_frequency: float) -> torch.Tensor:
    """Return squared frequency excess above a maximum without penalizing slow exploration."""
    if maximum_frequency <= 0.0:
        raise ValueError("Maximum frequency must be positive.")
    return torch.square((frequency - maximum_frequency).clamp_min(0.0))


def time_normalize_frequency_event_metric(metric: torch.Tensor, interval: torch.Tensor) -> torch.Tensor:
    """Scale an event metric by its interval so event count does not increase reward per unit time."""
    return metric * interval


def frequency_command_score(
    measured_frequency: torch.Tensor,
    commanded_frequency: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Score frequency tracking with a smooth kernel that retains gradients far from the target."""
    if tolerance <= 0.0:
        raise ValueError("Frequency command tolerance must be positive.")
    normalized_error_l2 = torch.square((measured_frequency - commanded_frequency) / tolerance)
    return torch.reciprocal(1.0 + normalized_error_l2)


def frequency_command_error_l2(
    measured_frequency: torch.Tensor,
    commanded_frequency: torch.Tensor,
) -> torch.Tensor:
    """Return squared error between measured and commanded stepping frequencies."""
    return torch.square(measured_frequency - commanded_frequency)


def frequency_command_relative_error_l2(
    measured_frequency: torch.Tensor,
    commanded_frequency: torch.Tensor,
) -> torch.Tensor:
    """Return squared relative cadence error so all commanded frequencies use one scale."""
    if torch.any(commanded_frequency <= 0.0):
        raise ValueError("Commanded frequencies must be positive.")
    return torch.square((measured_frequency - commanded_frequency) / commanded_frequency)


def foot_relative_position_x(
    foot_positions_w: torch.Tensor,
    root_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Return each foot's sagittal position relative to the other foot [m]."""
    foot_0_from_1_w = foot_positions_w[:, 0] - foot_positions_w[:, 1]
    foot_0_from_1_b = quat_apply_inverse(root_quat_w, foot_0_from_1_w)
    return torch.stack((foot_0_from_1_b[:, 0], -foot_0_from_1_b[:, 0]), dim=1)


def commanded_swing_foot_score(
    per_foot_value: torch.Tensor,
    target_side: torch.Tensor,
    target_value: float,
) -> torch.Tensor:
    """Score the commanded swing foot's positive progress toward a target value."""
    if per_foot_value.shape[1:] != (2,) or target_side.shape != per_foot_value.shape[:1]:
        raise ValueError("Per-foot values must have shape (num_envs, 2).")
    if target_value <= 0.0:
        raise ValueError("Swing-foot target value must be positive.")
    swing_index = torch.where(target_side > 0.0, 1, 0)
    environment_index = torch.arange(target_side.shape[0], device=target_side.device)
    return (per_foot_value[environment_index, swing_index] / target_value).clamp(0.0, 1.0)


def commanded_swing_foot_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    target_height: float,
    command_index: int = 1,
) -> torch.Tensor:
    """Densely reward lifting the scheduled swing foot above the support foot."""
    if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 2:
        raise ValueError("Swing-foot height reward requires exactly two foot bodies.")
    asset = cast("Articulation", env.scene[asset_cfg.name])
    heights = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    relative_heights = torch.stack((heights[:, 0] - heights[:, 1], heights[:, 1] - heights[:, 0]), dim=1)
    target_side = env.command_manager.get_command(command_name)[:, command_index]
    return commanded_swing_foot_score(relative_heights, target_side, target_height)


def commanded_swing_foot_forward_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
    command_name: str,
    target_distance: float,
    command_index: int = 1,
) -> torch.Tensor:
    """Densely reward moving the scheduled swing foot ahead of the support foot."""
    if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 2:
        raise ValueError("Swing-foot progress reward requires exactly two foot bodies.")
    if isinstance(body_cfg.body_ids, slice) or len(body_cfg.body_ids) != 1:
        raise ValueError("Swing-foot progress reward requires exactly one body-frame body.")
    asset = cast("Articulation", env.scene[asset_cfg.name])
    relative_x = foot_relative_position_x(
        asset.data.body_pos_w.torch[:, asset_cfg.body_ids],
        asset.data.body_quat_w.torch[:, body_cfg.body_ids[0]],
    )
    target_side = env.command_manager.get_command(command_name)[:, command_index]
    return commanded_swing_foot_score(relative_x, target_side, target_distance)


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
    updated_step_length = torch.where(valid_touchdown, forward_step_length, last_step_length)
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
        if isinstance(sensor_cfg.body_ids, slice) or len(sensor_cfg.body_ids) != 2:
            raise ValueError("Alternating touchdown reward requires exactly two resolved foot bodies.")
        self._foot_body_ids = sensor_cfg.body_ids
        self._contact_sensor = cast("ContactSensor", env.scene.sensors[sensor_cfg.name])
        self._last_landing_foot = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self._episode_alternations = torch.zeros(env.num_envs, device=env.device)
        self._time_since_alternation = torch.zeros(env.num_envs, device=env.device)
        self._has_previous_alternation = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._frequency_score = torch.zeros(env.num_envs, device=env.device)
        self._frequency_error_l2 = torch.zeros(env.num_envs, device=env.device)
        self._frequency_excess_l2 = torch.zeros(env.num_envs, device=env.device)
        self._measured_frequency = torch.zeros(env.num_envs, device=env.device)
        self._frequency_event_interval = torch.zeros(env.num_envs, device=env.device)
        self._has_frequency_event = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._last_frequency_command = torch.zeros(env.num_envs, device=env.device)

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

    @property
    def frequency_error_l2(self) -> torch.Tensor:
        """Squared distance of the latest alternating event from the target frequency band."""
        return self._frequency_error_l2

    @property
    def frequency_excess_l2(self) -> torch.Tensor:
        """Squared excess of the latest alternating event above the maximum target frequency."""
        return self._frequency_excess_l2

    @property
    def measured_frequency(self) -> torch.Tensor:
        """Measured frequency for the latest valid alternating-touchdown interval."""
        return self._measured_frequency

    @property
    def frequency_event_interval(self) -> torch.Tensor:
        """Duration of the latest valid alternating-touchdown interval."""
        return self._frequency_event_interval

    @property
    def has_frequency_event(self) -> torch.Tensor:
        """Whether the current step contains a valid frequency measurement."""
        return self._has_frequency_event

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear touchdown history for selected environments."""
        selected = slice(None) if env_ids is None else env_ids
        self._last_landing_foot[selected] = -1
        self._episode_alternations[selected] = 0.0
        self._time_since_alternation[selected] = 0.0
        self._has_previous_alternation[selected] = False
        self._frequency_score[selected] = 0.0
        self._frequency_error_l2[selected] = 0.0
        self._frequency_excess_l2[selected] = 0.0
        self._measured_frequency[selected] = 0.0
        self._frequency_event_interval[selected] = 0.0
        self._has_frequency_event[selected] = False
        self._last_frequency_command[selected] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        minimum_air_time: float,
        force_threshold: float,
        minimum_frequency: float,
        maximum_frequency: float,
        frequency_tolerance: float,
        frequency_command_name: str | None = None,
        minimum_alternation_interval: float = 0.0,
    ) -> torch.Tensor:
        """Return one for a valid alternating touchdown and zero otherwise."""
        if frequency_command_name is not None:
            frequency_command = env.command_manager.get_command(frequency_command_name)[:, 0]
            command_changed = ~torch.isclose(frequency_command, self._last_frequency_command)
            self._has_previous_alternation.masked_fill_(command_changed, False)
            self._time_since_alternation.masked_fill_(command_changed, 0.0)
            self._last_frequency_command.copy_(frequency_command)
        last_air_time = self._contact_sensor.data.last_air_time
        normal_force_history = self._contact_sensor.data.net_normal_forces_w_history
        if last_air_time is None or normal_force_history is None:
            raise RuntimeError("Contact sensor must enable air-time tracking for alternating touchdowns.")

        first_contact = self._contact_sensor.compute_first_contact(env.step_dt).torch[:, self._foot_body_ids].bool()
        air_time = last_air_time.torch[:, self._foot_body_ids]
        contact_force = normal_force_history.torch[:, :, self._foot_body_ids, :].norm(dim=-1).max(dim=1)[0]
        valid_touchdown = first_contact & (air_time >= minimum_air_time) & (contact_force >= force_threshold)
        self._time_since_alternation += env.step_dt
        valid_touchdown = filter_short_touchdown_intervals(
            valid_touchdown,
            self._last_landing_foot,
            self._time_since_alternation,
            minimum_alternation_interval,
        )
        reward, updated_history = update_alternating_touchdown_state(
            valid_touchdown,
            self._last_landing_foot,
        )
        self._last_landing_foot.copy_(updated_history)
        self._episode_alternations += reward
        alternating_event = reward > 0.0
        has_interval = alternating_event & self._has_previous_alternation
        interval = self._time_since_alternation.clamp_min(env.step_dt)
        frequency = torch.reciprocal(interval)
        self._measured_frequency.copy_(torch.where(has_interval, frequency, 0.0))
        self._frequency_event_interval.copy_(torch.where(has_interval, interval, 0.0))
        self._has_frequency_event.copy_(has_interval)
        score = frequency_band_score(
            frequency,
            minimum_frequency,
            maximum_frequency,
            frequency_tolerance,
        )
        normalized_score = time_normalize_frequency_event_metric(score, interval)
        self._frequency_score.copy_(torch.where(has_interval, normalized_score, 0.0))
        frequency_error = frequency_band_error_l2(frequency, minimum_frequency, maximum_frequency)
        normalized_error = time_normalize_frequency_event_metric(frequency_error, interval)
        self._frequency_error_l2.copy_(torch.where(has_interval, normalized_error, 0.0))
        frequency_excess = frequency_above_band_error_l2(frequency, maximum_frequency)
        normalized_excess = time_normalize_frequency_event_metric(frequency_excess, interval)
        self._frequency_excess_l2.copy_(torch.where(has_interval, normalized_excess, 0.0))
        self._has_previous_alternation |= alternating_event
        self._time_since_alternation.masked_fill_(alternating_event, 0.0)
        return reward


def alternating_step_frequency_score(
    env: ManagerBasedRLEnv,
    reward_term_name: str,
) -> torch.Tensor:
    """Return the current event score from an alternating-touchdown reward term."""
    term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    reward_term = cast(AlternatingFeetTouchdownReward, term_cfg.func)
    return reward_term.frequency_score


def alternating_step_frequency_error_l2(
    env: ManagerBasedRLEnv,
    reward_term_name: str,
) -> torch.Tensor:
    """Return the latest event's squared distance from the target frequency band."""
    term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    reward_term = cast(AlternatingFeetTouchdownReward, term_cfg.func)
    return reward_term.frequency_error_l2


def alternating_step_frequency_excess_l2(
    env: ManagerBasedRLEnv,
    reward_term_name: str,
) -> torch.Tensor:
    """Return squared excess above the maximum target frequency for the latest event."""
    term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    reward_term = cast(AlternatingFeetTouchdownReward, term_cfg.func)
    return reward_term.frequency_excess_l2


def commanded_step_frequency_score(
    env: ManagerBasedRLEnv,
    reward_term_name: str,
    command_name: str,
    tolerance: float,
    target_multiplier: float = 1.0,
) -> torch.Tensor:
    """Score the latest alternating-step frequency against the scalar frequency command."""
    if target_multiplier <= 0.0:
        raise ValueError("Frequency target multiplier must be positive.")
    term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    reward_term = cast(AlternatingFeetTouchdownReward, term_cfg.func)
    target = target_multiplier * env.command_manager.get_command(command_name)[:, 0]
    score = frequency_command_score(reward_term.measured_frequency, target, tolerance)
    normalized = time_normalize_frequency_event_metric(score, reward_term.frequency_event_interval)
    return torch.where(reward_term.has_frequency_event, normalized, 0.0)


def commanded_step_frequency_error_l2(
    env: ManagerBasedRLEnv,
    reward_term_name: str,
    command_name: str,
    target_multiplier: float = 1.0,
) -> torch.Tensor:
    """Penalize squared error from the scalar frequency command at valid frequency events."""
    if target_multiplier <= 0.0:
        raise ValueError("Frequency target multiplier must be positive.")
    term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    reward_term = cast(AlternatingFeetTouchdownReward, term_cfg.func)
    target = target_multiplier * env.command_manager.get_command(command_name)[:, 0]
    error = frequency_command_error_l2(reward_term.measured_frequency, target)
    normalized = time_normalize_frequency_event_metric(error, reward_term.frequency_event_interval)
    return torch.where(reward_term.has_frequency_event, normalized, 0.0)


def commanded_step_frequency_relative_error_l2(
    env: ManagerBasedRLEnv,
    reward_term_name: str,
    command_name: str,
    target_multiplier: float = 1.0,
) -> torch.Tensor:
    """Penalize relative cadence error at valid alternating-touchdown events."""
    if target_multiplier <= 0.0:
        raise ValueError("Frequency target multiplier must be positive.")
    term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    reward_term = cast(AlternatingFeetTouchdownReward, term_cfg.func)
    target = target_multiplier * env.command_manager.get_command(command_name)[:, 0]
    error = frequency_command_relative_error_l2(reward_term.measured_frequency, target)
    normalized = time_normalize_frequency_event_metric(error, reward_term.frequency_event_interval)
    return torch.where(reward_term.has_frequency_event, normalized, 0.0)


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
        maximum_rewarded_step_length: float | None = None,
    ) -> torch.Tensor:
        """Return positive step length or the latest left-right asymmetry."""
        if maximum_rewarded_step_length is not None and maximum_rewarded_step_length <= 0.0:
            raise ValueError("Maximum rewarded step length must be positive.")
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
        self._previous_touchdown_pos_w.copy_(
            torch.where(valid_touchdown.unsqueeze(-1), feet_pos_w, self._previous_touchdown_pos_w)
        )
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
        if maximum_rewarded_step_length is not None:
            return landing_step_length.clamp_max(maximum_rewarded_step_length)
        return landing_step_length
