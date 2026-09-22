# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

    from .actions import VelocityIntegratedJointPositionAction


def wrapped_heading_error(heading_w: torch.Tensor, target_heading: float = 0.0) -> torch.Tensor:
    """Return the signed shortest-angle error from ``heading_w`` to the target [rad]."""
    error = target_heading - heading_w
    return torch.atan2(torch.sin(error), torch.cos(error))


def heading_error(
    env: ManagerBasedEnv,
    target_heading: float = 0.0,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Return the robot's signed heading error as a one-dimensional policy observation."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: Articulation = env.scene[asset_cfg.name]
    return wrapped_heading_error(asset.data.heading_w.torch, target_heading).unsqueeze(-1)


def body_height_above_ground(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return selected body height relative to each environment origin [m]."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]


def binary_contact_state(normal_forces_w: torch.Tensor, force_threshold: float) -> torch.Tensor:
    """Convert per-body world-frame normal forces into binary contact observations."""
    if force_threshold <= 0.0:
        raise ValueError("Contact-force threshold must be positive.")
    return (torch.linalg.vector_norm(normal_forces_w, dim=-1) >= force_threshold).float()


def foot_contacts(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    """Return binary contact state for the selected feet."""
    sensor = cast("ContactSensor", env.scene.sensors[sensor_cfg.name])
    normal_forces_w = sensor.data.net_normal_forces_w
    if normal_forces_w is None:
        raise RuntimeError("Contact sensor must provide normal forces for foot-contact observations.")
    return binary_contact_state(normal_forces_w.torch[:, sensor_cfg.body_ids], force_threshold)


def whole_body_center_of_mass(body_com_pos_w: torch.Tensor, body_mass: torch.Tensor) -> torch.Tensor:
    """Return the mass-weighted center of mass of an articulation in world coordinates."""
    if body_com_pos_w.shape[:2] != body_mass.shape:
        raise ValueError("Body COM positions and masses must have matching environment and body dimensions.")
    total_mass = body_mass.sum(dim=1, keepdim=True)
    if torch.any(total_mass <= 0.0):
        raise ValueError("Total articulation mass must be positive.")
    return (body_com_pos_w * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass


def support_foot_planar_distances(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Return COM planar distances to the ideal and non-ideal support feet [m]."""
    if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 2:
        raise ValueError("Support-foot distance requires exactly two resolved foot bodies.")
    asset: Articulation = env.scene[asset_cfg.name]
    center_of_mass_w = whole_body_center_of_mass(asset.data.body_com_pos_w.torch, asset.data.body_mass.torch)
    feet_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    target_side = env.command_manager.get_command(command_name)[:, 0]
    target_index = torch.where(target_side > 0.0, 0, 1)
    other_index = 1 - target_index
    environment_index = torch.arange(center_of_mass_w.shape[0], device=center_of_mass_w.device)
    target_foot_w = feet_w[environment_index, target_index]
    other_foot_w = feet_w[environment_index, other_index]
    target_distance = torch.linalg.vector_norm(center_of_mass_w[:, :2] - target_foot_w[:, :2], dim=1)
    other_distance = torch.linalg.vector_norm(center_of_mass_w[:, :2] - other_foot_w[:, :2], dim=1)
    return torch.stack((target_distance, other_distance), dim=1)


def ideal_support_foot_planar_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Return COM-to-commanded-support-foot distance on the world ground plane [m]."""
    return support_foot_planar_distances(env, asset_cfg, command_name)[:, :1]


def normalized_support_foot_distance_error(
    ideal_distance: torch.Tensor,
    other_distance: torch.Tensor,
) -> torch.Tensor:
    """Return normalized ideal-minus-other support-foot distance error in ``[-1, 1]``."""
    if ideal_distance.shape != other_distance.shape:
        raise ValueError("Ideal and non-ideal support-foot distances must have matching shapes.")
    return (ideal_distance - other_distance) / (ideal_distance + other_distance).clamp_min(1.0e-6)


def ideal_support_foot_normalized_distance_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Return normalized COM planar-distance error for the commanded support foot."""
    distances = support_foot_planar_distances(env, asset_cfg, command_name)
    error = normalized_support_foot_distance_error(distances[:, 0], distances[:, 1])
    return error.unsqueeze(-1)


def foot_contact_forces(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return per-foot normal contact-force vectors in the selected body frame [N]."""
    if isinstance(body_cfg.body_ids, slice) or len(body_cfg.body_ids) != 1:
        raise ValueError("Foot-force observations require exactly one body-frame body.")
    sensor = cast("ContactSensor", env.scene.sensors[sensor_cfg.name])
    asset: Articulation = env.scene[body_cfg.name]
    normal_forces_w = sensor.data.net_normal_forces_w
    if normal_forces_w is None:
        raise RuntimeError("Contact sensor must provide normal forces for foot-force observations.")
    selected_forces_w = normal_forces_w.torch[:, sensor_cfg.body_ids]
    body_quat_w = asset.data.body_quat_w.torch[:, body_cfg.body_ids[0]]
    body_quat_per_foot = body_quat_w.unsqueeze(1).expand(-1, selected_forces_w.shape[1], -1)
    forces_b = quat_apply_inverse(body_quat_per_foot.reshape(-1, 4), selected_forces_w.reshape(-1, 3))
    return forces_b.reshape(selected_forces_w.shape[0], -1)


def center_of_mass_relative_to_feet(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the whole-body COM position relative to each foot in the robot body frame [m]."""
    if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 2:
        raise ValueError("COM-relative-foot observations require exactly two resolved foot bodies.")
    if isinstance(body_cfg.body_ids, slice) or len(body_cfg.body_ids) != 1:
        raise ValueError("COM-relative-foot observations require exactly one body-frame body.")
    asset: Articulation = env.scene[asset_cfg.name]
    center_of_mass_w = whole_body_center_of_mass(asset.data.body_com_pos_w.torch, asset.data.body_mass.torch)
    foot_positions_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    relative_positions_w = center_of_mass_w.unsqueeze(1) - foot_positions_w
    body_quat_w = asset.data.body_quat_w.torch[:, body_cfg.body_ids[0]]
    body_quat_per_foot = body_quat_w.unsqueeze(1).expand(-1, 2, -1)
    return quat_apply_inverse(body_quat_per_foot.reshape(-1, 4), relative_positions_w.reshape(-1, 3)).reshape(-1, 6)


def selected_body_lin_vel_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return one selected body's linear velocity in its own frame [m/s]."""
    if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 1:
        raise ValueError("Body linear velocity requires exactly one resolved body.")
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    return quat_apply_inverse(asset.data.body_quat_w.torch[:, body_id], asset.data.body_lin_vel_w.torch[:, body_id])


def selected_body_ang_vel_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return one selected body's angular velocity in its own frame [rad/s]."""
    if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 1:
        raise ValueError("Body angular velocity requires exactly one resolved body.")
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    return quat_apply_inverse(asset.data.body_quat_w.torch[:, body_id], asset.data.body_ang_vel_w.torch[:, body_id])


def selected_body_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the unit gravity vector expressed in one selected body's frame."""
    if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 1:
        raise ValueError("Projected gravity requires exactly one resolved body.")
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_quat_w.torch[:, asset_cfg.body_ids[0]]
    gravity_w = torch.zeros((body_quat_w.shape[0], 3), device=body_quat_w.device, dtype=body_quat_w.dtype)
    gravity_w[:, 2] = -1.0
    return quat_apply_inverse(body_quat_w, gravity_w)


def joint_velocity_limit(env: ManagerBasedEnv, action_name: str) -> torch.Tensor:
    """Return the sampled joint angular-velocity limit [rad/s]."""
    action_term = cast("VelocityIntegratedJointPositionAction", env.action_manager.get_term(action_name))
    return action_term.velocity_limit
