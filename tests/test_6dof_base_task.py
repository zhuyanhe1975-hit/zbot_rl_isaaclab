# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import math
from types import SimpleNamespace
from typing import Any, cast

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SphereCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg

from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_base.env_cfg import BaseEnvCfg
from zbot_rl_isaaclab.tasks.velocity.mdp.commands import periodic_support_side
from zbot_rl_isaaclab.tasks.velocity.mdp.observations import (
    center_of_mass_relative_to_feet,
    normalized_support_foot_distance_error,
    whole_body_center_of_mass,
)
from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import (
    commanded_support_force_contrast_score,
    ideal_support_foot_distance_score,
)
from zbot_rl_isaaclab.tasks.velocity.mdp.terminations import filtered_contact_above_threshold


def test_whole_body_center_of_mass_is_mass_weighted():
    positions = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]])
    masses = torch.tensor([[3.0, 1.0]])

    center = whole_body_center_of_mass(positions, masses)

    torch.testing.assert_close(center, torch.tensor([[0.5, 1.0, 1.5]]))


def test_support_foot_command_switches_every_half_period():
    steps = torch.tensor([0, 24, 25, 49, 50])

    side = periodic_support_side(torch.ones(5), steps, step_dt=0.02, period=1.0)

    torch.testing.assert_close(side, torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0]))


def test_center_of_mass_coordinates_use_base_body_not_articulation_root():
    half_sqrt = math.sqrt(0.5)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_com_pos_w=SimpleNamespace(torch=torch.tensor([[[1.0, 0.0, 0.0]] * 3])),
            body_mass=SimpleNamespace(torch=torch.ones(1, 3)),
            body_pos_w=SimpleNamespace(torch=torch.zeros(1, 3, 3)),
            body_quat_w=SimpleNamespace(
                torch=torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, half_sqrt, half_sqrt]]])
            ),
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
        )
    )

    class _Scene:
        def __getitem__(self, name: str) -> Any:
            assert name == "robot"
            return robot

    env: Any = SimpleNamespace(scene=_Scene())
    result = center_of_mass_relative_to_feet(
        env,
        SceneEntityCfg("robot", body_ids=[0, 1]),
        SceneEntityCfg("robot", body_ids=[2]),
    )

    torch.testing.assert_close(result, torch.tensor([[0.0, -1.0, 0.0, 0.0, -1.0, 0.0]]), atol=1.0e-6, rtol=0.0)


def test_support_foot_distance_reward_prefers_com_projection_near_target_foot():
    ideal_distance = torch.tensor([0.0, 0.025, 0.10])
    other_distance = torch.tensor([0.10, 0.05, 0.0])

    score = ideal_support_foot_distance_score(ideal_distance, other_distance)
    error = normalized_support_foot_distance_error(ideal_distance, other_distance)

    torch.testing.assert_close(score, -error)
    torch.testing.assert_close(score[0], torch.tensor(1.0))
    assert score[0] > score[1] > score[2]
    torch.testing.assert_close(score[2], torch.tensor(-1.0))


def test_periodic_support_phase_rewards_the_commanded_foot_force():
    target_side = torch.tensor([1.0, -1.0, 1.0])
    forces = torch.tensor([[20.0, 0.0], [0.0, 20.0], [0.0, 20.0]])

    force_score = commanded_support_force_contrast_score(forces, target_side, minimum_total_force=1.0)

    torch.testing.assert_close(force_score, torch.tensor([1.0, 1.0, -1.0]))


def test_support_force_contrast_is_dense_and_normalized():
    forces = torch.tensor([[15.0, 5.0], [10.0, 10.0], [0.0, 0.0]])

    score = commanded_support_force_contrast_score(forces, torch.ones(3), minimum_total_force=1.0)

    torch.testing.assert_close(score, torch.tensor([0.5, 0.0, -1.0]))


def test_filtered_foot_contact_detects_only_force_above_threshold():
    force_matrix = torch.tensor(
        [
            [[[-0.9, 0.0, 0.0]]],
            [[[0.0, 0.0, 1.1]]],
        ]
    )
    sensor = SimpleNamespace(data=SimpleNamespace(normal_force_matrix_w=SimpleNamespace(torch=force_matrix)))
    env: Any = SimpleNamespace(scene=SimpleNamespace(sensors={"foot_collision": sensor}))

    collision = filtered_contact_above_threshold(env, SceneEntityCfg("foot_collision"), threshold=1.0)

    torch.testing.assert_close(collision, torch.tensor([False, True]))


def test_base_task_contains_no_stepping_objective():
    cfg = BaseEnvCfg()
    robot_spawn = cast(UsdFileCfg, cfg.scene.robot.spawn)

    observation_names = set(vars(cfg.observations.policy))
    reward_names = set(vars(cfg.rewards))

    assert cfg.commands.weight_shift.period == 1.0
    assert cfg.commands.weight_shift.debug_vis
    marker = cast(SphereCfg, cfg.commands.weight_shift.com_projection_visualizer_cfg.markers["com_projection"])
    assert marker.radius == 0.035
    assert cfg.curriculum is None
    assert cfg.actions.joint_pos.velocity_limit_range == (2.0 * math.pi, 2.0 * math.pi)
    assert cfg.actions.joint_pos.position_offset_limit == 0.5 * math.pi
    assert cfg.scene.robot.actuators["legs"].actuator_velocity_limit == 2.0 * math.pi
    assert cfg.scene.robot.actuators["legs"].joint_velocity_limit == 2.0 * math.pi
    assert cfg.observations.policy.foot_contact_forces.scale == 0.01
    assert cfg.observations.policy.base_lin_vel.params["asset_cfg"].body_names == "base"
    assert cfg.observations.policy.base_ang_vel.params["asset_cfg"].body_names == "base"
    assert cfg.observations.policy.projected_gravity.params["asset_cfg"].body_names == "base"
    assert cfg.observations.policy.foot_contact_forces.params["body_cfg"].body_names == "base"
    assert cfg.observations.policy.com_relative_to_feet.params["asset_cfg"].body_names == "foot_.*"
    assert cfg.observations.policy.com_relative_to_feet.params["body_cfg"].body_names == "base"
    assert cfg.observations.policy.ideal_support_foot.params["command_name"] == "weight_shift"
    assert cfg.observations.policy.support_distance_error.params["command_name"] == "weight_shift"
    assert cfg.rewards.support_distance.weight == 5.0
    assert "std" not in cfg.rewards.support_distance.params
    assert cfg.rewards.support_force.weight == 2.0
    assert cfg.rewards.support_force.params["minimum_total_force"] == 1.0
    assert not hasattr(cfg.rewards, "swing_lift")
    assert cfg.rewards.alive.weight == 2.0
    assert not hasattr(cfg.rewards, "upright")
    assert not hasattr(cfg.rewards, "flat_orientation_l2")
    assert cfg.rewards.termination_penalty.weight == -20.0
    assert cfg.rewards.feet_collision_penalty.weight == -500.0
    assert cfg.rewards.feet_collision_penalty.params["term_keys"] == ["feet_collision"]
    assert cfg.rewards.joint_deviation_l1.weight == -0.2
    assert not hasattr(cfg.rewards, "feet_clearance")
    assert not hasattr(cfg.rewards, "joint_torques_l2")
    assert not hasattr(cfg.rewards, "double_support")
    assert not hasattr(cfg.rewards, "feet_position_xy_l2")
    assert not hasattr(cfg.rewards, "com_position")
    assert not hasattr(cfg.rewards, "weight_transfer")
    assert not hasattr(cfg.rewards, "com_shift")
    assert not hasattr(cfg.rewards, "com_velocity")
    assert not hasattr(cfg.observations.policy, "feet_positions_xy")
    assert not hasattr(cfg.observations.policy, "target_side")
    assert not hasattr(cfg.observations.policy, "weight_shift_command")
    assert not hasattr(cfg.observations.policy, "lateral_target")
    assert cfg.scene.foot_collision.prim_path.endswith("/foot_0")
    assert cfg.scene.foot_collision.filter_prim_paths_expr == ["{ENV_REGEX_NS}/Robot/foot_1"]
    assert isinstance(robot_spawn.articulation_props, PhysxArticulationRootPropertiesCfg)
    assert robot_spawn.articulation_props.enabled_self_collisions is True
    assert cfg.terminations.feet_collision.params["threshold"] == 1.0
    assert not ({"foot_contacts", "heading_error", "base_height", "joint_velocity_limit"} & observation_names)
    assert not ({"step_length", "alternating_touchdown", "feet_air_time"} & reward_names)
