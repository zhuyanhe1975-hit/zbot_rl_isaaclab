# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from pathlib import Path
from typing import cast

from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialBaseCfg
from isaaclab_newton.physics import MJWarpSolverCfg

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from zbot_rl_isaaclab.assets import ZBOT_6DOF_USD_PATH
from zbot_rl_isaaclab.pretraining import SHARED_OBSERVATION_SLICES, WALKING_OBSERVATION_DIM
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof.agents.rsl_rl_ppo_cfg import PPORunnerCfg
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof.env_cfg import VelocityEnvCfg, VelocityPhysicsCfg
from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import world_forward_velocity


def test_zbot_task_contract():
    """The baseline task must retain its six-action walking contract."""
    cfg = VelocityEnvCfg()
    physics = cast(VelocityPhysicsCfg, cfg.sim.physics)
    ground_spawn = cast(GroundPlaneCfg, cfg.scene.ground.spawn)
    robot_spawn = cast(UsdFileCfg, cfg.scene.robot.spawn)
    solver = cast(MJWarpSolverCfg, physics.newton_mjwarp.solver_cfg)
    ground_material = cast(RigidBodyMaterialBaseCfg, ground_spawn.physics_material)
    robot_material = cast(RigidBodyMaterialBaseCfg, robot_spawn.physics_material)

    assert Path(ZBOT_6DOF_USD_PATH).is_file()
    assert cfg.scene.robot.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert cfg.actions.joint_pos.joint_names == [f"joint{index}" for index in range(1, 7)]
    assert cfg.actions.joint_pos.preserve_order
    assert cfg.actions.joint_pos.velocity_limit_range == (2.0 * math.pi, 2.0 * math.pi)
    assert cfg.actions.joint_pos.position_offset_limit == 0.5 * math.pi
    assert cfg.scene.robot.actuators["legs"].actuator_velocity_limit == 2.0 * math.pi
    assert cfg.scene.robot.actuators["legs"].joint_velocity_limit == 2.0 * math.pi
    assert cfg.scene.contact_forces.track_air_time
    assert cfg.commands is None
    assert cfg.curriculum is None
    assert tuple(SHARED_OBSERVATION_SLICES) == (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "foot_contact_forces",
        "com_relative_to_feet",
    )
    assert WALKING_OBSERVATION_DIM == 44
    assert cfg.observations.policy.base_lin_vel.params["asset_cfg"].body_names == "base"
    assert cfg.observations.policy.base_ang_vel.params["asset_cfg"].body_names == "base"
    assert cfg.observations.policy.projected_gravity.params["asset_cfg"].body_names == "base"
    assert cfg.observations.policy.foot_contact_forces.params["sensor_cfg"].body_names == "foot_.*"
    assert cfg.observations.policy.com_relative_to_feet.params["asset_cfg"].body_names == "foot_.*"
    assert cfg.observations.policy.heading_error.params["asset_cfg"].body_names == "base"
    assert cfg.observations.policy.base_height.params["asset_cfg"].body_names == "base"
    assert cfg.observations.policy.foot_contacts.params["sensor_cfg"].body_names == "foot_.*"
    assert cfg.observations.policy.foot_contacts.params["force_threshold"] == 10.0
    assert cfg.rewards.body_forward_speed.func is world_forward_velocity
    assert cfg.rewards.body_forward_speed.weight == 5.0
    assert cfg.rewards.body_forward_speed.params["asset_cfg"].body_names == "base"
    assert cfg.rewards.body_lateral_vel_l2.params["asset_cfg"].body_names == "base"
    assert cfg.rewards.heading_error_l2.params["asset_cfg"].body_names == "base"
    assert cfg.rewards.yaw_rate_l2.params["asset_cfg"].body_names == "base"
    assert cfg.rewards.flat_orientation_l2.params["asset_cfg"].body_names == "base"
    assert cfg.rewards.ang_vel_xy_l2.params["asset_cfg"].body_names == "base"
    assert cfg.rewards.alive.weight == 0.2
    assert cfg.rewards.heading_error_l2.weight == -1.0
    assert cfg.rewards.yaw_rate_l2.weight == -0.2
    assert not hasattr(cfg.terminations, "heading_deviation")
    assert cfg.rewards.stage_one_horizontal_velocity_l2.weight == 0.0
    assert cfg.rewards.single_support_foot_height_l2.weight == -1.0
    assert cfg.rewards.single_support_foot_height_l2.params["force_threshold"] == 10.0
    assert cfg.rewards.single_support_foot_height_l2.params["maximum_height_difference"] == 0.05
    assert cfg.rewards.termination_penalty.weight == -200.0
    assert cfg.rewards.flat_orientation_l2.weight == 0.0
    assert cfg.rewards.ang_vel_xy_l2.weight == -0.05
    assert cfg.rewards.alternating_touchdown.weight == 1.0e-6
    assert cfg.rewards.alternating_touchdown.params["minimum_air_time"] == 0.05
    assert cfg.rewards.alternating_touchdown.params["force_threshold"] == 10.0
    assert "crossing_margin" not in cfg.rewards.alternating_touchdown.params
    assert "asset_cfg" not in cfg.rewards.alternating_touchdown.params
    assert cfg.rewards.alternating_touchdown.params["minimum_frequency"] == 1.0
    assert cfg.rewards.alternating_touchdown.params["maximum_frequency"] == 2.0
    assert cfg.rewards.stage_one_step_frequency.weight == 300.0
    assert cfg.rewards.step_frequency_error_l2.weight == -100.0
    assert cfg.rewards.step_frequency_excess_l2.weight == -25.0
    assert not hasattr(cfg.rewards, "feet_air_time")
    assert cfg.rewards.step_length.weight == 100.0
    assert cfg.rewards.step_length.params["return_symmetry_error"] is False
    assert cfg.rewards.step_length.params["crossing_margin"] == 0.0
    assert cfg.rewards.step_length_asymmetry.weight == -100.0
    assert cfg.rewards.step_length_asymmetry.params["return_symmetry_error"] is True
    assert not hasattr(cfg.rewards, "lin_vel_z_l2")
    assert cfg.decimation == 4
    assert cfg.sim.dt == 0.005
    assert solver.enable_multiccd
    assert solver.ls_iterations == 75
    assert solver.cone == "elliptic"
    assert solver.impratio == 100.0
    assert physics.newton_mjwarp.default_shape_cfg.gap == 0.001
    assert physics.newton_mjwarp.default_shape_cfg.ke == 5.0e4
    assert physics.newton_mjwarp.default_shape_cfg.kd == 5.0e2
    assert physics.newton_mjwarp.default_shape_cfg.mu == 1.0
    assert ground_material.dynamic_friction == 1.0
    assert robot_material.static_friction == 1.0
    assert robot_material.dynamic_friction == 1.0


def test_ppo_prioritizes_long_horizon_stability_with_bounded_initial_exploration():
    cfg = PPORunnerCfg()

    assert cfg.num_steps_per_env == 48
    assert cfg.save_interval == 50
    assert isinstance(cfg.actor.distribution_cfg, RslRlMLPModelCfg.GaussianDistributionCfg)
    assert cfg.actor.distribution_cfg.init_std == 0.4
    assert cfg.algorithm.learning_rate == 1.0e-4
    assert cfg.algorithm.schedule == "fixed"
    assert cfg.algorithm.entropy_coef == 0.002
    assert cfg.algorithm.gamma == 0.995
    assert cfg.algorithm.lam == 0.97
