# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal balance task with event-driven left/right weight transfer."""

import copy
import math
from typing import cast

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg

import isaaclab_tasks.core.locomotion.mdp as locomotion_mdp
import isaaclab_tasks.core.velocity.mdp as velocity_mdp

from zbot_rl_isaaclab.assets import ZBOT_6DOF_CFG

from ...mdp.actions_cfg import VelocityIntegratedJointPositionActionCfg
from ...mdp.commands_cfg import UniformStepFrequencyCommandCfg
from ...mdp.observations import (
    center_of_mass_relative_to_feet,
    foot_contact_forces,
    ideal_support_foot_normalized_distance_error,
    selected_body_ang_vel_b,
    selected_body_lin_vel_b,
    selected_body_projected_gravity,
)
from ...mdp.rewards import (
    periodic_support_distance_tracking,
    periodic_support_joint_force_tracking,
    selected_body_ang_vel_xy_l2,
    selected_body_lin_vel_z_l2,
)
from ...mdp.terminations import (
    body_height_below_minimum,
    filtered_contact_above_threshold,
)
from ..zbot_6dof.env_cfg import VelocityPhysicsCfg, VelocitySceneCfg


@configclass
class BaseSceneCfg(VelocitySceneCfg):
    """Base scene with a dedicated foot-to-foot collision sensor."""

    robot: ArticulationCfg = copy.deepcopy(ZBOT_6DOF_CFG)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    cast(UsdFileCfg, robot.spawn).articulation_props = PhysxArticulationRootPropertiesCfg(enabled_self_collisions=True)
    robot.actuators["legs"].actuator_velocity_limit = 2.0 * math.pi
    robot.actuators["legs"].joint_velocity_limit = 2.0 * math.pi

    foot_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/foot_0",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/foot_1"],
        update_period=0.0,
    )
    ankle_wrenches = JointWrenchSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        update_period=0.0,
    )


@configclass
class ActionsCfg:
    """Slow bounded joint motion suitable for stationary weight transfer."""

    joint_pos = VelocityIntegratedJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        preserve_order=True,
        velocity_limit_range=(2.0 * math.pi, 2.0 * math.pi),
        position_offset_limit=0.5 * math.pi,
    )


@configclass
class ObservationsCfg:
    """Small proprioceptive state needed for balance and weight transfer."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=selected_body_lin_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        base_ang_vel = ObsTerm(
            func=selected_body_ang_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        projected_gravity = ObsTerm(
            func=selected_body_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.005, n_max=0.005))
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        actions = ObsTerm(func=base_mdp.last_action)
        foot_contact_forces = ObsTerm(
            func=foot_contact_forces,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["foot_0", "foot_1"],
                    preserve_order=True,
                ),
                "body_cfg": SceneEntityCfg("robot", body_names="base"),
            },
            scale=0.01,
        )
        com_relative_to_feet = ObsTerm(
            func=center_of_mass_relative_to_feet,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["foot_0", "foot_1"],
                    preserve_order=True,
                ),
                "body_cfg": SceneEntityCfg("robot", body_names="base"),
            },
        )
        ideal_support_foot = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "weight_shift"})
        support_distance_error = ObsTerm(
            func=ideal_support_foot_normalized_distance_error,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["foot_0", "foot_1"],
                    preserve_order=True,
                ),
                "command_name": "weight_shift",
                "command_index": 1,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """Sample the support-foot switching frequency for each environment."""

    weight_shift = UniformStepFrequencyCommandCfg(
        asset_name="robot",
        minimum_frequency=0.5,
        maximum_frequency=2.0,
        resampling_time_range=(20.0, 20.0),
        include_phase_features=True,
        frequency_is_full_cycle=True,
        randomize_phase=False,
        curriculum_initial_maximum_frequency=0.5,
        curriculum_steps=24_000,
        frequency_reward_term_name=None,
        debug_vis=True,
    )


@configclass
class EventsCfg:
    """Reset to the same planted stance so foot position targets stay fixed."""

    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,  # pyright: ignore[reportArgumentType]
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    """Only rewards needed to remain upright and transfer weight without stepping."""

    alive = RewTerm(func=base_mdp.is_alive, weight=2.0)
    termination_penalty = RewTerm(func=locomotion_mdp.terminated_penalty, weight=-20.0)
    support_distance = RewTerm(
        func=periodic_support_distance_tracking,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["foot_0", "foot_1"],
                preserve_order=True,
            ),
            "command_name": "weight_shift",
            "phase_sin_index": 2,
            "target_amplitude": 0.6,
            "tolerance": 0.25,
        },
    )
    support_force = RewTerm(
        func=periodic_support_joint_force_tracking,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "ankle_wrenches",
                body_names=["b1", "foot_1"],
                preserve_order=True,
            ),
            "command_name": "weight_shift",
            "minimum_total_force": 1.0,
            "phase_sin_index": 2,
            "target_amplitude": 0.8,
            "tolerance": 0.3,
        },
    )
    feet_collision_penalty = RewTerm(
        func=base_mdp.is_terminated_term,  # pyright: ignore[reportArgumentType]
        weight=-500.0,
        params={"term_keys": ["feet_collision"]},
    )
    vertical_velocity_l2 = RewTerm(
        func=selected_body_lin_vel_z_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    angular_velocity_l2 = RewTerm(
        func=selected_body_ang_vel_xy_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    joint_deviation_l1 = RewTerm(func=base_mdp.joint_deviation_l1, weight=-0.2)
    feet_slide = RewTerm(
        func=velocity_mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
        },
    )


@configclass
class TerminationsCfg:
    """End on timeout, falling, non-foot ground contact, or contact between the feet."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=body_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "minimum_height": 0.18,
        },
    )
    illegal_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="a.*|b.*|base"),
            "threshold": 1.0,
        },
    )
    feet_collision = DoneTerm(
        func=filtered_contact_above_threshold,
        params={
            "sensor_cfg": SceneEntityCfg("foot_collision"),
            "threshold": 1.0,
        },
    )


@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    """Six-DoF ZBot base task for stationary double-support weight transfer."""

    sim: SimulationCfg = SimulationCfg(physics=VelocityPhysicsCfg())  # pyright: ignore[reportArgumentType]
    scene: BaseSceneCfg = BaseSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum = None

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.foot_collision.update_period = self.sim.dt
        self.scene.ankle_wrenches.update_period = self.sim.dt
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(3.5, 3.5, 2.0))

    def play_mode(self) -> None:
        """Display the full frequency range deterministically across environments."""
        self.commands.weight_shift.ordered = True
        self.commands.weight_shift.resampling_time_range = (1.0e9, 1.0e9)
