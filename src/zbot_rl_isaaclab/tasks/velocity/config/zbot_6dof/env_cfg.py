# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import math

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab_tasks.core.velocity.mdp as velocity_mdp
from isaaclab_tasks.utils import PresetCfg

from zbot_rl_isaaclab.assets import ZBOT_6DOF_CFG

from ...mdp.actions_cfg import VelocityIntegratedJointPositionActionCfg
from ...mdp.observations import joint_velocity_limit
from ...mdp.terminations import body_height_below_minimum


@configclass
class VelocityPhysicsCfg(PresetCfg):
    """Physics presets supported by the ZBot walking task."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=144,
            nconmax=36,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
        ),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class VelocitySceneCfg(InteractiveSceneCfg):
    """Flat-ground scene containing the six-DoF ZBot and foot contacts."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
    )
    robot: ArticulationCfg = copy.deepcopy(ZBOT_6DOF_CFG)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


@configclass
class CommandsCfg:
    """Forward walking commands."""

    base_velocity = base_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.1,
        heading_command=False,
        debug_vis=False,
        ranges=base_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.6),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Six velocity-limited actions integrated into joint-position targets."""

    joint_pos = VelocityIntegratedJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        preserve_order=True,
        velocity_limit_range=(0.2 * math.pi, 2.0 * math.pi),
        position_offset_limit=math.pi,
    )


@configclass
class ObservationsCfg:
    """Policy observations for velocity tracking and balance."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Low-dimensional proprioceptive observation group."""

        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        velocity_commands = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-0.1, n_max=0.1))
        actions = ObsTerm(func=base_mdp.last_action)
        joint_velocity_limit = ObsTerm(
            func=joint_velocity_limit,
            params={"action_name": "joint_pos"},
            scale=1.0 / math.pi,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Small reset perturbations for a robust baseline policy."""

    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,  # pyright: ignore[reportArgumentType]
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )
    reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Compact walking objective based on maintained locomotion terms."""

    track_lin_vel_xy = RewTerm(
        func=velocity_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z = RewTerm(
        func=velocity_mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    alive = RewTerm(func=base_mdp.is_alive, weight=0.2)
    termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-5.0)
    lin_vel_z_l2 = RewTerm(func=base_mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torques_l2 = RewTerm(func=base_mdp.joint_torques_l2, weight=-1.0e-6)
    joint_acc_l2 = RewTerm(func=base_mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    joint_pos_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-1.0)
    joint_deviation = RewTerm(func=base_mdp.joint_deviation_l1, weight=-0.03)
    feet_air_time = RewTerm(
        func=velocity_mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "threshold": 0.15,
        },
    )
    feet_slide = RewTerm(
        func=velocity_mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
        },
    )
    undesired_contacts = RewTerm(
        func=base_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="a.*|b.*"),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=body_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "minimum_height": 0.18,
        },
    )


@configclass
class VelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based flat-ground velocity task for the six-DoF ZBot."""

    sim: SimulationCfg = SimulationCfg(physics=VelocityPhysicsCfg())  # pyright: ignore[reportArgumentType]
    scene: VelocitySceneCfg = VelocitySceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Set simulation timing and visualization defaults."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.5, 2.5, 1.5))
