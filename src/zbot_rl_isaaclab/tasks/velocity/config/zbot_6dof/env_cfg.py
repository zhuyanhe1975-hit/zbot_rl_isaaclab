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
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab_tasks.core.velocity.mdp as velocity_mdp
from isaaclab_tasks.utils import PresetCfg

from zbot_rl_isaaclab.assets import ZBOT_6DOF_CFG

from ...mdp.actions_cfg import VelocityIntegratedJointPositionActionCfg
from ...mdp.observations import (
    body_height_above_ground,
    center_of_mass_relative_to_feet,
    foot_contact_forces,
    foot_contacts,
    heading_error,
    joint_velocity_limit,
    selected_body_ang_vel_b,
    selected_body_lin_vel_b,
    selected_body_projected_gravity,
)
from ...mdp.rewards import (
    AlternatingFeetTouchdownReward,
    StepLengthReward,
    alternating_step_frequency_error_l2,
    alternating_step_frequency_excess_l2,
    alternating_step_frequency_score,
    body_horizontal_velocity_l2,
    body_lateral_velocity_l2,
    heading_error_l2,
    selected_body_ang_vel_xy_l2,
    selected_body_flat_orientation_l2,
    single_support_foot_height_difference_l2,
    world_forward_velocity,
    yaw_rate_l2,
)
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
            ls_iterations=75,
            cone="elliptic",
            impratio=100.0,
            integrator="implicitfast",
            enable_multiccd=True,
        ),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=True,
        default_shape_cfg=NewtonShapeCfg(
            margin=0.0,
            gap=0.001,
            ke=5.0e4,
            kd=5.0e2,
            mu=1.0,
        ),
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
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )
    robot: ArticulationCfg = copy.deepcopy(ZBOT_6DOF_CFG)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    robot.actuators["legs"].actuator_velocity_limit = 2.0 * math.pi
    robot.actuators["legs"].joint_velocity_limit = 2.0 * math.pi
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
class ActionsCfg:
    """Six velocity-limited actions integrated into joint-position targets."""

    joint_pos = VelocityIntegratedJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        preserve_order=True,
        velocity_limit_range=(2.0 * math.pi, 2.0 * math.pi),
        position_offset_limit=0.5 * math.pi,
    )


@configclass
class ObservationsCfg:
    """Policy observations for velocity tracking and balance."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Walking observations with a balance-prior-compatible 39-value prefix."""

        # Keep this prefix in lockstep with the balance task. The walking actor prior copies these
        # first-layer columns by semantic slice rather than by an unsafe raw tensor prefix.
        base_lin_vel = ObsTerm(
            func=selected_body_lin_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel = ObsTerm(
            func=selected_body_ang_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        projected_gravity = ObsTerm(
            func=selected_body_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-0.1, n_max=0.1))
        actions = ObsTerm(func=base_mdp.last_action)
        foot_contact_forces = ObsTerm(
            func=foot_contact_forces,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
                "body_cfg": SceneEntityCfg("robot", body_names="base"),
            },
            scale=0.01,
        )
        com_relative_to_feet = ObsTerm(
            func=center_of_mass_relative_to_feet,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
                "body_cfg": SceneEntityCfg("robot", body_names="base"),
            },
        )

        # Walking-only suffix. Its first-layer prior weights start at zero and learn from the
        # walking rewards; no balance feature is silently reinterpreted as one of these signals.
        heading_error = ObsTerm(
            func=heading_error,
            params={
                "target_heading": 0.0,
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        base_height = ObsTerm(
            func=body_height_above_ground,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        foot_contacts = ObsTerm(
            func=foot_contacts,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
                "force_threshold": 10.0,
            },
        )
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

    body_forward_speed = RewTerm(
        func=world_forward_velocity,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    alive = RewTerm(func=base_mdp.is_alive, weight=0.2)
    # Make early termination decisively worse than accumulating the direct-walking objectives.
    termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-200.0)
    body_lateral_vel_l2 = RewTerm(
        func=body_lateral_velocity_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    heading_error_l2 = RewTerm(
        func=heading_error_l2,
        weight=-1.0,
        params={
            "target_heading": 0.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )
    yaw_rate_l2 = RewTerm(
        func=yaw_rate_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    flat_orientation_l2 = RewTerm(
        func=selected_body_flat_orientation_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    ang_vel_xy_l2 = RewTerm(
        func=selected_body_ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    stage_one_horizontal_velocity_l2 = RewTerm(
        func=body_horizontal_velocity_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    single_support_foot_height_l2 = RewTerm(
        func=single_support_foot_height_difference_l2,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
            "force_threshold": 10.0,
            "maximum_height_difference": 0.05,
        },
    )
    joint_torques_l2 = RewTerm(func=base_mdp.joint_torques_l2, weight=-1.0e-6)
    joint_acc_l2 = RewTerm(func=base_mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    joint_pos_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-1.0)
    joint_deviation = RewTerm(func=base_mdp.joint_deviation_l1, weight=-0.1)
    alternating_touchdown = RewTerm(
        func=AlternatingFeetTouchdownReward,  # pyright: ignore[reportArgumentType]
        # Stateful detector only. A non-zero sentinel keeps it updating for the
        # frequency-band reward without rewarding touchdown count.
        weight=1.0e-6,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "minimum_air_time": 0.05,
            "force_threshold": 10.0,
            "minimum_frequency": 1.0,
            "maximum_frequency": 2.0,
            "frequency_tolerance": 0.5,
        },
    )
    stage_one_step_frequency = RewTerm(
        func=alternating_step_frequency_score,
        weight=300.0,
        params={"reward_term_name": "alternating_touchdown"},
    )
    step_frequency_error_l2 = RewTerm(
        func=alternating_step_frequency_error_l2,
        weight=-100.0,
        params={"reward_term_name": "alternating_touchdown"},
    )
    step_frequency_excess_l2 = RewTerm(
        func=alternating_step_frequency_excess_l2,
        weight=-25.0,
        params={"reward_term_name": "alternating_touchdown"},
    )
    step_length = RewTerm(
        func=StepLengthReward,  # pyright: ignore[reportArgumentType]
        weight=100.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
            "minimum_air_time": 0.05,
            "force_threshold": 10.0,
            "crossing_margin": 0.0,
            "return_symmetry_error": False,
        },
    )
    step_length_asymmetry = RewTerm(
        func=StepLengthReward,  # pyright: ignore[reportArgumentType]
        weight=-100.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
            "minimum_air_time": 0.05,
            "force_threshold": 10.0,
            "crossing_margin": 0.0,
            "return_symmetry_error": True,
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
    """Direct flat-ground walking task initialized from the balance actor prior."""

    sim: SimulationCfg = SimulationCfg(physics=VelocityPhysicsCfg())  # pyright: ignore[reportArgumentType]
    scene: VelocitySceneCfg = VelocitySceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum = None

    def __post_init__(self) -> None:
        """Set simulation timing and visualization defaults."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.5, 2.5, 1.5))
