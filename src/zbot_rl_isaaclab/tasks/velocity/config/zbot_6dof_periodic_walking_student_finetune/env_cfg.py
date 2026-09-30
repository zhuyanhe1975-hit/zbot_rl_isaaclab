# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Narrow PhysX randomization for student policy adaptation."""

import copy

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ...mdp.events import ExternalForcePulse
from ..zbot_6dof_base.env_cfg import EventsCfg as BaseEventsCfg
from ..zbot_6dof_periodic_walking.quality_rewards_cfg import QualityRewardsCfg as StudentFinetuneRewardsCfg
from ..zbot_6dof_periodic_walking_student.env_cfg import StudentEnvCfg, StudentObservationsCfg

_reset_events = BaseEventsCfg()


@configclass
class RobustEventsCfg(BaseEventsCfg):
    """PhysX-only, conservative uncertainty around the nominal teacher model."""

    robot_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,  # pyright: ignore[reportArgumentType]
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    base_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,  # pyright: ignore[reportArgumentType]
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    actuator_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,  # pyright: ignore[reportArgumentType]
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="joint[1-6]"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    reset_base = copy.deepcopy(_reset_events.reset_base)
    reset_base.params["pose_range"] = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (0.0, 0.0)}
    reset_base.params["velocity_range"] = {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    reset_joints = copy.deepcopy(_reset_events.reset_joints)
    reset_joints.params["position_range"] = (-0.02, 0.02)
    reset_joints.params["velocity_range"] = (-0.05, 0.05)
    external_force_pulse = EventTerm(
        func=ExternalForcePulse,  # pyright: ignore[reportArgumentType]
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mean_interval_s": 8.0,
            "duration_s": 0.16,
            "minimum_delta_velocity": 0.0,
            "initial_delta_velocity": 0.05,
            "maximum_delta_velocity": 0.2,
            "ramp_steps": 10_000,
            "ramp_offset_steps": 0,
            "visualize_force": False,
        },
    )


@configclass
class StudentFinetuneEnvCfg(StudentEnvCfg):
    """Nominal PPO adaptation with sensor-only actor and privileged critic."""

    observations: StudentObservationsCfg = StudentObservationsCfg()
    events: BaseEventsCfg = BaseEventsCfg()
    rewards: StudentFinetuneRewardsCfg = StudentFinetuneRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.weight_shift.curriculum_initial_maximum_frequency = None
        self.commands.weight_shift.frequency_reward_term_name = "alternating_touchdown"


@configclass
class StudentRobustEnvCfg(StudentFinetuneEnvCfg):
    """Add conservative PhysX variation after nominal student recovery."""

    events: RobustEventsCfg = RobustEventsCfg()

    def play_mode(self) -> None:
        super().play_mode()
        self.events = BaseEventsCfg()  # pyright: ignore[reportAttributeAccessIssue]
