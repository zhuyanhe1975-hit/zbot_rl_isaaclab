# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Student sees only IMU, joint telemetry, and known controller signals."""

import copy

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ..zbot_6dof_base.env_cfg import BaseSceneCfg
from ..zbot_6dof_periodic_walking.env_cfg import ObservationsCfg as TeacherObservationsCfg
from ..zbot_6dof_periodic_walking.env_cfg import PeriodicWalkingEnvCfg

_teacher_policy = TeacherObservationsCfg().policy
_teacher_policy.enable_corruption = False


@configclass
class StudentSceneCfg(BaseSceneCfg):
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base", update_period=0.0)


@configclass
class StudentPolicyCfg(ObsGroup):
    """38 observable values; no global velocity, foot force, or COM."""

    base_ang_vel = ObsTerm(
        func=base_mdp.imu_ang_vel,
        params={"asset_cfg": SceneEntityCfg("imu")},
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    projected_gravity = copy.deepcopy(_teacher_policy.projected_gravity)
    heading_error = copy.deepcopy(_teacher_policy.heading_error)
    base_lin_acc = ObsTerm(
        func=base_mdp.imu_lin_acc,
        params={"asset_cfg": SceneEntityCfg("imu")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        scale=0.1,
    )
    joint_pos = copy.deepcopy(_teacher_policy.joint_pos)
    joint_vel = copy.deepcopy(_teacher_policy.joint_vel)
    joint_effort = ObsTerm(
        func=base_mdp.joint_effort,
        noise=Unoise(n_min=-0.5, n_max=0.5),
        scale=0.01,
    )
    actions = copy.deepcopy(_teacher_policy.actions)
    ideal_support_foot = copy.deepcopy(_teacher_policy.ideal_support_foot)

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class StudentObservationsCfg:
    policy: StudentPolicyCfg = StudentPolicyCfg()
    teacher: TeacherObservationsCfg.PolicyCfg = copy.deepcopy(_teacher_policy)


@configclass
class StudentEnvCfg(PeriodicWalkingEnvCfg):
    """Same physics, actions, rewards, and command as the frozen teacher task."""

    scene: StudentSceneCfg = StudentSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.imu.update_period = self.sim.dt
        self.commands.weight_shift.curriculum_initial_maximum_frequency = 1.0
        self.commands.weight_shift.curriculum_steps = 4_800
        self.commands.weight_shift.endpoint_probability = 0.25
