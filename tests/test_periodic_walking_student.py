from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rsl_rl.algorithms import Distillation

import isaaclab.envs.mdp as base_mdp

from zbot_rl_isaaclab.student_distillation import GuidedDistillation
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking.env_cfg import PeriodicWalkingEnvCfg
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student.agents.rsl_rl_distillation_cfg import (
    DistillationRunnerCfg,
)
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student.env_cfg import StudentEnvCfg


def _terms(group) -> list[str]:
    return [field.name for field in fields(group) if hasattr(getattr(group, field.name), "func")]


def test_student_observations_are_limited_to_robot_telemetry_and_known_commands():
    cfg = StudentEnvCfg()

    assert _terms(cfg.observations.policy) == [
        "base_ang_vel",
        "projected_gravity",
        "heading_error",
        "base_lin_acc",
        "joint_pos",
        "joint_vel",
        "joint_effort",
        "actions",
        "ideal_support_foot",
    ]
    assert cfg.observations.policy.joint_effort.func is base_mdp.joint_effort
    assert cfg.observations.policy.base_ang_vel.func is base_mdp.imu_ang_vel
    assert cfg.observations.policy.base_lin_acc.func is base_mdp.imu_lin_acc
    assert cfg.observations.policy.base_lin_acc.scale == 0.1
    assert cfg.scene.imu.prim_path.endswith("/base")
    assert cfg.observations.policy.joint_effort.scale == 0.01
    assert cfg.observations.policy.enable_corruption
    assert cfg.commands.weight_shift.curriculum_initial_maximum_frequency == 1.0
    assert cfg.commands.weight_shift.curriculum_steps == 4_800
    assert cfg.commands.weight_shift.endpoint_probability == 0.25


def test_teacher_retains_frozen_ppo_observation_contract():
    student_cfg = StudentEnvCfg()
    teacher_cfg = PeriodicWalkingEnvCfg()

    assert _terms(student_cfg.observations.teacher) == _terms(teacher_cfg.observations.policy)
    for name in _terms(student_cfg.observations.teacher):
        term = getattr(student_cfg.observations.teacher, name)
        original = getattr(teacher_cfg.observations.policy, name)
        assert term.func is original.func
        assert term.scale == original.scale
    assert not student_cfg.observations.teacher.enable_corruption
    assert type(student_cfg.actions) is type(teacher_cfg.actions)
    assert type(student_cfg.rewards) is type(teacher_cfg.rewards)


def test_distillation_uses_recurrent_student_and_the_original_teacher_architecture():
    cfg = DistillationRunnerCfg()

    assert cfg.class_name == "DistillationRunner"
    assert cfg.obs_groups == {"student": ["policy"], "teacher": ["teacher"]}
    assert cfg.student.class_name == "RNNModel"
    assert cfg.student.rnn_type == "gru"
    assert cfg.teacher.hidden_dims == [256, 128, 128]
    assert cfg.teacher.obs_normalization is False
    assert cfg.algorithm.class_name == "zbot_rl_isaaclab.student_distillation:GuidedDistillation"


def test_teacher_guidance_fades_without_disappearing():
    algorithm = GuidedDistillation.__new__(GuidedDistillation)
    algorithm.num_updates = 0
    algorithm.teacher_fraction_start = 1.0
    algorithm.teacher_fraction_end = 0.25
    algorithm.teacher_fraction_decay_updates = 500

    assert algorithm.teacher_fraction() == 1.0
    algorithm.num_updates = 100
    assert algorithm.teacher_fraction() == 0.85
    algorithm.num_updates = 600
    assert algorithm.teacher_fraction() == 0.25


def test_guided_schedule_resumes_from_existing_distillation_checkpoint():
    algorithm = GuidedDistillation.__new__(GuidedDistillation)
    algorithm.num_updates = 0
    with patch.object(Distillation, "load", return_value=True):
        assert algorithm.load({"student_state_dict": {}, "iter": 99}, None, True)
    assert algorithm.num_updates == 100


def test_resumed_optimizer_uses_the_current_learning_rate():
    algorithm = GuidedDistillation.__new__(GuidedDistillation)
    algorithm.num_updates = 0
    algorithm.learning_rate = 1.0e-4
    algorithm.optimizer = SimpleNamespace(param_groups=[{"lr": 5.0e-4}])
    with patch.object(Distillation, "load", return_value=True):
        algorithm.load({"student_state_dict": {}, "optimizer_state_dict": {}, "iter": 275}, None, True)

    assert algorithm.optimizer.param_groups[0]["lr"] == 1.0e-4


def test_guided_rollout_blends_teacher_and_student_actions_without_switching():
    algorithm = GuidedDistillation.__new__(GuidedDistillation)
    algorithm.num_updates = 0
    algorithm.teacher_fraction_start = 0.5
    algorithm.teacher_fraction_end = 0.5
    algorithm.teacher_fraction_decay_updates = 1
    algorithm.student = lambda _obs, stochastic_output: torch.tensor([[1.0], [1.0]])
    algorithm.teacher = lambda _obs: torch.tensor([[2.0], [2.0]])
    algorithm.transition = SimpleNamespace()

    first_actions = algorithm.act(SimpleNamespace())
    second_actions = algorithm.act(SimpleNamespace())

    torch.testing.assert_close(first_actions, torch.tensor([[1.5], [1.5]]))
    torch.testing.assert_close(second_actions, first_actions)
