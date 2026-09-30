from types import SimpleNamespace
from unittest.mock import Mock

import torch

from zbot_rl_isaaclab.student_transfer import load_student_actor_prior
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student.agents.rsl_rl_distillation_cfg import (
    DistillationRunnerCfg,
)
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student_finetune.agents.rsl_rl_ppo_cfg import (
    PPORunnerCfg,
    RobustRunnerCfg,
)
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student_finetune.env_cfg import (
    StudentFinetuneEnvCfg,
    StudentRobustEnvCfg,
)


def test_ppo_actor_is_sensor_only_and_nominal_training_has_no_randomization():
    cfg = StudentFinetuneEnvCfg()

    assert hasattr(cfg.observations, "teacher")
    assert not hasattr(cfg.events, "robot_material")
    assert cfg.commands.weight_shift.curriculum_initial_maximum_frequency is None
    assert cfg.commands.weight_shift.frequency_reward_term_name == "alternating_touchdown"
    assert cfg.rewards.commanded_step_frequency.params["target_multiplier"] == 2.0
    assert cfg.rewards.commanded_step_frequency.weight > 0.0
    assert cfg.rewards.support_distance.weight > 0.0
    assert cfg.rewards.support_force.weight > 0.0
    assert cfg.rewards.touchdown_impact.weight < 0.0
    assert cfg.rewards.body_forward_speed.weight == 0.0
    assert cfg.rewards.single_support_foot_height_l2.params["maximum_height_difference"] == 0.04
    assert cfg.rewards.joint_torques_l2.weight == -1.0e-3
    assert cfg.rewards.step_length.weight == 100.0
    assert cfg.rewards.step_length.params["maximum_rewarded_step_length"] == 0.35


def test_robust_stage_randomizes_only_during_training():
    cfg = StudentRobustEnvCfg()

    assert cfg.events.robot_material.mode == "startup"
    assert cfg.events.base_mass.mode == "startup"
    assert cfg.events.actuator_gains.mode == "startup"
    assert cfg.events.external_force_pulse.mode == "interval"
    assert cfg.events.external_force_pulse.params["duration_s"] == 0.16
    assert cfg.events.external_force_pulse.params["maximum_delta_velocity"] == 0.2
    assert cfg.events.external_force_pulse.params["minimum_delta_velocity"] == 0.0
    assert cfg.events.external_force_pulse.params["ramp_steps"] == 10_000
    assert cfg.events.external_force_pulse.params["ramp_offset_steps"] == 0
    assert not cfg.events.external_force_pulse.params["visualize_force"]

    cfg.play_mode()
    assert not hasattr(cfg.events, "robot_material")
    assert not hasattr(cfg.events, "external_force_pulse")
    assert cfg.commands.weight_shift.ordered


def test_finetune_actor_matches_distilled_student_architecture():
    distilled = DistillationRunnerCfg()
    finetune = PPORunnerCfg()

    assert finetune.obs_groups == {"actor": ["policy"], "critic": ["teacher"]}
    assert finetune.actor.class_name == distilled.student.class_name == "RNNModel"
    assert finetune.actor.rnn_hidden_dim == distilled.student.rnn_hidden_dim
    assert finetune.actor.hidden_dims == distilled.student.hidden_dims
    assert finetune.max_iterations == 100
    assert finetune.save_interval == 25
    assert finetune.algorithm.learning_rate == 1.0e-5
    assert finetune.algorithm.num_learning_epochs == 2
    assert finetune.algorithm.entropy_coef == 0.0
    assert finetune.algorithm.clip_param == 0.1
    assert finetune.algorithm.class_name == "zbot_rl_isaaclab.student_ppo:AnchoredPPO"
    assert finetune.algorithm.max_relative_drift == 0.02
    assert RobustRunnerCfg().experiment_name != finetune.experiment_name


def test_actor_only_student_transfer_leaves_other_runner_state_fresh(tmp_path):
    actor = torch.nn.Linear(2, 1)
    critic = torch.nn.Linear(2, 1)
    before_critic = {name: tensor.clone() for name, tensor in critic.state_dict().items()}
    state = {name: torch.full_like(tensor, 0.5) for name, tensor in actor.state_dict().items()}
    path = tmp_path / "student.pt"
    torch.save({"student_state_dict": state, "teacher_state_dict": {"unused": 1}, "iter": 99}, path)

    set_reference_actor = Mock()
    runner = SimpleNamespace(
        alg=SimpleNamespace(get_policy=lambda: actor, set_reference_actor=set_reference_actor), critic=critic
    )
    assert load_student_actor_prior(runner, path) == 99
    set_reference_actor.assert_called_once_with()
    assert all(torch.equal(tensor, state[name]) for name, tensor in actor.state_dict().items())
    assert all(torch.equal(tensor, before_critic[name]) for name, tensor in critic.state_dict().items())

    ppo_prior = tmp_path / "ppo.pt"
    torch.save({"actor_state_dict": state, "iter": 100}, ppo_prior)
    assert load_student_actor_prior(runner, ppo_prior) == 100


def test_finetune_can_reduce_exploration_without_changing_actor_weights(tmp_path):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 1)
            self.distribution = torch.nn.Module()
            self.distribution.std_param = torch.nn.Parameter(torch.full((1,), 0.1))

    actor = Policy()
    original_layer = {name: value.clone() for name, value in actor.layer.state_dict().items()}
    checkpoint = tmp_path / "student.pt"
    torch.save({"student_state_dict": actor.state_dict(), "iter": 200}, checkpoint)
    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: actor))

    assert load_student_actor_prior(runner, checkpoint, initial_std=0.03) == 200
    torch.testing.assert_close(actor.distribution.std_param, torch.full((1,), 0.03))
    for name, value in actor.layer.state_dict().items():
        torch.testing.assert_close(value, original_layer[name])

    assert load_student_actor_prior(runner, checkpoint, initial_std=0.015, freeze_std=True) == 200
    torch.testing.assert_close(actor.distribution.std_param, torch.full((1,), 0.015))
    assert not actor.distribution.std_param.requires_grad
