# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import math
from types import SimpleNamespace
from typing import Any, cast

import torch

import isaaclab.envs.mdp as base_mdp

from isaaclab_tasks.utils.hydra import resolve_task_config

from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_frequency_command.env_cfg import (
    FREQUENCY_COMMAND_OBSERVATION_DIM,
    FrequencyCommandEnvCfg,
)
from zbot_rl_isaaclab.tasks.velocity.mdp.commands import (
    UniformStepFrequencyCommand,
    advance_frequency_phase,
    curriculum_maximum_frequency,
    frequency_metric_label,
    frequency_phase_features,
    ordered_step_frequencies,
    support_side_from_phase,
)
from zbot_rl_isaaclab.tasks.velocity.mdp.commands_cfg import UniformStepFrequencyCommandCfg
from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import (
    commanded_step_frequency_relative_error_l2,
    commanded_step_frequency_score,
    commanded_support_force_contrast,
)


def test_frequency_command_task_exposes_command_to_policy_and_reward():
    cfg = FrequencyCommandEnvCfg()

    assert cfg.commands.step_frequency.minimum_frequency == 0.5
    assert cfg.commands.step_frequency.maximum_frequency == 2.0
    assert cfg.commands.step_frequency.resampling_time_range == (8.0, 12.0)
    assert not cfg.commands.step_frequency.ordered
    assert cfg.observations.policy.step_frequency_command.func is base_mdp.generated_commands
    assert cfg.observations.policy.step_frequency_command.params["command_name"] == "step_frequency"
    assert FREQUENCY_COMMAND_OBSERVATION_DIM == 46
    assert cfg.rewards.alternating_touchdown.params["minimum_frequency"] == 0.5
    assert cfg.rewards.alternating_touchdown.params["maximum_frequency"] == 2.0
    assert cfg.rewards.alternating_touchdown.params["frequency_command_name"] == "step_frequency"

    assert cfg.rewards.stage_one_step_frequency.weight == 0.0
    assert cfg.rewards.step_frequency_error_l2.weight == 0.0
    assert cfg.rewards.step_frequency_excess_l2.weight == 0.0
    assert cfg.rewards.commanded_step_frequency.func is commanded_step_frequency_score
    assert cfg.rewards.commanded_step_frequency.weight == 200.0
    assert cfg.rewards.commanded_step_frequency.params == {
        "reward_term_name": "alternating_touchdown",
        "command_name": "step_frequency",
        "tolerance": 0.15,
    }
    assert cfg.rewards.commanded_step_frequency_error_l2.func is commanded_step_frequency_relative_error_l2
    assert cfg.rewards.commanded_step_frequency_error_l2.weight == -100.0
    assert cfg.rewards.commanded_support_force.func is commanded_support_force_contrast
    assert cfg.rewards.commanded_support_force.weight == 2.0
    assert cfg.rewards.commanded_support_force.params["command_index"] == 1


def test_frequency_command_task_preserves_walking_dynamics_and_safety_settings():
    cfg = FrequencyCommandEnvCfg()

    assert cfg.actions.joint_pos.velocity_limit_range[0] == cfg.actions.joint_pos.velocity_limit_range[1]
    assert cfg.curriculum is None
    assert cfg.rewards.body_forward_speed.weight == 5.0
    assert cfg.rewards.body_forward_speed.params["target_speed"] == 0.15
    assert cfg.rewards.alive.weight == 1.0
    assert cfg.rewards.termination_penalty.weight == -200.0
    assert cfg.rewards.step_length.weight == 100.0
    assert cfg.rewards.yaw_rate_l2.weight == -0.05
    assert cfg.rewards.ang_vel_xy_l2.weight == -0.02
    assert cfg.rewards.step_length_asymmetry.weight == -50.0
    assert cfg.rewards.single_support_foot_height_l2.weight == -1.0
    assert cfg.rewards.alternating_touchdown.params["minimum_air_time"] == 0.15
    assert cfg.rewards.step_length.params["minimum_air_time"] == 0.15
    assert cfg.rewards.step_length.params["crossing_margin"] == 0.01
    assert cfg.rewards.commanded_swing_height.weight == 5.0
    assert cfg.rewards.commanded_swing_height.params["target_height"] == 0.03
    assert cfg.rewards.commanded_swing_forward.weight == 5.0
    assert cfg.rewards.commanded_swing_forward.params["target_distance"] == 0.05
    assert cfg.terminations.heading_deviation.params["maximum_deviation"] == math.pi / 4.0
    assert cfg.terminations.heading_deviation.params["target_heading"] == 0.0
    assert cfg.terminations.heading_deviation.params["asset_cfg"].body_names == "base"


def test_frequency_command_bounds_can_be_overridden_from_the_cli():
    cfg, _ = resolve_task_config(
        "ZbotRlIsaaclab-Velocity-Zbot-6DOF-FrequencyCommand",
        "rsl_rl_cfg_entry_point",
        play_mode=True,
        overrides=(
            "env.commands.step_frequency.minimum_frequency=1.5",
            "env.commands.step_frequency.maximum_frequency=1.5",
        ),
    )

    assert cfg.commands.step_frequency.minimum_frequency == 1.5
    assert cfg.commands.step_frequency.maximum_frequency == 1.5
    assert cfg.commands.step_frequency.ordered
    assert cfg.commands.step_frequency.resampling_time_range == (1.0e9, 1.0e9)


def test_play_frequency_commands_are_ordered_by_environment_index():
    frequencies = ordered_step_frequencies(4, 0.5, 2.0)

    torch.testing.assert_close(frequencies, torch.tensor([0.5, 1.0, 1.5, 2.0]))


def test_frequency_curriculum_expands_from_slowest_to_full_range():
    assert curriculum_maximum_frequency(0.5, 2.0, 0.5, step=0, curriculum_steps=24_000) == 0.5
    assert curriculum_maximum_frequency(0.5, 2.0, 0.5, step=12_000, curriculum_steps=24_000) == 1.25
    assert curriculum_maximum_frequency(0.5, 2.0, 0.5, step=24_000, curriculum_steps=24_000) == 2.0
    assert curriculum_maximum_frequency(0.5, 2.0, 0.5, step=48_000, curriculum_steps=24_000) == 2.0


def test_training_frequency_resampling_writes_back_to_selected_environments():
    command = object.__new__(UniformStepFrequencyCommand)
    command.cfg = UniformStepFrequencyCommandCfg(
        ordered=False,
        minimum_frequency=0.5,
        maximum_frequency=2.0,
    )
    command._frequency = torch.zeros(4, 1)
    command._phase = torch.zeros(4)
    command._target_side = torch.ones(4, 1)
    command._ordered_frequency = ordered_step_frequencies(4, 0.5, 2.0)

    command._resample_command([1, 3])

    torch.testing.assert_close(command._frequency[[0, 2]], torch.zeros(2, 1))
    assert torch.all(command._frequency[[1, 3]] >= 0.5)
    assert torch.all(command._frequency[[1, 3]] <= 2.0)
    assert torch.all(command._frequency[[1, 3]] > 0.0)
    assert command.command.shape == (4, 2)

    command.cfg.include_phase_features = True
    phase_command = command.command
    assert phase_command.shape == (4, 4)
    torch.testing.assert_close(phase_command[:, 2:], torch.tensor([[0.0, 1.0]]).expand(4, -1))


def test_endpoint_sampling_rehearses_exact_frequency_limits():
    command = object.__new__(UniformStepFrequencyCommand)
    command.cfg = UniformStepFrequencyCommandCfg(
        ordered=False,
        minimum_frequency=0.5,
        maximum_frequency=2.0,
        endpoint_probability=0.5,
    )
    command._frequency = torch.zeros(1024, 1)
    command._phase = torch.zeros(1024)
    command._target_side = torch.ones(1024, 1)

    command._resample_command(torch.arange(1024))

    frequencies = command._frequency[:, 0]
    assert torch.all((frequencies == 0.5) | (frequencies == 2.0))
    assert torch.any(frequencies == 0.5)
    assert torch.any(frequencies == 2.0)


def test_frequency_phase_switches_support_side_at_commanded_step_rate():
    phase = torch.tensor([0.0, 0.0])
    frequency = torch.tensor([0.5, 2.0])

    advanced = advance_frequency_phase(phase, frequency, step_dt=0.5)

    torch.testing.assert_close(advanced, torch.tensor([0.125, 0.5]))
    torch.testing.assert_close(support_side_from_phase(advanced), torch.tensor([1.0, -1.0]))


def test_base_frequency_can_represent_complete_cycles_per_second():
    phase = torch.tensor([0.0, 0.0])
    frequency = torch.tensor([0.5, 1.0])

    advanced = advance_frequency_phase(
        phase,
        frequency,
        step_dt=0.5,
        frequency_is_full_cycle=True,
    )

    torch.testing.assert_close(advanced, torch.tensor([0.25, 0.5]))
    torch.testing.assert_close(support_side_from_phase(advanced), torch.tensor([1.0, -1.0]))


def test_frequency_phase_features_are_continuous_and_periodic():
    phase = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

    features = frequency_phase_features(phase)

    torch.testing.assert_close(
        features,
        torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, -1.0],
                [-1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_frequency_metrics_are_accumulated_by_nearest_command_bin():
    reward_term = SimpleNamespace(
        measured_frequency=torch.tensor([0.6, 1.1, 1.4, 1.9]),
        has_frequency_event=torch.ones(4, dtype=torch.bool),
    )
    reward_manager = SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(func=reward_term))
    command = object.__new__(UniformStepFrequencyCommand)
    command.cfg = UniformStepFrequencyCommandCfg()
    command._rl_env = cast(Any, SimpleNamespace(reward_manager=reward_manager))
    command._frequency = torch.tensor([[0.5], [1.0], [1.5], [2.0]])
    command._frequency_sum = torch.zeros(4)
    command._step_count = torch.zeros(4)
    command._evaluation_frequencies = torch.tensor([0.5, 1.0, 1.5, 2.0])
    command._measured_frequency_sum = torch.zeros(4, 4)
    command._signed_frequency_error_sum = torch.zeros(4, 4)
    command._absolute_frequency_error_sum = torch.zeros(4, 4)
    command._frequency_measurement_count = torch.zeros(4, 4)

    command._update_metrics()

    torch.testing.assert_close(command._measured_frequency_sum.diagonal(), reward_term.measured_frequency)
    torch.testing.assert_close(command._absolute_frequency_error_sum.diagonal(), torch.full((4,), 0.1))

    command.cfg.frequency_is_full_cycle = True
    reward_term.measured_frequency *= 2.0
    command._measured_frequency_sum.zero_()
    command._signed_frequency_error_sum.zero_()
    command._absolute_frequency_error_sum.zero_()
    command._frequency_measurement_count.zero_()
    command._update_metrics()

    torch.testing.assert_close(command._measured_frequency_sum.diagonal(), reward_term.measured_frequency / 2.0)
    torch.testing.assert_close(command._absolute_frequency_error_sum.diagonal(), torch.full((4,), 0.1))
    torch.testing.assert_close(command._frequency_measurement_count.diagonal(), torch.ones(4))
    assert frequency_metric_label(0.5) == "0p5"


def test_frequency_metrics_ignore_environments_without_touchdown_events():
    reward_term = SimpleNamespace(
        measured_frequency=torch.tensor([0.6, 50.0, 1.4, 50.0]),
        has_frequency_event=torch.tensor([True, False, True, False]),
    )
    command = object.__new__(UniformStepFrequencyCommand)
    command.cfg = UniformStepFrequencyCommandCfg()
    command._rl_env = cast(
        Any,
        SimpleNamespace(reward_manager=SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(func=reward_term))),
    )
    command._frequency = torch.tensor([[0.5], [1.0], [1.5], [2.0]])
    command._frequency_sum = torch.zeros(4)
    command._step_count = torch.zeros(4)
    command._evaluation_frequencies = torch.tensor([0.5, 1.0, 1.5, 2.0])
    command._measured_frequency_sum = torch.zeros(4, 4)
    command._signed_frequency_error_sum = torch.zeros(4, 4)
    command._absolute_frequency_error_sum = torch.zeros(4, 4)
    command._frequency_measurement_count = torch.zeros(4, 4)

    command._update_metrics()

    torch.testing.assert_close(command._measured_frequency_sum.sum(dim=0), torch.tensor([0.6, 0.0, 1.4, 0.0]))
    torch.testing.assert_close(command._frequency_measurement_count.sum(dim=0), torch.tensor([1.0, 0.0, 1.0, 0.0]))
