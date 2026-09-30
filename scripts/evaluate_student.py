#!/usr/bin/env python3
"""Evaluate a periodic walking teacher or student at four fixed frequencies."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import zbot_rl_isaaclab.tasks  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="ZbotRlIsaaclab-6DOF-Periodic-Walking-Student")
parser.add_argument("--agent", default="rsl_rl_distillation_cfg_entry_point")
parser.add_argument("--steps", type=int, default=1_000)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument(
    "--fixed-frequency", type=float, help="Evaluate all environments at one commanded gait frequency [Hz]."
)
parser.add_argument(
    "--disturbances", action="store_true", help="Enable robust-stage randomization and force pulses."
)
parser.add_argument("--show-force-arrows", action="store_true", help="Draw the applied force pulses during playback.")
parser.add_argument("--force-mean-interval", type=float, help="Override mean time between force-pulse starts [s].")
parser.add_argument("--force-duration", type=float, help="Override force-pulse duration [s].")
parser.add_argument("--force-minimum-delta-velocity", type=float, help="Minimum equivalent velocity kick [m/s].")
parser.add_argument("--force-maximum-delta-velocity", type=float, help="Maximum equivalent velocity kick [m/s].")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *remaining_args]


@hydra_task_config(args_cli.task, args_cli.agent, play_mode=True)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    if args_cli.checkpoint is None or args_cli.steps <= 0 or args_cli.num_envs <= 0:
        raise ValueError("Provide --checkpoint, positive --steps, and positive --num_envs.")
    if args_cli.fixed_frequency is not None and not 0.5 <= args_cli.fixed_frequency <= 2.0:
        raise ValueError("Fixed evaluation frequency must be between 0.5 and 2.0 Hz.")
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.seed = agent_cfg.seed
    frequency_minimum = args_cli.fixed_frequency if args_cli.fixed_frequency is not None else 0.5
    frequency_maximum = args_cli.fixed_frequency if args_cli.fixed_frequency is not None else 2.0
    env_cfg.commands.weight_shift.minimum_frequency = frequency_minimum
    env_cfg.commands.weight_shift.maximum_frequency = frequency_maximum
    env_cfg.commands.weight_shift.ordered = True
    env_cfg.commands.weight_shift.resampling_time_range = (1.0e9, 1.0e9)
    if args_cli.disturbances:
        from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student_finetune.env_cfg import (
            RobustEventsCfg,
        )

        env_cfg.events = RobustEventsCfg()
        env_cfg.events.external_force_pulse.params["ramp_steps"] = 0
        env_cfg.events.external_force_pulse.params["visualize_force"] = args_cli.show_force_arrows
        if args_cli.force_mean_interval is not None:
            env_cfg.events.external_force_pulse.params["mean_interval_s"] = args_cli.force_mean_interval
        if args_cli.force_duration is not None:
            env_cfg.events.external_force_pulse.params["duration_s"] = args_cli.force_duration
        if args_cli.force_minimum_delta_velocity is not None:
            env_cfg.events.external_force_pulse.params["minimum_delta_velocity"] = args_cli.force_minimum_delta_velocity
        if args_cli.force_maximum_delta_velocity is not None:
            env_cfg.events.external_force_pulse.params["maximum_delta_velocity"] = args_cli.force_maximum_delta_velocity
    elif args_cli.show_force_arrows:
        raise ValueError("Force arrows require --disturbances.")
    elif any(
        value is not None
        for value in (
            args_cli.force_mean_interval,
            args_cli.force_duration,
            args_cli.force_minimum_delta_velocity,
            args_cli.force_maximum_delta_velocity,
        )
    ):
        raise ValueError("Force-pulse overrides require --disturbances.")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))

    with launch_simulation(env_cfg, args_cli):
        base_env = cast(
            "ManagerBasedRLEnv",
            create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False),
        )
        manager_env = cast("ManagerBasedRLEnv", base_env.unwrapped)
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=manager_env.device)
        robot = manager_env.scene["robot"]
        base_index = robot.body_names.index("base")
        foot_body_ids = [robot.body_names.index(name) for name in ("foot_0", "foot_1")]
        contact_sensor = cast("ContactSensor", manager_env.scene.sensors["contact_forces"])
        assert contact_sensor.body_names is not None
        foot_sensor_ids = [contact_sensor.body_names.index(name) for name in ("foot_0", "foot_1")]
        heading_term = manager_env.reward_manager.get_term_cfg("heading_error_l2")
        cadence_term = manager_env.reward_manager.get_term_cfg("alternating_touchdown").func
        frequencies = manager_env.command_manager.get_command("weight_shift")[:, 0].cpu().tolist()
        num_envs = args_cli.num_envs
        episode_steps = torch.zeros(num_envs, device=manager_env.device)
        completed_steps = torch.zeros(num_envs, device=manager_env.device)
        completed_episodes = torch.zeros(num_envs, device=manager_env.device)
        forward_distance = torch.zeros(num_envs, device=manager_env.device)
        swing_heights = torch.full((args_cli.steps, num_envs), float("nan"), device=manager_env.device)
        impact_forces = torch.full((args_cli.steps, num_envs, 2), float("nan"), device=manager_env.device)
        step_lengths = torch.full((args_cli.steps, num_envs), float("nan"), device=manager_env.device)
        heading_errors = torch.zeros((args_cli.steps, num_envs), device=manager_env.device)
        cadence_errors = torch.full((args_cli.steps, num_envs), float("nan"), device=manager_env.device)
        touchdown_intervals = torch.full((args_cli.steps, num_envs), float("nan"), device=manager_env.device)
        previous_touchdown_pos_w = robot.data.body_pos_w.torch[:, foot_body_ids].clone()
        obs = env.get_observations()

        with torch.inference_mode():
            for step in range(args_cli.steps):
                x_before = robot.data.body_pos_w.torch[:, base_index, 0].clone()
                obs, _, dones, _ = env.step(policy(obs))
                force_history = contact_sensor.data.net_normal_forces_w_history
                last_air_time = contact_sensor.data.last_air_time
                if force_history is None or last_air_time is None:
                    raise RuntimeError("Evaluation requires foot force history and air-time tracking.")
                peak_forces = force_history.torch[:, :, foot_sensor_ids, :].norm(dim=-1).max(dim=1)[0]
                foot_positions_w = robot.data.body_pos_w.torch[:, foot_body_ids]
                foot_heights = foot_positions_w[:, :, 2]
                single_support = (peak_forces >= 10.0).sum(dim=1) == 1
                swing_heights[step] = torch.where(
                    single_support, (foot_heights[:, 0] - foot_heights[:, 1]).abs(), float("nan")
                )
                first_contact = contact_sensor.compute_first_contact(manager_env.step_dt).torch[:, foot_sensor_ids]
                first_contact = first_contact.bool()
                valid_touchdown = first_contact & (last_air_time.torch[:, foot_sensor_ids] >= 0.05)
                impact_forces[step] = torch.where(valid_touchdown, peak_forces, float("nan"))
                valid_touchdown &= peak_forces >= 10.0
                valid_touchdown &= (valid_touchdown.int().sum(dim=1) == 1).unsqueeze(1)
                root_quat_w = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, 2, -1)
                displacement = foot_positions_w - previous_touchdown_pos_w
                forward_length = quat_apply_inverse(root_quat_w, displacement)[..., 0].clamp_min(0.0)
                landing_length = (forward_length * valid_touchdown).sum(dim=1)
                step_lengths[step] = torch.where(valid_touchdown.any(dim=1), landing_length, float("nan"))
                previous_touchdown_pos_w.copy_(
                    torch.where(valid_touchdown.unsqueeze(-1), foot_positions_w, previous_touchdown_pos_w)
                )
                heading_error_l2 = cast(torch.Tensor, heading_term.func(manager_env, **heading_term.params))
                heading_errors[step] = torch.sqrt(heading_error_l2)
                target_frequency = manager_env.command_manager.get_command("weight_shift")[:, 0]
                cadence_errors[step] = torch.where(
                    cadence_term.has_frequency_event,
                    (cadence_term.measured_frequency * 0.5 - target_frequency).abs(),
                    float("nan"),
                )
                touchdown_intervals[step] = torch.where(
                    cadence_term.has_frequency_event,
                    cadence_term.frequency_event_interval,
                    float("nan"),
                )
                done_mask = dones.bool()
                policy.reset(done_mask)
                x_after = robot.data.body_pos_w.torch[:, base_index, 0]
                forward_distance += torch.where(done_mask, 0.0, x_after - x_before)
                episode_steps += 1
                completed_steps += torch.where(done_mask, episode_steps, 0.0)
                completed_episodes += done_mask.float()
                episode_steps[done_mask] = 0
                previous_touchdown_pos_w[done_mask] = robot.data.body_pos_w.torch[done_mask][:, foot_body_ids]

        def finite_percentile(values: torch.Tensor, quantile: float) -> float:
            values = values[torch.isfinite(values)]
            return float(torch.quantile(values, quantile).item()) if values.numel() else float("nan")

        def finite_mean(values: torch.Tensor) -> float:
            values = values[torch.isfinite(values)]
            return float(values.mean().item()) if values.numel() else float("nan")

        if args_cli.disturbances:
            pulse_term = manager_env.event_manager.get_term_cfg("external_force_pulse").func
            print(f"external_force_pulses={pulse_term.pulse_count.cpu().tolist()}")

        print(
            "frequency_hz,completed_episodes,mean_observed_episode_steps,forward_distance_m,"
            "p90_swing_height_m,p90_touchdown_force_n,p90_step_length_m,mean_abs_heading_rad,"
            "mean_abs_frequency_error_hz,median_abs_frequency_error_hz,frequency_events,"
            "p10_touchdown_interval_s,median_touchdown_interval_s,p90_touchdown_interval_s,"
            "fraction_intervals_below_half_target"
        )
        for index, frequency in enumerate(frequencies):
            mean_steps = (completed_steps[index] + episode_steps[index]) / (
                completed_episodes[index] + (episode_steps[index] > 0).float()
            ).clamp_min(1)
            intervals = touchdown_intervals[:, index]
            valid_intervals = intervals[torch.isfinite(intervals)]
            short_fraction = (
                float((valid_intervals < 0.25 / frequency).float().mean().item())
                if valid_intervals.numel()
                else float("nan")
            )
            print(
                f"{frequency:.2f},{int(completed_episodes[index].item())},"
                f"{mean_steps.item():.1f},{forward_distance[index].item():.3f},"
                f"{finite_percentile(swing_heights[:, index], 0.9):.3f},"
                f"{finite_percentile(impact_forces[:, index], 0.9):.1f},"
                f"{finite_percentile(step_lengths[:, index], 0.9):.3f},"
                f"{heading_errors[:, index].mean().item():.3f},"
                f"{finite_mean(cadence_errors[:, index]):.3f},"
                f"{finite_percentile(cadence_errors[:, index], 0.5):.3f},"
                f"{torch.isfinite(cadence_errors[:, index]).sum().item()},"
                f"{finite_percentile(intervals, 0.1):.3f},"
                f"{finite_percentile(intervals, 0.5):.3f},"
                f"{finite_percentile(intervals, 0.9):.3f},{short_fraction:.3f}"
            )
        env.close()


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
