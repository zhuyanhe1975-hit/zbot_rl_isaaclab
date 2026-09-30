#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportOptionalMemberAccess=false
"""Evaluate Base COM and ankle-load phase tracking at fixed frequencies."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import sys
from pathlib import Path

import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import add_frontend_args, create_isaaclab_env
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

from zbot_rl_isaaclab.tasks.velocity.mdp.observations import whole_body_center_of_mass
from zbot_rl_isaaclab.tasks.velocity.mdp.rewards import (
    signed_support_distance_contrast,
    signed_support_force_contrast,
)

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="ZbotRlIsaaclab-6DOF-Base")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--steps", type=int, default=2_000)
parser.add_argument("--warmup-steps", type=int, default=100)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *remaining_args]


def _correlation(
    count: torch.Tensor,
    x_sum: torch.Tensor,
    y_sum: torch.Tensor,
    xx_sum: torch.Tensor,
    yy_sum: torch.Tensor,
    xy_sum: torch.Tensor,
) -> torch.Tensor:
    numerator = count * xy_sum - x_sum * y_sum
    denominator = torch.sqrt(
        (count * xx_sum - x_sum.square()) * (count * yy_sum - y_sum.square())
    ).clamp_min(1.0e-8)
    return numerator / denominator


@hydra_task_config(args_cli.task, args_cli.agent, play_mode=True)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    if args_cli.steps <= args_cli.warmup_steps:
        raise ValueError("Evaluation steps must exceed warmup steps.")
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required.")
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env_cfg.scene.num_envs = 4
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.commands.weight_shift.minimum_frequency = 0.5
    env_cfg.commands.weight_shift.maximum_frequency = 2.0
    env_cfg.commands.weight_shift.ordered = True
    env_cfg.commands.weight_shift.resampling_time_range = (1.0e9, 1.0e9)

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    with launch_simulation(env_cfg, args_cli):
        base_env = create_isaaclab_env(
            args_cli.task,
            env_cfg,
            args_cli,
            convert_marl_to_single_agent=False,
        )
        manager_env = base_env.unwrapped
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=manager_env.device)

        distance_cfg = manager_env.reward_manager.get_term_cfg("support_distance")
        force_cfg = manager_env.reward_manager.get_term_cfg("support_force")
        asset_cfg = distance_cfg.params["asset_cfg"]
        sensor_cfg = force_cfg.params["sensor_cfg"]
        asset = manager_env.scene[asset_cfg.name]
        sensor = manager_env.scene.sensors[sensor_cfg.name]

        obs = env.get_observations()
        shape = (manager_env.num_envs,)
        count = torch.zeros(shape, device=manager_env.device)
        com_sum = torch.zeros(shape, device=manager_env.device)
        force_sum = torch.zeros(shape, device=manager_env.device)
        target_sum = torch.zeros(shape, device=manager_env.device)
        com_sq_sum = torch.zeros(shape, device=manager_env.device)
        force_sq_sum = torch.zeros(shape, device=manager_env.device)
        target_sq_sum = torch.zeros(shape, device=manager_env.device)
        com_target_sum = torch.zeros(shape, device=manager_env.device)
        force_target_sum = torch.zeros(shape, device=manager_env.device)
        com_mae_sum = torch.zeros(shape, device=manager_env.device)
        force_mae_sum = torch.zeros(shape, device=manager_env.device)
        com_score_sum = torch.zeros(shape, device=manager_env.device)
        force_score_sum = torch.zeros(shape, device=manager_env.device)
        com_sign_sum = torch.zeros(shape, device=manager_env.device)
        force_sign_sum = torch.zeros(shape, device=manager_env.device)
        sign_count = torch.zeros(shape, device=manager_env.device)
        done_count = torch.zeros(shape, device=manager_env.device)

        with torch.inference_mode():
            for step in range(args_cli.steps):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)

                if step < args_cli.warmup_steps:
                    continue
                command = manager_env.command_manager.get_command("weight_shift")
                target = command[:, 2]
                center_of_mass_w = whole_body_center_of_mass(
                    asset.data.body_com_pos_w.torch,
                    asset.data.body_mass.torch,
                )
                feet_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
                distances = torch.linalg.vector_norm(center_of_mass_w[:, None, :2] - feet_w[:, :, :2], dim=2)
                com = signed_support_distance_contrast(distances)
                ankle_forces = sensor.data.force.torch[:, sensor_cfg.body_ids].norm(dim=-1)
                force = signed_support_force_contrast(ankle_forces, force_cfg.params["minimum_total_force"])
                com_target = distance_cfg.params["target_amplitude"] * target
                force_target = force_cfg.params["target_amplitude"] * target
                com_score = distance_cfg.func(manager_env, **distance_cfg.params)
                force_score = force_cfg.func(manager_env, **force_cfg.params)

                count += 1.0
                com_sum += com
                force_sum += force
                target_sum += target
                com_sq_sum += com.square()
                force_sq_sum += force.square()
                target_sq_sum += target.square()
                com_target_sum += com * target
                force_target_sum += force * target
                com_mae_sum += torch.abs(com - com_target)
                force_mae_sum += torch.abs(force - force_target)
                com_score_sum += com_score
                force_score_sum += force_score
                active_target = torch.abs(target) >= 0.2
                com_sign_sum += (torch.sign(com) == torch.sign(target)).float() * active_target
                force_sign_sum += (torch.sign(force) == torch.sign(target)).float() * active_target
                sign_count += active_target
                done_count += dones.float()

        frequencies = manager_env.command_manager.get_command("weight_shift")[:, 0]
        com_corr = _correlation(count, com_sum, target_sum, com_sq_sum, target_sq_sum, com_target_sum)
        force_corr = _correlation(count, force_sum, target_sum, force_sq_sum, target_sq_sum, force_target_sum)
        print("frequency_hz,com_score,force_score,com_mae,force_mae,com_corr,force_corr,com_sign,force_sign,mean_episode_steps")
        for index in range(manager_env.num_envs):
            episodes = done_count[index].clamp_min(1.0)
            print(
                f"{frequencies[index].item():.3f},"
                f"{(com_score_sum[index] / count[index]).item():.4f},"
                f"{(force_score_sum[index] / count[index]).item():.4f},"
                f"{(com_mae_sum[index] / count[index]).item():.4f},"
                f"{(force_mae_sum[index] / count[index]).item():.4f},"
                f"{com_corr[index].item():.4f},"
                f"{force_corr[index].item():.4f},"
                f"{(com_sign_sum[index] / sign_count[index].clamp_min(1.0)).item():.4f},"
                f"{(force_sign_sum[index] / sign_count[index].clamp_min(1.0)).item():.4f},"
                f"{(count[index] / episodes).item():.1f}"
            )
        env.close()


if __name__ == "__main__":
    main()
