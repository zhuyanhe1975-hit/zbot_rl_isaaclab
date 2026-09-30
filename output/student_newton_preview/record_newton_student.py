"""Record a fixed-frequency student rollout from Newton GL without MoviePy."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation

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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="ZbotRlIsaaclab-6DOF-Periodic-Walking-Student")
parser.add_argument("--agent", default="rsl_rl_distillation_cfg_entry_point")
parser.add_argument("--output", required=True)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--frequency", type=float)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
add_frontend_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *remaining_args]


@hydra_task_config(args_cli.task, args_cli.agent, play_mode=True)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    checkpoint = Path(args_cli.checkpoint).resolve()
    output = Path(args_cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    env_cfg.scene.num_envs = 1 if args_cli.frequency is not None else 4
    env_cfg.seed = agent_cfg.seed
    env_cfg.commands.weight_shift.minimum_frequency = args_cli.frequency or 0.5
    env_cfg.commands.weight_shift.maximum_frequency = args_cli.frequency or 2.0
    env_cfg.commands.weight_shift.ordered = True
    env_cfg.commands.weight_shift.resampling_time_range = (1.0e9, 1.0e9)
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
        visualizer = next(v for v in manager_env.sim.visualizers if v.cfg.visualizer_type == "newton_gl")
        robot = manager_env.scene["robot"]
        base_index = robot.body_names.index("base")
        frequencies = manager_env.command_manager.get_command("weight_shift")[:, 0].cpu().tolist()
        print(f"Frequencies by environment: {frequencies}", flush=True)
        obs = env.get_observations()
        encoder = None
        resets = [0] * env_cfg.scene.num_envs
        captured = 0
        try:
            import torch

            with torch.inference_mode():
                for step in range(args_cli.steps):
                    obs, _, dones, _ = env.step(policy(obs))
                    policy.reset(dones)
                    for index, done in enumerate(dones.cpu().tolist()):
                        resets[index] += int(done)
                    if step % 2:
                        continue
                    if args_cli.frequency is not None:
                        base = robot.data.body_pos_w.torch[0, base_index].cpu().tolist()
                        target = (base[0], base[1], base[2])
                        eye = (base[0] + 1.1, base[1] - 1.35, base[2] + 0.65)
                        visualizer.set_camera_view(eye, target)
                    frame = np.asarray(visualizer.render_rgb_array())[:, :, :3]
                    frame = np.ascontiguousarray(frame, dtype=np.uint8)
                    if encoder is None:
                        height, width = frame.shape[:2]
                        encoder = subprocess.Popen(
                            [
                                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                                "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                                "-r", "25", "-i", "-", "-an", "-c:v", "libx264",
                                "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
                                str(output),
                            ],
                            stdin=subprocess.PIPE,
                        )
                        print(f"Capturing {width}x{height} Newton frames", flush=True)
                    assert encoder.stdin is not None
                    encoder.stdin.write(frame.tobytes())
                    captured += 1
                    if step % 100 == 0:
                        print(f"step={step} resets={resets}", flush=True)
        finally:
            if encoder is not None:
                assert encoder.stdin is not None
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError("ffmpeg failed")
            env.close()
        print(f"Saved {output}; frames={captured}; resets={resets}", flush=True)


if __name__ == "__main__":
    main()
