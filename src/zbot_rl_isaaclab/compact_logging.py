"""Compact, stable console rendering for RSL-RL training metrics."""

from __future__ import annotations

import contextlib
import datetime
import io
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _mean_episode_metrics(ep_extras: Sequence[Mapping[str, Any]], device: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    keys = {key for episode in ep_extras for key in episode}
    for key in keys:
        values = [torch.as_tensor(episode[key], device=device).reshape(-1) for episode in ep_extras if key in episode]
        if values:
            metrics[key] = float(torch.cat(values).float().mean().item())
    return metrics


def _sum_metrics(metrics: Mapping[str, float], keys: Sequence[str]) -> float:
    return sum(metrics.get(key, 0.0) for key in keys)


def format_compact_training_log(
    *,
    iteration: int,
    total_iterations: int,
    total_steps: int,
    steps_per_second: int,
    iteration_time: float,
    elapsed_seconds: float,
    eta_seconds: float,
    losses: Mapping[str, float],
    learning_rate: float,
    action_std: float,
    mean_reward: float | None,
    mean_episode_length: float | None,
    metrics: Mapping[str, float],
    task_profile: str = "walking",
) -> str:
    """Format the important training signals into fixed, categorized rows."""
    stage = int(round(metrics.get("Curriculum/walking_stages/stage", 1.0)))
    ready = metrics.get("Curriculum/walking_stages/promotion_ready", 0.0) >= 0.5
    blend = metrics.get("Curriculum/walking_stages/stage_two_blend", 0.0)
    survival = metrics.get("Curriculum/walking_stages/survival_ratio_ema", 0.0)
    cadence = metrics.get("Curriculum/walking_stages/alternation_rate_ema", 0.0)

    balance_penalty = _sum_metrics(
        metrics,
        (
            "Episode_Reward/body_lateral_vel_l2",
            "Episode_Reward/stage_one_horizontal_velocity_l2",
            "Episode_Reward/single_support_foot_height_l2",
            "Episode_Reward/feet_slide",
            "Episode_Reward/step_length_asymmetry",
        ),
    )
    control_penalty = _sum_metrics(
        metrics,
        (
            "Episode_Reward/joint_acc_l2",
            "Episode_Reward/joint_torques_l2",
            "Episode_Reward/joint_deviation",
            "Episode_Reward/action_rate_l2",
            "Episode_Reward/joint_pos_limits",
        ),
    )
    safety_penalty = _sum_metrics(
        metrics,
        ("Episode_Reward/termination_penalty", "Episode_Reward/undesired_contacts"),
    )

    loss_text = " ".join(f"{name}={float(value):.4f}" for name, value in losses.items())
    reward_text = "n/a" if mean_reward is None else f"{mean_reward:.2f}"
    length_text = "n/a" if mean_episode_length is None else f"{mean_episode_length:.1f}"
    elapsed = datetime.timedelta(seconds=int(elapsed_seconds))
    eta = datetime.timedelta(seconds=int(eta_seconds))

    common_rows = (
        "=" * 96,
        f"Training {iteration}/{total_iterations} | elapsed {elapsed} | ETA {eta}",
        f"[Performance] steps={total_steps:,}  throughput={steps_per_second:,}/s  iteration={iteration_time:.2f}s",
        f"[Optimization] {loss_text}  lr={learning_rate:.2e}  action_std={action_std:.3f}",
    )
    if task_profile == "base":
        motion_penalty = _sum_metrics(
            metrics,
            ("Episode_Reward/vertical_velocity_l2", "Episode_Reward/angular_velocity_l2"),
        )
        base_control_penalty = _sum_metrics(
            metrics,
            ("Episode_Reward/action_rate_l2",),
        )
        task_rows = (
            f"[Episode] reward={reward_text}  length={length_text}",
            "[Rewards] "
            f"support_distance={metrics.get('Episode_Reward/support_distance', 0.0):+.3f}  "
            f"support_force={metrics.get('Episode_Reward/support_force', 0.0):+.3f}  "
            f"alive={metrics.get('Episode_Reward/alive', 0.0):+.3f}",
            "[Penalties] "
            f"joint_pose={metrics.get('Episode_Reward/joint_deviation_l1', 0.0):+.3f}  "
            f"motion={motion_penalty:+.3f}  control={base_control_penalty:+.3f}",
            "[Termination] "
            f"timeout={metrics.get('Episode_Termination/time_out', 0.0):.1%}  "
            f"fall={metrics.get('Episode_Termination/base_height', 0.0):.1%}  "
            f"body_contact={metrics.get('Episode_Termination/illegal_contact', 0.0):.1%}  "
            f"feet_collision={metrics.get('Episode_Termination/feet_collision', 0.0):.1%}",
        )
        return "\n".join(common_rows + task_rows)

    walking_rows = (
        f"[Episode] reward={reward_text}  length={length_text}  survival={survival:.1%}",
        f"[Curriculum] stage={stage}  ready={'yes' if ready else 'no'}  blend={blend:.2f}  cadence={cadence:.2f}Hz",
        "[Rewards] "
        f"alive={metrics.get('Episode_Reward/alive', 0.0):+.3f}  "
        f"touchdown={metrics.get('Episode_Reward/alternating_touchdown', 0.0):+.3f}  "
        f"frequency={metrics.get('Episode_Reward/stage_one_step_frequency', 0.0):+.3f}  "
        f"forward={metrics.get('Episode_Reward/body_forward_speed', 0.0):+.3f}  "
        f"step={metrics.get('Episode_Reward/step_length', 0.0):+.3f}",
        f"[Penalties] balance={balance_penalty:+.3f}  control={control_penalty:+.3f}  safety={safety_penalty:+.3f}",
        "[Termination] "
        f"timeout={metrics.get('Episode_Termination/time_out', 0.0):.1%}  "
        f"base_height={metrics.get('Episode_Termination/base_height', 0.0):.1%}",
    )
    return "\n".join(common_rows + walking_rows)


def install_compact_rsl_rl_logging() -> None:
    """Patch the RSL-RL console logger while retaining all TensorBoard metrics."""
    from rsl_rl.utils.logger import Logger

    original_log = Logger.log
    if getattr(original_log, "_zbot_compact", False):
        return

    def compact_log(
        self: Logger,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict[str, float],
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        del print_minimal, width, pad
        metrics = _mean_episode_metrics(self.ep_extras, self.device)
        iteration_time = collect_time + learn_time
        collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
        fps = int(collection_size / iteration_time)

        with contextlib.redirect_stdout(io.StringIO()):
            original_log(
                self,
                it,
                start_it,
                total_it,
                collect_time,
                learn_time,
                loss_dict,
                learning_rate,
                action_std,
                rnd_weight,
                print_minimal=True,
            )

        completed_iterations = it + 1 - start_it
        remaining_iterations = total_it - start_it - completed_iterations
        eta_seconds = self.tot_time / completed_iterations * remaining_iterations
        mean_reward = statistics.mean(self.rewbuffer) if self.rewbuffer else None
        mean_episode_length = statistics.mean(self.lenbuffer) if self.lenbuffer else None
        print(
            format_compact_training_log(
                iteration=it,
                total_iterations=total_it,
                total_steps=self.tot_timesteps,
                steps_per_second=fps,
                iteration_time=iteration_time,
                elapsed_seconds=self.tot_time,
                eta_seconds=eta_seconds,
                losses=loss_dict,
                learning_rate=learning_rate,
                action_std=float(action_std.mean().item()),
                mean_reward=mean_reward,
                mean_episode_length=mean_episode_length,
                metrics=metrics,
                task_profile="base" if self.cfg.get("experiment_name") == "zbot_6dof_base" else "walking",
            )
        )

    compact_log._zbot_compact = True  # type: ignore[attr-defined]
    Logger.log = compact_log  # type: ignore[method-assign]
