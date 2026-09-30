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


def _nonzero_reward_weights(env_cfg: Mapping[str, Any] | object) -> dict[str, float]:
    """Return configured reward weights in declaration order, excluding disabled terms."""
    rewards = env_cfg.get("rewards") if isinstance(env_cfg, Mapping) else getattr(env_cfg, "rewards", None)
    if rewards is None:
        return {}

    terms = rewards.items() if isinstance(rewards, Mapping) else vars(rewards).items()
    weights: dict[str, float] = {}
    for name, term in terms:
        weight = term.get("weight") if isinstance(term, Mapping) else getattr(term, "weight", None)
        if weight is None:
            continue
        numeric_weight = float(weight)
        if numeric_weight != 0.0:
            weights[name] = numeric_weight
    return weights


def _format_reward_rows(
    metrics: Mapping[str, float],
    reward_weights: Mapping[str, float],
    *,
    max_width: int = 120,
) -> tuple[str, ...]:
    """Render every enabled reward term, wrapping long groups across console rows."""

    def wrap(label: str, entries: Sequence[str]) -> list[str]:
        rows: list[str] = []
        current = label
        for entry in entries:
            candidate = f"{current}  {entry}"
            if current != label and len(candidate) > max_width:
                rows.append(current)
                current = f"{label}  {entry}"
            else:
                current = candidate
        if current != label:
            rows.append(current)
        return rows

    positive: list[str] = []
    negative: list[str] = []
    for name, weight in reward_weights.items():
        if weight == 0.0:
            continue
        value = metrics.get(f"Episode_Reward/{name}", 0.0)
        displayed_value = f"{value:+.3e}" if 0.0 < abs(value) < 0.0005 else f"{value:+.3f}"
        entry = f"{name}={displayed_value} (scale={weight:g})"
        (positive if weight > 0.0 else negative).append(entry)
    return tuple(wrap("[Rewards]", positive) + wrap("[Penalties]", negative))


def _format_frequency_tracking_rows(metrics: Mapping[str, float]) -> tuple[str, ...]:
    """Render command-to-measured cadence for populated evaluation bins."""
    entries: list[str] = []
    for center, label in ((0.5, "0p5"), (1.0, "1"), (1.5, "1p5"), (2.0, "2")):
        prefix = "Metrics/step_frequency/"
        samples = metrics.get(f"{prefix}samples_{label}", 0.0)
        if samples <= 0.0:
            continue
        measured = metrics.get(f"{prefix}measured_sum_{label}", 0.0) / samples
        bias = abs(metrics.get(f"{prefix}signed_error_sum_{label}", 0.0) / samples)
        mae = metrics.get(f"{prefix}absolute_error_sum_{label}", 0.0) / samples
        entries.append(f"{center:g}->{measured:.2f}Hz (bias={bias:.2f}, mae={mae:.2f}, n={samples:.0f})")
    if not entries:
        return ()
    return ("[Frequency] " + "  ".join(entries),)


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
    reward_weights: Mapping[str, float] | None = None,
    task_profile: str = "walking",
) -> str:
    """Format training signals and every reward term whose configured weight is non-zero."""
    has_walking_curriculum = "Curriculum/walking_stages/stage" in metrics
    stage = int(round(metrics.get("Curriculum/walking_stages/stage", 2.0)))
    ready = metrics.get("Curriculum/walking_stages/promotion_ready", 0.0) >= 0.5
    blend = metrics.get("Curriculum/walking_stages/stage_two_blend", 1.0)
    survival = metrics.get(
        "Curriculum/walking_stages/survival_ratio_ema",
        metrics.get("Episode_Termination/time_out", 0.0),
    )
    cadence = metrics.get("Curriculum/walking_stages/alternation_rate_ema", 0.0)

    if reward_weights is None:
        reward_weights = {
            key.removeprefix("Episode_Reward/"): 1.0 for key in metrics if key.startswith("Episode_Reward/")
        }
    reward_rows = _format_reward_rows(metrics, reward_weights)
    command_rows = ()
    target_frequency_keys = (
        "Metrics/step_frequency/mean_target_frequency",
        "Metrics/weight_shift/mean_target_frequency",
    )
    target_frequency_key = next((key for key in target_frequency_keys if key in metrics), None)
    if target_frequency_key is not None:
        command_rows = (f"[Command] target_frequency={metrics[target_frequency_key]:.3f}Hz",)
    frequency_tracking_rows = _format_frequency_tracking_rows(metrics)

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
        task_rows = (
            f"[Episode] reward={reward_text}  length={length_text}",
            *reward_rows,
            "[Termination] "
            f"timeout={metrics.get('Episode_Termination/time_out', 0.0):.1%}  "
            f"fall={metrics.get('Episode_Termination/base_height', 0.0):.1%}  "
            f"body_contact={metrics.get('Episode_Termination/illegal_contact', 0.0):.1%}  "
            f"feet_collision={metrics.get('Episode_Termination/feet_collision', 0.0):.1%}",
        )
        return "\n".join(common_rows + task_rows)

    mode_row = (
        f"[Curriculum] stage={stage}  ready={'yes' if ready else 'no'}  blend={blend:.2f}  cadence={cadence:.2f}Hz"
        if has_walking_curriculum
        else "[Mode] direct walking"
    )
    walking_rows = (
        f"[Episode] reward={reward_text}  length={length_text}  survival={survival:.1%}",
        mode_row,
        *command_rows,
        *frequency_tracking_rows,
        *reward_rows,
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
                reward_weights=_nonzero_reward_weights(self.env_cfg),
                task_profile=(
                    "base"
                    if self.cfg.get("experiment_name") in {"zbot_6dof_base", "zbot_6dof_frequency_balance"}
                    else "walking"
                ),
            )
        )

    compact_log._zbot_compact = True  # type: ignore[attr-defined]
    Logger.log = compact_log  # type: ignore[method-assign]
