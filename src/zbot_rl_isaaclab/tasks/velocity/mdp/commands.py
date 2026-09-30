# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import PeriodicSupportFootCommandCfg, UniformStepFrequencyCommandCfg


def periodic_support_side(
    initial_side: torch.Tensor,
    episode_length_buf: torch.Tensor,
    step_dt: float,
    period: float,
) -> torch.Tensor:
    """Return the ideal support-foot side for a fixed full-cycle period."""
    if period <= 0.0 or step_dt <= 0.0:
        raise ValueError("Support-foot period and environment step must be positive.")
    if initial_side.shape != episode_length_buf.shape:
        raise ValueError("Initial side and episode length must have matching shapes.")
    half_cycle = torch.floor(episode_length_buf.float() * step_dt / (0.5 * period)).long()
    phase_sign = torch.where(half_cycle.remainder(2) == 0, 1.0, -1.0)
    return initial_side * phase_sign


def ordered_step_frequencies(
    num_envs: int,
    minimum_frequency: float,
    maximum_frequency: float,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return an ascending frequency command for each environment index."""
    if num_envs <= 0:
        raise ValueError("Number of environments must be positive.")
    if minimum_frequency <= 0.0 or maximum_frequency < minimum_frequency:
        raise ValueError("Frequency bounds must satisfy 0 < minimum_frequency <= maximum_frequency.")
    return torch.linspace(minimum_frequency, maximum_frequency, num_envs, device=device)


def curriculum_maximum_frequency(
    minimum_frequency: float,
    maximum_frequency: float,
    initial_maximum_frequency: float,
    step: int,
    curriculum_steps: int,
) -> float:
    """Expand the sampled frequency ceiling linearly over training steps."""
    if not minimum_frequency <= initial_maximum_frequency <= maximum_frequency:
        raise ValueError("Initial curriculum frequency must lie inside the configured frequency range.")
    if step < 0 or curriculum_steps <= 0:
        raise ValueError("Curriculum step must be non-negative and duration must be positive.")
    progress = min(step / curriculum_steps, 1.0)
    return initial_maximum_frequency + progress * (maximum_frequency - initial_maximum_frequency)


def advance_frequency_phase(
    phase: torch.Tensor,
    frequency: torch.Tensor,
    step_dt: float,
    *,
    frequency_is_full_cycle: bool = False,
) -> torch.Tensor:
    """Advance phase using either full-cycle frequency or support-switch frequency."""
    if phase.shape != frequency.shape:
        raise ValueError("Phase and frequency tensors must have matching shapes.")
    if step_dt <= 0.0:
        raise ValueError("Environment step must be positive.")
    phase_rate = frequency if frequency_is_full_cycle else 0.5 * frequency
    return torch.remainder(phase + phase_rate * step_dt, 1.0)


def support_side_from_phase(phase: torch.Tensor) -> torch.Tensor:
    """Return ``+1`` during the first half-cycle and ``-1`` during the second."""
    return torch.where(phase < 0.5, 1.0, -1.0)


def frequency_phase_features(phase: torch.Tensor) -> torch.Tensor:
    """Encode cyclic phase without a discontinuity at the phase wrap."""
    angle = 2.0 * torch.pi * phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def frequency_metric_label(frequency: float) -> str:
    """Return a TensorBoard-safe label for a frequency-bin center."""
    return f"{frequency:g}".replace(".", "p")


class PeriodicSupportFootCommand(CommandTerm):
    """Alternate the ideal support foot on a fixed time schedule."""

    cfg: PeriodicSupportFootCommandCfg

    def __init__(self, cfg: PeriodicSupportFootCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        if cfg.period <= 0.0:
            raise ValueError("Support-foot command period must be positive.")
        self._rl_env = env
        self._asset = cast("Articulation", env.scene[cfg.asset_name])
        self._target_side = torch.ones(env.num_envs, 1, device=env.device)
        self._initial_side = torch.ones_like(self._target_side)
        self.metrics["switches"] = torch.zeros(env.num_envs, device=env.device)

    @property
    def command(self) -> torch.Tensor:
        """Return ``+1`` for foot 0 and ``-1`` for foot 1."""
        return self._target_side

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        count = self._target_side[env_ids].shape[0]
        random_side = torch.randint(0, 2, (count, 1), device=self.device, dtype=torch.long)
        sampled_side = random_side.float() * 2.0 - 1.0
        self._initial_side[env_ids] = sampled_side
        self._target_side[env_ids] = sampled_side

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        updated_target = periodic_support_side(
            self._initial_side[:, 0],
            self._rl_env.episode_length_buf,
            self._rl_env.step_dt,
            self.cfg.period,
        ).unsqueeze(-1)
        switched = updated_target[:, 0] != self._target_side[:, 0]
        self._target_side.copy_(updated_target)
        self.metrics["switches"] += switched.float()

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "com_projection_visualizer"):
                self.com_projection_visualizer = VisualizationMarkers(self.cfg.com_projection_visualizer_cfg)
            self.com_projection_visualizer.set_visibility(True)
        elif hasattr(self, "com_projection_visualizer"):
            self.com_projection_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        del event
        if not self._asset.is_initialized:
            return
        body_com_pos_w = self._asset.data.body_com_pos_w.torch
        body_mass = self._asset.data.body_mass.torch
        center_of_mass_w = (body_com_pos_w * body_mass.unsqueeze(-1)).sum(dim=1) / body_mass.sum(dim=1, keepdim=True)
        projected_position = center_of_mass_w.clone()
        projected_position[:, 2] = self._env.scene.env_origins[:, 2] + 0.035
        self.com_projection_visualizer.visualize(
            translations=projected_position,
            environment_ids=self._env.scene._ALL_INDICES,
        )


class UniformStepFrequencyCommand(CommandTerm):
    """Generate a target frequency and its time-driven support-foot schedule."""

    cfg: UniformStepFrequencyCommandCfg

    def __init__(self, cfg: UniformStepFrequencyCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        if cfg.minimum_frequency <= 0.0 or cfg.maximum_frequency < cfg.minimum_frequency:
            raise ValueError("Frequency bounds must satisfy 0 < minimum_frequency <= maximum_frequency.")
        if not 0.0 <= cfg.endpoint_probability <= 0.5:
            raise ValueError("Endpoint probability must be between 0 and 0.5.")
        if cfg.curriculum_initial_maximum_frequency is not None:
            curriculum_maximum_frequency(
                cfg.minimum_frequency,
                cfg.maximum_frequency,
                cfg.curriculum_initial_maximum_frequency,
                step=0,
                curriculum_steps=cfg.curriculum_steps,
            )
        self._step_dt = env.step_dt
        self._rl_env = env
        self._asset = cast("Articulation", env.scene[cfg.asset_name])
        self._frequency = torch.zeros(env.num_envs, 1, device=env.device)
        self._phase = torch.zeros(env.num_envs, device=env.device)
        self._target_side = torch.ones(env.num_envs, 1, device=env.device)
        self._ordered_frequency = ordered_step_frequencies(
            env.num_envs,
            cfg.minimum_frequency,
            cfg.maximum_frequency,
            device=env.device,
        )
        self._frequency_sum = torch.zeros(env.num_envs, device=env.device)
        self._step_count = torch.zeros(env.num_envs, device=env.device)
        self._evaluation_frequencies = torch.tensor(cfg.evaluation_frequencies, device=env.device)
        if self._evaluation_frequencies.numel() == 0 or torch.any(self._evaluation_frequencies <= 0.0):
            raise ValueError("Evaluation frequencies must contain positive values.")
        bin_shape = (env.num_envs, self._evaluation_frequencies.numel())
        self._measured_frequency_sum = torch.zeros(bin_shape, device=env.device)
        self._signed_frequency_error_sum = torch.zeros(bin_shape, device=env.device)
        self._absolute_frequency_error_sum = torch.zeros(bin_shape, device=env.device)
        self._frequency_measurement_count = torch.zeros(bin_shape, device=env.device)
        self.metrics["mean_target_frequency"] = torch.zeros(env.num_envs, device=env.device)

    @property
    def command(self) -> torch.Tensor:
        """Return frequency and support side, optionally followed by sine/cosine phase."""
        command = torch.cat((self._frequency, self._target_side), dim=1)
        if self.cfg.include_phase_features:
            command = torch.cat((command, frequency_phase_features(self._phase)), dim=1)
        return command

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if self.cfg.ordered:
            self._frequency[env_ids, 0] = self._ordered_frequency[env_ids]
        else:
            count = self._frequency[env_ids].shape[0]
            maximum_frequency = self.cfg.maximum_frequency
            if self.cfg.curriculum_initial_maximum_frequency is not None:
                maximum_frequency = curriculum_maximum_frequency(
                    self.cfg.minimum_frequency,
                    self.cfg.maximum_frequency,
                    self.cfg.curriculum_initial_maximum_frequency,
                    self._rl_env.common_step_counter,
                    self.cfg.curriculum_steps,
                )
            sampled = torch.empty(count, 1, device=self._frequency.device).uniform_(
                self.cfg.minimum_frequency,
                maximum_frequency,
            )
            if self.cfg.endpoint_probability > 0.0:
                endpoint_draw = torch.rand(count, 1, device=self._frequency.device)
                sampled = torch.where(
                    endpoint_draw < self.cfg.endpoint_probability, self.cfg.minimum_frequency, sampled
                )
                sampled = torch.where(endpoint_draw >= 1.0 - self.cfg.endpoint_probability, maximum_frequency, sampled)
            self._frequency[env_ids] = sampled
        self._target_side[env_ids, 0] = support_side_from_phase(self._phase[env_ids])

    def _update_metrics(self) -> None:
        self._frequency_sum += self._frequency[:, 0]
        self._step_count += 1.0
        if self.cfg.frequency_reward_term_name is None:
            return
        term_cfg = self._rl_env.reward_manager.get_term_cfg(self.cfg.frequency_reward_term_name)
        reward_term = term_cfg.func
        # The detector measures alternating touchdowns. Two such events make
        # one full cycle when the command is expressed as gait-cycle Hz.
        measured_frequency = reward_term.measured_frequency * (0.5 if self.cfg.frequency_is_full_cycle else 1.0)
        valid_measurement = reward_term.has_frequency_event
        target_frequency = self._frequency[:, 0]
        bin_index = torch.argmin(
            torch.abs(target_frequency.unsqueeze(1) - self._evaluation_frequencies.unsqueeze(0)),
            dim=1,
        ).unsqueeze(1)
        event_weight = valid_measurement.to(measured_frequency.dtype)
        frequency_error = measured_frequency - target_frequency
        self._measured_frequency_sum.scatter_add_(1, bin_index, (measured_frequency * event_weight).unsqueeze(1))
        self._signed_frequency_error_sum.scatter_add_(1, bin_index, (frequency_error * event_weight).unsqueeze(1))
        self._absolute_frequency_error_sum.scatter_add_(
            1, bin_index, (frequency_error.abs() * event_weight).unsqueeze(1)
        )
        self._frequency_measurement_count.scatter_add_(1, bin_index, event_weight.unsqueeze(1))

    def _update_command(self) -> None:
        self._phase.copy_(
            advance_frequency_phase(
                self._phase,
                self._frequency[:, 0],
                self._step_dt,
                frequency_is_full_cycle=self.cfg.frequency_is_full_cycle,
            )
        )
        self._target_side[:, 0] = support_side_from_phase(self._phase)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Log the episode-mean target frequency before sampling the next command."""
        selected = slice(None) if env_ids is None else env_ids
        denominator = self._step_count[selected].clamp_min(1.0)
        self.metrics["mean_target_frequency"][selected] = self._frequency_sum[selected] / denominator
        measured_sum = self._measured_frequency_sum[selected].sum(dim=0)
        signed_error_sum = self._signed_frequency_error_sum[selected].sum(dim=0)
        error_sum = self._absolute_frequency_error_sum[selected].sum(dim=0)
        measurement_count = self._frequency_measurement_count[selected].sum(dim=0)
        measured_values, signed_values, error_values, sample_values = (
            torch.stack((measured_sum, signed_error_sum, error_sum, measurement_count)).cpu().tolist()
        )
        count = self._phase[selected].shape[0]
        if self.cfg.ordered or not self.cfg.randomize_phase:
            self._phase[selected] = 0.0
        else:
            self._phase[selected] = torch.rand(count, device=self._phase.device)
        extras = super().reset(env_ids)
        for index, center in enumerate(self.cfg.evaluation_frequencies):
            label = frequency_metric_label(center)
            extras[f"measured_sum_{label}"] = float(measured_values[index])
            extras[f"signed_error_sum_{label}"] = float(signed_values[index])
            extras[f"absolute_error_sum_{label}"] = float(error_values[index])
            extras[f"samples_{label}"] = float(sample_values[index])
        self._frequency_sum[selected] = 0.0
        self._step_count[selected] = 0.0
        self._measured_frequency_sum[selected] = 0.0
        self._signed_frequency_error_sum[selected] = 0.0
        self._absolute_frequency_error_sum[selected] = 0.0
        self._frequency_measurement_count[selected] = 0.0
        return extras

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "com_projection_visualizer"):
                self.com_projection_visualizer = VisualizationMarkers(self.cfg.com_projection_visualizer_cfg)
            self.com_projection_visualizer.set_visibility(True)
        elif hasattr(self, "com_projection_visualizer"):
            self.com_projection_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        del event
        if not self._asset.is_initialized:
            return
        body_com_pos_w = self._asset.data.body_com_pos_w.torch
        body_mass = self._asset.data.body_mass.torch
        center_of_mass_w = (body_com_pos_w * body_mass.unsqueeze(-1)).sum(dim=1) / body_mass.sum(dim=1, keepdim=True)
        projected_position = center_of_mass_w.clone()
        projected_position[:, 2] = self._env.scene.env_origins[:, 2] + 0.035
        self.com_projection_visualizer.visualize(
            translations=projected_position,
            environment_ids=self._env.scene._ALL_INDICES,
        )
