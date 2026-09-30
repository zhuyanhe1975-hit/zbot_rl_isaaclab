"""Teacher-guided rollout schedule for sensor-limited student distillation."""

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict


class GuidedDistillation(Distillation):
    """Blend teacher and student actions while reducing teacher guidance."""

    def __init__(
        self,
        *args,
        teacher_fraction_start: float = 1.0,
        teacher_fraction_end: float = 0.25,
        teacher_fraction_decay_updates: int = 500,
        **kwargs,
    ) -> None:
        if not 0.0 <= teacher_fraction_end <= teacher_fraction_start <= 1.0:
            raise ValueError("Teacher-action fractions must satisfy 0 <= end <= start <= 1.")
        if teacher_fraction_decay_updates <= 0:
            raise ValueError("Teacher-action decay must be positive.")
        super().__init__(*args, **kwargs)
        self.teacher_fraction_start = teacher_fraction_start
        self.teacher_fraction_end = teacher_fraction_end
        self.teacher_fraction_decay_updates = teacher_fraction_decay_updates

    def teacher_fraction(self) -> float:
        progress = min(self.num_updates / self.teacher_fraction_decay_updates, 1.0)
        return self.teacher_fraction_start + progress * (self.teacher_fraction_end - self.teacher_fraction_start)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Use privileged teacher actions for a decaying share of environments."""
        student_actions = self.student(obs, stochastic_output=True).detach()
        teacher_actions = self.teacher(obs).detach()
        teacher_fraction = self.teacher_fraction()
        actions = teacher_fraction * teacher_actions + (1.0 - teacher_fraction) * student_actions
        self.transition.actions = actions
        self.transition.privileged_actions = teacher_actions
        self.transition.observations = obs
        return actions

    def update(self) -> dict[str, float]:
        fraction = self.teacher_fraction()
        losses = super().update()
        losses["teacher_fraction"] = fraction
        return losses

    def save(self) -> dict:
        checkpoint = super().save()
        checkpoint["guided_updates"] = self.num_updates
        return checkpoint

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        restore_iteration = super().load(loaded_dict, load_cfg, strict)
        if "optimizer_state_dict" in loaded_dict:
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
        if "student_state_dict" in loaded_dict and restore_iteration:
            self.num_updates = int(loaded_dict.get("guided_updates", loaded_dict["iter"] + 1))
        return restore_iteration
