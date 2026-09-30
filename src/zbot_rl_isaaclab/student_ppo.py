"""PPO with a bounded actor change from the selected student checkpoint."""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO


class AnchoredPPO(PPO):
    """Keep the actor near its initial policy while the critic adapts freely."""

    def __init__(self, *args, max_relative_drift: float = 0.02, **kwargs) -> None:
        if max_relative_drift <= 0.0:
            raise ValueError("Maximum relative actor drift must be positive.")
        super().__init__(*args, **kwargs)
        self.max_relative_drift = max_relative_drift
        self._actor_reference: dict[str, torch.Tensor] | None = None

    def set_reference_actor(self) -> None:
        """Snapshot actor parameters after loading the selected student prior."""
        self._actor_reference = {
            name: parameter.detach().clone() for name, parameter in self._raw_actor.named_parameters()
        }

    def _project_actor(self) -> float:
        if self._actor_reference is None:
            raise RuntimeError("Actor reference must be set before PPO updates.")
        largest_drift = 0.0
        with torch.no_grad():
            for name, parameter in self._raw_actor.named_parameters():
                reference = self._actor_reference[name]
                scale = reference.square().mean().sqrt().clamp_min(0.01)
                delta = parameter - reference
                relative_drift = delta.square().mean().sqrt() / scale
                largest_drift = max(largest_drift, relative_drift.item())
                factor = (self.max_relative_drift / relative_drift.clamp_min(1.0e-12)).clamp(max=1.0)
                parameter.copy_(reference + factor * delta)
        return largest_drift

    def update(self) -> dict[str, float]:
        losses = super().update()
        losses["actor_drift_before_projection"] = self._project_actor()
        return losses

    def save(self) -> dict:
        checkpoint = super().save()
        if self._actor_reference is None:
            raise RuntimeError("Actor reference must be set before saving PPO checkpoints.")
        checkpoint["actor_reference_state_dict"] = self._actor_reference
        return checkpoint

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        restore_iteration = super().load(loaded_dict, load_cfg, strict)
        reference = loaded_dict.get("actor_reference_state_dict")
        if reference is None:
            self.set_reference_actor()
        else:
            self._actor_reference = {
                name: reference[name].to(device=parameter.device, dtype=parameter.dtype).detach().clone()
                for name, parameter in self._raw_actor.named_parameters()
            }
        return restore_iteration
