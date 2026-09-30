"""Initialize a PPO actor from a distilled student without restoring its optimizer."""

from pathlib import Path
from typing import Any

import torch


def freeze_student_action_std(runner: Any) -> None:
    """Keep the checkpoint's action noise fixed during PPO adaptation."""
    actor = runner.alg.get_policy()
    std_param = getattr(getattr(actor, "distribution", None), "std_param", None)
    if not isinstance(std_param, torch.nn.Parameter):
        raise TypeError("Student actor must have a Gaussian standard deviation parameter.")
    std_param.requires_grad_(False)


def load_student_actor_prior(
    runner: Any,
    checkpoint_path: str | Path,
    initial_std: float | None = None,
    freeze_std: bool = False,
) -> int:
    """Strictly load a distilled student or PPO actor into a matching PPO actor."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("student_state_dict") or checkpoint.get("actor_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Student prior requires student_state_dict or actor_state_dict.")
    actor = runner.alg.get_policy()
    actor.load_state_dict(state, strict=True)
    std_param = getattr(getattr(actor, "distribution", None), "std_param", None)
    if initial_std is not None:
        if initial_std <= 0.0:
            raise ValueError("Initial action standard deviation must be positive.")
        if not isinstance(std_param, torch.Tensor):
            raise TypeError("Student actor must have a learnable Gaussian standard deviation.")
        with torch.no_grad():
            std_param.fill_(initial_std)
    if freeze_std:
        freeze_student_action_std(runner)
    set_reference = getattr(runner.alg, "set_reference_actor", None)
    if callable(set_reference):
        set_reference()
    return int(checkpoint["iter"])


def install_student_actor_prior(checkpoint_path: str | Path, initial_std: float = 0.015) -> None:
    """Attach actor-only initialization to the normal RSL-RL PPO runner."""
    from rsl_rl.runners import OnPolicyRunner

    source = Path(checkpoint_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    original_init = OnPolicyRunner.__init__

    def student_initialized_runner(self: OnPolicyRunner, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        iteration = load_student_actor_prior(self, source, initial_std=initial_std, freeze_std=True)
        print(f"[INFO] Initialized PPO actor from student-policy checkpoint iteration {iteration}: {source}")
        print(f"[INFO] PPO action standard deviation is fixed at {initial_std:g}.")
        print("[INFO] PPO critic, optimizer, and iteration start fresh.")

    OnPolicyRunner.__init__ = student_initialized_runner


def install_student_resume_constraints() -> None:
    """Freeze the Gaussian standard deviation before loading a PPO checkpoint."""
    from rsl_rl.runners import OnPolicyRunner

    original_init = OnPolicyRunner.__init__

    def constrained_runner(self: OnPolicyRunner, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        freeze_student_action_std(self)

    OnPolicyRunner.__init__ = constrained_runner
