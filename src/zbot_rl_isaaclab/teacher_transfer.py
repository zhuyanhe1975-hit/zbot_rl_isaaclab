"""Initialize a quality-task teacher from the old teacher actor only."""

from pathlib import Path
from typing import Any

from .student_transfer import load_student_actor_prior


def install_teacher_actor_prior(checkpoint_path: str | Path, initial_std: float = 0.1) -> None:
    """Load matching teacher actor weights while starting PPO state fresh."""
    from rsl_rl.runners import OnPolicyRunner

    source = Path(checkpoint_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    original_init = OnPolicyRunner.__init__

    def teacher_initialized_runner(self: OnPolicyRunner, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        iteration = load_student_actor_prior(self, source, initial_std=initial_std)
        print(f"[INFO] Initialized quality teacher actor from iteration {iteration}: {source}")
        print(f"[INFO] Teacher action standard deviation starts at {initial_std:g}.")
        print("[INFO] Teacher critic, optimizer, and iteration start fresh.")

    OnPolicyRunner.__init__ = teacher_initialized_runner
