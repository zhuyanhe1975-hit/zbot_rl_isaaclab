#!/usr/bin/env python3
"""Train or resume sensor-only PPO after distillation, with optional robust physics."""

import sys

from isaaclab.cli import cli

from zbot_rl_isaaclab.student_transfer import install_student_actor_prior, install_student_resume_constraints


def _pop_student_prior(argv: list[str]) -> str | None:
    for index, argument in enumerate(argv):
        if argument == "--student_prior":
            if index + 1 >= len(argv):
                raise ValueError("--student_prior requires a distillation checkpoint path.")
            path = argv[index + 1]
            del argv[index : index + 2]
            return path
        if argument.startswith("--student_prior="):
            del argv[index]
            return argument.split("=", 1)[1]
    return None


def main() -> None:
    student_prior = _pop_student_prior(sys.argv)
    has_checkpoint = any(arg == "--checkpoint" or arg.startswith("--checkpoint=") for arg in sys.argv)
    if student_prior is not None and has_checkpoint:
        raise ValueError("Use --student_prior to start fine-tuning or --checkpoint to resume, not both.")
    if student_prior is None and not has_checkpoint:
        raise ValueError("Provide --student_prior for a fresh fine-tune or --checkpoint to resume one.")
    if student_prior is not None:
        install_student_actor_prior(student_prior)
    else:
        install_student_resume_constraints()
    sys.argv.insert(1, "train")
    cli()


if __name__ == "__main__":
    main()
