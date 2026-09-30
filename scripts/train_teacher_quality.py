#!/usr/bin/env python3
"""Train the quality-objective teacher from actor-only prior or resume its PPO state."""

import sys

from isaaclab.cli import cli

from zbot_rl_isaaclab.teacher_transfer import install_teacher_actor_prior


def _pop_teacher_prior(argv: list[str]) -> str | None:
    for index, argument in enumerate(argv):
        if argument == "--teacher_prior":
            if index + 1 >= len(argv):
                raise ValueError("--teacher_prior requires a checkpoint path.")
            path = argv[index + 1]
            del argv[index : index + 2]
            return path
        if argument.startswith("--teacher_prior="):
            del argv[index]
            return argument.split("=", 1)[1]
    return None


def main() -> None:
    teacher_prior = _pop_teacher_prior(sys.argv)
    has_checkpoint = any(arg == "--checkpoint" or arg.startswith("--checkpoint=") for arg in sys.argv)
    if teacher_prior is not None and has_checkpoint:
        raise ValueError("Use --teacher_prior for actor-only initialization or --checkpoint to resume, not both.")
    if teacher_prior is None and not has_checkpoint:
        raise ValueError("Provide --teacher_prior or --checkpoint.")
    if teacher_prior is not None:
        install_teacher_actor_prior(teacher_prior)
    sys.argv.insert(1, "train")
    cli()


if __name__ == "__main__":
    main()
