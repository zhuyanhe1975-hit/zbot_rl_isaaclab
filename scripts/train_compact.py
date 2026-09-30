#!/usr/bin/env python3
"""Launch Isaac Lab training with the compact ZBot console logger."""

import sys

from isaaclab.cli import cli

from zbot_rl_isaaclab.compact_logging import install_compact_rsl_rl_logging
from zbot_rl_isaaclab.pretraining import install_actor_prior


def _pop_actor_prior(argv: list[str]) -> str | None:
    """Remove the project-local actor-prior option before Isaac Lab parses its CLI."""
    for index, argument in enumerate(argv):
        if argument == "--actor_prior":
            if index + 1 >= len(argv):
                raise ValueError("--actor_prior requires a checkpoint path.")
            path = argv[index + 1]
            del argv[index : index + 2]
            return path
        if argument.startswith("--actor_prior="):
            del argv[index]
            return argument.split("=", 1)[1]
    return None


install_compact_rsl_rl_logging()
actor_prior = _pop_actor_prior(sys.argv)
if actor_prior is not None:
    install_actor_prior(actor_prior)
sys.argv.insert(1, "train")
cli()
