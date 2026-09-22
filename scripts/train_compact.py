#!/usr/bin/env python3
"""Launch Isaac Lab training with the compact ZBot console logger."""

import sys

from isaaclab.cli import cli

from zbot_rl_isaaclab.compact_logging import install_compact_rsl_rl_logging

install_compact_rsl_rl_logging()
sys.argv.insert(1, "train")
cli()
