# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Deployable-observation student task for periodic walking distillation."""

import gymnasium as gym

from . import agents

_agent_cfg = f"{agents.__name__}.rsl_rl_distillation_cfg:DistillationRunnerCfg"

gym.register(
    id="ZbotRlIsaaclab-6DOF-Periodic-Walking-Student",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:StudentEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": _agent_cfg,
        "rsl_rl_distillation_cfg_entry_point": _agent_cfg,
    },
)

_quality_agent_cfg = f"{agents.__name__}.rsl_rl_quality_distillation_cfg:QualityDistillationRunnerCfg"

gym.register(
    id="ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Quality",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quality_env_cfg:QualityStudentEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": _quality_agent_cfg,
        "rsl_rl_distillation_cfg_entry_point": _quality_agent_cfg,
    },
)
