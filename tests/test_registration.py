# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the generated task registrations."""

import gymnasium as gym

import zbot_rl_isaaclab.tasks  # noqa: F401


def test_task_registrations():
    """The generated tasks must expose valid environment and agent entry points."""
    expected = {
        "ZbotRlIsaaclab-6DOF-Base": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_base.env_cfg:BaseEnvCfg",
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_base.agents.rsl_rl_ppo_cfg:PPORunnerCfg"
            ),
        },
        "ZbotRlIsaaclab-Velocity-Zbot-6DOF": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof.env_cfg:VelocityEnvCfg",
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof.agents.rsl_rl_ppo_cfg:PPORunnerCfg"
            ),
        },
        "ZbotRlIsaaclab-Velocity-Zbot-6DOF-FrequencyCommand": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_frequency_command.env_cfg:FrequencyCommandEnvCfg"
            ),
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_frequency_command.agents.rsl_rl_ppo_cfg:PPORunnerCfg"
            ),
        },
        "ZbotRlIsaaclab-6DOF-Frequency-Balance": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_frequency_balance.env_cfg:FrequencyBalanceEnvCfg"
            ),
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_frequency_balance.agents.rsl_rl_ppo_cfg:PPORunnerCfg"
            ),
        },
        "ZbotRlIsaaclab-6DOF-Periodic-Walking": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking.env_cfg:PeriodicWalkingEnvCfg"
            ),
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking.agents.rsl_rl_ppo_cfg:PPORunnerCfg"
            ),
        },
        "ZbotRlIsaaclab-6DOF-Periodic-Walking-Teacher-Quality": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking.quality_env_cfg:QualityTeacherEnvCfg"
            ),
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking.agents."
                "rsl_rl_quality_ppo_cfg:QualityPPORunnerCfg"
            ),
        },
        "ZbotRlIsaaclab-6DOF-Periodic-Walking-Student": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student.env_cfg:StudentEnvCfg"
            ),
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student.agents."
                "rsl_rl_distillation_cfg:DistillationRunnerCfg"
            ),
        },
        "ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Quality": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student.quality_env_cfg:"
                "QualityStudentEnvCfg"
            ),
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student.agents."
                "rsl_rl_quality_distillation_cfg:QualityDistillationRunnerCfg"
            ),
        },
        "ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Finetune": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student_finetune.env_cfg:"
                "StudentFinetuneEnvCfg"
            ),
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student_finetune.agents."
                "rsl_rl_ppo_cfg:PPORunnerCfg"
            ),
        },
        "ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Robust": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student_finetune.env_cfg:"
                "StudentRobustEnvCfg"
            ),
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": (
                "zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student_finetune.agents."
                "rsl_rl_ppo_cfg:RobustRunnerCfg"
            ),
        },
    }

    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        assert spec.kwargs["env_cfg_entry_point"] == expected_values["env_cfg_entry_point"]
        if "default_agent" in expected_values:
            assert spec.kwargs["default_agent"] == expected_values["default_agent"]
        assert spec.kwargs["rsl_rl_cfg_entry_point"] == expected_values["rsl_rl_cfg_entry_point"]
