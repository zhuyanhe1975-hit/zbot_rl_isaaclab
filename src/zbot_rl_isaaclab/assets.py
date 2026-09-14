# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from importlib.resources import files

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ZBOT_6DOF_USD_PATH = str(files("zbot_rl_isaaclab").joinpath("assets/zbot_6s_new.usd"))
"""Packaged six-DoF ZBot USD asset path."""

ZBOT_6DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ZBOT_6DOF_USD_PATH,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, -0.06, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "joint1": 0.312,
            "joint2": 0.837,
            "joint3": -2.02,
            "joint4": 2.02,
            "joint5": -0.837,
            "joint6": -0.312,
        },
        joint_vel={"joint[1-6]": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            actuator_effort_limit=2000.0,
            actuator_velocity_limit=1000.0,
            joint_effort_limit=2000.0,
            joint_velocity_limit=1000.0,
            stiffness=50.0,
            damping=5.0,
            friction=0.0,
        ),
    },
)
"""Configuration for the six-DoF ZBot articulation."""
