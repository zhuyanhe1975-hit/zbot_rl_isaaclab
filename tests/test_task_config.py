# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from pathlib import Path

from zbot_rl_isaaclab.assets import ZBOT_6DOF_USD_PATH
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof.env_cfg import VelocityEnvCfg


def test_zbot_task_contract():
    """The baseline task must retain its six-action walking contract."""
    cfg = VelocityEnvCfg()

    assert Path(ZBOT_6DOF_USD_PATH).is_file()
    assert cfg.scene.robot.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert cfg.actions.joint_pos.joint_names == [f"joint{index}" for index in range(1, 7)]
    assert cfg.actions.joint_pos.preserve_order
    assert cfg.actions.joint_pos.velocity_limit_range == (0.2 * math.pi, 2.0 * math.pi)
    assert cfg.actions.joint_pos.position_offset_limit == math.pi
    assert cfg.scene.contact_forces.track_air_time
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.0, 0.6)
    assert cfg.commands.base_velocity.ranges.lin_vel_y == (0.0, 0.0)
    assert cfg.commands.base_velocity.ranges.ang_vel_z == (0.0, 0.0)
    assert cfg.decimation == 4
    assert cfg.sim.dt == 0.005
