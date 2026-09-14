# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import JointActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .actions import VelocityIntegratedJointPositionAction


@configclass
class VelocityIntegratedJointPositionActionCfg(JointActionCfg):
    """Configuration for velocity-limited integration into joint-position targets."""

    class_type: type[VelocityIntegratedJointPositionAction] | str = (
        "{DIR}.actions:VelocityIntegratedJointPositionAction"
    )

    velocity_limit_range: tuple[float, float] = (0.2 * math.pi, 2.0 * math.pi)
    """Range used to sample one joint angular-velocity limit per environment [rad/s]."""

    position_offset_limit: float = math.pi
    """Symmetric bound on accumulated position offset from the default pose [rad]."""
