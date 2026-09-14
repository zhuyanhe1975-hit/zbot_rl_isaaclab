#!/usr/bin/env bash
PYTHONPATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/src${PYTHONPATH:+:$PYTHONPATH}" /home/yhzhu/AI/IsaacLab/.venv/bin/isaaclab train --rl_library rsl_rl physics=newton_mjwarp --task ZbotRlIsaaclab-Velocity-Zbot-6DOF "$@"

# PYTHONPATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/src${PYTHONPATH:+:$PYTHONPATH}" /home/yhzhu/AI/IsaacLab/.venv/bin/isaaclab train --rl_library rsl_rl physics=ovphysx --task ZbotRlIsaaclab-Velocity-Zbot-6DOF #--viz newton_rtx
