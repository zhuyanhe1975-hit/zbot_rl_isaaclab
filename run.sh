#!/usr/bin/env bash
PYTHONPATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/src${PYTHONPATH:+:$PYTHONPATH}" /home/yhzhu/AI/IsaacLab/.venv/bin/isaaclab play --rl_library rsl_rl physics=newton_mjwarp --task ZbotRlIsaaclab-Velocity-Zbot-6DOF --checkpoint latest --num_envs 16 --viz newton

# PYTHONPATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/src${PYTHONPATH:+:$PYTHONPATH}" /home/yhzhu/AI/IsaacLab/.venv/bin/isaaclab play --rl_library rsl_rl physics=ovphysx --task ZbotRlIsaaclab-Velocity-Zbot-6DOF --checkpoint latest --num_envs 16 --viz newton
