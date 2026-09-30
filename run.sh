# physics=newton_mjwarp
# PYTHONPATH=/home/yhzhu/myWorks_vips/zbot_rl_isaaclab/src /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/isaaclab play --rl_library rsl_rl physics=newton_mjwarp --task ZbotRlIsaaclab-6DOF-Base --checkpoint latest --num_envs 16 --viz newton
# physics=ovphysx balance task
# PYTHONPATH=/home/yhzhu/myWorks_vips/zbot_rl_isaaclab/src /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/isaaclab play --rl_library rsl_rl physics=ovphysx --task ZbotRlIsaaclab-6DOF-Base --checkpoint latest --num_envs 16 --viz newton

# physics=ovphysx walking task
# PYTHONPATH=/home/yhzhu/myWorks_vips/zbot_rl_isaaclab/src /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/isaaclab play --rl_library rsl_rl physics=ovphysx --task ZbotRlIsaaclab-Velocity-Zbot-6DOF --checkpoint latest --num_envs 16 --viz newton
# PYTHONPATH=/home/yhzhu/myWorks_vips/zbot_rl_isaaclab/src /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/isaaclab play --rl_library rsl_rl physics=ovphysx --task ZbotRlIsaaclab-Velocity-Zbot-6DOF --checkpoint logs/rsl_rl/zbot_6dof_velocity/2026-09-24_14-14-07/model_50.pt --num_envs 16 --viz newton

# Commanded-frequency walking over 0.5-2.0 Hz; set both bounds equal to test one fixed frequency.
# PYTHONPATH=/home/yhzhu/myWorks_vips/zbot_rl_isaaclab/src /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/isaaclab play --rl_library rsl_rl physics=ovphysx --task ZbotRlIsaaclab-Velocity-Zbot-6DOF-FrequencyCommand --checkpoint latest --num_envs 4 --viz newton env.commands.step_frequency.minimum_frequency=0.5 env.commands.step_frequency.maximum_frequency=2.0

# Frequency-conditioned stationary weight-transfer task (activate after its first checkpoint is available).
# PYTHONPATH=/home/yhzhu/myWorks_vips/zbot_rl_isaaclab/src /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/isaaclab play --rl_library rsl_rl physics=ovphysx --task ZbotRlIsaaclab-6DOF-Frequency-Balance --checkpoint latest --num_envs 4 --viz newton env.commands.weight_shift.minimum_frequency=0.5 env.commands.weight_shift.maximum_frequency=2.0

# Periodic walking: 16 environments receive ordered frequencies from 1.0 to 2.0 Hz.
PYTHONPATH=/home/yhzhu/myWorks_vips/zbot_rl_isaaclab/src \
  /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/isaaclab play \
  --rl_library rsl_rl \
  physics=isaacsim_physx \
  --task ZbotRlIsaaclab-6DOF-Periodic-Walking \
  --checkpoint latest \
  --num_envs 16 \
  --viz newton \
  env.commands.weight_shift.minimum_frequency=1.0 \
  env.commands.weight_shift.maximum_frequency=1.0 \
  env.commands.weight_shift.ordered=true
