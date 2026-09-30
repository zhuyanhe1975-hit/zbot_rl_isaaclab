#!/usr/bin/env bash

# rsl_rl physics=newton_mjwarp
# PYTHONPATH=/home/yhzhu/myWorks_vips/zbot_rl_isaaclab/src /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/python /home/yhzhu/myWorks_vips/zbot_rl_isaaclab/scripts/train_compact.py --rl_library rsl_rl physics=newton_mjwarp --task ZbotRlIsaaclab-6DOF-Base --num_envs 1024 #--viz newton

# Train periodic walking from scratch with heading error in the policy observations.
PYTHONPATH=/home/yhzhu/myWorks_vips/zbot_rl_isaaclab/src \
  /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/python \
  /home/yhzhu/myWorks_vips/zbot_rl_isaaclab/scripts/train_compact.py \
  --rl_library rsl_rl \
  physics=isaacsim_physx \
  --task ZbotRlIsaaclab-6DOF-Periodic-Walking \
  --num_envs 4096 \
  --viz newton
  # --actor_prior /home/yhzhu/myWorks_vips/zbot_rl_isaaclab/pretrained/zbot_6dof_base_full_frequency_ovphysx_iter400/periodic_walking_actor_prior.pt \
  # --run_name base_iter400_periodic_walking \


# 6-DoF commanded-frequency walking initialized from the best frequency-balance actor.
# PYTHONPATH=/home/yhzhu/myWorks_vips/zbot_rl_isaaclab/src /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/python /home/yhzhu/myWorks_vips/zbot_rl_isaaclab/scripts/train_compact.py --rl_library rsl_rl physics=ovphysx --task ZbotRlIsaaclab-Velocity-Zbot-6DOF-FrequencyCommand --num_envs 4096 --actor_prior /home/yhzhu/myWorks_vips/zbot_rl_isaaclab/pretrained/zbot_6dof_frequency_balance_iter300_walking_prior.pt
