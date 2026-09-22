#!/usr/bin/env bash

# 行走任务
# PYTHONPATH=/home/yhzhu/myWorks_vips/zbot_rl_isaaclab/src /home/yhzhu/AI/IsaacLab/.venv/bin/python /home/yhzhu/myWorks_vips/zbot_rl_isaaclab/scripts/train_compact.py --rl_library rsl_rl physics=ovphysx --task ZbotRlIsaaclab-Velocity-Zbot-6DOF

# 基础双脚支撑重心转移任务（需要训练此任务时直接执行下面这一行）
PYTHONPATH=/home/yhzhu/myWorks_vips/zbot_rl_isaaclab/src /home/yhzhu/AI/IsaacLab/.venv/bin/python /home/yhzhu/myWorks_vips/zbot_rl_isaaclab/scripts/train_compact.py --rl_library rsl_rl physics=ovphysx --task ZbotRlIsaaclab-6DOF-Base --num_envs 1024 #--viz newton
