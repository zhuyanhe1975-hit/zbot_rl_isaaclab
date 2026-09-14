# zbot_rl_isaaclab

基于当前 Isaac Lab manager-based 工作流实现的 6-DoF ZBot 双足行走基线项目。

项目仅参考 `zbot_rl_student` 中最基础的机器人资产、初始关节姿态和执行器设置，没有迁移旧工程的
Direct 环境、三阶段课程、网页、训练脚本或 checkpoint。观测、奖励、注册和训练入口均使用当前
Isaac Lab API 重新组织。

## 任务

- Gym ID：`ZbotRlIsaaclab-Velocity-Zbot-6DOF`
- 工作流：manager-based，single-agent
- 动作：`joint1` 至 `joint6` 的 6 维归一化关节速度，积分后转换为关节位置目标
- 观测：基座线/角速度、投影重力、速度命令、关节位置/速度、上一动作、当前速度上限，共 31 维
- 目标：在平地跟踪 `0.0` 至 `0.6 m/s` 的前向速度命令
- 训练器：RSL-RL PPO
- 默认物理后端：Newton MJWarp；也保留 PhysX/OvPhysX 预设

奖励由速度跟踪、存活、足端腾空组成，并惩罚竖直速度、横滚/俯仰角速度、关节力矩与加速度、
动作突变、关节限位、足端滑移以及非足端接触。机身 `base` 高度低于 `0.18 m` 时终止回合。

每个环境在 reset 时从 `[0.2π, 2.0π] rad/s` 采样一个关节速度上限。策略动作先经过 `tanh`，再按
`速度 × 环境步长` 积分到默认姿态的关节位置偏移；累计偏移限制在 `[-π, π] rad`。策略观测中的
速度上限除以 `π`，因此对应范围为 `[0.2, 2.0]`。

## 安装

本项目由 `/home/yhzhu/AI/IsaacLab` 源码树生成，`pyproject.toml` 使用指向该源码树的可编辑相对路径：

```bash
cd /home/yhzhu/myWorks_vips/zbot_rl_isaaclab
uv sync
```

## 验证环境

```bash
uv run python scripts/list_envs.py --show_presets
uv run isaaclab zero_agent --task ZbotRlIsaaclab-Velocity-Zbot-6DOF --num_envs 4
uv run isaaclab random_agent --task ZbotRlIsaaclab-Velocity-Zbot-6DOF --num_envs 16
```

如需使用 Isaac Sim PhysX：

```bash
uv run --extra isaacsim isaaclab random_agent \
  --task ZbotRlIsaaclab-Velocity-Zbot-6DOF \
  --num_envs 16 physics=isaacsim_physx --viz kit
```

## 训练与回放

先进行小规模 smoke training：

```bash
uv run isaaclab train \
  --rl_library rsl_rl \
  --task ZbotRlIsaaclab-Velocity-Zbot-6DOF \
  --num_envs 64 --max_iterations 10
```

确认环境、观测和奖励正常后，再使用默认规模训练：

```bash
uv run isaaclab train --task ZbotRlIsaaclab-Velocity-Zbot-6DOF
uv run isaaclab play \
  --task ZbotRlIsaaclab-Velocity-Zbot-6DOF \
  --checkpoint latest --num_envs 16 --viz newton
```

## 开发检查

```bash
uv run pytest
uv run pre-commit run --all-files
```

主要配置位于：

- `src/zbot_rl_isaaclab/assets.py`：USD、初始姿态和执行器
- `src/zbot_rl_isaaclab/tasks/velocity/config/zbot_6dof/env_cfg.py`：场景、MDP 和物理配置
- `src/zbot_rl_isaaclab/tasks/velocity/config/zbot_6dof/agents/rsl_rl_ppo_cfg.py`：PPO 配置
