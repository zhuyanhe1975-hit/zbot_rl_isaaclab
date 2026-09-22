# zbot_rl_isaaclab

基于当前 Isaac Lab manager-based 工作流实现的 6-DoF ZBot 双足行走基线项目。

项目仅参考 `zbot_rl_student` 中最基础的机器人资产、初始关节姿态和执行器设置，没有迁移旧工程的
Direct 环境、三阶段课程、网页、训练脚本或 checkpoint。观测、奖励、注册和训练入口均使用当前
Isaac Lab API 重新组织。

## 任务

### 最基础的双脚支撑任务

- Gym ID：`ZbotRlIsaaclab-6DOF-Base`
- 目标：保持直立并在左右脚之间反复换载；达到一侧目标后切换到另一侧，不限制转移周期
- 观测：身体 `base` 的速度/姿态、关节位置/速度、上一动作、左右脚三轴接触力、身体坐标系下整机重心相对左右脚的坐标、当前理想支撑脚，以及 COM 地面投影到该脚的平面距离误差
- 奖励：存活、COM 地面投影到当前理想支撑脚的平面距离、归一化足底力差、初始关节姿态，以及最小的速度/动作平滑正则项

该任务不包含抬脚、步频、落脚、步长、速度命令或课程逻辑。任务单独启用了机器人自碰撞，
双脚之间设有独立的过滤接触传感器，
任意一只脚以超过 `1 N` 的力碰到另一只脚都会立即终止回合。任务不使用足底承重比例奖励；
当前理想支撑脚由周期 command 指定，而不是由实际接触力或距离阈值选择。完整左右循环周期为 `1.0 s`，
每 `0.5 s` 切换一次目标脚；COM 到目标脚的平面距离只参与观测和奖励，不控制 command 切换。
距离奖励采用归一化对比形式 `(d_other - d_ideal) / (d_ideal + d_other)`：靠近目标脚趋近 `+1`，
位于两脚中间趋近 `0`，停在非目标脚趋近 `-1`；观测中的距离误差也采用相反符号的同尺度归一化值。
周期阶段使用归一化足底力差 `(F_ideal - F_swing) / (F_ideal + F_swing)` 作为稠密奖励；理想支撑脚
承重越大越正，非支撑脚承重越大越负。不设置摆动脚抬高目标。
base 任务的目标轨迹积分速度与实际 actuator/simulation 关节速度上限均固定为 `2π rad/s`。
重心使用所有刚体质心按质量加权计算，并用刚体 `base` 的完整姿态转换到机器人身体坐标系。
启用可视化时，地面上的绿色小球显示整机重心在地面的世界坐标投影。

### 行走任务

- Gym ID：`ZbotRlIsaaclab-Velocity-Zbot-6DOF`
- 工作流：manager-based，single-agent
- 动作：`joint1` 至 `joint6` 的 6 维归一化关节速度，积分后转换为关节位置目标
- 观测：基座线/角速度、投影重力、关节位置/速度、上一动作、当前速度上限，共 28 维
- 目标：不使用速度命令，直接最大化身体坐标系 `+X` 正前方的行走速度
- 训练器：RSL-RL PPO
- 默认物理后端：Newton MJWarp；也保留 PhysX/OvPhysX 预设

奖励由身体正前方速度、存活、左右脚交替落地和前向步幅组成，并惩罚身体横向速度、
左右步幅差、关节力矩与加速度、动作突变、关节限位、足端滑移以及非足端接触。当前暂不使用
`lin_vel_z_l2` 和 `ang_vel_xy_l2`。
单脚支撑期间会惩罚两脚高度差的平方，不设固定高度区间；正常低幅摆动代价较小，抬腿越高
惩罚增长越快，双脚支撑阶段不触发。
交替落地仅在单脚首次接触、触地前腾空至少 `0.05 s` 且接触力至少 `10 N` 时计分；此外，摆动脚
必须曾位于另一只脚后方，并在落地时越过到另一只脚前方，才构成一次有效迈步。首次落脚、同脚
重复落地、未完成前后交换和双脚同时落地均不计分。机身 `base` 高度低于 `0.18 m` 时终止回合。
步幅按同一只脚相邻有效落地点在身体 `+X` 方向上的距离计算，不设目标或上限，正向步幅越大
奖励越高；同时惩罚最近左右脚步幅的绝对差，使两侧尽量对称。仅单脚有效落地时结算，避免
双脚跳跃刷步幅。

每个环境在 reset 时从 `[0.2π, 2.0π] rad/s` 采样一个关节速度上限。策略动作先经过 `tanh`，再按
`速度 × 环境步长` 积分到默认姿态的关节位置偏移；累计偏移限制在 `[-π, π] rad`。策略观测中的
速度上限除以 `π`，因此对应范围为 `[0.2, 2.0]`。

## 两阶段课程

阶段 1 只学习稳定原地踏步：关闭前向速度、步幅和步幅对称奖励，保留交替跨越奖励，并惩罚
身体水平漂移。训练至少 5,000 个环境步后，当近期平均回合寿命达到最大时长的 60%，且有效
交替落脚频率稳定在 `1–2 Hz` 时，自动晋级阶段 2。这里的频率定义为每秒完成的有效交替
落脚事件数，对应相邻事件间隔 `0.5–1.0 s`；该区间内频率奖励最高，区间外平滑衰减。

阶段 2 在 5,000 个环境步内平滑开启身体前向速度、步幅和左右步幅对称奖励，同时关闭阶段 1
的原地水平速度惩罚与频率奖励。课程阶段、混合比例、存活率和交替频率记录在 TensorBoard 的
`Curriculum/walking_stages/*` 指标中。

## Python 环境

本机直接复用 `/home/yhzhu/AI/IsaacLab/.venv`，不在项目目录中创建第二个虚拟环境。运行命令前设置：

```bash
cd /home/yhzhu/myWorks_vips/zbot_rl_isaaclab
export ISAACLAB_ENV=/home/yhzhu/AI/IsaacLab/.venv
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

项目的任务入口已经以 `--no-deps` 方式注册到该环境；`PYTHONPATH` 保证修改 `src/` 后立即生效。
不要在本目录运行 `uv sync` 或普通 `uv run`，否则会重新创建项目专属 `.venv`。

## 验证环境

```bash
$ISAACLAB_ENV/bin/python scripts/list_envs.py --show_presets
$ISAACLAB_ENV/bin/isaaclab zero_agent --task ZbotRlIsaaclab-Velocity-Zbot-6DOF --num_envs 4
$ISAACLAB_ENV/bin/isaaclab random_agent --task ZbotRlIsaaclab-Velocity-Zbot-6DOF --num_envs 16
```

如需使用 Isaac Sim PhysX：

```bash
$ISAACLAB_ENV/bin/isaaclab random_agent \
  --task ZbotRlIsaaclab-Velocity-Zbot-6DOF \
  --num_envs 16 physics=isaacsim_physx --viz kit
```

## 训练与回放

训练最基础的双脚支撑任务：

```bash
$ISAACLAB_ENV/bin/isaaclab train \
  --rl_library rsl_rl \
  --task ZbotRlIsaaclab-6DOF-Base \
  --num_envs 64 --max_iterations 10
```

先进行小规模 smoke training：

```bash
$ISAACLAB_ENV/bin/isaaclab train \
  --rl_library rsl_rl \
  --task ZbotRlIsaaclab-Velocity-Zbot-6DOF \
  --num_envs 64 --max_iterations 10
```

确认环境、观测和奖励正常后，再使用默认规模训练：

```bash
$ISAACLAB_ENV/bin/isaaclab train --task ZbotRlIsaaclab-Velocity-Zbot-6DOF
$ISAACLAB_ENV/bin/isaaclab play \
  --task ZbotRlIsaaclab-Velocity-Zbot-6DOF \
  --checkpoint latest --num_envs 16 --viz newton
```

## 开发检查

```bash
uv run --project /home/yhzhu/AI/IsaacLab --extra isaacsim --with pytest python -m pytest
uv run --project /home/yhzhu/AI/IsaacLab --extra isaacsim pre-commit run --all-files
```

主要配置位于：

- `src/zbot_rl_isaaclab/assets.py`：USD、初始姿态和执行器
- `src/zbot_rl_isaaclab/tasks/velocity/config/zbot_6dof/env_cfg.py`：场景、MDP 和物理配置
- `src/zbot_rl_isaaclab/tasks/velocity/config/zbot_6dof/agents/rsl_rl_ppo_cfg.py`：PPO 配置
