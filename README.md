# zbot_rl_isaaclab

基于当前 Isaac Lab manager-based 工作流实现的 6-DoF ZBot 双足行走基线项目。

项目仅参考 `zbot_rl_student` 中最基础的机器人资产、初始关节姿态和执行器设置，没有迁移旧工程的
Direct 环境、三阶段课程、网页、训练脚本或 checkpoint。观测、奖励、注册和训练入口均使用当前
Isaac Lab API 重新组织。

## 任务

### 最基础的双脚支撑任务

- Gym ID：`ZbotRlIsaaclab-6DOF-Base`
- 目标：保持直立，并按 `0.5–2.0 Hz` 的目标频率在左右脚之间反复换载
- 观测：身体 `base` 的速度/姿态、关节位置/速度、上一动作、左右脚三轴接触力、身体坐标系下整机重心相对左右脚的坐标、目标换载频率、当前理想支撑脚、连续周期的 `sin/cos` 相位，以及 COM 地面投影到该脚的平面距离误差
- 奖励：存活、COM 地面投影到当前理想支撑脚的平面距离、归一化足底力差、初始关节姿态，以及最小的速度/动作平滑正则项

该任务不包含抬脚、落脚、步长、速度命令或课程逻辑，也不把实际落脚事件作为步频奖励。任务单独启用了机器人自碰撞，
双脚之间设有独立的过滤接触传感器，
任意一只脚以超过 `1 N` 的力碰到另一只脚都会立即终止回合。任务不使用足底承重比例奖励；
当前理想支撑脚由频率 command 指定，而不是由实际接触力或距离阈值选择。训练时每个环境从
训练从固定 `0.5 Hz` 开始，在前 `24,000` 个环境步内逐渐把采样上限扩展到 `2.0 Hz`；每个环境采样后在一个 `20 s` 回合内保持频率不变。Base 中的频率表示每秒完成的完整左右往返周期数，因此相邻目标侧切换间隔为 `0.25–1.0 s`。回放时，各环境按索引在完整频率区间内均匀排列并保持不变。
COM 到目标脚的平面距离只参与观测和奖励，不控制 command 切换。每回合从两脚中点相位开始；COM 与左右踝关节反力目标随周期相位按正弦曲线连续变化，统一规定正值偏向 `foot_0`、负值偏向 `foot_1`。OvPhysX 下继续沿用旧成功任务验证过的 `b1`、`foot_1` 踝反力，而不使用后端相关的足底接触流形聚合值。策略可利用 `sin/cos` 相位提前减速和换向，而不是等支撑脚方波翻转后再追赶。两项跟踪奖励均为有符号值：准确跟踪趋近 `+1`，误差超过容差后为负，避免不跟踪时仍持续获得正奖励。
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
单脚支撑期间允许两脚最多 `0.05 m` 的正常高度差；只对超过该阈值的部分施加平方惩罚，权重为
`-1`。因此正常摆动脚离地不受罚，只有过高抬腿的代价会增长，双脚支撑阶段也不触发。
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

本机直接复用 `/home/yhzhu/AI/IsaacLab-3.0/.venv`，不在项目目录中创建第二个虚拟环境。运行命令前设置：

```bash
cd /home/yhzhu/myWorks_vips/zbot_rl_isaaclab
export ISAACLAB_ENV=/home/yhzhu/AI/IsaacLab-3.0/.venv
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
uv pip install --python "$ISAACLAB_ENV/bin/python" --no-deps -e .
```

安装命令会在该环境注册 `isaaclab.tasks` 入口，使 `isaaclab train` 和 `isaaclab play` 能发现本项目的任务；新建或更换虚拟环境后需重新执行。`PYTHONPATH` 保证修改 `src/` 后立即生效。
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

## 学生网络与 sim-to-real

面向真机的周期行走学生网络使用 IMU、编码器与 MIT 电机力矩反馈。教师蒸馏、传感器输入顺序、
随机化微调和部署侧动作定义见 [学生网络 sim-to-real 说明](docs/student_sim2real.md)；
训练入口依次为 `train_teacher_quality.sh`、`train_student.sh`、`finetune_student.sh` 和 `robust_student.sh`。

## 开发检查

```bash
uv run --project /home/yhzhu/AI/IsaacLab-3.0 --extra isaacsim --with pytest python -m pytest
uv run --project /home/yhzhu/AI/IsaacLab-3.0 --extra isaacsim pre-commit run --all-files
```

主要配置位于：

- `src/zbot_rl_isaaclab/assets.py`：USD、初始姿态和执行器
- `src/zbot_rl_isaaclab/tasks/velocity/config/zbot_6dof/env_cfg.py`：场景、MDP 和物理配置
- `src/zbot_rl_isaaclab/tasks/velocity/config/zbot_6dof/agents/rsl_rl_ppo_cfg.py`：PPO 配置
