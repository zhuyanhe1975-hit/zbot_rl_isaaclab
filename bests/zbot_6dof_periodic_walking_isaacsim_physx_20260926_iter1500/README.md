# ZBot 6DOF Periodic Walking — 1500 iterations

Final snapshot of the fresh, no-prior Isaac Sim PhysX training run from
`logs/rsl_rl/zbot_6dof_periodic_walking/2026-09-26_13-00-49`.

- Task: `ZbotRlIsaaclab-6DOF-Periodic-Walking`
- Checkpoint: `model_1499.pt` (zero-based iteration 1499, after 1500 iterations)
- Isaac Lab: `/home/yhzhu/AI/IsaacLab-3.0`
- `params/`: saved environment and PPO configuration from this run
- `exported/`: inference policy in TorchScript and ONNX formats; keep `policy.onnx.data` beside `policy.onnx`
- `training.tfevents`: TensorBoard scalars for this run

Replay this archived checkpoint from the project root:

```bash
PYTHONPATH="$PWD/src" /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/isaaclab play \
  --rl_library rsl_rl physics=isaacsim_physx \
  --task ZbotRlIsaaclab-6DOF-Periodic-Walking \
  --checkpoint "$PWD/bests/zbot_6dof_periodic_walking_isaacsim_physx_20260926_iter1500/model_1499.pt" \
  --num_envs 16 --viz newton \
  env.commands.weight_shift.minimum_frequency=1.0 \
  env.commands.weight_shift.maximum_frequency=2.0 \
  env.commands.weight_shift.ordered=true
```
