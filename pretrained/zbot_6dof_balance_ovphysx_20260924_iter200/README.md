# ZBot 6-DoF balance prior (OvPhysX)

This artifact preserves the successful double-support weight-transfer policy as a pretraining prior for later walking work.

## Provenance

- Source run: `logs/rsl_rl/zbot_6dof_base/2026-09-24_12-20-40`
- Task: `ZbotRlIsaaclab-6DOF-Base`
- Physics backend: `ovphysx`
- Parallel environments: 8192
- Seed: 42
- Checkpoint: iteration 200
- Observation dimension: 41
- Action dimension: 6
- Repository base commit: `0cbf1533a4a7771d9d7fc9ddbb384231bf7ae17c`

## Training evidence

The run continued through TensorBoard step 232 after the final periodic checkpoint. Tail-20 means at shutdown:

- Mean reward: 47.6625
- Mean episode length: 949.6095
- Support-force reward: 1.1129
- Support-distance reward: 0.5766
- Feet-slide penalty: -0.0491
- Timeout fraction: 90.26%
- Base-height termination fraction: 9.04%
- Throughput: 111,692 steps/s

## Files

- `model_200.pt`: resumable RSL-RL checkpoint, including actor, critic, and optimizer states.
- `exported/policy.pt`: TorchScript inference policy.
- `exported/policy.onnx` plus `policy.onnx.data`: ONNX inference policy.
- `params/env.yaml` and `params/agent.yaml`: exact resolved training configuration.
- `training.tfevents`: TensorBoard history used for the metrics above.
- `git/IsaacLab.diff`: Isaac Lab dependency diff captured by the training runner.
- `walking_actor_prior.pt`: 44-input walking actor prior with eight semantically aligned observation terms.
- `SHA256SUMS`: integrity hashes for the preserved artifacts.

The stable alias `pretrained/zbot_6dof_balance_ovphysx_latest` points to this directory.

## Reuse

Resume the same task:

```bash
PYTHONPATH=src /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/python scripts/train_compact.py \
  --rl_library rsl_rl physics=ovphysx \
  --task ZbotRlIsaaclab-6DOF-Base \
  --checkpoint pretrained/zbot_6dof_balance_ovphysx_latest/model_200.pt
```

For a walking task whose observation layout differs from the 41-element balance observation, reuse compatible hidden/action layers instead of loading the full actor input layer blindly. Preserve the first-layer observation mapping explicitly when implementing that transfer.

The included walking prior performs that mapping explicitly: 39 shared values are copied by named slice, while the 5 walking-only input columns start at zero. Balance-only support-command inputs are excluded. Only the actor MLP transfers; the walking critic, optimizer, iteration counter, and exploration standard deviation start from the walking configuration because their semantics depend on the walking rewards.
