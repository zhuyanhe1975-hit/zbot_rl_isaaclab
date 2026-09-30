#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 5 || ( "${1:-}" != "--resume" && $# -gt 4 ) ]]; then
  echo "Usage: $0 [old_teacher_checkpoint] [num_envs] [iterations] [run_name]" >&2
  echo "       $0 --resume quality_teacher_checkpoint [num_envs] [iterations] [run_name]" >&2
  exit 2
fi
if [[ "${1:-}" == "--resume" ]]; then
  CHECKPOINT="${2:-}"
  NUM_ENVS="${3:-4096}"
  ITERATIONS="${4:-300}"
  RUN_NAME="${5:-teacher_quality_resume}"
  CHECKPOINT_FLAG=--checkpoint
else
  CHECKPOINT="${1:-$PROJECT_DIR/bests/zbot_6dof_periodic_walking_isaacsim_physx_20260926_iter1500/model_1499.pt}"
  NUM_ENVS="${2:-4096}"
  ITERATIONS="${3:-300}"
  RUN_NAME="${4:-teacher_quality}"
  CHECKPOINT_FLAG=--teacher_prior
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Teacher checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi
CHECKPOINT="$(realpath -- "$CHECKPOINT")"
cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/python \
  "$PROJECT_DIR/scripts/train_teacher_quality.py" \
  --rl_library rsl_rl \
  physics=isaacsim_physx \
  --task ZbotRlIsaaclab-6DOF-Periodic-Walking-Teacher-Quality \
  "$CHECKPOINT_FLAG" "$CHECKPOINT" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$ITERATIONS" \
  --run_name "$RUN_NAME"
