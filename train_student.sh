#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--resume" ]]; then
  CHECKPOINT="${2:-}"
  NUM_ENVS="${3:-4096}"
  ITERATIONS="${4:-600}"
  RUN_NAME="${5:-student_quality_distillation_resume}"
  EXPECTED_TASK=ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Quality
  MAX_ARGS=5
  EXTRA_ENV_ARGS=(env.commands.weight_shift.curriculum_initial_maximum_frequency=null)
else
  CHECKPOINT="${1:-}"
  NUM_ENVS="${2:-4096}"
  ITERATIONS="${3:-600}"
  RUN_NAME="${4:-student_quality_distillation}"
  EXPECTED_TASK=ZbotRlIsaaclab-6DOF-Periodic-Walking-Teacher-Quality
  MAX_ARGS=4
  EXTRA_ENV_ARGS=()
fi
if [[ $# -gt "$MAX_ARGS" || ! -f "$CHECKPOINT" ]]; then
  echo "Usage: $0 /path/to/approved-quality-teacher/model_<iteration>.pt [num_envs] [iterations] [run_name]" >&2
  echo "       $0 --resume /path/to/quality-student/model_<iteration>.pt [num_envs] [iterations] [run_name]" >&2
  exit 2
fi
CHECKPOINT="$(realpath -- "$CHECKPOINT")"
RUN_JSON="$(dirname -- "$CHECKPOINT")/run.json"
if [[ ! -f "$RUN_JSON" ]]; then
  echo "Run metadata not found beside checkpoint: $RUN_JSON" >&2
  exit 2
fi
CHECKPOINT_TASK="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task"])' "$RUN_JSON")"
if [[ "$CHECKPOINT_TASK" != "$EXPECTED_TASK" ]]; then
  echo "Checkpoint belongs to $CHECKPOINT_TASK, expected $EXPECTED_TASK." >&2
  exit 2
fi
cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/isaaclab train \
  --rl_library rsl_rl \
  physics=isaacsim_physx \
  --task ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Quality \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint "$CHECKPOINT" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$ITERATIONS" \
  --run_name "$RUN_NAME" \
  "${EXTRA_ENV_ARGS[@]}"
