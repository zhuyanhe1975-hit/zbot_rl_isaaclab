#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 || ( "${1:-}" != "--resume" && $# -gt 4 ) ]]; then
  echo "Usage: $0 quality_student_checkpoint [num_envs] [iterations] [run_name]" >&2
  echo "       $0 --resume nominal_checkpoint [num_envs] [iterations] [run_name]" >&2
  exit 2
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--resume" ]]; then
  CHECKPOINT="${2:-}"
  NUM_ENVS="${3:-4096}"
  ITERATIONS="${4:-100}"
  RUN_NAME="${5:-student_quality_nominal_resume}"
  CHECKPOINT_FLAG=--checkpoint
else
  CHECKPOINT="$1"
  NUM_ENVS="${2:-4096}"
  ITERATIONS="${3:-100}"
  RUN_NAME="${4:-student_quality_nominal}"
  CHECKPOINT_FLAG=--student_prior
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Student checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi
CHECKPOINT="$(realpath -- "$CHECKPOINT")"
RUN_JSON="$(dirname -- "$CHECKPOINT")/run.json"
if [[ ! -f "$RUN_JSON" ]]; then
  echo "Run metadata not found beside checkpoint: $RUN_JSON" >&2
  exit 2
fi
CHECKPOINT_TASK="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task"])' "$RUN_JSON")"
if [[ "${1:-}" == "--resume" ]]; then
  EXPECTED_TASK=ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Finetune
else
  EXPECTED_TASK=ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Quality
fi
if [[ "$CHECKPOINT_TASK" != "$EXPECTED_TASK" ]]; then
  echo "Checkpoint belongs to $CHECKPOINT_TASK, expected $EXPECTED_TASK." >&2
  exit 2
fi
cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/python \
  "$PROJECT_DIR/scripts/train_student_finetune.py" \
  --rl_library rsl_rl \
  physics=isaacsim_physx \
  --task ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Finetune \
  "$CHECKPOINT_FLAG" "$CHECKPOINT" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$ITERATIONS" \
  --run_name "$RUN_NAME"
