#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 || ! -f $1 ]]; then
  echo "Usage: $0 /path/to/model_<iteration>.pt [quality_teacher|quality_student|distilled|nominal|robust] [steps] [num_envs] [nominal|disturbed]" >&2
  exit 2
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="$(realpath -- "$1")"
MODE="${2:-nominal}"
STEPS="${3:-1000}"
NUM_ENVS="${4:-4}"
DISTURBANCE_MODE="${5:-nominal}"
case "$DISTURBANCE_MODE" in
  nominal) EXTRA_ARGS=() ;;
  disturbed) EXTRA_ARGS=(--disturbances) ;;
  *) echo "Evaluation mode must be nominal or disturbed." >&2; exit 2 ;;
esac
case "$MODE" in
  quality_teacher)
    TASK=ZbotRlIsaaclab-6DOF-Periodic-Walking-Teacher-Quality
    AGENT=rsl_rl_cfg_entry_point
    ;;
  quality_student)
    TASK=ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Quality
    AGENT=rsl_rl_distillation_cfg_entry_point
    ;;
  distilled)
    TASK=ZbotRlIsaaclab-6DOF-Periodic-Walking-Student
    AGENT=rsl_rl_distillation_cfg_entry_point
    ;;
  nominal)
    TASK=ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Finetune
    AGENT=rsl_rl_cfg_entry_point
    ;;
  robust)
    TASK=ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Robust
    AGENT=rsl_rl_cfg_entry_point
    ;;
  *)
    echo "Mode must be quality_teacher, quality_student, distilled, nominal, or robust." >&2
    exit 2
    ;;
esac

cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/python \
  "$PROJECT_DIR/scripts/evaluate_student.py" \
  --task "$TASK" \
  --agent "$AGENT" \
  --checkpoint "$CHECKPOINT" \
  --steps "$STEPS" \
  --num_envs "$NUM_ENVS" \
  "${EXTRA_ARGS[@]}" \
  physics=isaacsim_physx
