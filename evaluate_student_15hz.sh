#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: ./evaluate_student_15hz.sh [quality_or_robust_checkpoint] [nominal|training|strong] [steps] [num_envs]"
  echo "       ./evaluate_student_15hz.sh --show-command [checkpoint] [mode] [steps] [num_envs]"
  echo "The legacy 'disturbed' mode is an alias for strong (0.25–0.5 m/s)."
  exit 0
fi
show_only=false
if [[ "${1:-}" == "--show-command" ]]; then
  show_only=true
  shift
fi
if [[ $# -gt 4 ]]; then
  echo "Usage: $0 [quality_or_robust_checkpoint] [nominal|disturbed] [steps] [num_envs]" >&2
  exit 2
fi

checkpoint="${1:-$repo_root/logs/rsl_rl/zbot_6dof_periodic_walking_student_robust/2026-09-27_16-22-11_student_force_anchored_stage1/model_25.pt}"
mode="${2:-disturbed}"
steps="${3:-1000}"
num_envs="${4:-64}"
if [[ ! -f "$checkpoint" || ! "$steps" =~ ^[1-9][0-9]*$ || ! "$num_envs" =~ ^[1-9][0-9]*$ ]]; then
  echo "Provide a student checkpoint, positive steps, and positive num_envs." >&2
  exit 2
fi
checkpoint="$(realpath -- "$checkpoint")"
run_json="$(dirname -- "$checkpoint")/run.json"
if [[ ! -f "$run_json" ]]; then
  echo "Run metadata not found beside checkpoint: $run_json" >&2
  exit 2
fi
checkpoint_task="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task"])' "$run_json")"
case "$checkpoint_task" in
  ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Quality)
    task="$checkpoint_task"
    agent=rsl_rl_distillation_cfg_entry_point
    ;;
  ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Robust)
    task="$checkpoint_task"
    agent=rsl_rl_cfg_entry_point
    ;;
  *) echo "Checkpoint must belong to a quality or robust student run." >&2; exit 2 ;;
esac

case "$mode" in
  nominal) disturbance_args=() ;;
  training)
    force_min="${ZBOT_FORCE_MIN_DELTA_V:-0.05}"
    force_max="${ZBOT_FORCE_MAX_DELTA_V:-0.25}"
    disturbance_args=(
      --disturbances
      --force-mean-interval 1.0
      --force-duration 0.3
      --force-minimum-delta-velocity "$force_min"
      --force-maximum-delta-velocity "$force_max"
    )
    ;;
  strong|disturbed)
    disturbance_args=(
      --disturbances
      --force-mean-interval 1.0
      --force-duration 0.3
      --force-minimum-delta-velocity 0.25
      --force-maximum-delta-velocity 0.5
    )
    ;;
  *) echo "Mode must be nominal, training, or strong." >&2; exit 2 ;;
esac

command=(
  /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/python
  "$repo_root/scripts/evaluate_student.py"
  --task "$task"
  --agent "$agent"
  --checkpoint "$checkpoint"
  --fixed-frequency 1.5
  --steps "$steps"
  --num_envs "$num_envs"
  "${disturbance_args[@]}"
  physics=isaacsim_physx
)

echo "Evaluating 1.5 Hz $mode checkpoint: $checkpoint"
if [[ "$show_only" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec env PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "${command[@]}"
