#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: ./train_student_15hz.sh [robust_checkpoint] [num_envs] [additional_iterations] [run_name]"
  echo "       ./train_student_15hz.sh --resume-latest [num_envs] [additional_iterations] [run_name]"
  echo "       ./train_student_15hz.sh --show-command [other arguments]"
  echo "Without --resume-latest or a checkpoint, training restarts from robust model_25.pt."
  echo "Optional: ZBOT_FORCE_MIN_DELTA_V and ZBOT_FORCE_MAX_DELTA_V set the disturbance range in m/s."
  exit 0
fi
show_only=false
if [[ "${1:-}" == "--show-command" ]]; then
  show_only=true
  shift
fi
resume_latest=false
if [[ "${1:-}" == "--resume-latest" ]]; then
  resume_latest=true
  shift
fi
if [[ ( "$resume_latest" == true && $# -gt 3 ) || ( "$resume_latest" == false && $# -gt 4 ) ]]; then
  echo "Usage: $0 [robust_checkpoint] [num_envs] [additional_iterations] [run_name]" >&2
  exit 2
fi

if [[ "$resume_latest" == true ]]; then
  log_root="$repo_root/logs/rsl_rl/zbot_6dof_periodic_walking_student_robust"
  latest_run="$(find "$log_root" -mindepth 1 -maxdepth 1 -type d -name '*student_15hz_force_specialization*' \
    -printf '%p\n' | sort -r | head -n 1)"
  if [[ -z "$latest_run" ]]; then
    echo "No previous 1.5 Hz specialization run found in $log_root" >&2
    exit 2
  fi
  checkpoint_name="$(find "$latest_run" -mindepth 1 -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' |
    sort -V | tail -n 1)"
  if [[ -z "$checkpoint_name" ]]; then
    echo "No checkpoint found in $latest_run" >&2
    exit 2
  fi
  checkpoint="$latest_run/$checkpoint_name"
  num_envs="${1:-4096}"
  iterations="${2:-250}"
  run_name="${3:-student_15hz_force_specialization_resume}"
else
  checkpoint="${1:-$repo_root/logs/rsl_rl/zbot_6dof_periodic_walking_student_robust/2026-09-27_16-22-11_student_force_anchored_stage1/model_25.pt}"
  num_envs="${2:-4096}"
  iterations="${3:-250}"
  run_name="${4:-student_15hz_force_specialization}"
fi
force_min="${ZBOT_FORCE_MIN_DELTA_V:-0.05}"
force_max="${ZBOT_FORCE_MAX_DELTA_V:-0.25}"
if [[ ! -f "$checkpoint" || ! "$num_envs" =~ ^[1-9][0-9]*$ || ! "$iterations" =~ ^[1-9][0-9]*$ ]]; then
  echo "Provide a robust checkpoint, positive num_envs, and positive additional_iterations." >&2
  exit 2
fi
if ! python3 -c 'import sys; a,b=map(float,sys.argv[1:]); sys.exit(not 0 <= a <= b)' "$force_min" "$force_max"; then
  echo "Force limits must satisfy 0 <= ZBOT_FORCE_MIN_DELTA_V <= ZBOT_FORCE_MAX_DELTA_V." >&2
  exit 2
fi
checkpoint="$(realpath -- "$checkpoint")"
run_json="$(dirname -- "$checkpoint")/run.json"
if [[ ! -f "$run_json" ]] || [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task"])' "$run_json")" \
  != "ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Robust" ]]; then
  echo "Checkpoint must belong to a robust student run: $checkpoint" >&2
  exit 2
fi

command=(
  /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/python
  "$repo_root/scripts/train_student_finetune.py"
  --rl_library rsl_rl
  physics=isaacsim_physx
  --task ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Robust
  --checkpoint "$checkpoint"
  --num_envs "$num_envs"
  --max_iterations "$iterations"
  --run_name "$run_name"
  env.commands.weight_shift.minimum_frequency=1.5
  env.commands.weight_shift.maximum_frequency=1.5
  env.commands.weight_shift.ordered=true
  "env.commands.weight_shift.resampling_time_range=(1000000000.0,1000000000.0)"
  env.events.external_force_pulse.params.mean_interval_s=1.0
  env.events.external_force_pulse.params.duration_s=0.3
  "env.events.external_force_pulse.params.minimum_delta_velocity=$force_min"
  "env.events.external_force_pulse.params.maximum_delta_velocity=$force_max"
  env.events.external_force_pulse.params.ramp_steps=0
)

echo "Specializing robust student at 1.5 Hz from: $checkpoint"
echo "Random physical force pulses: mean interval 1 s, duration 0.3 s, equivalent impulse $force_min–$force_max m/s."
if [[ "$resume_latest" == false && $# -eq 0 ]]; then
  echo "Fresh specialization from model_25.pt. Use --resume-latest to continue the most recent specialization."
fi
if [[ "$show_only" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec env PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "${command[@]}"
