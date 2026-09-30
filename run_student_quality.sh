#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: ./run_student_quality.sh [frequency_hz|range] [quality_student_checkpoint]"
  echo "       ./run_student_quality.sh --show-checkpoint [frequency_hz|range] [quality_student_checkpoint]"
  echo "Examples: ./run_student_quality.sh 0.5"
  echo "          ./run_student_quality.sh range"
  exit 0
fi

show_only=false
if [[ "${1:-}" == "--show-checkpoint" ]]; then
  show_only=true
  shift
fi
if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [frequency_hz|range] [quality_student_checkpoint]" >&2
  exit 2
fi

frequency="${1:-1.5}"
if [[ "$frequency" == "range" ]]; then
  num_envs=4
  min_frequency=0.5
  max_frequency=2.0
elif [[ "$frequency" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  if ! python3 -c 'import sys; sys.exit(not 0.5 <= float(sys.argv[1]) <= 2.0)' "$frequency"; then
    echo "Frequency must be between 0.5 and 2.0 Hz." >&2
    exit 2
  fi
  num_envs=1
  min_frequency="$frequency"
  max_frequency="$frequency"
else
  echo "Frequency must be a number in Hz or 'range'." >&2
  exit 2
fi

if [[ $# -eq 2 ]]; then
  checkpoint="$(realpath -- "$2")"
else
  log_root="$repo_root/logs/rsl_rl/zbot_6dof_periodic_walking_student_quality"
  run_dir="$(
    find "$log_root" -mindepth 1 -maxdepth 1 -type d ! -name '*smoke*' \
      -printf '%p\n' | sort -r | head -n 1
  )"
  if [[ -z "$run_dir" ]]; then
    echo "No quality student run found in $log_root" >&2
    exit 1
  fi
  checkpoint_name="$(
    find "$run_dir" -mindepth 1 -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' |
      sort -V | tail -n 1
  )"
  if [[ -z "$checkpoint_name" ]]; then
    echo "No checkpoint found in $run_dir" >&2
    exit 1
  fi
  checkpoint="$run_dir/$checkpoint_name"
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "Quality student checkpoint not found: $checkpoint" >&2
  exit 1
fi

run_json="$(dirname -- "$checkpoint")/run.json"
if [[ ! -f "$run_json" ]] || [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task"])' "$run_json")" \
  != "ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Quality" ]]; then
  echo "Checkpoint does not belong to a quality student run: $checkpoint" >&2
  exit 2
fi

echo "Playing quality student checkpoint: $checkpoint"
if [[ "$show_only" == true ]]; then
  exit 0
fi

exec env PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/isaaclab play \
  --rl_library rsl_rl \
  --task ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Quality \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint "$checkpoint" \
  --num_envs "$num_envs" \
  --viz newton_gl \
  physics=isaacsim_physx \
  "env.commands.weight_shift.minimum_frequency=$min_frequency" \
  "env.commands.weight_shift.maximum_frequency=$max_frequency" \
  env.commands.weight_shift.ordered=true
