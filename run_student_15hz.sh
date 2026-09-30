#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: ./run_student_15hz.sh [robust_checkpoint] [training|nominal]"
  echo "       ./run_student_15hz.sh --show-command [robust_checkpoint] [training|nominal]"
  echo "Defaults to the newest 1.5 Hz specialist checkpoint and its saved training disturbance."
  exit 0
fi

show_only=false
if [[ "${1:-}" == "--show-command" ]]; then
  show_only=true
  shift
fi
if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [robust_checkpoint] [training|nominal]" >&2
  exit 2
fi

if [[ "${1:-}" == "training" || "${1:-}" == "nominal" ]]; then
  mode="$1"
  checkpoint=""
else
  checkpoint="${1:-}"
  mode="${2:-training}"
fi
if [[ "$mode" != "training" && "$mode" != "nominal" ]]; then
  echo "Mode must be training or nominal." >&2
  exit 2
fi

if [[ -z "$checkpoint" ]]; then
  log_root="$repo_root/logs/rsl_rl/zbot_6dof_periodic_walking_student_robust"
  latest_run="$(find "$log_root" -mindepth 1 -maxdepth 1 -type d -name '*student_15hz_force_specialization*' \
    -printf '%p\n' | sort -r | head -n 1)"
  if [[ -z "$latest_run" ]]; then
    echo "No 1.5 Hz specialist run found in $log_root" >&2
    exit 1
  fi
  checkpoint_name="$(find "$latest_run" -mindepth 1 -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' |
    sort -V | tail -n 1)"
  if [[ -z "$checkpoint_name" ]]; then
    echo "No checkpoint found in $latest_run" >&2
    exit 1
  fi
  checkpoint="$latest_run/$checkpoint_name"
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "Student checkpoint not found: $checkpoint" >&2
  exit 1
fi
checkpoint="$(realpath -- "$checkpoint")"
env_yaml="$(dirname -- "$checkpoint")/params/env.yaml"
if [[ ! -f "$env_yaml" ]]; then
  echo "Saved environment configuration not found: $env_yaml" >&2
  exit 2
fi
if ! /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/python - "$env_yaml" <<'PY'
import sys
import yaml
with open(sys.argv[1]) as file:
    cfg = yaml.load(file, Loader=yaml.BaseLoader)
command = cfg['commands']['weight_shift']
if float(command['minimum_frequency']) != 1.5 or float(command['maximum_frequency']) != 1.5:
    raise SystemExit('Checkpoint was not trained with a fixed 1.5 Hz command.')
PY
then
  exit 2
fi

if [[ "$show_only" == true ]]; then
  exec "$repo_root/run_student_robust.sh" --show-command 1.5 "$checkpoint" "$mode"
fi
exec "$repo_root/run_student_robust.sh" 1.5 "$checkpoint" "$mode"
