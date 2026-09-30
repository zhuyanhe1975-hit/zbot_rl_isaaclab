#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: ./run_student_robust.sh [frequency_hz|range] [robust_checkpoint] [nominal|training|disturbed]"
  echo "       ./run_student_robust.sh --show-command [frequency_hz|range] [robust_checkpoint] [mode]"
  echo "Examples: ./run_student_robust.sh 0.5"
  echo "          ./run_student_robust.sh range disturbed"
  exit 0
fi

show_only=false
if [[ "${1:-}" == "--show-command" ]]; then
  show_only=true
  shift
fi
if [[ $# -gt 3 ]]; then
  echo "Usage: $0 [frequency_hz|range] [robust_checkpoint] [nominal|disturbed]" >&2
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

default_checkpoint="$repo_root/logs/rsl_rl/zbot_6dof_periodic_walking_student_robust/2026-09-27_16-22-11_student_force_anchored_stage1/model_25.pt"
if [[ $# -eq 2 && ( "$2" == "nominal" || "$2" == "disturbed" ) ]]; then
  checkpoint="$default_checkpoint"
  mode="$2"
else
  checkpoint="${2:-$default_checkpoint}"
  mode="${3:-nominal}"
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "Robust student checkpoint not found: $checkpoint" >&2
  exit 1
fi
checkpoint="$(realpath -- "$checkpoint")"
run_json="$(dirname -- "$checkpoint")/run.json"
if [[ ! -f "$run_json" ]] || [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task"])' "$run_json")" \
  != "ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Robust" ]]; then
  echo "Checkpoint does not belong to a robust student run: $checkpoint" >&2
  exit 2
fi

case "$mode" in
  nominal) extra_args=() ;;
  training)
    env_yaml="$(dirname -- "$checkpoint")/params/env.yaml"
    if [[ ! -f "$env_yaml" ]]; then
      echo "Saved environment configuration not found: $env_yaml" >&2
      exit 2
    fi
    force_values="$(/home/yhzhu/AI/IsaacLab-3.0/.venv/bin/python - "$env_yaml" <<'PY'
import sys
import yaml
with open(sys.argv[1]) as file:
    cfg = yaml.load(file, Loader=yaml.BaseLoader)
force = cfg['events']['external_force_pulse']['params']
print(*(force[key] for key in (
    'mean_interval_s', 'duration_s', 'minimum_delta_velocity',
    'maximum_delta_velocity', 'ramp_steps', 'ramp_offset_steps'
)))
PY
)"
    read -r force_interval force_duration force_min force_max ramp_steps ramp_offset <<< "$force_values"
    extra_args=(
      --train_env_cfg
      "env.events.external_force_pulse.params.visualize_force=true"
      "env.events.external_force_pulse.params.mean_interval_s=$force_interval"
      "env.events.external_force_pulse.params.duration_s=$force_duration"
      "env.events.external_force_pulse.params.minimum_delta_velocity=$force_min"
      "env.events.external_force_pulse.params.maximum_delta_velocity=$force_max"
      "env.events.external_force_pulse.params.ramp_steps=$ramp_steps"
      "env.events.external_force_pulse.params.ramp_offset_steps=$ramp_offset"
      "env.commands.weight_shift.resampling_time_range=(1000000000.0,1000000000.0)"
    )
    ;;
  disturbed)
    extra_args=(
      --train_env_cfg
      "env.events.external_force_pulse.params.ramp_steps=0"
      "env.events.external_force_pulse.params.visualize_force=true"
      "env.events.external_force_pulse.params.mean_interval_s=1.0"
      "env.events.external_force_pulse.params.duration_s=0.3"
      "env.events.external_force_pulse.params.minimum_delta_velocity=0.25"
      "env.events.external_force_pulse.params.maximum_delta_velocity=0.5"
      "env.commands.weight_shift.resampling_time_range=(1000000000.0,1000000000.0)"
    )
    ;;
  *) echo "Mode must be nominal, training, or disturbed." >&2; exit 2 ;;
esac

command=(
  /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/isaaclab play
  --rl_library rsl_rl
  --task ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Robust
  --agent rsl_rl_cfg_entry_point
  --checkpoint "$checkpoint"
  --num_envs "$num_envs"
  --viz newton_gl
  physics=isaacsim_physx
  "env.commands.weight_shift.minimum_frequency=$min_frequency"
  "env.commands.weight_shift.maximum_frequency=$max_frequency"
  env.commands.weight_shift.ordered=true
  "${extra_args[@]}"
)

printf 'Playing %s checkpoint: %s\n' "$mode" "$checkpoint"
if [[ "$mode" == "disturbed" ]]; then
  echo "Visual stress test: random force pulses every 1 s on average, 0.3 s each, equivalent to 0.25–0.5 m/s."
  echo "This is stronger than the training/evaluation disturbance range (up to 0.2 m/s)."
fi
if [[ "$mode" == "training" ]]; then
  echo "Saved training force pulses: mean interval $force_interval s, duration $force_duration s, equivalent impulse $force_min–$force_max m/s."
fi
if [[ "$show_only" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec env PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "${command[@]}"
