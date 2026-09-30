#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: ./play_student.sh [frequency_hz|range] [checkpoint]"
  echo "Examples: ./play_student.sh 0.5"
  echo "          ./play_student.sh range"
  exit 0
fi

frequency="${1:-1.5}"
checkpoint="${2:-$repo_root/logs/rsl_rl/zbot_6dof_periodic_walking_student/2026-09-26_17-19-41/model_200.pt}"
if [[ ! -f "$checkpoint" ]]; then
  echo "Student checkpoint not found: $checkpoint" >&2
  exit 1
fi

if [[ "$frequency" == "range" ]]; then
  num_envs=4
  min_frequency=0.5
  max_frequency=2.0
elif [[ "$frequency" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  num_envs=1
  min_frequency="$frequency"
  max_frequency="$frequency"
else
  echo "Frequency must be a number in Hz or 'range'." >&2
  exit 1
fi

exec env PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/isaaclab play \
  --rl_library rsl_rl \
  --task ZbotRlIsaaclab-6DOF-Periodic-Walking-Student \
  --agent rsl_rl_distillation_cfg_entry_point \
  --checkpoint "$checkpoint" \
  --num_envs "$num_envs" \
  --viz newton_gl \
  physics=isaacsim_physx \
  "env.commands.weight_shift.minimum_frequency=$min_frequency" \
  "env.commands.weight_shift.maximum_frequency=$max_frequency" \
  env.commands.weight_shift.ordered=true
