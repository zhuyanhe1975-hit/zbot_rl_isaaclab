#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: ./run_student.sh [frequency_hz|range] [nominal_finetune_checkpoint]"
  echo "Examples: ./run_student.sh"
  echo "          ./run_student.sh 0.5"
  echo "          ./run_student.sh range"
  exit 0
fi
if [[ $# -gt 2 ]]; then
  echo "Usage: ./run_student.sh [frequency_hz|range] [nominal_finetune_checkpoint]" >&2
  exit 2
fi

frequency="${1:-1.5}"
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
  exit 2
fi

if [[ $# -eq 2 ]]; then
  checkpoint="$(realpath -- "$2")"
else
  log_root="$repo_root/logs/rsl_rl/zbot_6dof_periodic_walking_student_finetune"
  checkpoint="$(
    find "$log_root" -mindepth 2 -maxdepth 2 -type f -name 'model_*.pt' \
      ! -path '*smoke*' ! -path '*pilot*' -printf '%T@ %p\n' |
      sort -nr |
      sed -n '1s/^[^ ]* //p'
  )"
fi
if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
  echo "No nominal student fine-tuning checkpoint found: ${checkpoint:-<none>}" >&2
  exit 1
fi

echo "Playing checkpoint: $checkpoint"
exec env PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  /home/yhzhu/AI/IsaacLab-3.0/.venv/bin/isaaclab play \
  --rl_library rsl_rl \
  --task ZbotRlIsaaclab-6DOF-Periodic-Walking-Student-Finetune \
  --agent rsl_rl_cfg_entry_point \
  --checkpoint "$checkpoint" \
  --num_envs "$num_envs" \
  --viz newton_gl \
  physics=isaacsim_physx \
  "env.commands.weight_shift.minimum_frequency=$min_frequency" \
  "env.commands.weight_shift.maximum_frequency=$max_frequency" \
  env.commands.weight_shift.ordered=true
