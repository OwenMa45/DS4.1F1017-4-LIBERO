#!/usr/bin/env bash
# Explicit five-task launch. Repeating the command skips recorded trials.
set -euo pipefail
out=${1:?Usage: bash scripts/run_five_tasks.sh OUTPUT_PLAN_DIRECTORY}
sim=${LIBERO_SIM_RUNNER:-/mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh}
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ -f "$sim" ]] || { printf '%s\n' 'Simulation runner missing.' >&2; exit 2; }
if [[ -z ${DEEPSEEK_API_KEY:-} && ! -s "$project_root/ds_API.txt" ]]; then
  printf '%s\n' 'Export DEEPSEEK_API_KEY or fill project ds_API.txt first.' >&2
  exit 2
fi
if [[ ! -f "$out/plan.json" ]]; then
  bash "$sim" -B -m ds_libero plan-five \
    --config "$project_root/configs/direct_v2_geometry_1024_high.json" --out "$out"
fi
bash "$sim" -u -B -m ds_libero run-plan "$out"
printf 'Read per-task results, cost and latency in %s/aggregate.json and %s/episodes.csv\n' "$out" "$out"
