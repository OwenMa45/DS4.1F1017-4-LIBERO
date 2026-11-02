#!/usr/bin/env bash
# Creates a new frozen plan and runs the five GPT trials sequentially.
set -euo pipefail
out=${1:?Usage: bash scripts/run_gpt_five.sh NEW_OUTPUT_DIRECTORY}
sim=${LIBERO_SIM_RUNNER:-/mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh}
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
config=configs/gpt61_sol_medium_220.json
[[ -f "$sim" ]] || { printf '%s\n' 'Simulation runner missing.' >&2; exit 2; }
[[ ! -e "$out" ]] || { printf '%s\n' 'Output exists; inspect it and use run-plan to continue only unstarted trials.' >&2; exit 2; }
bash "$sim" -B -m ds_libero check-config --config "$config"
bash "$sim" -B -m ds_libero plan-five --config "$config" --out "$out"
exec bash "$sim" -u -B -m ds_libero run-plan "$out"
