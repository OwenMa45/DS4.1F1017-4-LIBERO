#!/usr/bin/env bash
# SAM3 worker must already be running; never install GPU dependencies in LIBERO.
set -euo pipefail
out=${1:?Usage: bash scripts/run_sam3_five.sh NEW_OUTPUT_DIRECTORY [CONFIG]}
config=${2:-configs/deepseek_sam3_220.json}
sim=${LIBERO_SIM_RUNNER:-/mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh}
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
[[ -f "$sim" ]] || { printf '%s\n' 'Simulation runner missing.' >&2; exit 2; }
[[ ! -e "$out" && ! -e "${out}-sam3-check" ]] || { printf '%s\n' 'Output exists; use new output paths.' >&2; exit 2; }
bash "$sim" -B -m ds_libero check-config --config "$config"
# Real SAM3 forward pass on a blank image. No DeepSeek/API charge or robot motion.
bash "$sim" -u -B -m ds_libero.sam3_server check --config "$config" --out "${out}-sam3-check"
bash "$sim" -B -m ds_libero microbench plan --config "$config" \
  --policy-id deepseek-v4.1-flash-1017-sam3 --revision sam3-tools-v3-action-repair --out "$out"
exec bash "$sim" -u -B -m ds_libero microbench run "$out"
