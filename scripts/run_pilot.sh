#!/usr/bin/env bash
# Foreground pipeline: process exit is visible; no indefinitely running tail -f.
set -euo pipefail
config=${1:-configs/direct_v2_high.json}
out=${2:?Usage: bash scripts/run_pilot.sh CONFIG NEW_OUTPUT_DIRECTORY}
sim=${LIBERO_SIM_RUNNER:-/mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh}
if [[ -z ${DEEPSEEK_API_KEY:-} ]]; then
  printf '%s\n' 'Cannot start: export DEEPSEEK_API_KEY in this shell first.' >&2
  exit 2
fi
[[ -f "$config" && -f "$sim" ]] || { printf '%s\n' 'Config or simulation runner missing.' >&2; exit 2; }
mkdir -p "$(dirname "$out")"
mkdir "$out" # Refuse to overwrite an existing campaign.
check_pass() {
  bash "$sim" -B -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("status")=="pass" else 2)' "$1/result.json"
}
bash "$sim" -u -B -m ds_libero calibrate --config "$config" --out "$out/calibration" 2>&1 | tee "$out/calibration.console.log"
check_pass "$out/calibration"
bash "$sim" -u -B -m ds_libero api-check --config "$config" --calibration "$out/calibration" --out "$out/api-check" 2>&1 | tee "$out/api-check.console.log"
# invalid_output is a completed diagnostic with exit code 0; it must still stop this pipeline.
check_pass "$out/api-check"
bash "$sim" -u -B -m ds_libero run --config "$config" --calibration "$out/calibration" --api-check "$out/api-check" --out "$out/episode" 2>&1 | tee "$out/episode.console.log"
printf 'Pipeline finished. Read status and task_success in %s/episode/summary.json\n' "$out"
