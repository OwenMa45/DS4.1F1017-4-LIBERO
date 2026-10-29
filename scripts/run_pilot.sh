#!/usr/bin/env bash
# Foreground pipeline: process exit is visible; no indefinitely running tail -f.
set -euo pipefail
config=${1:-configs/direct_v2_high.json}
out=${2:?Usage: bash scripts/run_pilot.sh CONFIG NEW_OUTPUT_DIRECTORY}
sim=${LIBERO_SIM_RUNNER:-/mnt/cpfs/users/mrq/libero_cpu/run_cpu.sh}
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z ${DEEPSEEK_API_KEY:-} && ! -s "$project_root/ds_API.txt" ]]; then
  printf '%s\n' 'Cannot start: export DEEPSEEK_API_KEY or fill project ds_API.txt first.' >&2
  exit 2
fi
[[ -f "$config" && -f "$sim" ]] || { printf '%s\n' 'Config or simulation runner missing.' >&2; exit 2; }
mkdir -p "$(dirname "$out")"
mkdir "$out" # Refuse to overwrite an existing campaign.
finish_report() {
  local result=$?
  trap - EXIT
  # Runs even if a paid API-check failed; keep the pipeline's original exit status.
  if ! bash "$sim" -B -m ds_libero report-campaign "$out" > "$out/campaign_report.console.log" 2>&1; then
    printf 'Campaign report failed; inspect %s/campaign_report.console.log\n' "$out" >&2
    if [[ $result -eq 0 ]]; then result=2; fi
  fi
  exit "$result"
}
trap finish_report EXIT
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
