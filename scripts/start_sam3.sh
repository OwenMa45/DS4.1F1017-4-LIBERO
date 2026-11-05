#!/usr/bin/env bash
# Activate the prepared SAM3 environment first, or set SAM3_PYTHON explicitly.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
sam3_python=${SAM3_PYTHON:-python}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$sam3_python" -u -B -m ds_libero.sam3_server serve \
  --sam3-code /mnt/cpfs/users/mrq/package/sam3/sam3 \
  --checkpoint /mnt/cpfs/users/mrq/package/sam3/checkpoints "$@"
