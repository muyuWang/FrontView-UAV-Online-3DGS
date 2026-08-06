#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wmy/workspace_vla/Online-3DGS-Monocular"
PYTHON="/home/wmy/anaconda3/envs/worldvln/bin/python"

cd "$ROOT"
exec "$PYTHON" scripts/run_360dvo_orbmono_fixed_all_gpu4_7.py \
  --gpu-ids 4,5,6,7 \
  --skip-existing-success \
  "$@"
