#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wmy/workspace_vla/Online-3DGS-Monocular"
exec bash "${ROOT}/scripts/run_pano360_scene_best_tgbr75.sh" \
  --scene NSK \
  --gpu "${GPU_ID:-6}" \
  "$@"
