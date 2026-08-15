#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wmy/workspace_vla/Online-3DGS-Monocular"
STAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_ROOT="${ROOT}/Logs_pano360_ftp_nsk_launch"
mkdir -p "${LAUNCH_LOG_ROOT}"

FTP_LOG="${LAUNCH_LOG_ROOT}/ftp_gpu5_${STAMP}.log"
NSK_LOG="${LAUNCH_LOG_ROOT}/nsk_gpu6_${STAMP}.log"

echo "Starting FTP on GPU 5; console log: ${FTP_LOG}"
bash "${ROOT}/scripts/run_pano360_ftp_best_tgbr75.sh" --gpu 5 "$@" \
  >"${FTP_LOG}" 2>&1 &
FTP_PID=$!

echo "Starting NSK on GPU 6; console log: ${NSK_LOG}"
bash "${ROOT}/scripts/run_pano360_nsk_best_tgbr75.sh" --gpu 6 "$@" \
  >"${NSK_LOG}" 2>&1 &
NSK_PID=$!

set +e
wait "${FTP_PID}"
FTP_STATUS=$?
wait "${NSK_PID}"
NSK_STATUS=$?
set -e

echo "FTP exit status: ${FTP_STATUS}"
echo "NSK exit status: ${NSK_STATUS}"
if [[ "${FTP_STATUS}" -ne 0 || "${NSK_STATUS}" -ne 0 ]]; then
  exit 1
fi
