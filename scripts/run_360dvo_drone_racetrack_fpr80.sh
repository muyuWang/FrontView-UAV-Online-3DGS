#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wmy/workspace_vla/Online-3DGS-Monocular"
CONFIG="configs/360dvo/360DVO_drone_racetrack_fpr80.yaml"
DATASET_NAME="360DVO-FPR80-drone-racetrack"
RUN_ROOT="${ROOT}/Logs_360dvo_fpr80/${DATASET_NAME}"
PYTHON_BIN="${PYTHON_BIN:-/home/wmy/anaconda3/envs/worldvln/bin/python}"
GPU_ID="${GPU_ID:-4}"
SEED="${SEED:-42}"
STAMP="$(date +%Y-%m-%d-%H-%M-%S)"
EXP_NAME="${EXP_NAME:-360dvo_drone_racetrack_fpr80_seed${SEED}_gpu${GPU_ID}_${STAMP}}"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" slam_new.py \
  --config "${CONFIG}" \
  --exp_name "${EXP_NAME}" \
  --seed "${SEED}"

RUN_DIR="$({
  find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${EXP_NAME}" -printf '%T@ %p\n'
} | sort -nr | sed -n '1p' | cut -d' ' -f2-)"

if [[ -z "${RUN_DIR}" || ! -f "${RUN_DIR}/point_cloud.ply" ]]; then
  echo "Could not locate the completed run for ${EXP_NAME}" >&2
  exit 1
fi

"${PYTHON_BIN}" render.py \
  --run_dir "${RUN_DIR}" \
  --output_dir "${RUN_DIR}/videos_full" \
  --fps 24 \
  --max_frames -1 \
  --device cuda:0 \
  --far_gs_depth_threshold 80 \
  --skip_novel \
  --skip_primitives

"${PYTHON_BIN}" - "${RUN_DIR}/results.json" <<'PY' | tee "${RUN_DIR}/metrics_summary.txt"
import json
import sys

results = json.load(open(sys.argv[1], "r", encoding="utf-8"))
metrics = results["eval_res"]
fpr = results.get("frontview_far_field", {})
print(f"PSNR: {metrics['psnr']:.6f} dB")
print(f"SSIM: {metrics['ssim']:.6f}")
print(f"LPIPS: {metrics['lpips']:.6f}")
print(f"Online reconstruction time: {results['online_recon_time']:.3f} s")
print(f"Gaussians: {results['num_gaussians']}")
print(f"FPR hash bypass rows: {fpr.get('hash_bypass_rows', 0)}")
print(f"FPR projective rejected rows: {fpr.get('projective_rejected_rows', 0)}")
PY

echo "Run directory: ${RUN_DIR}"
echo "Metrics summary: ${RUN_DIR}/metrics_summary.txt"
echo "Detailed render metrics: ${RUN_DIR}/videos_full/render_metrics.json"
echo "RGB comparison video: ${RUN_DIR}/videos_full/render_vs_gt.mp4"
echo "Depth video: ${RUN_DIR}/videos_full/render_depth.mp4"
