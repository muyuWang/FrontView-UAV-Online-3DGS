#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wmy/workspace_vla/Online-3DGS-Monocular"
CONFIG="configs/lvba/HKU_Cultural_Center_01_geometry_locked_sh2_full.yaml"
RUN_ROOT="${ROOT}/Logs_lvba_quality/LVBA-HKU_Cultural_Center_01-geometry-locked-sh2-full"
GPU_ID="4"
EXP_NAME="${1:-lvba_hku_cultural_center_01_geometry_locked_sh2_full_v1_gpu${GPU_ID}}"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

conda run --no-capture-output -n worldvln \
  python slam_new.py \
    --config "${CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --seed 42

RUN_DIR="$({
  find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${EXP_NAME}" -printf '%T@ %p\n'
} | sort -nr | sed -n '1p' | cut -d' ' -f2-)"

if [[ -z "${RUN_DIR}" || ! -f "${RUN_DIR}/point_cloud.ply" ]]; then
  echo "Could not locate the completed run for ${EXP_NAME}" >&2
  exit 1
fi

conda run --no-capture-output -n worldvln \
  python render.py \
    --run_dir "${RUN_DIR}" \
    --output_dir "${RUN_DIR}/videos_full" \
    --fps 24 \
    --max_frames -1 \
    --device cuda:0 \
    --skip_novel \
    --skip_primitives

echo "Run directory: ${RUN_DIR}"
echo "RGB comparison: ${RUN_DIR}/videos_full/render_vs_gt.mp4"
echo "Depth video: ${RUN_DIR}/videos_full/render_depth.mp4"
echo "Metrics: ${RUN_DIR}/videos_full/render_metrics.json"
