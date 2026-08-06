#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wmy/workspace_vla/Online-3DGS-Monocular"
CONFIG="configs/experiments_7_22/road_evidence_lod_online25.yaml"
RUN_ROOT="${ROOT}/Logs_far_near_optimization_7_22"
DATASET_NAME="HorizonGS-road-street1-evidence-lod-online25"
GPU_ID="${GPU_ID:-4}"
EXP_NAME="${1:-road_evidence_lod_v1_gpu${GPU_ID}}"
STAMP="$(date +%Y-%m-%d-%H-%M-%S)"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

conda run --no-capture-output -n worldvln \
  python slam_new.py \
    --config "${CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --seed 42

ONLINE_RUN="$({
  find "${RUN_ROOT}/${DATASET_NAME}" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${EXP_NAME}" -printf '%T@ %p\n'
} | sort -nr | sed -n '1p' | cut -d' ' -f2-)"

if [[ -z "${ONLINE_RUN}" || ! -f "${ONLINE_RUN}/point_cloud.ply" ]]; then
  echo "Could not locate the completed online run for ${EXP_NAME}" >&2
  exit 1
fi

REFINED_RUN="${RUN_ROOT}/Road-evidence-lod-final/${STAMP}_${EXP_NAME}"
conda run --no-capture-output -n worldvln \
  python scripts/refine_saved_map_appearance.py \
    --source-run "${ONLINE_RUN}" \
    --output-run "${REFINED_RUN}" \
    --steps 400 \
    --sh-degree 2 \
    --appearance-lod evidence \
    --sh-lr 0.002 \
    --opacity-lr 0.02 \
    --hard-fraction 0 \
    --color-loss-type l2 \
    --ssim-weight 0 \
    --seed 42

conda run --no-capture-output -n worldvln \
  python render.py \
    --run_dir "${REFINED_RUN}" \
    --output_dir "${REFINED_RUN}/videos_full" \
    --fps 24 \
    --max_frames -1 \
    --device cuda:0 \
    --skip_novel \
    --skip_primitives

echo "Online run: ${ONLINE_RUN}"
echo "Refined run: ${REFINED_RUN}"
echo "RGB comparison: ${REFINED_RUN}/videos_full/render_vs_gt.mp4"
echo "Depth video: ${REFINED_RUN}/videos_full/render_depth.mp4"
echo "Metrics: ${REFINED_RUN}/videos_full/render_metrics.json"
