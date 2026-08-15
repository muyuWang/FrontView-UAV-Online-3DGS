#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wmy/workspace_vla/Online-3DGS-Monocular"
PYTHON_BIN="${PYTHON_BIN:-/home/wmy/anaconda3/envs/worldvln/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
SEED="${SEED:-43}"
VIDEO_FPS="${VIDEO_FPS:-30}"
SCENE=""
GPU_ID=""
DRY_RUN=0
EXP_NAME=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_pano360_scene_best_tgbr75.sh --scene SCENE [options]

Supported scenes: FTP, NSK, NSC

Options:
  --scene NAME       Pano360 scene name (required).
  --dry-run          Validate data/config and print commands without reconstruction.
  --gpu ID           Physical GPU ID (defaults: FTP=5, NSK=6, NSC=4).
  --seed N           Random seed (default: 43, or SEED environment variable).
  --exp-name NAME    Explicit experiment name.
  -h, --help         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)
      SCENE="${2^^}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --exp-name)
      EXP_NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${SCENE}" in
  FTP)
    SCENE_LOWER="ftp"
    DEFAULT_GPU=5
    ;;
  NSK)
    SCENE_LOWER="nsk"
    DEFAULT_GPU=6
    ;;
  NSC)
    SCENE_LOWER="nsc"
    DEFAULT_GPU=4
    ;;
  *)
    echo "--scene must be one of: FTP, NSK, NSC" >&2
    usage >&2
    exit 2
    ;;
esac

GPU_ID="${GPU_ID:-${DEFAULT_GPU}}"
CONFIG="${ROOT}/configs/pano360/${SCENE_LOWER}_frontview_best_tgbr75.yaml"
DATASET="${ROOT}/data/Online3DGS_pano360/${SCENE}"
DATASET_NAME="Pano360-${SCENE}-FrontViewBest-TGBR75"
LOG_ROOT="${ROOT}/Logs_pano360_${SCENE_LOWER}_best_tgbr75"

if [[ -z "${EXP_NAME}" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  EXP_NAME="${SCENE_LOWER}_frontview_best_tgbr75_seed${SEED}_gpu${GPU_ID}_${STAMP}"
fi

for required in "${PYTHON_BIN}" "${CONFIG}" "${DATASET}/trajectory_orb.json" \
  "${DATASET}/conversion_stats.json" "${DATASET}/image_list.txt"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

cd "${ROOT}"

"${PYTHON_BIN}" - "${CONFIG}" "${DATASET}" "${SCENE}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

from utils_new.frontview_coverage_recovery import (
    validate_front_view_coverage_recovery_config,
)
from utils_new.frontview_directional_layer import (
    validate_front_view_directional_layer_config,
)
from utils_new.frontview_far_field import validate_front_view_far_field_config
from utils_new.frontview_sampling import validate_front_view_sampling_config
from utils_new.frontview_scale_cover import validate_front_view_scale_cover_config
from utils_new.streaming_appearance_lod import (
    validate_streaming_appearance_lod_config,
)
from utils_new.tool_utils import load_config

config_path = Path(sys.argv[1]).resolve()
dataset = Path(sys.argv[2]).resolve()
scene = sys.argv[3]
config = load_config(str(config_path))

stats = json.loads((dataset / "conversion_stats.json").read_text(encoding="utf-8"))
trajectory = json.loads((dataset / "trajectory_orb.json").read_text(encoding="utf-8"))
image_list = [
    line.strip()
    for line in (dataset / "image_list.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
cameras = trajectory.get("cameras", [])
frame_count = int(stats.get("frame_count", -1))
if stats.get("scene") != scene:
    raise RuntimeError(
        f"Scene contract failed: expected={scene}, stats={stats.get('scene')!r}"
    )
if frame_count <= 0 or len(cameras) != frame_count or len(image_list) != frame_count:
    raise RuntimeError(
        "Frame contract failed: "
        f"stats={frame_count}, cameras={len(cameras)}, image_list={len(image_list)}"
    )

for index, camera in enumerate(cameras):
    image = dataset / "rectified" / camera["image"]
    points = dataset / "orb_point_clouds" / f"point_cloud_{index}.npy"
    point_ids = dataset / "orb_point_ids" / f"point_ids_{index}.npy"
    for required in (image, points, point_ids):
        if not required.is_file():
            raise FileNotFoundError(required)
    point_shape = np.load(points, mmap_mode="r").shape
    id_shape = np.load(point_ids, mmap_mode="r").shape
    if len(point_shape) != 2 or point_shape[1] != 3 or point_shape[0] != id_shape[0]:
        raise RuntimeError(
            f"Sparse point/ID mismatch at frame {index}: {point_shape} vs {id_shape}"
        )

expected_name = f"Pano360-{scene}-FrontViewBest-TGBR75"
if Path(config["Dataset"]["dataset_path"]).resolve() != dataset:
    raise RuntimeError("Dataset.dataset_path does not match the validated dataset")
if Path(config["Testset"]["dataset_path"]).resolve() != dataset:
    raise RuntimeError("Testset.dataset_path does not match the validated dataset")
if config["Dataset"]["name"] != expected_name:
    raise RuntimeError(f"Unexpected dataset name: {config['Dataset']['name']!r}")

validate_front_view_coverage_recovery_config(config["FrontViewCoverageRecovery"])
validate_front_view_sampling_config(config["FrontViewSampling"])
validate_front_view_far_field_config(config["FrontViewFarField"])
validate_front_view_scale_cover_config(config["FrontViewScaleCover"])
validate_streaming_appearance_lod_config(config["StreamingAppearanceLOD"])
validate_front_view_directional_layer_config(config["FrontViewDirectionalLayer"])

assert config["HashBlock"]["use_hash"] is False
assert config["StreamingAppearanceLOD"]["enabled"] is True
assert config["StreamingAppearanceLOD"]["selection_mode"] == "gradient_agreement"
assert config["StreamingAppearanceLOD"]["max_target_fraction"] == 0.75
assert config["StreamingAppearanceLOD"]["compute_routing"] is False
assert config["StreamingAppearanceLOD"]["bounded_replay_residency_enabled"] is False
assert not config["StreamingAppearanceLOD"].get("spectral_residency_enabled", False)
assert config["FrontViewFarField"]["depth_m"] == 8.0
assert config["FrontViewSampling"]["depth_edges_m"] == [2.0, 5.0]
print(
    f"Validated {scene}: {frame_count} frames, aligned sparse points/IDs, "
    "hash-free PBSD/TSC/FPR + dense TGBR-75 config."
)
PY

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/torch_extensions/online3dgs_gpu${GPU_ID}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

SLAM_CMD=(
  "${PYTHON_BIN}" slam_new.py
  --config "${CONFIG}"
  --exp_name "${EXP_NAME}"
  --seed "${SEED}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  "${PYTHON_BIN}" slam_new.py --help >/dev/null
  printf 'SLAM: '
  printf '%q ' "${SLAM_CMD[@]}"
  printf '\n'
  echo "GPU: physical ${GPU_ID} (CUDA_VISIBLE_DEVICES=${GPU_ID})"
  echo "After reconstruction, the script renders all ${VIDEO_FPS} FPS frames, computes PSNR/SSIM only, and writes render_vs_gt.mp4."
  echo "Output root: ${LOG_ROOT}"
  echo "Dry-run complete; reconstruction was not started."
  exit 0
fi

"${SLAM_CMD[@]}"

RUN_PARENT="${LOG_ROOT}/${DATASET_NAME}"
RUN_DIR="$({
  find "${RUN_PARENT}" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${EXP_NAME}" -printf '%T@ %p\n'
} | sort -nr | sed -n '1p' | cut -d' ' -f2-)"

if [[ -z "${RUN_DIR}" || ! -f "${RUN_DIR}/point_cloud.ply" ]]; then
  echo "Could not locate the completed run for ${EXP_NAME}" >&2
  exit 1
fi

"${PYTHON_BIN}" render.py \
  --run_dir "${RUN_DIR}" \
  --output_dir "${RUN_DIR}/videos_full" \
  --fps "${VIDEO_FPS}" \
  --max_frames -1 \
  --device cuda:0 \
  --far_gs_depth_threshold 8 \
  --skip_lpips \
  --skip_novel \
  --skip_primitives \
  --ignore_cached_renders

"${PYTHON_BIN}" - "${RUN_DIR}" <<'PY' | tee "${RUN_DIR}/metrics_summary.txt"
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
metrics = json.loads(
    (run_dir / "videos_full" / "render_metrics.json").read_text(encoding="utf-8")
)["mean"]
frames = int(results["num_processed_frames"])
seconds = float(results["online_recon_time"])
print(f"PSNR: {metrics['psnr']:.6f} dB")
print(f"SSIM: {metrics['ssim']:.6f}")
print(f"Online reconstruction time: {seconds:.3f} s")
print(f"Online speed: {frames / seconds:.4f} FPS")
print(f"Gaussians: {int(results['num_gaussians'])}")
PY

echo "Run directory: ${RUN_DIR}"
echo "Metrics: ${RUN_DIR}/videos_full/render_metrics.json"
echo "RGB comparison video: ${RUN_DIR}/videos_full/render_vs_gt.mp4"
echo "Depth video: ${RUN_DIR}/videos_full/render_depth.mp4"
