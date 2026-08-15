#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wmy/workspace_vla/Online-3DGS-Monocular"
PYTHON_BIN="${PYTHON_BIN:-/home/wmy/anaconda3/envs/worldvln/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
SEED="${SEED:-43}"
VIDEO_FPS="${VIDEO_FPS:-10}"
SCENE=""
GPU_ID=""
EXP_NAME=""
RESULT_JSON=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_mydata_scene_best_tgbr75.sh --scene villageN --gpu ID [options]

Supported scenes: village1, village2, village3, village4

Options:
  --scene NAME         Scene name (required).
  --gpu ID             Physical GPU ID (required).
  --seed N             Random seed (default: 43).
  --exp-name NAME      Explicit experiment name.
  --result-json PATH   Write a machine-readable stage summary to PATH.
  --dry-run            Validate inputs and print commands without reconstruction.
  -h, --help           Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)
      SCENE="${2,,}"
      shift 2
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
    --result-json)
      RESULT_JSON="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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
  village1|village2|village3|village4) ;;
  *)
    echo "--scene must be one of: village1, village2, village3, village4" >&2
    exit 2
    ;;
esac
if [[ -z "${GPU_ID}" ]]; then
  echo "--gpu is required" >&2
  exit 2
fi

CONFIG="${ROOT}/configs/mydata/${SCENE}_best_tgbr75.yaml"
DATASET="${ROOT}/data/mydata/${SCENE}"
DATASET_NAME="MyData-${SCENE}-FrontViewBest-TGBR75"
LOG_ROOT="${ROOT}/Logs_mydata_villages_best_tgbr75"
if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="${SCENE}_best_tgbr75_seed${SEED}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S)"
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

from utils_new.frontview_coverage_recovery import validate_front_view_coverage_recovery_config
from utils_new.frontview_directional_layer import validate_front_view_directional_layer_config
from utils_new.frontview_far_field import validate_front_view_far_field_config
from utils_new.frontview_sampling import validate_front_view_sampling_config
from utils_new.frontview_scale_cover import validate_front_view_scale_cover_config
from utils_new.streaming_appearance_lod import validate_streaming_appearance_lod_config
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
    raise RuntimeError(f"Scene mismatch: expected={scene}, actual={stats.get('scene')!r}")
if frame_count <= 0 or len(cameras) != frame_count or len(image_list) != frame_count:
    raise RuntimeError(
        f"Frame contract failed: stats={frame_count}, cameras={len(cameras)}, "
        f"image_list={len(image_list)}"
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
        raise RuntimeError(f"Sparse point/ID mismatch at frame {index}: {point_shape} vs {id_shape}")

expected_name = f"MyData-{scene}-FrontViewBest-TGBR75"
for section in ("Dataset", "Testset"):
    if Path(config[section]["dataset_path"]).resolve() != dataset:
        raise RuntimeError(f"{section}.dataset_path does not match {dataset}")
    if config[section]["name"] != expected_name:
        raise RuntimeError(f"Unexpected {section}.name: {config[section]['name']!r}")

validate_front_view_coverage_recovery_config(config["FrontViewCoverageRecovery"])
validate_front_view_sampling_config(config["FrontViewSampling"])
validate_front_view_far_field_config(config["FrontViewFarField"])
validate_front_view_scale_cover_config(config["FrontViewScaleCover"])
validate_streaming_appearance_lod_config(config["StreamingAppearanceLOD"])
validate_front_view_directional_layer_config(config["FrontViewDirectionalLayer"])
assert config["HashBlock"]["use_hash"] is False
assert config["Results"]["save_online_stage"] is True
assert config["FrontViewSampling"]["depth_edges_m"] == [2.0, 5.0]
assert config["FrontViewFarField"]["depth_m"] == 8.0
assert config["StreamingAppearanceLOD"]["max_target_fraction"] == 0.75
assert config["StreamingAppearanceLOD"]["compute_routing"] is False
assert not config["StreamingAppearanceLOD"].get("spectral_residency_enabled", False)
print(
    f"Validated {scene}: {frame_count} frames, aligned persistent points/IDs, "
    "hash-free PBSD/TSC/FPR + dense TGBR-75."
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
  printf '\nGPU: physical %s\n' "${GPU_ID}"
  echo "Online and post-refinement stages will each be rendered at ${VIDEO_FPS} FPS with PSNR/SSIM only."
  echo "Output root: ${LOG_ROOT}"
  exit 0
fi

"${SLAM_CMD[@]}"

RUN_PARENT="${LOG_ROOT}/${DATASET_NAME}"
RUN_DIR="$({
  find "${RUN_PARENT}" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${EXP_NAME}" -printf '%T@ %p\n'
} | sort -nr | sed -n '1p' | cut -d' ' -f2-)"
if [[ -z "${RUN_DIR}" || ! -f "${RUN_DIR}/point_cloud.ply" ]]; then
  echo "Could not locate completed run for ${EXP_NAME}" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/online_stage/point_cloud.ply" ]]; then
  echo "Missing exact pre-refinement snapshot in ${RUN_DIR}/online_stage" >&2
  exit 1
fi
cp "${RUN_DIR}/config.yaml" "${RUN_DIR}/online_stage/config.yaml"

RENDER_ARGS=(
  --fps "${VIDEO_FPS}"
  --max_frames -1
  --device cuda:0
  --far_gs_depth_threshold 8
  --skip_lpips
  --skip_depth
  --skip_novel
  --skip_primitives
  --ignore_cached_renders
)
"${PYTHON_BIN}" render.py \
  --run_dir "${RUN_DIR}/online_stage" \
  --output_dir "${RUN_DIR}/online_stage/videos_full" \
  "${RENDER_ARGS[@]}"
"${PYTHON_BIN}" render.py \
  --run_dir "${RUN_DIR}" \
  --output_dir "${RUN_DIR}/videos_full" \
  "${RENDER_ARGS[@]}"

if [[ -z "${RESULT_JSON}" ]]; then
  RESULT_JSON="${RUN_DIR}/two_stage_summary.json"
fi
mkdir -p "$(dirname "${RESULT_JSON}")"
"${PYTHON_BIN}" - "${SCENE}" "${RUN_DIR}" "${RESULT_JSON}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

scene = sys.argv[1]
run_dir = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def gaussian_count(path):
    with path.open("rb") as stream:
        for raw in stream:
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    raise RuntimeError(f"No vertex count in {path}")

def codec(path):
    return subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt", "-of", "json", str(path),
        ],
        text=True,
    ).strip()

results = read_json(run_dir / "results.json")
online_metrics = read_json(run_dir / "online_stage/videos_full/render_metrics.json")["mean"]
post_metrics = read_json(run_dir / "videos_full/render_metrics.json")["mean"]
online_video = run_dir / "online_stage/videos_full/render_vs_gt.mp4"
post_video = run_dir / "videos_full/render_vs_gt.mp4"
payload = {
    "scene": scene,
    "run_dir": str(run_dir),
    "frame_count": int(results["num_processed_frames"]),
    "online_mapping_seconds": float(results["online_mapping_seconds"]),
    "online_fps": int(results["num_processed_frames"]) / float(results["online_mapping_seconds"]),
    "online_stage_export_seconds": float(results["online_stage_export_seconds"]),
    "post_refinement_seconds": float(results["post_refinement_seconds"]),
    "total_pipeline_seconds": float(results["online_recon_time"]),
    "online": {
        "psnr": float(online_metrics["psnr"]),
        "ssim": float(online_metrics["ssim"]),
        "gaussians": gaussian_count(run_dir / "online_stage/point_cloud.ply"),
        "video": str(online_video),
        "video_probe": json.loads(codec(online_video)),
    },
    "post_refinement": {
        "psnr": float(post_metrics["psnr"]),
        "ssim": float(post_metrics["ssim"]),
        "gaussians": gaussian_count(run_dir / "point_cloud.ply"),
        "video": str(post_video),
        "video_probe": json.loads(codec(post_video)),
    },
    "metric_protocol": "same-trajectory reconstruction PSNR/SSIM; LPIPS disabled",
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
(run_dir / "two_stage_summary.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY

echo "Completed ${SCENE}: ${RESULT_JSON}"
