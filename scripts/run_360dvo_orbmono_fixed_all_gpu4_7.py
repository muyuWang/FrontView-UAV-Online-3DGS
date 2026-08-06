#!/usr/bin/env python3
"""Run image-certified pose-contract TGBR-75 on all 360DVO scenes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "Online3DGS_360DVO"
DEFAULT_PREPARED_ROOT = ROOT / "data" / "Online3DGS_360DVO_pose_contract_v4"
DEFAULT_WORK_ROOT = ROOT / "data" / "Online3DGS_360DVO_pose_contract_v4_work"
DEFAULT_CACHE_ROOT = ROOT / "data" / "Online3DGS_360DVO_orbmono_fixed_dense_v3_cache"
DEFAULT_SAVE_ROOT = ROOT / "Logs_360dvo_pose_contract_v4"
DEFAULT_CONFIG_DIR = ROOT / "configs" / "360dvo_pose_contract_v4_runtime"
DEFAULT_PYTHON = Path("/home/wmy/anaconda3/envs/worldvln/bin/python")
DEFAULT_CUDA_HOME = Path("/usr/local/cuda-11.8")
DEFAULT_ORB_BINARY = Path(
    "/home/wmy/workspace_vla/third_party/ORB_SLAM3/Examples/Monocular/mono_tum_vi"
)
DEFAULT_ORB_VOCABULARY = Path(
    "/home/wmy/workspace_vla/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt"
)
DEFAULT_ORB_SETTINGS = ROOT / "configs" / "360dvo" / "orbslam3_grove_frontview.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--cuda-home", type=Path, default=DEFAULT_CUDA_HOME)
    parser.add_argument("--orb-binary", type=Path, default=DEFAULT_ORB_BINARY)
    parser.add_argument("--orb-vocabulary", type=Path, default=DEFAULT_ORB_VOCABULARY)
    parser.add_argument("--orb-settings", type=Path, default=DEFAULT_ORB_SETTINGS)
    parser.add_argument("--gpu-ids", default="4,5,6,7")
    parser.add_argument("--scenes", default=None)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--video-fps", type=float, default=24.0)
    parser.add_argument("--skip-existing-success", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    errors = parser.add_mutually_exclusive_group()
    errors.add_argument("--continue-on-error", action="store_true", default=True)
    errors.add_argument(
        "--stop-on-error", action="store_false", dest="continue_on_error"
    )
    return parser.parse_args()


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_csv_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("Comma-separated option contains no values")
    return names


def discover_scenes(data_root: Path, scene_filter: str | None) -> list[dict[str, Any]]:
    requested = parse_csv_names(scene_filter)
    if requested is None:
        candidates = sorted(path.name for path in data_root.iterdir() if path.is_dir())
    else:
        candidates = requested
    scenes = []
    for scene in candidates:
        source = data_root / scene
        config = ROOT / "configs" / "360dvo" / f"360DVO_{scene}_orb.yaml"
        if not config.is_file():
            if requested is not None:
                raise FileNotFoundError(f"Missing source config for {scene}: {config}")
            continue
        stats_path = source / "conversion_stats.json"
        trajectory_path = source / "trajectory_orb.json"
        if not stats_path.is_file() or not trajectory_path.is_file():
            raise FileNotFoundError(f"Incomplete source dataset: {source}")
        stats = read_json(stats_path)
        cameras = read_json(trajectory_path).get("cameras", [])
        frame_count = int(stats["frame_count"])
        if len(cameras) != frame_count:
            raise RuntimeError(
                f"{scene} frame mismatch: stats={frame_count}, cameras={len(cameras)}"
            )
        missing = [
            camera["image"]
            for camera in cameras
            if not (source / "rectified" / camera["image"]).is_file()
        ]
        if missing:
            raise RuntimeError(f"{scene} has {len(missing)} missing rectified images")
        scenes.append(
            {
                "scene": scene,
                "source": str(source.resolve()),
                "source_config": str(config.resolve()),
                "frame_count": frame_count,
            }
        )
    if requested is None and len(scenes) != 20:
        raise RuntimeError(
            f"Expected exactly 20 formal 360DVO scenes, found {len(scenes)}"
        )
    return sorted(scenes, key=lambda item: (-item["frame_count"], item["scene"]))


def capacity(frame_count: int) -> tuple[int, int]:
    if frame_count <= 1000:
        return 5000, 3200
    if frame_count <= 2000:
        return 1500, 1200
    return 800, 600


def pin_keyframes_on_gpu(frame_count: int) -> bool:
    return frame_count <= 1000


def dataset_name(scene: str) -> str:
    return f"360DVO-PoseContractV4-TGBR75-{scene.replace('_', '-')}"


def prepared_scene_path(prepared_root: Path, scene: str) -> Path:
    pose_contract = prepared_root / f"{scene}_auto_pose_contract_tracks"
    if (pose_contract / "conversion_stats.json").is_file():
        return pose_contract
    spline = prepared_root / f"{scene}_orbmono_epipolar_spline_gtcenter_tracks"
    if (spline / "conversion_stats.json").is_file():
        return spline
    epipolar = prepared_root / f"{scene}_orbmono_epipolar_gtcenter_tracks"
    if (epipolar / "conversion_stats.json").is_file():
        return epipolar
    return prepared_root / f"{scene}_orbmono_windowed_gtcenter_tracks"


def update_runtime_dataset_path(config_path: Path, dataset_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    resolved = str(dataset_path.resolve())
    changed = False
    for section in ("Dataset", "Testset"):
        if config[section]["dataset_path"] != resolved:
            config[section]["dataset_path"] = resolved
            changed = True
    if changed:
        temporary = config_path.with_suffix(config_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        os.replace(temporary, config_path)


def write_runtime_config(
    info: dict[str, Any], prepared_root: Path, args: argparse.Namespace
) -> Path:
    scene = info["scene"]
    max_points, extra_points = capacity(info["frame_count"])
    name = dataset_name(scene)
    dataset_path = prepared_scene_path(prepared_root, scene)
    config = {
        "inherit_from": info["source_config"],
        "Dataset": {
            "name": name,
            "dataset_path": str(dataset_path),
            "max_pts_num": max_points,
        },
        "Testset": {
            "name": name,
            "dataset_path": str(dataset_path),
            "max_pts_num": max_points,
        },
        "Results": {
            "save_dir": str(args.save_root),
            "save_gt": False,
            "save_exr": False,
            "save_mesh": False,
            "skip_eval": False,
        },
        "Mapper": {
            "use_multi_reso": False,
            "pin_kf_gpu": pin_keyframes_on_gpu(info["frame_count"]),
            "initialization_frames": 4,
            "optimization_iters": 10,
            "initialization_iters": 10,
            "post_refinement": {"max_steps": 100, "opt_cam": False},
            "KFGraph": {
                "kf_interval": 1,
                "global_window_size": 4,
            },
            "CameraOptimizer": {"pose_refine_init_steps": 0, "pose_opt_steps": 4},
        },
        "Model": {
            "sh_degree": 3,
            "extra_pts_num": extra_points,
            "err_threshold": 0.05,
            "camera_scale_rescalar": 0.25,
            "scene_scale": 1.0,
        },
        "HashBlock": {"use_hash": False},
        "FrontViewCoverageRecovery": {
            "enabled": True,
            "min_frame_gap": 40,
            "min_translation_m": 1.0,
            "min_rotation_deg": 3.0,
            "residual_threshold": 0.08,
            "min_failure_fraction": 0.15,
            "opacity_threshold": 0.50,
            "depth_fallback_enabled": True,
            "depth_prior_window_frames": 240,
            "depth_prior_quantile": 0.90,
            "depth_prior_min_m": 20.0,
            "depth_prior_max_m": 120.0,
            "depth_fallback_min_valid": 128,
            "depth_fallback_confidence": 0.50,
            "depth_fallback_cell_px": 18,
            "depth_fallback_scale_multiplier": 10.0,
            "depth_fallback_motion_floor_enabled": True,
            "depth_fallback_max_projected_drift_px": 8.0,
            "depth_fallback_motion_floor_max_m": 500.0,
            "depth_fallback_map_enabled": False,
            "newborn_optimization_iters": 5,
            "newborn_max_scale_expansion": 1.0,
            "newborn_freeze_positions": True,
            "tracking_update_interval": 5,
            "tracking_optimization_iters": 1,
        },
        "FrontViewSampling": {
            "enabled": True,
            "selection_mode": "depth_stratified",
            "pool_multiplier": 2,
            "evidence_fraction": 0.5,
            "reference_frames": 2,
            "photo_sigma": 0.08,
            "photo_mode": "consistency",
            "parallax_reference_deg": 2.0,
            "parallax_floor": 0.25,
            "confidence_power": 1.0,
            "shuffle_evidence": False,
            "shuffle_seed": 42,
            "depth_edges_m": [20.0, 50.0],
            "depth_fractions": [0.25, 0.45, 0.30],
        },
        "FrontViewFarField": {
            "enabled": True,
            "depth_m": 80.0,
            "projective_cell_px": 12,
            "depth_bin_ratio": 1.10,
            "shuffle_responsibility": False,
            "shuffle_seed": 42,
        },
        "FrontViewScaleCover": {
            "enabled": True,
            "radius_multiplier": 0.3,
            "scale_compatibility": 1.0,
            "neighbors": 32,
            "rebuild_rows": 8192,
            "color_distance_threshold": 0.15,
            "shuffle_occupancy": False,
            "shuffle_seed": 42,
        },
        "StreamingAppearanceLOD": {
            "enabled": True,
            "birth_degree": 2,
            "target_degree": 3,
            "min_views": 2,
            "max_target_fraction": 0.75,
            "promotion_interval": 10,
            "selection_mode": "gradient_agreement",
            "utility_ema_decay": 0.9,
            "shuffle_seed": 43,
        },
    }
    path = args.config_dir / f"360DVO_{scene}_orbmono_fixed_tgbr75.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def process_env(gpu: str, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["CUDA_HOME"] = str(args.cuda_home)
    env["PATH"] = f"{args.cuda_home / 'bin'}:{env.get('PATH', '')}"
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env["TORCH_EXTENSIONS_DIR"] = str(
        Path.home() / ".cache" / "torch_extensions" / f"online3dgs_gpu{gpu}"
    )
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        return subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def find_run_dir(save_root: Path, name: str, exp_name: str) -> Path:
    parent = save_root / name
    matches = sorted(
        parent.glob(f"*_{exp_name}"), key=lambda path: path.stat().st_mtime
    )
    if not matches:
        raise FileNotFoundError(f"No run matching {exp_name} under {parent}")
    return matches[-1]


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def video_is_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return False
    try:
        stream = json.loads(probe.stdout)["streams"][0]
        return (
            stream["codec_name"] == "h264"
            and stream["pix_fmt"] == "yuv420p"
            and int(stream["width"]) > 0
            and int(stream["height"]) > 0
        )
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def collect_metrics(run_dir: Path) -> dict[str, Any]:
    results = read_json(run_dir / "results.json")
    final = read_json(run_dir / "eval" / "final_result.json")
    render = read_json(run_dir / "videos_full" / "render_metrics.json")
    source_metrics = results.get("eval_res", {})
    final_metrics = final.get("mean", {})
    render_metrics = render.get("mean", {})
    for label, metrics in (
        ("source", source_metrics),
        ("final", final_metrics),
        ("render", render_metrics),
    ):
        for key in ("psnr", "ssim", "lpips"):
            if not finite_number(metrics.get(key)):
                raise RuntimeError(f"Invalid {label}.{key} in {run_dir}")
    return {
        "psnr": float(source_metrics["psnr"]),
        "ssim": float(source_metrics["ssim"]),
        "lpips": float(source_metrics["lpips"]),
        "final_psnr": float(final_metrics["psnr"]),
        "final_ssim": float(final_metrics["ssim"]),
        "final_lpips": float(final_metrics["lpips"]),
        "render_psnr": float(render_metrics["psnr"]),
        "render_ssim": float(render_metrics["ssim"]),
        "render_lpips": float(render_metrics["lpips"]),
        "online_recon_time": results.get("online_recon_time"),
        "eval_time": results.get("eval_time"),
        "num_processed_frames": results.get("num_processed_frames"),
        "num_keyframes": results.get("num_keyframes"),
        "num_gaussians": results.get("num_gaussians"),
        "render_frame_count": render.get("frame_count"),
    }


def validate_completed_run(run_dir: Path) -> dict[str, Any]:
    required = (
        run_dir / "point_cloud.ply",
        run_dir / "results.json",
        run_dir / "eval" / "final_result.json",
        run_dir / "videos_full" / "render_metrics.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing artifacts: " + ", ".join(missing))
    videos = (
        run_dir / "videos_full" / "render_vs_gt.mp4",
        run_dir / "videos_full" / "render_depth.mp4",
    )
    invalid = [str(path) for path in videos if not video_is_valid(path)]
    if invalid:
        raise RuntimeError("Invalid videos: " + ", ".join(invalid))
    return {
        **collect_metrics(run_dir),
        "render_vs_gt_video": str(videos[0]),
        "render_depth_video": str(videos[1]),
    }


def latest_success(save_root: Path, scene: str) -> tuple[Path, dict[str, Any]] | None:
    parent = save_root / dataset_name(scene)
    if not parent.is_dir():
        return None
    for run_dir in sorted(
        parent.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True
    ):
        if not run_dir.is_dir():
            continue
        try:
            return run_dir, validate_completed_run(run_dir)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            continue
    return None


CSV_FIELDS = [
    "scene",
    "status",
    "gpu",
    "frame_count",
    "pose_selection",
    "selected_dataset",
    "accepted_point_count",
    "epipolar_inlier_fraction",
    "psnr",
    "ssim",
    "lpips",
    "render_psnr",
    "render_ssim",
    "render_lpips",
    "online_recon_time",
    "num_gaussians",
    "wall_time",
    "run_dir",
    "render_vs_gt_video",
    "render_depth_video",
    "preprocess_log",
    "slam_log",
    "render_log",
    "error",
]


class BatchState:
    def __init__(self, payload: dict[str, Any], summary: Path, metrics_csv: Path):
        self.payload = payload
        self.summary = summary
        self.metrics_csv = metrics_csv
        self.latest = summary.parent / "batch_status_latest.json"
        self.lock = threading.Lock()

    def update(self, scene: str, **values: Any) -> None:
        with self.lock:
            self.payload["records"][scene].update(values)
            self.payload["updated_at"] = iso_now()
            self._write_locked()

    def write(self) -> None:
        with self.lock:
            self._write_locked()

    def finish(self) -> None:
        with self.lock:
            self.payload["finished_at"] = iso_now()
            self.payload["updated_at"] = iso_now()
            self._write_locked()

    def _write_locked(self) -> None:
        for path in (self.summary, self.latest):
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self.payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, path)
        temporary_csv = self.metrics_csv.with_suffix(self.metrics_csv.suffix + ".tmp")
        with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for record in self.payload["records"].values():
                writer.writerow({key: record.get(key, "") for key in CSV_FIELDS})
        os.replace(temporary_csv, self.metrics_csv)


def preprocess_command(info: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(ROOT / "scripts" / "preprocess_360dvo_orbmono_fixed.py"),
        "--scene",
        info["scene"],
        "--source",
        info["source"],
        "--work-root",
        str(args.work_root),
        "--output-root",
        str(args.prepared_root),
        "--cache-root",
        str(args.cache_root),
        "--python",
        str(args.python),
        "--orb-binary",
        str(args.orb_binary),
        "--orb-vocabulary",
        str(args.orb_vocabulary),
        "--orb-settings",
        str(args.orb_settings),
        "--seed",
        str(args.seed),
        "--device",
        "cuda:0",
    ]


def run_scene(
    info: dict[str, Any],
    gpu: str,
    config_path: Path,
    args: argparse.Namespace,
    batch_tag: str,
    state: BatchState,
) -> None:
    scene = info["scene"]
    started = time.monotonic()
    logs = args.save_root / "batch_logs"
    preprocess_log = logs / f"{scene}_gpu{gpu}_{batch_tag}_preprocess.log"
    slam_log = logs / f"{scene}_gpu{gpu}_{batch_tag}_slam.log"
    render_log = logs / f"{scene}_gpu{gpu}_{batch_tag}_render.log"
    state.update(
        scene,
        status="running_preprocess",
        gpu=gpu,
        started_at=iso_now(),
        preprocess_log=str(preprocess_log),
        slam_log=str(slam_log),
        render_log=str(render_log),
        config=str(config_path),
    )
    print(f"[GPU {gpu}] PREP  {scene} ({info['frame_count']} frames)", flush=True)
    env = process_env(gpu, args)
    code = run_logged(preprocess_command(info, args), preprocess_log, env)
    if code != 0:
        raise RuntimeError(
            f"preprocessing exited with code {code}; see {preprocess_log}"
        )
    preprocess_status = read_json(args.work_root / scene / "preprocess_status.json")
    if preprocess_status.get("status") != "success":
        raise RuntimeError(
            f"Invalid preprocess status for {scene}: {preprocess_status}"
        )
    selected_dataset = Path(preprocess_status["track_dataset"])
    update_runtime_dataset_path(config_path, selected_dataset)
    epipolar_fraction = preprocess_status.get("epipolar_inlier_fraction")
    epipolar_label = (
        f"{float(epipolar_fraction):.3f}" if finite_number(epipolar_fraction) else "n/a"
    )
    state.update(
        scene,
        status="running_slam",
        accepted_point_count=preprocess_status["accepted_point_count"],
        epipolar_inlier_fraction=epipolar_fraction,
        pose_selection=preprocess_status.get("pose_selection"),
        selected_dataset=str(selected_dataset),
    )
    print(
        f"[GPU {gpu}] SLAM  {scene}: tracks={preprocess_status['accepted_point_count']} "
        f"epi={epipolar_label} selection={preprocess_status.get('pose_selection')}",
        flush=True,
    )
    exp_name = f"pose_contract_v4_tgbr75_{scene}_seed{args.seed}_gpu{gpu}_{batch_tag}"
    slam_command = [
        str(args.python),
        "slam_new.py",
        "--config",
        str(config_path),
        "--exp_name",
        exp_name,
        "--seed",
        str(args.seed),
    ]
    code = run_logged(slam_command, slam_log, env)
    if code != 0:
        raise RuntimeError(f"slam_new.py exited with code {code}; see {slam_log}")
    run_dir = find_run_dir(args.save_root, dataset_name(scene), exp_name)
    state.update(scene, status="running_render", run_dir=str(run_dir))
    print(f"[GPU {gpu}] VIDEO {scene}", flush=True)
    render_command = [
        str(args.python),
        "render.py",
        "--run_dir",
        str(run_dir),
        "--output_dir",
        str(run_dir / "videos_full"),
        "--fps",
        str(args.video_fps),
        "--max_frames",
        "-1",
        "--device",
        "cuda:0",
        "--far_gs_depth_threshold",
        "80",
        "--skip_novel",
        "--skip_primitives",
        "--ignore_cached_renders",
    ]
    code = run_logged(render_command, render_log, env)
    if code != 0:
        raise RuntimeError(f"render.py exited with code {code}; see {render_log}")
    metrics = validate_completed_run(run_dir)
    state.update(
        scene,
        status="success",
        finished_at=iso_now(),
        wall_time=time.monotonic() - started,
        **metrics,
    )
    print(
        f"[GPU {gpu}] DONE  {scene}: PSNR={metrics['psnr']:.5f} "
        f"SSIM={metrics['ssim']:.6f} LPIPS={metrics['lpips']:.6f}",
        flush=True,
    )


def worker(
    gpu: str,
    jobs: queue.Queue[tuple[dict[str, Any], Path]],
    args: argparse.Namespace,
    batch_tag: str,
    state: BatchState,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            info, config = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            run_scene(info, gpu, config, args, batch_tag, state)
        except (
            Exception
        ) as error:  # noqa: BLE001 - preserve failures and continue the batch.
            state.update(
                info["scene"], status="failed", finished_at=iso_now(), error=str(error)
            )
            print(
                f"[GPU {gpu}] FAIL  {info['scene']}: {error}",
                file=sys.stderr,
                flush=True,
            )
            if not args.continue_on_error:
                stop_event.set()
        finally:
            jobs.task_done()


def main() -> int:
    args = parse_args()
    for name in (
        "data_root",
        "prepared_root",
        "work_root",
        "cache_root",
        "save_root",
        "config_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    gpu_ids = parse_csv_names(args.gpu_ids)
    assert gpu_ids is not None
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"Duplicate GPU IDs: {gpu_ids}")
    for required in (
        args.python,
        args.cuda_home / "bin" / "nvcc",
        args.orb_binary,
        args.orb_vocabulary,
        args.orb_settings,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    scenes = discover_scenes(args.data_root, args.scenes)
    for directory in (
        args.prepared_root,
        args.work_root,
        args.cache_root,
        args.save_root,
        args.config_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    batch_tag = now_tag()
    summary = args.save_root / f"batch_summary_{batch_tag}.json"
    metrics_csv = args.save_root / f"batch_metrics_{batch_tag}.csv"
    payload: dict[str, Any] = {
        "batch_tag": batch_tag,
        "started_at": iso_now(),
        "updated_at": iso_now(),
        "repo_root": str(ROOT),
        "data_root": str(args.data_root),
        "prepared_root": str(args.prepared_root),
        "save_root": str(args.save_root),
        "gpu_ids": gpu_ids,
        "seed": args.seed,
        "method": (
            "held-out image-certified 360DVO pose contract + persistent "
            "DISK-LightGlue world tracks with unsafe fallback rejection + "
            "hash-free TGBR-75 + residual-certified sparse-dropout recovery"
        ),
        "records": {
            info["scene"]: {
                "scene": info["scene"],
                "status": "pending",
                "frame_count": info["frame_count"],
                "source": info["source"],
            }
            for info in scenes
        },
    }
    state = BatchState(payload, summary, metrics_csv)
    jobs: queue.Queue[tuple[dict[str, Any], Path]] = queue.Queue()
    queued = 0
    for info in scenes:
        config = write_runtime_config(info, args.prepared_root, args)
        if args.skip_existing_success:
            existing = latest_success(args.save_root, info["scene"])
            if existing is not None:
                run_dir, metrics = existing
                state.update(
                    info["scene"],
                    status="skipped_existing_success",
                    config=str(config),
                    run_dir=str(run_dir),
                    **metrics,
                )
                continue
        if args.dry_run:
            max_points, extra_points = capacity(info["frame_count"])
            state.update(
                info["scene"],
                status="dry_run",
                config=str(config),
                max_points=max_points,
                extra_points=extra_points,
                preprocess_command=" ".join(preprocess_command(info, args)),
            )
            continue
        jobs.put((info, config))
        queued += 1

    state.write()
    print(f"Discovered {len(scenes)} formal scenes; queued {queued}", flush=True)
    print(f"GPUs: {','.join(gpu_ids)}", flush=True)
    print(f"Summary: {summary}", flush=True)
    if args.dry_run or queued == 0:
        state.finish()
        return 0

    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=worker,
            name=f"gpu-{gpu}",
            args=(gpu, jobs, args, batch_tag, state, stop_event),
        )
        for gpu in gpu_ids
    ]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        print("Interrupted; active child processes are allowed to exit", flush=True)
        for thread in threads:
            thread.join()
    state.finish()
    failed = [
        row for row in state.payload["records"].values() if row["status"] == "failed"
    ]
    print(
        f"Batch finished: {len(scenes) - len(failed)}/{len(scenes)} without failure",
        flush=True,
    )
    print(f"Metrics: {metrics_csv}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
