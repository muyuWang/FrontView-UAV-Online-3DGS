#!/usr/bin/env python3
"""Validate the current far-field method on four UAV scenes in parallel."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON = Path("/home/wmy/anaconda3/envs/worldvln/bin/python")
CUDA_HOME = Path("/usr/local/cuda-11.8")
METHOD_CONFIG = (
    ROOT
    / "configs/360dvo_coverage_recovery"
    / "mountains_uncertainty_bootstrap_repair_best.yaml"
)
OUTPUT_ROOT = ROOT / "Logs_cross_scene_current_best_8_15"
SHARED_GSPLAT_CACHE = Path.home() / ".cache/torch_extensions/online3dgs_dual_shared"
GSPLAT_EXTENSION = SHARED_GSPLAT_CACHE / "gsplat_cuda/gsplat_cuda.so"
GSPLAT_HOOK = ROOT / "scripts/gsplat_prebuilt_site"
METHOD_BLOCKS = (
    "FrontViewSampling",
    "FrontViewFarField",
    "FrontViewScaleCover",
    "FrontViewCoverageRecovery",
    "CausalDualResponsibility",
    "FrontViewDirectionalLayer",
    "StreamingAppearanceLOD",
)


SCENES: dict[str, dict[str, Any]] = {
    "bridge_night": {
        "gpu": "2",
        "fps": 30.0,
        "frames": 518,
        "base_config": ROOT
        / "configs/360dvo_pose_contract_v4"
        / "bridge_night_full.yaml",
        "previous": (
            {
                "label": "previous_uncertified_epipolar_input",
                "stage": "post_refinement",
                "metrics": ROOT
                / "Logs_360dvo_orbmono_fixed_tgbr75"
                / "360DVO-ORBMonoFixed-TGBR75-bridge-night"
                / "2026-07-31-00-59-59_orbmono_fixed_tgbr75_bridge_night_seed43_gpu5_20260730_221359"
                / "videos_full/render_metrics.json",
            },
            {
                "label": "previous_best_logged_different_input_contract",
                "stage": "post_refinement",
                "metrics": ROOT
                / "Logs_360dvo_tgbr75"
                / "360DVO-TGBR75-bridge-night"
                / "2026-07-30-15-25-49_all360dvo_tgbr75_bridge_night_seed43_gpu1_20260730_120013"
                / "videos_full/render_metrics.json",
            },
        ),
    },
    "village4": {
        "gpu": "5",
        "fps": 10.0,
        "frames": 1329,
        "base_config": ROOT / "configs/mydata/village4_best_tgbr75.yaml",
        "previous": (
            {
                "label": "previous_before_finite_certificate_decoupling_online",
                "stage": "online",
                "metrics": ROOT
                / "Logs_cross_scene_current_best_8_15"
                / "validation_20260815_current_best_seed43_v2"
                / "runs/CrossSceneCurrentBest-village4"
                / "2026-08-15-01-24-57_current_best_village4_seed43_gpu3_validation_20260815_current_best_seed43_v2"
                / "online_stage/videos_full/render_metrics.json",
            },
            {
                "label": "previous_before_finite_certificate_decoupling_post",
                "stage": "post_refinement",
                "metrics": ROOT
                / "Logs_cross_scene_current_best_8_15"
                / "validation_20260815_current_best_seed43_v2"
                / "runs/CrossSceneCurrentBest-village4"
                / "2026-08-15-01-24-57_current_best_village4_seed43_gpu3_validation_20260815_current_best_seed43_v2"
                / "videos_full/render_metrics.json",
            },
        ),
    },
    "mountains": {
        "gpu": "5",
        "fps": 24.0,
        "frames": 765,
        "base_config": ROOT / "configs/360dvo_pose_contract_v4/mountains_full.yaml",
        "previous": (
            {
                "label": "previous_best_k96_without_finite_depth_certificate",
                "stage": "post_refinement",
                "metrics": ROOT
                / "Logs_mountains_artifact_repair_8_14"
                / "online_full765_archive_k96_verified"
                / "batch_20260814_231307"
                / "gpu6_C_temporally_resolved_archive_k96"
                / "360DVO-CRR-gpu6_C_temporally_resolved_archive_k96-mountains"
                / "2026-08-14-23-13-13_mountains_crr_gpu6_C_temporally_resolved_archive_k96_seed43_gpu6_20260814_231307"
                / "videos_psnr_ssim/render_metrics.json",
            },
        ),
    },
    "nsc": {
        "gpu": "6",
        "fps": 30.0,
        "frames": 1859,
        "base_config": ROOT / "configs/pano360/nsc_frontview_best_tgbr75.yaml",
        "previous": (
            {
                "label": "previous_buggy_finite_certificate_online",
                "stage": "online",
                "metrics": ROOT
                / "Logs_cross_scene_current_best_8_15"
                / "sky_metric_certificate_cross_scene_20260815"
                / "runs/CrossSceneCurrentBest-nsc"
                / "2026-08-15-19-47-29_current_best_nsc_seed43_gpu6_sky_metric_certificate_cross_scene_20260815"
                / "online_stage/videos_full/render_metrics.json",
            },
            {
                "label": "previous_buggy_finite_certificate_post",
                "stage": "post_refinement",
                "metrics": ROOT
                / "Logs_cross_scene_current_best_8_15"
                / "sky_metric_certificate_cross_scene_20260815"
                / "runs/CrossSceneCurrentBest-nsc"
                / "2026-08-15-19-47-29_current_best_nsc_seed43_gpu6_sky_metric_certificate_cross_scene_20260815"
                / "videos_full/render_metrics.json",
            },
        ),
    },
    "panoair": {
        "gpu": "4",
        "fps": 30.0,
        "frames": 2230,
        "base_config": ROOT
        / "configs/frontview_uav/panoair_orbvi_tsc_tgbr75_sh3_full.yaml",
        "previous": (
            {
                "label": "previous_buggy_finite_certificate_online",
                "stage": "online",
                "metrics": ROOT
                / "Logs_cross_scene_current_best_8_15"
                / "sky_metric_certificate_cross_scene_20260815"
                / "runs/CrossSceneCurrentBest-panoair"
                / "2026-08-15-19-47-29_current_best_panoair_seed43_gpu4_sky_metric_certificate_cross_scene_20260815"
                / "online_stage/videos_full/render_metrics.json",
            },
            {
                "label": "previous_buggy_finite_certificate_post",
                "stage": "post_refinement",
                "metrics": ROOT
                / "Logs_cross_scene_current_best_8_15"
                / "sky_metric_certificate_cross_scene_20260815"
                / "runs/CrossSceneCurrentBest-panoair"
                / "2026-08-15-19-47-29_current_best_panoair_seed43_gpu4_sky_metric_certificate_cross_scene_20260815"
                / "videos_full/render_metrics.json",
            },
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--batch-name", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--scenes", nargs="+", choices=tuple(SCENES), default=tuple(SCENES)
    )
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_metric_mean(path: Path) -> dict[str, float]:
    payload = load_json(path)
    mean = payload["mean"]
    result = {}
    for key in ("psnr", "ssim"):
        value = float(mean[key])
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite {key} in {path}: {value}")
        result[key] = value
    result["frame_count"] = int(payload["frame_count"])
    return result


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def validate_dataset(config: dict[str, Any], expected_frames: int) -> None:
    dataset = Path(config["Dataset"]["dataset_path"]).expanduser().resolve()
    for required in (
        dataset / "trajectory_orb.json",
        dataset / "conversion_stats.json",
        dataset / "rectified",
        dataset / "orb_point_clouds",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    trajectory = load_json(dataset / "trajectory_orb.json")
    stats = load_json(dataset / "conversion_stats.json")
    cameras = trajectory.get("cameras", [])
    frame_count = int(stats.get("frame_count", len(cameras)))
    if frame_count != expected_frames or len(cameras) != expected_frames:
        raise RuntimeError(
            f"Frame contract failed for {dataset}: stats={frame_count}, "
            f"trajectory={len(cameras)}, expected={expected_frames}"
        )
    image_list = dataset / "image_list.txt"
    if image_list.is_file():
        listed = sum(bool(line.strip()) for line in image_list.read_text().splitlines())
        if listed != expected_frames:
            raise RuntimeError(
                f"Image list mismatch for {dataset}: {listed} != {expected_frames}"
            )


def build_runtime_config(
    scene: str, spec: dict[str, Any], batch_dir: Path
) -> tuple[Path, dict[str, Any]]:
    from utils_new.frontview_directional_layer import (
        validate_front_view_directional_layer_config,
    )
    from utils_new.frontview_dual_responsibility import (
        validate_causal_dual_responsibility_config,
    )
    from utils_new.frontview_coverage_recovery import (
        validate_front_view_coverage_recovery_config,
    )
    from utils_new.frontview_far_field import validate_front_view_far_field_config
    from utils_new.frontview_sampling import validate_front_view_sampling_config
    from utils_new.frontview_scale_cover import validate_front_view_scale_cover_config
    from utils_new.streaming_appearance_lod import (
        validate_streaming_appearance_lod_config,
    )
    from utils_new.tool_utils import load_config

    config = copy.deepcopy(load_config(str(spec["base_config"])))
    method = load_config(str(METHOD_CONFIG))
    for block in METHOD_BLOCKS:
        config[block] = copy.deepcopy(method[block])
    config.setdefault("HashBlock", {})["use_hash"] = False
    config.setdefault("Model", {}).setdefault("frequency_sampling", {})[
        "preserve_sparse_track_geometry"
    ] = False
    lod = config["StreamingAppearanceLOD"]
    lod["compute_routing"] = False
    lod["bounded_replay_residency_enabled"] = False
    lod["spectral_residency_enabled"] = False
    dataset_name = f"CrossSceneCurrentBest-{scene}"
    config["Dataset"]["name"] = dataset_name
    config["Testset"]["name"] = dataset_name
    config.setdefault("Results", {}).update(
        {
            "save_dir": str((batch_dir / "runs").resolve()),
            "save_gt": False,
            "save_exr": False,
            "save_mesh": False,
            "skip_eval": True,
            "save_online_stage": True,
        }
    )

    validate_dataset(config, int(spec["frames"]))
    validate_front_view_sampling_config(config["FrontViewSampling"])
    validate_front_view_far_field_config(config["FrontViewFarField"])
    validate_front_view_scale_cover_config(config["FrontViewScaleCover"])
    validate_front_view_coverage_recovery_config(config["FrontViewCoverageRecovery"])
    validate_causal_dual_responsibility_config(config["CausalDualResponsibility"])
    validate_front_view_directional_layer_config(config["FrontViewDirectionalLayer"])
    validate_streaming_appearance_lod_config(config["StreamingAppearanceLOD"])
    assert config["HashBlock"]["use_hash"] is False
    assert config["FrontViewSampling"]["selection_mode"] == "adaptive_log_depth_random"
    assert config["FrontViewFarField"]["routing_mode"] == "adaptive_observability"
    assert config["CausalDualResponsibility"]["enabled"] is True
    assert config["CausalDualResponsibility"][
        "finite_depth_certificate_enabled"
    ] is True
    assert config["CausalDualResponsibility"][
        "finite_depth_preserve_appearance_ownership"
    ] is True
    assert config["FrontViewDirectionalLayer"]["max_anchors"] == 96
    assert config["FrontViewDirectionalLayer"]["source_fusion"] == "first"
    assert config["FrontViewDirectionalLayer"]["uncertainty_bootstrap_enabled"] is True
    assert config["Results"]["save_online_stage"] is True

    output = batch_dir / "runtime_configs" / f"{scene}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return output, config


def process_env(gpu: str, cpu_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["CUDA_HOME"] = str(CUDA_HOME)
    env["PATH"] = f"{CUDA_HOME / 'bin'}:{env.get('PATH', '')}"
    env["TORCH_EXTENSIONS_DIR"] = str(SHARED_GSPLAT_CACHE)
    env["GSPLAT_PREBUILT_EXTENSION"] = str(GSPLAT_EXTENSION)
    env["PYTHONPATH"] = f"{GSPLAT_HOOK}:{env.get('PYTHONPATH', '')}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["OMP_NUM_THREADS"] = str(cpu_threads)
    env["MKL_NUM_THREADS"] = str(cpu_threads)
    env["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command exited with code {completed.returncode}; see {log_path}"
        )


def find_run_dir(config: dict[str, Any], exp_name: str) -> Path:
    parent = Path(config["Results"]["save_dir"]) / config["Dataset"]["name"]
    matches = sorted(parent.glob(f"*_{exp_name}"), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No run matching {exp_name} under {parent}")
    return matches[-1]


def gaussian_count(path: Path) -> int:
    with path.open("rb") as stream:
        for raw in stream:
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    raise RuntimeError(f"No vertex count in {path}")


def probe_video(path: Path, expected_frames: int) -> dict[str, Any]:
    payload = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,nb_frames,width,height:stream_tags=encoder",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    stream = payload["streams"][0]
    frames = int(stream["nb_frames"])
    if stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p":
        raise RuntimeError(f"Unexpected video format for {path}: {stream}")
    if frames != expected_frames:
        raise RuntimeError(f"Video frame mismatch for {path}: {frames} != {expected_frames}")
    return stream


def render_command(run_dir: Path, output_dir: Path, fps: float) -> list[str]:
    return [
        str(PYTHON),
        "render.py",
        "--run_dir",
        str(run_dir),
        "--output_dir",
        str(output_dir),
        "--fps",
        str(fps),
        "--max_frames",
        "-1",
        "--device",
        "cuda:0",
        "--far_gs_depth_threshold",
        "80",
        "--video_encoder",
        "auto",
        "--skip_lpips",
        "--skip_novel",
        "--skip_primitives",
        "--ignore_cached_renders",
    ]


def previous_comparisons(
    spec: dict[str, Any], current: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    rows = []
    for reference in spec["previous"]:
        old = finite_metric_mean(reference["metrics"])
        stage = reference["stage"]
        new = current[stage]
        rows.append(
            {
                "label": reference["label"],
                "stage": stage,
                "metrics_path": str(reference["metrics"]),
                "previous": old,
                "current": new,
                "delta": {
                    "psnr": new["psnr"] - old["psnr"],
                    "ssim": new["ssim"] - old["ssim"],
                },
            }
        )
    return rows


def collect_scene_result(scene: str, spec: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    frames = int(spec["frames"])
    results = load_json(run_dir / "results.json")
    online_dir = run_dir / "online_stage/videos_full"
    post_dir = run_dir / "videos_full"
    online = finite_metric_mean(online_dir / "render_metrics.json")
    post = finite_metric_mean(post_dir / "render_metrics.json")
    for metrics in (online, post):
        if metrics["frame_count"] != frames:
            raise RuntimeError(
                f"Metric frame mismatch for {scene}: {metrics['frame_count']} != {frames}"
            )
    videos = {}
    for stage, directory in (("online", online_dir), ("post_refinement", post_dir)):
        stage_videos = {}
        for name in ("render_vs_gt.mp4", "render_depth.mp4", "render_vs_gt_depth.mp4"):
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(path)
            stage_videos[name] = {
                "path": str(path),
                "probe": probe_video(path, frames),
            }
        videos[stage] = stage_videos
    online_mapping_seconds = float(
        results.get("online_mapping_seconds", results["online_recon_time"])
    )
    current = {"online": online, "post_refinement": post}
    return {
        "scene": scene,
        "gpu": spec["gpu"],
        "run_dir": str(run_dir),
        "frame_count": frames,
        "online_mapping_seconds": online_mapping_seconds,
        "online_fps": frames / online_mapping_seconds,
        "online_stage_export_seconds": float(
            results.get("online_stage_export_seconds", 0.0)
        ),
        "post_refinement_seconds": float(results.get("post_refinement_seconds", 0.0)),
        "online": {
            **online,
            "gaussians": gaussian_count(run_dir / "online_stage/point_cloud.ply"),
        },
        "post_refinement": {
            **post,
            "gaussians": gaussian_count(run_dir / "point_cloud.ply"),
        },
        "post_minus_online": {
            "psnr": post["psnr"] - online["psnr"],
            "ssim": post["ssim"] - online["ssim"],
        },
        "videos": videos,
        "previous_comparisons": previous_comparisons(spec, current),
    }


def run_scene(
    scene: str,
    spec: dict[str, Any],
    batch_dir: Path,
    seed: int,
    cpu_threads: int,
    dry_run: bool,
) -> dict[str, Any]:
    config_path, config = build_runtime_config(scene, spec, batch_dir)
    gpu = str(spec["gpu"])
    exp_name = f"current_best_{scene}_seed{seed}_gpu{gpu}_{batch_dir.name}"
    status_path = batch_dir / "status" / f"{scene}.json"
    status: dict[str, Any] = {
        "scene": scene,
        "gpu": gpu,
        "frames": int(spec["frames"]),
        "runtime_config": str(config_path),
        "status": "dry_run" if dry_run else "running_online_reconstruction",
        "started_at": iso_now(),
    }
    write_json(status_path, status)
    slam = [
        str(PYTHON),
        "slam_new.py",
        "--config",
        str(config_path),
        "--exp_name",
        exp_name,
        "--seed",
        str(seed),
        "--cpu_threads",
        str(cpu_threads),
    ]
    status["commands"] = {"slam": slam}
    write_json(status_path, status)
    if dry_run:
        return status

    env = process_env(gpu, cpu_threads)
    logs = batch_dir / "launcher_logs"
    try:
        run_logged(slam, logs / f"{scene}_slam.log", env)
        run_dir = find_run_dir(config, exp_name)
        online_stage = run_dir / "online_stage"
        for required in (
            run_dir / "point_cloud.ply",
            online_stage / "point_cloud.ply",
            online_stage / "frontview_directional_layer.pt",
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        shutil.copy2(run_dir / "config.yaml", online_stage / "config.yaml")

        online_render = render_command(
            online_stage, online_stage / "videos_full", float(spec["fps"])
        )
        status.update(
            status="rendering_online_stage",
            run_dir=str(run_dir),
        )
        status["commands"]["online_render"] = online_render
        write_json(status_path, status)
        run_logged(online_render, logs / f"{scene}_render_online.log", env)

        post_render = render_command(
            run_dir, run_dir / "videos_full", float(spec["fps"])
        )
        status["status"] = "rendering_post_refinement"
        status["commands"]["post_render"] = post_render
        write_json(status_path, status)
        run_logged(post_render, logs / f"{scene}_render_post.log", env)

        result = collect_scene_result(scene, spec, run_dir)
        result.update(status="success", finished_at=iso_now())
        write_json(run_dir / "two_stage_summary.json", result)
        write_json(status_path, result)
        return result
    except Exception as error:
        status.update(status="failed", finished_at=iso_now(), error=str(error))
        write_json(status_path, status)
        return status


def write_summary(batch_dir: Path, results: list[dict[str, Any]]) -> None:
    ordered = sorted(results, key=lambda row: row["scene"])
    successful = [row for row in ordered if row.get("status") == "success"]
    payload = {
        "schema_version": 1,
        "method": (
            "adaptive PBSD + observability-typed TSC/FPR + footprint trust + "
            "causal dual responsibility with finite-depth certificate + "
            "K96 first-source directional archive + uncertainty bootstrap + TGBR-75"
        ),
        "metric_protocol": (
            "same-trajectory full-resolution PSNR/SSIM at online snapshot and "
            "post-refinement; LPIPS disabled"
        ),
        "status": "success" if len(successful) == len(ordered) else "failed",
        "scenes": ordered,
    }
    write_json(batch_dir / "summary.json", payload)
    with (batch_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "scene",
            "status",
            "gpu",
            "frame_count",
            "online_fps",
            "online_psnr",
            "online_ssim",
            "post_psnr",
            "post_ssim",
            "post_psnr_delta",
            "post_ssim_delta",
            "run_dir",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in ordered:
            output = {key: row.get(key) for key in fields}
            if row.get("status") == "success":
                output.update(
                    online_psnr=row["online"]["psnr"],
                    online_ssim=row["online"]["ssim"],
                    post_psnr=row["post_refinement"]["psnr"],
                    post_ssim=row["post_refinement"]["ssim"],
                    post_psnr_delta=row["post_minus_online"]["psnr"],
                    post_ssim_delta=row["post_minus_online"]["ssim"],
                )
            writer.writerow(output)


def main() -> int:
    args = parse_args()
    for required in (PYTHON, METHOD_CONFIG, GSPLAT_EXTENSION, GSPLAT_HOOK):
        if not required.exists():
            raise FileNotFoundError(required)
    selected = [scene for scene in SCENES if scene in set(args.scenes)]
    stamp = args.batch_name or datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    batch_dir = OUTPUT_ROOT / stamp
    batch_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "batch": stamp,
        "status": "dry_run" if args.dry_run else "running",
        "seed": args.seed,
        "cpu_threads_per_scene": args.cpu_threads,
        "method_config": str(METHOD_CONFIG),
        "schedule": {
            scene: {"gpu": SCENES[scene]["gpu"], "frames": SCENES[scene]["frames"]}
            for scene in selected
        },
        "started_at": iso_now(),
    }
    write_json(batch_dir / "manifest.json", manifest)

    results = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        futures = {
            executor.submit(
                run_scene,
                scene,
                SCENES[scene],
                batch_dir,
                args.seed,
                args.cpu_threads,
                args.dry_run,
            ): scene
            for scene in selected
        }
        for future in as_completed(futures):
            row = future.result()
            with lock:
                results.append(row)
    if args.dry_run:
        manifest.update(status="dry_run_complete", finished_at=iso_now())
        write_json(batch_dir / "manifest.json", manifest)
        print(f"Dry-run complete: {batch_dir}")
        return 0

    write_summary(batch_dir, results)
    successful = all(row.get("status") == "success" for row in results)
    manifest.update(
        status="success" if successful else "failed",
        finished_at=iso_now(),
        summary=str(batch_dir / "summary.json"),
    )
    write_json(batch_dir / "manifest.json", manifest)
    print(f"Summary: {batch_dir / 'summary.json'}")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
