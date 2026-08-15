#!/usr/bin/env python3
"""Run the current directional front-view method on one 360DVO scene."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
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
DEFAULT_PYTHON = Path("/home/wmy/anaconda3/envs/worldvln/bin/python")
DEFAULT_CUDA_HOME = Path("/usr/local/cuda-11.8")
DEFAULT_ORB_BINARY = Path(
    "/home/wmy/workspace_vla/third_party/ORB_SLAM3/Examples/Monocular/mono_tum_vi"
)
DEFAULT_ORB_VOCABULARY = Path(
    "/home/wmy/workspace_vla/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt"
)
DEFAULT_ORB_SETTINGS = ROOT / "configs" / "360dvo" / "orbslam3_grove_frontview.yaml"


SCENES = {
    "bridge_night": 518,
    "canyon_line": 786,
    "canyon_loop": 1220,
    "city_driving": 2250,
    "downhill_biking": 576,
    "dragon_boat": 2125,
    "drone_racetrack": 504,
    "field": 499,
    "grove": 551,
    "hongkong_central": 1600,
    "hongkong_wanchai": 1740,
    "indoor_RC_car": 1350,
    "london_bridge": 1402,
    "mountains": 765,
    "ridge_to_lake": 552,
    "shanghai_street": 685,
    "snowmobile": 830,
    "snowy_mountain_road": 728,
    "tokyo_citywalk": 799,
    "wingsuit": 528,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=sorted(SCENES))
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--video-fps", type=float, default=24.0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--cuda-home", type=Path, default=DEFAULT_CUDA_HOME)
    parser.add_argument("--orb-binary", type=Path, default=DEFAULT_ORB_BINARY)
    parser.add_argument("--orb-vocabulary", type=Path, default=DEFAULT_ORB_VOCABULARY)
    parser.add_argument("--orb-settings", type=Path, default=DEFAULT_ORB_SETTINGS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the runtime config and print commands without executing them.",
    )
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def capacity(frame_count: int) -> tuple[int, int]:
    if frame_count <= 1000:
        return 5000, 3200
    if frame_count <= 2000:
        return 1500, 1200
    return 800, 600


def dataset_name(scene: str) -> str:
    return f"360DVO-DirectionalFrontView-{scene.replace('_', '-')}"


def prepared_scene_path(prepared_root: Path, scene: str) -> Path:
    candidates = (
        f"{scene}_auto_pose_contract_tracks",
        f"{scene}_orbmono_epipolar_spline_gtcenter_tracks",
        f"{scene}_orbmono_epipolar_gtcenter_tracks",
        f"{scene}_orbmono_windowed_gtcenter_tracks",
    )
    for candidate in candidates[:-1]:
        path = prepared_root / candidate
        if (path / "conversion_stats.json").is_file():
            return path
    return prepared_root / candidates[-1]


def validate_scene(args: argparse.Namespace) -> dict[str, Any]:
    source = args.data_root / args.scene
    source_config = ROOT / "configs" / "360dvo" / f"360DVO_{args.scene}_orb.yaml"
    stats_path = source / "conversion_stats.json"
    trajectory_path = source / "trajectory_orb.json"
    for required in (source_config, stats_path, trajectory_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    stats = read_json(stats_path)
    cameras = read_json(trajectory_path).get("cameras", [])
    frame_count = int(stats["frame_count"])
    expected = SCENES[args.scene]
    if frame_count != expected or len(cameras) != expected:
        raise RuntimeError(
            f"{args.scene} frame mismatch: embedded={expected}, "
            f"stats={frame_count}, trajectory={len(cameras)}"
        )
    missing = [
        camera["image"]
        for camera in cameras
        if not (source / "rectified" / camera["image"]).is_file()
    ]
    if missing:
        raise RuntimeError(f"{args.scene} has {len(missing)} missing rectified images")
    return {
        "scene": args.scene,
        "source": source.resolve(),
        "source_config": source_config.resolve(),
        "frame_count": frame_count,
    }


def runtime_config(info: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    max_points, extra_points = capacity(info["frame_count"])
    name = dataset_name(info["scene"])
    dataset_path = prepared_scene_path(args.prepared_root, info["scene"])
    return {
        "inherit_from": str(info["source_config"]),
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
            "save_dir": str(args.save_dir),
            "save_gt": False,
            "save_exr": False,
            "save_mesh": False,
            "skip_eval": False,
        },
        "Mapper": {
            "use_multi_reso": False,
            "pin_kf_gpu": info["frame_count"] <= 1000,
            "initialization_frames": 4,
            "optimization_iters": 10,
            "initialization_iters": 10,
            "post_refinement": {"max_steps": 100, "opt_cam": False},
            "KFGraph": {"kf_interval": 1, "global_window_size": 4},
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
        "FrontViewDirectionalLayer": {
            "enabled": True,
            "sparse_point_threshold": 10,
            "anchor_interval_frames": 20,
            "max_anchors": 12,
            "min_anchors": 2,
            "far_depth_m": 80.0,
            "low_opacity_threshold": 0.5,
            "consistency_threshold": 0.12,
            "blend_weight": 0.75,
            "exclude_exact_frame": True,
            "causal_only": True,
            "use_geometry_gate": False,
        },
    }


def write_runtime_config(
    info: dict[str, Any], args: argparse.Namespace
) -> Path:
    config_dir = args.save_dir / "runtime_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"360DVO_{info['scene']}_directional.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(runtime_config(info, args), handle, sort_keys=False)
    return path


def update_dataset_path(config_path: Path, dataset_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    resolved = str(dataset_path.resolve())
    config["Dataset"]["dataset_path"] = resolved
    config["Testset"]["dataset_path"] = resolved
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def process_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["CUDA_HOME"] = str(args.cuda_home)
    env["PATH"] = f"{args.cuda_home / 'bin'}:{env.get('PATH', '')}"
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env["TORCH_EXTENSIONS_DIR"] = str(
        Path.home() / ".cache" / "torch_extensions" / f"online3dgs_gpu{args.gpu}"
    )
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command exited with code {result.returncode}; see {log_path}"
        )


def find_run_dir(save_dir: Path, scene: str, exp_name: str) -> Path:
    parent = save_dir / dataset_name(scene)
    matches = sorted(
        parent.glob(f"*_{exp_name}"), key=lambda path: path.stat().st_mtime
    )
    if not matches:
        raise FileNotFoundError(f"No run matching {exp_name} under {parent}")
    return matches[-1]


def finite_metrics(metrics: dict[str, Any], label: str) -> dict[str, float]:
    result = {}
    for key in ("psnr", "ssim", "lpips"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"Invalid {label}.{key}: {value}")
        result[key] = float(value)
    return result


def collect_results(run_dir: Path) -> dict[str, Any]:
    online = read_json(run_dir / "results.json")
    final = read_json(run_dir / "eval" / "final_result.json")
    render = read_json(run_dir / "videos_full" / "render_metrics.json")
    videos = {
        "render_vs_gt": run_dir / "videos_full" / "render_vs_gt.mp4",
        "render_depth": run_dir / "videos_full" / "render_depth.mp4",
    }
    missing = [str(path) for path in videos.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing rendered videos: " + ", ".join(missing))
    return {
        "online_metrics": finite_metrics(online.get("eval_res", {}), "online"),
        "final_metrics": finite_metrics(final.get("mean", {}), "final"),
        "render_metrics": finite_metrics(render.get("mean", {}), "render"),
        "online_recon_time": online.get("online_recon_time"),
        "eval_time": online.get("eval_time"),
        "num_processed_frames": online.get("num_processed_frames"),
        "num_keyframes": online.get("num_keyframes"),
        "num_gaussians": online.get("num_gaussians"),
        "render_frame_count": render.get("frame_count"),
        "videos": {key: str(path) for key, path in videos.items()},
    }


def preprocess_command(info: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(ROOT / "scripts" / "preprocess_360dvo_orbmono_fixed.py"),
        "--scene",
        info["scene"],
        "--source",
        str(info["source"]),
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


def commands(
    info: dict[str, Any], config_path: Path, args: argparse.Namespace, tag: str
) -> tuple[list[str], str]:
    exp_name = f"directional_{info['scene']}_seed{args.seed}_gpu{args.gpu}_{tag}"
    slam = [
        str(args.python),
        "slam_new.py",
        "--config",
        str(config_path),
        "--exp_name",
        exp_name,
        "--seed",
        str(args.seed),
    ]
    return slam, exp_name


def render_command(run_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
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


def resolve_paths(args: argparse.Namespace) -> None:
    for name in (
        "save_dir",
        "data_root",
        "prepared_root",
        "work_root",
        "cache_root",
        "python",
        "cuda_home",
        "orb_binary",
        "orb_vocabulary",
        "orb_settings",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())


def main() -> int:
    args = parse_args()
    resolve_paths(args)
    info = validate_scene(args)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_runtime_config(info, args)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = args.save_dir / "launcher_logs"
    summary_path = args.save_dir / f"{args.scene}_{tag}_summary.json"
    prep = preprocess_command(info, args)
    slam, exp_name = commands(info, config_path, args, tag)
    summary: dict[str, Any] = {
        "scene": args.scene,
        "frame_count": info["frame_count"],
        "gpu": str(args.gpu),
        "seed": args.seed,
        "status": "dry_run" if args.dry_run else "running_preprocess",
        "started_at": iso_now(),
        "save_dir": str(args.save_dir),
        "runtime_config": str(config_path),
        "method": "pose-contract persistent tracks + hashless front-view LOD + causal directional far-field layer",
        "commands": {"preprocess": prep, "slam": slam},
    }
    write_json(summary_path, summary)

    if args.dry_run:
        print(f"Runtime config: {config_path}")
        print(f"PREPROCESS: {shlex.join(prep)}")
        print(f"SLAM:       {shlex.join(slam)}")
        print("RENDER:     generated after SLAM resolves its timestamped run directory")
        print(f"Summary: {summary_path}")
        return 0

    required_files = (
        args.python,
        args.cuda_home / "bin" / "nvcc",
        args.orb_binary,
        args.orb_vocabulary,
        args.orb_settings,
    )
    for required in required_files:
        if not required.is_file():
            raise FileNotFoundError(required)

    env = process_env(args)
    started = time.monotonic()
    try:
        run_logged(prep, logs / f"{args.scene}_{tag}_preprocess.log", env)
        preprocess_status = read_json(
            args.work_root / args.scene / "preprocess_status.json"
        )
        if preprocess_status.get("status") != "success":
            raise RuntimeError(f"Invalid preprocess status: {preprocess_status}")
        selected_dataset = Path(preprocess_status["track_dataset"])
        update_dataset_path(config_path, selected_dataset)
        summary.update(
            status="running_slam",
            selected_dataset=str(selected_dataset),
            pose_selection=preprocess_status.get("pose_selection"),
            accepted_point_count=preprocess_status.get("accepted_point_count"),
            epipolar_inlier_fraction=preprocess_status.get(
                "epipolar_inlier_fraction"
            ),
        )
        write_json(summary_path, summary)

        run_logged(slam, logs / f"{args.scene}_{tag}_slam.log", env)
        run_dir = find_run_dir(args.save_dir, args.scene, exp_name)
        render = render_command(run_dir, args)
        summary.update(
            status="running_render",
            run_dir=str(run_dir),
            commands={"preprocess": prep, "slam": slam, "render": render},
        )
        write_json(summary_path, summary)

        run_logged(render, logs / f"{args.scene}_{tag}_render.log", env)
        summary.update(
            status="success",
            finished_at=iso_now(),
            wall_time=time.monotonic() - started,
            **collect_results(run_dir),
        )
        write_json(summary_path, summary)
        metrics = summary["render_metrics"]
        print(
            f"DONE {args.scene}: PSNR={metrics['psnr']:.5f} "
            f"SSIM={metrics['ssim']:.6f} LPIPS={metrics['lpips']:.6f}"
        )
        print(f"Run directory: {run_dir}")
        print(f"Summary: {summary_path}")
        return 0
    except Exception as error:  # noqa: BLE001 - persist the failed stage.
        summary.update(
            status="failed",
            finished_at=iso_now(),
            wall_time=time.monotonic() - started,
            error=str(error),
        )
        write_json(summary_path, summary)
        print(f"FAILED: {error}", file=sys.stderr)
        print(f"Summary: {summary_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
