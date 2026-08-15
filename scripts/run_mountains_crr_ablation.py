#!/usr/bin/env python3
"""Run matched Mountains causal-resolution responsibility ablations on GPUs 4-7."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils_new.tool_utils import load_config


DEFAULT_CONFIG = (
    ROOT / "configs/360dvo_coverage_recovery/mountains_crr_ablation_base.yaml"
)
DEFAULT_OUTPUT = ROOT / "Logs_mountains_crr_ablation_8_12"
DEFAULT_PYTHON = Path("/home/wmy/anaconda3/envs/worldvln/bin/python")

VARIANTS: dict[str, dict[str, Any]] = {
    "A_fixed_depth": {},
    "B_causal_route": {
        "far_field": {
            "routing_mode": "causal_observability",
        },
    },
    "C_causal_projective": {
        "far_field": {
            "routing_mode": "causal_observability",
            "map_redundancy_gate": True,
            "projective_nms_mode": "gaussian_support",
        },
    },
    "D_full_crr": {
        "far_field": {
            "routing_mode": "causal_observability",
            "map_redundancy_gate": True,
            "projective_nms_mode": "gaussian_support",
        },
        "sampling": {"selection_mode": "residual_importance"},
        "scale_cover": {
            "target_size_mode": "gaussian_support",
            "distance_mode": "gaussian_overlap",
            "color_distance_threshold": -1.0,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpus", nargs="+", default=("4", "5", "6", "7"))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional subset of variant names to run",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--frames", type=int, default=0, help="0 runs all frames")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--render-begin", type=int, default=None)
    parser.add_argument("--render-end", type=int, default=None)
    parser.add_argument("--fixed-far-mask-dir", type=Path, default=None)
    parser.add_argument("--fixed-far-begin", type=int, default=None)
    parser.add_argument("--fixed-far-end", type=int, default=None)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="Maximum CPU threads for each reconstruction subprocess",
    )
    parser.add_argument(
        "--scale-cover-backend",
        choices=("config", "scipy_kdtree", "pytorch3d_knn"),
        default="config",
        help="Optional TSC neighbor-query backend override",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def process_env(gpu: str, cpu_threads: int = 1) -> dict[str, str]:
    env = os.environ.copy()
    cuda_home = Path("/usr/local/cuda-11.8")
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["CUDA_HOME"] = str(cuda_home)
    env["PATH"] = f"{cuda_home / 'bin'}:{env.get('PATH', '')}"
    env["TORCH_EXTENSIONS_DIR"] = str(
        Path.home() / ".cache/torch_extensions" / f"online3dgs_gpu{gpu}"
    )
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("MAX_JOBS", "2")
    thread_count = str(cpu_threads)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[variable] = thread_count
    env["PYTHONUNBUFFERED"] = "1"
    return env


def trajectory_length(dataset_path: Path) -> int:
    with (dataset_path / "trajectory_orb.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cameras = payload.get("cameras") if isinstance(payload, dict) else payload
    if not isinstance(cameras, list) or not cameras:
        raise RuntimeError("trajectory_orb.json has no cameras")
    return len(cameras)


def build_config(
    baseline: dict[str, Any],
    variant: str,
    changes: dict[str, Any],
    variant_root: Path,
    frames: int,
) -> dict[str, Any]:
    config = copy.deepcopy(baseline)
    config.pop("inherit_from", None)
    config["Results"].update(
        save_dir=str(variant_root),
        skip_eval=True,
        save_gt=False,
        save_exr=False,
        save_mesh=False,
    )
    config["Dataset"]["name"] = f"360DVO-CRR-{variant}-mountains"
    config["Testset"]["name"] = config["Dataset"]["name"]
    if frames > 0:
        total = trajectory_length(Path(config["Dataset"]["dataset_path"]))
        if frames > total:
            raise ValueError(f"Requested {frames} frames, dataset has {total}")
        cutoff = total - frames
        config["Dataset"]["end_cutoff"] = cutoff
        config["Testset"]["end_cutoff"] = cutoff
    config["FrontViewFarField"].update(changes.get("far_field", {}))
    config["FrontViewSampling"].update(changes.get("sampling", {}))
    config["FrontViewScaleCover"].update(changes.get("scale_cover", {}))
    config.setdefault("FrontViewObservability", {}).update(
        changes.get("observability", {})
    )
    config.setdefault("FrontViewCoverageRecovery", {}).update(
        changes.get("coverage_recovery", {})
    )
    config.setdefault("FrontViewInverseDepthCertificate", {}).update(
        changes.get("inverse_depth_certificate", {})
    )
    config.setdefault("CausalMetricBirth", {}).update(
        changes.get("causal_metric_birth", {})
    )
    config.setdefault("CausalPersistentLandmarkMemory", {}).update(
        changes.get("causal_landmark_memory", {})
    )
    config.setdefault("CausalDualResponsibility", {}).update(
        changes.get("causal_dual_responsibility", {})
    )
    config.setdefault("FrontViewDirectionalLayer", {}).update(
        changes.get("directional_layer", {})
    )
    return config


def run_command(command: list[str], log, env: dict[str, str]) -> None:
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
    if result.returncode:
        raise RuntimeError(f"Command exited with code {result.returncode}")


def find_run_dir(variant_root: Path, dataset_name: str, exp_name: str) -> Path:
    matches = sorted(
        (variant_root / dataset_name).glob(f"*_{exp_name}"),
        key=lambda path: path.stat().st_mtime,
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one run for {exp_name}, found {matches}")
    return matches[0]


def verify_h264(path: Path) -> None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode or result.stdout.strip() != "h264":
        raise RuntimeError(f"Video is not H.264: {path}")


def collect_result(
    variant: str,
    gpu: str,
    run_dir: Path,
    fixed_far_metrics_path: Path | None = None,
) -> dict[str, Any]:
    with (run_dir / "results.json").open("r", encoding="utf-8") as handle:
        online = json.load(handle)
    metrics_path = run_dir / "videos_psnr_ssim" / "render_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    fixed_far_metrics = None
    if fixed_far_metrics_path is not None:
        with fixed_far_metrics_path.open("r", encoding="utf-8") as handle:
            fixed_far_metrics = json.load(handle)
    if "lpips" in metrics["mean"]:
        raise RuntimeError("LPIPS was not disabled")
    videos = {
        "render_vs_gt": run_dir / "videos_psnr_ssim/render_vs_gt.mp4",
        "render_depth": run_dir / "videos_psnr_ssim/render_depth.mp4",
    }
    for path in videos.values():
        verify_h264(path)
    frames = int(online["num_processed_frames"])
    seconds = float(online["online_recon_time"])
    return {
        "status": "success",
        "variant": variant,
        "gpu": gpu,
        "run_dir": str(run_dir),
        "metrics": {
            "psnr": float(metrics["mean"]["psnr"]),
            "ssim": float(metrics["mean"]["ssim"]),
            **(
                {
                    "fixed_far_psnr": float(metrics["mean"]["fixed_far_psnr"]),
                    "fixed_far_ssim": float(metrics["mean"]["fixed_far_ssim"]),
                }
                if "fixed_far_psnr" in metrics["mean"]
                else {}
            ),
            **(
                {
                    "fixed_far_psnr": float(
                        fixed_far_metrics["mean"]["fixed_far_psnr"]
                    ),
                    "fixed_far_ssim": float(
                        fixed_far_metrics["mean"]["fixed_far_ssim"]
                    ),
                }
                if fixed_far_metrics is not None
                else {}
            ),
        },
        "metric_frames": int(metrics["frame_count"]),
        "num_processed_frames": frames,
        "num_keyframes": int(online["num_keyframes"]),
        "num_gaussians": int(online["num_gaussians"]),
        "online_recon_time": seconds,
        "online_fps": frames / seconds,
        "frontview_sampling": online.get("frontview_sampling", {}),
        "frontview_inverse_depth_certificate": online.get(
            "frontview_inverse_depth_certificate", {}
        ),
        "frontview_far_field": online.get("frontview_far_field", {}),
        "frontview_observability": online.get("frontview_observability", {}),
        "frontview_scale_cover": online.get("frontview_scale_cover", {}),
        "frontview_coverage_recovery": online.get(
            "frontview_coverage_recovery", {}
        ),
        "causal_metric_birth": online.get("causal_metric_birth", {}),
        "causal_persistent_landmark_memory": online.get(
            "causal_persistent_landmark_memory", {}
        ),
        "causal_dual_responsibility": online.get(
            "causal_dual_responsibility", {}
        ),
        "streaming_appearance_lod": online.get("streaming_appearance_lod", {}),
        "videos": {name: str(path) for name, path in videos.items()},
    }


def run_variant(
    args: argparse.Namespace,
    batch_root: Path,
    variant: str,
    gpu: str,
    config_path: Path,
    dataset_name: str,
    tag: str,
) -> dict[str, Any]:
    variant_root = batch_root / variant
    exp_name = f"mountains_crr_{variant}_seed{args.seed}_gpu{gpu}_{tag}"
    log_path = batch_root / "launcher_logs" / f"{variant}.log"
    status_path = batch_root / "status" / f"{variant}.json"
    env = process_env(gpu, args.cpu_threads)
    started = time.monotonic()
    status: dict[str, Any] = {
        "status": "running_slam",
        "variant": variant,
        "gpu": gpu,
        "config": str(config_path),
        "log": str(log_path),
    }
    write_json(status_path, status)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x", encoding="utf-8") as log:
            run_command(
                [
                    str(args.python),
                    "slam_new.py",
                    "--config",
                    str(config_path),
                    "--exp_name",
                    exp_name,
                    "--seed",
                    str(args.seed),
                    "--cpu_threads",
                    str(args.cpu_threads),
                ],
                log,
                env,
            )
            run_dir = find_run_dir(variant_root, dataset_name, exp_name)
            status.update(status="running_render", run_dir=str(run_dir))
            write_json(status_path, status)
            render_command = [
                str(args.python),
                "render.py",
                "--run_dir",
                str(run_dir),
                "--output_dir",
                str(run_dir / "videos_psnr_ssim"),
                "--fps",
                str(args.fps),
                "--max_frames",
                "-1",
                "--device",
                "cuda:0",
                "--far_gs_depth_threshold",
                "80",
                "--skip_lpips",
                "--skip_novel",
                "--skip_primitives",
                "--ignore_cached_renders",
            ]
            if args.render_begin is not None:
                render_command.extend(("--render_begin", str(args.render_begin)))
            if args.render_end is not None:
                render_command.extend(("--render_end", str(args.render_end)))
            if (
                args.fixed_far_mask_dir is not None
                and args.fixed_far_begin is None
            ):
                render_command.extend(
                    ("--fixed_far_mask_dir", str(args.fixed_far_mask_dir))
                )
            run_command(render_command, log, env)
            fixed_far_metrics_path = None
            if args.fixed_far_begin is not None:
                fixed_far_output = run_dir / "videos_fixed_far_psnr_ssim"
                fixed_far_command = [
                    str(args.python),
                    "render.py",
                    "--run_dir",
                    str(run_dir),
                    "--output_dir",
                    str(fixed_far_output),
                    "--fps",
                    str(args.fps),
                    "--max_frames",
                    "-1",
                    "--device",
                    "cuda:0",
                    "--far_gs_depth_threshold",
                    "80",
                    "--skip_lpips",
                    "--skip_novel",
                    "--skip_primitives",
                    "--ignore_cached_renders",
                    "--render_begin",
                    str(args.fixed_far_begin),
                    "--render_end",
                    str(args.fixed_far_end),
                    "--fixed_far_mask_dir",
                    str(args.fixed_far_mask_dir),
                ]
                run_command(fixed_far_command, log, env)
                fixed_far_metrics_path = fixed_far_output / "render_metrics.json"
        result = collect_result(
            variant,
            gpu,
            run_dir,
            fixed_far_metrics_path=fixed_far_metrics_path,
        )
        result["wall_time"] = time.monotonic() - started
        write_json(status_path, result)
        return result
    except Exception as error:  # noqa: BLE001 - persist every worker failure.
        status.update(
            status="failed",
            error=str(error),
            wall_time=time.monotonic() - started,
        )
        write_json(status_path, status)
        return status


def metric_delta(lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(lhs["metrics"][key]) - float(rhs["metrics"][key])
        for key in lhs["metrics"].keys() & rhs["metrics"].keys()
    }


def main() -> int:
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    if args.fixed_far_mask_dir is not None:
        args.fixed_far_mask_dir = args.fixed_far_mask_dir.expanduser().resolve()
    if not args.config.is_file() or not args.python.is_file():
        raise FileNotFoundError(f"Missing config or Python: {args.config}, {args.python}")
    selected_variants = (
        list(VARIANTS.items())
        if args.variants is None
        else [(name, VARIANTS[name]) for name in args.variants]
    )
    if not selected_variants:
        raise ValueError("At least one variant is required")
    if len(set(args.gpus)) != len(selected_variants):
        raise ValueError("One distinct GPU is required per selected variant")
    if args.frames < 0:
        raise ValueError("--frames must be nonnegative")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if (args.render_begin is None) != (args.render_end is None):
        raise ValueError("--render-begin and --render-end must be provided together")
    if args.render_begin is not None and args.render_end < args.render_begin:
        raise ValueError("--render-end must be >= --render-begin")
    if args.fixed_far_mask_dir is not None and not args.fixed_far_mask_dir.is_dir():
        raise FileNotFoundError(args.fixed_far_mask_dir)
    if (args.fixed_far_begin is None) != (args.fixed_far_end is None):
        raise ValueError(
            "--fixed-far-begin and --fixed-far-end must be provided together"
        )
    if args.fixed_far_begin is not None:
        if args.fixed_far_mask_dir is None:
            raise ValueError("A fixed-far range requires --fixed-far-mask-dir")
        if args.fixed_far_end < args.fixed_far_begin:
            raise ValueError("--fixed-far-end must be >= --fixed-far-begin")

    baseline = load_config(str(args.config))
    if args.scale_cover_backend != "config":
        baseline.setdefault("FrontViewScaleCover", {})["query_backend"] = (
            args.scale_cover_backend
        )
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = args.save_dir / f"batch_{tag}"
    config_dir = batch_root / "runtime_configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    jobs = []
    for (variant, changes), gpu in zip(selected_variants, args.gpus):
        variant_root = batch_root / variant
        config = build_config(baseline, variant, changes, variant_root, args.frames)
        config_path = config_dir / f"{variant}.yaml"
        with config_path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        jobs.append((variant, str(gpu), config_path, config["Dataset"]["name"]))

    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "tag": tag,
        "base_config": str(args.config),
        "frames": args.frames or "all",
        "seed": args.seed,
        "cpu_threads_per_process": args.cpu_threads,
        "scale_cover_backend": args.scale_cover_backend,
        "render_range": [args.render_begin, args.render_end],
        "fixed_far_mask_dir": (
            None
            if args.fixed_far_mask_dir is None
            else str(args.fixed_far_mask_dir)
        ),
        "fixed_far_range": [args.fixed_far_begin, args.fixed_far_end],
        "jobs": [
            {"variant": variant, "gpu": gpu, "config": str(config_path)}
            for variant, gpu, config_path, _ in jobs
        ],
    }
    write_json(batch_root / "manifest.json", manifest)
    print(f"Batch: {batch_root}", flush=True)
    if args.dry_run:
        return 0

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                run_variant,
                args,
                batch_root,
                variant,
                gpu,
                config_path,
                dataset_name,
                tag,
            ): variant
            for variant, gpu, config_path, dataset_name in jobs
        }
        for future in as_completed(futures):
            variant = futures[future]
            results[variant] = future.result()
            print(f"{variant}: {results[variant]['status']}", flush=True)

    successful = all(row.get("status") == "success" for row in results.values())
    summary: dict[str, Any] = {
        "status": "success" if successful else "failed",
        "protocol": "matched online budget; saved checkpoint PSNR/SSIM; LPIPS disabled",
        "results": results,
    }
    if successful:
        baseline_variant = selected_variants[0][0]
        baseline_result = results[baseline_variant]
        summary["baseline_variant"] = baseline_variant
        summary["versus_baseline"] = {
            variant: metric_delta(row, baseline_result)
            for variant, row in results.items()
            if variant != baseline_variant
        }
    write_json(batch_root / "summary.json", summary)
    manifest["status"] = summary["status"]
    manifest["summary"] = str(batch_root / "summary.json")
    write_json(batch_root / "manifest.json", manifest)
    print(f"Summary: {batch_root / 'summary.json'}", flush=True)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
