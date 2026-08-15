#!/usr/bin/env python3
"""Decompose late Mountains map corruption on GPUs 4-7."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

import run_mountains_crr_ablation as runner


SHARED_GSPLAT_CACHE = Path.home() / ".cache/torch_extensions/online3dgs_dual_shared"
GSPLAT_PREBUILT_EXTENSION = SHARED_GSPLAT_CACHE / "gsplat_cuda/gsplat_cuda.so"
GSPLAT_PREBUILT_SITE = Path(__file__).resolve().parent / "gsplat_prebuilt_site"
_BASE_PROCESS_ENV = runner.process_env


def shared_process_env(gpu, cpu_threads=1):
    env = _BASE_PROCESS_ENV(gpu, cpu_threads)
    env["TORCH_EXTENSIONS_DIR"] = str(SHARED_GSPLAT_CACHE)
    env["GSPLAT_PREBUILT_EXTENSION"] = str(GSPLAT_PREBUILT_EXTENSION)
    env["PYTHONPATH"] = f"{GSPLAT_PREBUILT_SITE}:{env.get('PYTHONPATH', '')}"
    return env


runner.process_env = shared_process_env

BASE_CONFIG = (
    runner.ROOT
    / "Logs_mountains_adaptive_goal_8_12_8_13"
    / "final/stage35_full_765/batch_20260813_095909"
    / "runtime_configs/A_visible_residual_detail_real.yaml"
)
OUTPUT_ROOT = (
    runner.ROOT
    / "Logs_mountains_far_depth_goal_8_13/depth_cause_decomposition"
)
FIXED_FAR_MASKS = (
    runner.ROOT
    / "Logs_mountains_adaptive_goal_8_12_8_13/evaluation"
    / "fixed_far_masks_q80_545_619_v2"
)

METHODS = {
    "A_stage35": {},
    "B_stop_birth_after_620": {
        "enabled": True,
        "start_frame": 620,
        "stop_birth": True,
    },
    "C_stop_pruning_after_620": {
        "enabled": True,
        "start_frame": 620,
        "stop_opacity_pruning": True,
    },
    "D_freeze_prior_geometry": {
        "enabled": True,
        "start_frame": 620,
        "freeze_existing_geometry": True,
        "isolate_future_births": True,
    },
}
BASELINE_VARIANT = "A_stage35"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--save-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=runner.DEFAULT_PYTHON)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    raise RuntimeError(f"PLY has no vertex count: {path}")


def render_stage(args, run_dir, stage_dir, output_name, log, env):
    output_dir = run_dir / output_name
    command = [
        str(args.python),
        "render.py",
        "--run_dir",
        str(stage_dir),
        "--output_dir",
        str(output_dir),
        "--fps",
        "24",
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
    runner.run_command(command, log, env)
    fixed_output = run_dir / f"{output_name}_fixed_far"
    fixed_command = command.copy()
    fixed_command[fixed_command.index(str(output_dir))] = str(fixed_output)
    fixed_command.extend(
        [
            "--render_begin",
            "545",
            "--render_end",
            "619",
            "--fixed_far_mask_dir",
            str(args.fixed_far_mask_dir),
        ]
    )
    runner.run_command(fixed_command, log, env)
    return output_dir, fixed_output


def read_stage_metrics(output_dir, fixed_output):
    with (output_dir / "render_metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (fixed_output / "render_metrics.json").open("r", encoding="utf-8") as handle:
        fixed = json.load(handle)
    if "lpips" in metrics["mean"] or "lpips" in fixed["mean"]:
        raise RuntimeError("LPIPS was not disabled")
    videos = {
        "render_vs_gt": output_dir / "render_vs_gt.mp4",
        "render_depth": output_dir / "render_depth.mp4",
    }
    for path in videos.values():
        runner.verify_h264(path)
    return {
        "psnr": float(metrics["mean"]["psnr"]),
        "ssim": float(metrics["mean"]["ssim"]),
        "fixed_far_psnr": float(fixed["mean"]["fixed_far_psnr"]),
        "fixed_far_ssim": float(fixed["mean"]["fixed_far_ssim"]),
        "metric_frames": int(metrics["frame_count"]),
        "videos": {name: str(path) for name, path in videos.items()},
    }


def run_variant(args, batch_root, variant, gpu, config_path, dataset_name, tag):
    variant_root = batch_root / variant
    exp_name = f"mountains_depth_audit_{variant}_seed{args.seed}_gpu{gpu}_{tag}"
    log_path = batch_root / "launcher_logs" / f"{variant}.log"
    status_path = batch_root / "status" / f"{variant}.json"
    env = runner.process_env(gpu, args.cpu_threads)
    started = time.monotonic()
    status = {
        "status": "running_slam",
        "variant": variant,
        "gpu": gpu,
        "config": str(config_path),
        "log": str(log_path),
    }
    runner.write_json(status_path, status)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x", encoding="utf-8") as log:
            runner.run_command(
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
            run_dir = runner.find_run_dir(variant_root, dataset_name, exp_name)
            with (run_dir / "results.json").open("r", encoding="utf-8") as handle:
                run_results = json.load(handle)

            online_stage = run_dir / "online_stage"
            online_ply = online_stage / "point_cloud.ply"
            if not online_ply.is_file():
                raise FileNotFoundError(online_ply)
            shutil.copy2(run_dir / "config.yaml", online_stage / "config.yaml")

            status.update(status="running_online_render", run_dir=str(run_dir))
            runner.write_json(status_path, status)
            online_output, online_fixed = render_stage(
                args,
                run_dir,
                online_stage,
                "videos_online_psnr_ssim",
                log,
                env,
            )
            status["status"] = "running_final_render"
            runner.write_json(status_path, status)
            final_output, final_fixed = render_stage(
                args,
                run_dir,
                run_dir,
                "videos_final_psnr_ssim",
                log,
                env,
            )

        result = {
            "status": "success",
            "variant": variant,
            "gpu": gpu,
            "run_dir": str(run_dir),
            "online_stage": read_stage_metrics(online_output, online_fixed),
            "final_stage": read_stage_metrics(final_output, final_fixed),
            "online_num_gaussians": ply_vertex_count(online_ply),
            "final_num_gaussians": int(run_results["num_gaussians"]),
            "num_processed_frames": int(run_results["num_processed_frames"]),
            "num_keyframes": int(run_results["num_keyframes"]),
            "online_recon_time": float(run_results["online_recon_time"]),
            "online_fps": (
                float(run_results["num_processed_frames"])
                / float(run_results["online_recon_time"])
            ),
            "post_refinement_seconds": float(
                run_results.get("post_refinement_seconds", 0.0)
            ),
            "causal_depth_audit": run_results.get("causal_depth_audit", {}),
            "wall_time": time.monotonic() - started,
        }
        runner.write_json(status_path, result)
        return result
    except Exception as error:  # noqa: BLE001 - persist worker failures.
        status.update(
            status="failed",
            error=str(error),
            wall_time=time.monotonic() - started,
        )
        runner.write_json(status_path, status)
        return status


def stage_delta(lhs, rhs, stage):
    return {
        key: float(lhs[stage][key]) - float(rhs[stage][key])
        for key in ("psnr", "ssim", "fixed_far_psnr", "fixed_far_ssim")
    }


def main():
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    args.fixed_far_mask_dir = FIXED_FAR_MASKS.resolve()
    args.gpus = ("4", "5", "6", "7")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    for path in (
        args.config,
        args.python,
        args.fixed_far_mask_dir,
        GSPLAT_PREBUILT_EXTENSION,
        GSPLAT_PREBUILT_SITE / "sitecustomize.py",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    baseline = runner.load_config(str(args.config))
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = args.save_dir / f"batch_{tag}"
    config_dir = batch_root / "runtime_configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    jobs = []
    for (variant, audit), gpu in zip(METHODS.items(), args.gpus):
        variant_root = batch_root / variant
        config = runner.build_config(baseline, variant, {}, variant_root, 765)
        config["Results"]["save_online_stage"] = True
        config["CausalDepthAudit"] = audit
        config_path = config_dir / f"{variant}.yaml"
        with config_path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        jobs.append((variant, gpu, config_path, config["Dataset"]["name"]))

    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "protocol": (
            "Matched 765-frame Stage35 causal decomposition at frame 620; "
            "online and final checkpoints are evaluated on all frames and fixed "
            "545-619 far masks; LPIPS disabled"
        ),
        "tag": tag,
        "base_config": str(args.config),
        "seed": args.seed,
        "fixed_far_mask_dir": str(args.fixed_far_mask_dir),
        "jobs": [
            {"variant": variant, "gpu": gpu, "config": str(config_path)}
            for variant, gpu, config_path, _ in jobs
        ],
    }
    runner.write_json(batch_root / "manifest.json", manifest)
    print(f"Batch: {batch_root}", flush=True)
    if args.dry_run:
        return 0

    results = {}
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
    summary = {
        "status": "success" if successful else "failed",
        "protocol": manifest["protocol"],
        "results": results,
    }
    if successful:
        baseline_result = results[BASELINE_VARIANT]
        summary["versus_baseline"] = {
            variant: {
                "online_stage": stage_delta(row, baseline_result, "online_stage"),
                "final_stage": stage_delta(row, baseline_result, "final_stage"),
                "online_num_gaussians": (
                    row["online_num_gaussians"]
                    - baseline_result["online_num_gaussians"]
                ),
            }
            for variant, row in results.items()
            if variant != BASELINE_VARIANT
        }
        summary["post_refinement_delta"] = {
            variant: stage_delta(row, row, "final_stage")
            for variant, row in results.items()
        }
        for variant, row in results.items():
            summary["post_refinement_delta"][variant] = {
                key: row["final_stage"][key] - row["online_stage"][key]
                for key in ("psnr", "ssim", "fixed_far_psnr", "fixed_far_ssim")
            }
    runner.write_json(batch_root / "summary.json", summary)
    manifest["status"] = summary["status"]
    manifest["summary"] = str(batch_root / "summary.json")
    runner.write_json(batch_root / "manifest.json", manifest)
    print(f"Summary: {batch_root / 'summary.json'}", flush=True)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
