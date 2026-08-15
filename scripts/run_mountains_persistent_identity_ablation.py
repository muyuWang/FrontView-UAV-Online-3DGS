#!/usr/bin/env python3
"""Run the four-way mountains persistent-identity ablation on separate GPUs."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/home/wmy/anaconda3/envs/worldvln/bin/python")
DEFAULT_OUTPUT = ROOT / "Logs_mountains_pir_ablation_8_6"
DEFAULT_BASELINE = (
    ROOT
    / "Logs_360dvo_directional_layer_8_3"
    / "360DVO-DirectionalLayer-mountains"
    / "2026-08-03-02-19-21_directional_causal_v2_seed43_gpu4_20260803"
    / "config.yaml"
)

VARIANTS = {
    "A_source_raster": {
        "preserve_sparse_track_geometry": False,
        "responsibility_basis": "source",
        "sparse_track_identity_enabled": False,
        "shuffle_sparse_track_identity": False,
    },
    "B_source_exact": {
        "preserve_sparse_track_geometry": True,
        "responsibility_basis": "source",
        "sparse_track_identity_enabled": False,
        "shuffle_sparse_track_identity": False,
    },
    "C_identity_exact": {
        "preserve_sparse_track_geometry": True,
        "responsibility_basis": "persistent_identity",
        "sparse_track_identity_enabled": True,
        "shuffle_sparse_track_identity": False,
    },
    "D_identity_shuffled": {
        "preserve_sparse_track_geometry": True,
        "responsibility_basis": "persistent_identity",
        "sparse_track_identity_enabled": True,
        "shuffle_sparse_track_identity": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpus", nargs=4, default=("4", "5", "6", "7"))
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--video-fps", type=float, default=24.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def finite_metrics(metrics: dict[str, Any], label: str) -> dict[str, float]:
    result = {}
    for key in ("psnr", "ssim", "lpips"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"Invalid {label}.{key}: {value}")
        result[key] = float(value)
    return result


def build_config(
    baseline: dict[str, Any],
    variant: str,
    options: dict[str, Any],
    save_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(baseline)
    config.pop("inherit_from", None)
    config["Results"]["save_dir"] = str((save_dir / variant).resolve())
    config["Results"]["save_gt"] = False
    config["Results"]["save_exr"] = False
    config["Results"]["save_mesh"] = False
    config["Results"]["skip_eval"] = False

    frequency = config["Model"].setdefault("frequency_sampling", {})
    frequency["preserve_sparse_track_geometry"] = bool(
        options["preserve_sparse_track_geometry"]
    )
    config["FrontViewFarField"]["responsibility_basis"] = options[
        "responsibility_basis"
    ]
    config["FrontViewScaleCover"]["sparse_track_identity_enabled"] = bool(
        options["sparse_track_identity_enabled"]
    )
    config["FrontViewScaleCover"]["shuffle_sparse_track_identity"] = bool(
        options["shuffle_sparse_track_identity"]
    )
    return config


def process_env(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    cuda_home = Path("/usr/local/cuda-11.8")
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["CUDA_HOME"] = str(cuda_home)
    env["PATH"] = f"{cuda_home / 'bin'}:{env.get('PATH', '')}"
    env["TORCH_EXTENSIONS_DIR"] = str(
        Path.home() / ".cache" / "torch_extensions" / f"online3dgs_gpu{gpu}"
    )
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def find_run_dir(variant_root: Path, dataset_name: str, exp_name: str) -> Path:
    parent = variant_root / dataset_name
    matches = sorted(
        parent.glob(f"*_{exp_name}"), key=lambda path: path.stat().st_mtime
    )
    if not matches:
        raise FileNotFoundError(f"No run matching {exp_name} under {parent}")
    return matches[-1]


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


def collect_result(
    variant: str,
    gpu: str,
    options: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    with (run_dir / "results.json").open("r", encoding="utf-8") as handle:
        online = json.load(handle)
    with (run_dir / "eval" / "final_result.json").open(
        "r", encoding="utf-8"
    ) as handle:
        final = json.load(handle)
    with (run_dir / "videos_full" / "render_metrics.json").open(
        "r", encoding="utf-8"
    ) as handle:
        render = json.load(handle)

    frame_count = int(online["num_processed_frames"])
    recon_time = float(online["online_recon_time"])
    videos = {
        "render_vs_gt": run_dir / "videos_full" / "render_vs_gt.mp4",
        "render_depth": run_dir / "videos_full" / "render_depth.mp4",
    }
    missing = [str(path) for path in videos.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing rendered videos: " + ", ".join(missing))

    return {
        "status": "success",
        "variant": variant,
        "gpu": gpu,
        "options": options,
        "run_dir": str(run_dir),
        "pure_metrics": finite_metrics(final.get("mean", {}), "pure"),
        "online_metrics": finite_metrics(online.get("eval_res", {}), "online"),
        "directional_metrics": finite_metrics(render.get("mean", {}), "directional"),
        "num_processed_frames": frame_count,
        "num_keyframes": int(online["num_keyframes"]),
        "num_gaussians": int(online["num_gaussians"]),
        "online_recon_time": recon_time,
        "online_fps": frame_count / recon_time,
        "render_frame_count": int(render["frame_count"]),
        "frontview_far_field": online.get("frontview_far_field", {}),
        "frontview_scale_cover": online.get("frontview_scale_cover", {}),
        "videos": {name: str(path) for name, path in videos.items()},
    }


def run_variant(
    args: argparse.Namespace,
    variant: str,
    gpu: str,
    options: dict[str, Any],
    config_path: Path,
    dataset_name: str,
    tag: str,
) -> dict[str, Any]:
    variant_root = args.save_dir / variant
    exp_name = f"mountains_pir_{variant}_seed{args.seed}_gpu{gpu}_{tag}"
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
    env = process_env(gpu)
    log_path = args.save_dir / "launcher_logs" / f"{variant}.log"
    status_path = args.save_dir / "status" / f"{variant}.json"
    started = time.monotonic()
    status: dict[str, Any] = {
        "status": "running_slam",
        "variant": variant,
        "gpu": gpu,
        "options": options,
        "config": str(config_path),
        "log": str(log_path),
    }
    write_json(status_path, status)

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            run_command(slam, log, env)
            run_dir = find_run_dir(variant_root, dataset_name, exp_name)
            render = [
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
            status.update(status="running_render", run_dir=str(run_dir))
            write_json(status_path, status)
            run_command(render, log, env)

        result = collect_result(variant, gpu, options, run_dir)
        result["wall_time"] = time.monotonic() - started
        write_json(status_path, result)
        return result
    except Exception as error:  # noqa: BLE001 - persist failures for batch runs.
        status.update(
            status="failed",
            error=str(error),
            wall_time=time.monotonic() - started,
        )
        write_json(status_path, status)
        return status


def metric_delta(
    results: dict[str, dict[str, Any]], lhs: str, rhs: str
) -> dict[str, dict[str, float]]:
    delta = {}
    for family in ("pure_metrics", "directional_metrics"):
        delta[family] = {
            key: results[lhs][family][key] - results[rhs][family][key]
            for key in ("psnr", "ssim", "lpips")
        }
    return delta


def main() -> int:
    args = parse_args()
    args.save_dir = args.save_dir.expanduser().resolve()
    args.baseline_config = args.baseline_config.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    if not args.baseline_config.is_file():
        raise FileNotFoundError(args.baseline_config)
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    if len(set(args.gpus)) != 4:
        raise ValueError("Four distinct GPUs are required")

    with args.baseline_config.open("r", encoding="utf-8") as handle:
        baseline = yaml.safe_load(handle)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_dir = args.save_dir / "runtime_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for (variant, options), gpu in zip(VARIANTS.items(), args.gpus):
        config = build_config(baseline, variant, options, args.save_dir)
        config_path = config_dir / f"{variant}.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        jobs.append(
            (variant, str(gpu), options, config_path, config["Dataset"]["name"])
        )

    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "tag": tag,
        "baseline_config": str(args.baseline_config),
        "seed": args.seed,
        "jobs": [
            {
                "variant": variant,
                "gpu": gpu,
                "options": options,
                "config": str(config_path),
            }
            for variant, gpu, options, config_path, _ in jobs
        ],
    }
    write_json(args.save_dir / "manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                run_variant,
                args,
                variant,
                gpu,
                options,
                config_path,
                dataset_name,
                tag,
            ): variant
            for variant, gpu, options, config_path, dataset_name in jobs
        }
        for future in as_completed(futures):
            variant = futures[future]
            results[variant] = future.result()
            print(f"{variant}: {results[variant]['status']}", flush=True)

    successful = all(row.get("status") == "success" for row in results.values())
    summary: dict[str, Any] = {
        "status": "success" if successful else "failed",
        "tag": tag,
        "baseline_config": str(args.baseline_config),
        "seed": args.seed,
        "results": results,
    }
    if successful:
        summary["comparisons"] = {
            "B_minus_A_exact_geometry": metric_delta(
                results, "B_source_exact", "A_source_raster"
            ),
            "C_minus_B_identity_responsibility": metric_delta(
                results, "C_identity_exact", "B_source_exact"
            ),
            "C_minus_D_true_vs_shuffled_identity": metric_delta(
                results, "C_identity_exact", "D_identity_shuffled"
            ),
        }
    write_json(args.save_dir / "summary.json", summary)
    manifest["status"] = summary["status"]
    write_json(args.save_dir / "manifest.json", manifest)
    print(f"Summary: {args.save_dir / 'summary.json'}")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
