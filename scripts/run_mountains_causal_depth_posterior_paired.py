#!/usr/bin/env python3
"""Run same-GPU serial causal-depth controls on currently free GPUs 6 and 7."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

import run_mountains_causal_depth_posterior as experiment
import run_mountains_crr_ablation as runner


SHARED_GSPLAT_CACHE = Path.home() / ".cache/torch_extensions/online3dgs_dual_shared"
GSPLAT_PREBUILT_EXTENSION = SHARED_GSPLAT_CACHE / "gsplat_cuda/gsplat_cuda.so"
GSPLAT_PREBUILT_SITE = Path(__file__).resolve().parent / "gsplat_prebuilt_site"
GSPLAT_PREBUILT_HOOK = GSPLAT_PREBUILT_SITE / "sitecustomize.py"
_BASE_PROCESS_ENV = runner.process_env


def shared_process_env(gpu, cpu_threads=1):
    env = _BASE_PROCESS_ENV(gpu, cpu_threads)
    env["TORCH_EXTENSIONS_DIR"] = str(SHARED_GSPLAT_CACHE)
    env["GSPLAT_PREBUILT_EXTENSION"] = str(GSPLAT_PREBUILT_EXTENSION)
    env["PYTHONPATH"] = f"{GSPLAT_PREBUILT_SITE}:{env.get('PYTHONPATH', '')}"
    return env


runner.process_env = shared_process_env

BASE_CONFIG = experiment.runner.DEFAULT_CONFIG
OUTPUT_ROOT = (
    runner.ROOT / "Logs_mountains_far_depth_goal_8_13/causal_depth_posterior_paired_620"
)
FIXED_FAR_MASKS = (
    runner.ROOT
    / "Logs_mountains_adaptive_goal_8_12_8_13/evaluation"
    / "fixed_far_masks_q80_545_619_v2"
)
PAIR_SCHEDULE = {
    "6": ("A_stage35", "B_causal_posterior_real"),
    "7": ("D_matched_observe_only", "C_causal_posterior_shuffled"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--save-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=runner.DEFAULT_PYTHON)
    parser.add_argument("--frames", type=int, default=620)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_gpu_sequence(args, batch_root, gpu, jobs, tag):
    results = []
    for sequence_index, (run_name, config_path, dataset_name) in enumerate(jobs):
        result = runner.run_variant(
            args,
            batch_root,
            run_name,
            gpu,
            config_path,
            dataset_name,
            tag,
        )
        result["sequence_index"] = sequence_index
        results.append(result)
        if result.get("status") != "success":
            break
    return gpu, results


def paired_delta(first, second):
    return {
        key: float(second["metrics"][key]) - float(first["metrics"][key])
        for key in first["metrics"].keys() & second["metrics"].keys()
    }


def main():
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    args.fixed_far_mask_dir = FIXED_FAR_MASKS.resolve()
    args.fixed_far_begin = 545
    args.fixed_far_end = 619
    args.render_begin = None
    args.render_end = None
    args.fps = 24.0
    args.scale_cover_backend = "config"
    args.gpus = tuple(PAIR_SCHEDULE)
    if args.frames != 620:
        raise ValueError("The fixed-far protocol is defined for exactly 620 frames")
    for path in (
        args.config,
        args.python,
        args.fixed_far_mask_dir,
        GSPLAT_PREBUILT_EXTENSION,
        GSPLAT_PREBUILT_HOOK,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    baseline = runner.load_config(str(args.config))
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = args.save_dir / f"batch_{tag}"
    config_dir = batch_root / "runtime_configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    jobs_by_gpu = {}
    manifest_jobs = []
    for gpu, methods in PAIR_SCHEDULE.items():
        jobs = []
        for sequence_index, method in enumerate(methods):
            run_name = f"gpu{gpu}_{sequence_index}_{method}"
            config = runner.build_config(
                baseline,
                run_name,
                experiment.runner.VARIANTS[method],
                batch_root / run_name,
                args.frames,
            )
            config_path = config_dir / f"{run_name}.yaml"
            with config_path.open("x", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            jobs.append((run_name, config_path, config["Dataset"]["name"]))
            manifest_jobs.append(
                {
                    "gpu": gpu,
                    "sequence_index": sequence_index,
                    "method": method,
                    "config": str(config_path),
                }
            )
        jobs_by_gpu[gpu] = jobs

    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "protocol": (
            "same-GPU serial pairs; PBSD-selected UV/RGB/budget unchanged; "
            "real, identity-depth shuffle, and matched observe-only; no LPIPS"
        ),
        "base_commit": "23b1731",
        "base_config": str(args.config),
        "frames": args.frames,
        "seed": args.seed,
        "cpu_threads_per_process": args.cpu_threads,
        "fixed_far_mask_dir": str(args.fixed_far_mask_dir),
        "fixed_far_range": [args.fixed_far_begin, args.fixed_far_end],
        "jobs": manifest_jobs,
    }
    runner.write_json(batch_root / "manifest.json", manifest)
    print(f"Batch: {batch_root}", flush=True)
    if args.dry_run:
        return 0

    results_by_gpu = {}
    with ThreadPoolExecutor(max_workers=len(jobs_by_gpu)) as executor:
        futures = {
            executor.submit(run_gpu_sequence, args, batch_root, gpu, jobs, tag): gpu
            for gpu, jobs in jobs_by_gpu.items()
        }
        for future in as_completed(futures):
            gpu, results = future.result()
            results_by_gpu[gpu] = results
            print(f"GPU {gpu}: {len(results)} runs, {results[-1]['status']}", flush=True)

    successful = all(
        len(rows) == len(PAIR_SCHEDULE[gpu])
        and all(row.get("status") == "success" for row in rows)
        for gpu, rows in results_by_gpu.items()
    )
    summary = {
        "status": "success" if successful else "failed",
        "protocol": manifest["protocol"],
        "results_by_gpu": results_by_gpu,
        "paired_deltas_second_minus_first": {
            gpu: paired_delta(rows[0], rows[1])
            for gpu, rows in results_by_gpu.items()
            if len(rows) == 2
            and all(row.get("status") == "success" for row in rows)
        },
    }
    runner.write_json(batch_root / "summary.json", summary)
    manifest["status"] = summary["status"]
    manifest["summary"] = str(batch_root / "summary.json")
    runner.write_json(batch_root / "manifest.json", manifest)
    print(f"Summary: {batch_root / 'summary.json'}", flush=True)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
