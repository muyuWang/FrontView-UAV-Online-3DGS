#!/usr/bin/env python3
"""Test causal landmark metric/responsibility coordinate decoupling."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

import run_mountains_landmark_admitted_mean as admitted


runner = admitted.runner
BASE_CONFIG = admitted.BASE_CONFIG
FIXED_FAR_MASKS = admitted.FIXED_FAR_MASKS
OUTPUT_ROOT = (
    runner.ROOT
    / "Logs_mountains_far_depth_goal_8_13/landmark_coordinate_decoupling"
)


def memory(*, responsibility="metric", shuffle=False):
    return {
        "enabled": True,
        "conditioning_mode": "admitted_mean",
        "minimum_observations": 1,
        "maximum_conditioning_points": 500,
        "transport_rule": "full",
        "propagate_conditioned_uncertainty": False,
        "responsibility_coordinate": responsibility,
        "shuffle_depths": bool(shuffle),
        "shuffle_seed": 43,
    }


METHODS = {
    "A_stage35": {},
    "B_landmark_metric_responsibility": {
        "causal_landmark_memory": memory(responsibility="metric"),
    },
    "C_landmark_original_responsibility": {
        "causal_landmark_memory": memory(responsibility="original_posterior"),
    },
    "D_shuffled_original_responsibility": {
        "causal_landmark_memory": memory(
            responsibility="original_posterior", shuffle=True
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--save-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=runner.DEFAULT_PYTHON)
    parser.add_argument("--frames", type=int, default=620)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--gpus", nargs=4, default=("4", "5", "6", "7"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    args.fixed_far_mask_dir = (
        FIXED_FAR_MASKS.resolve() if args.frames >= 620 else None
    )
    args.fixed_far_begin = 545 if args.frames >= 620 else None
    args.fixed_far_end = 619 if args.frames >= 620 else None
    args.render_begin = None
    args.render_end = None
    args.fps = 24.0
    args.scale_cover_backend = "config"
    if args.frames < 1 or args.frames > 765:
        raise ValueError("--frames must lie in [1, 765]")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    for path in (
        args.config,
        args.python,
        admitted.GSPLAT_PREBUILT_EXTENSION,
        admitted.GSPLAT_PREBUILT_SITE / "sitecustomize.py",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    baseline = runner.load_config(str(args.config))
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = args.save_dir / f"frames{args.frames}_batch_{tag}"
    config_dir = batch_root / "runtime_configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    jobs = []
    for (variant, changes), gpu in zip(METHODS.items(), args.gpus):
        variant_root = batch_root / variant
        config = runner.build_config(
            baseline, variant, changes, variant_root, args.frames
        )
        config_path = config_dir / f"{variant}.yaml"
        with config_path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        jobs.append((variant, str(gpu), config_path, config["Dataset"]["name"]))

    protocol = (
        "Matched online candidate UV, original uncertainty admission, confidence, "
        "and per-frame birth budget. Landmark transport changes metric birth depth; "
        "original-posterior variants retain the unconditioned posterior only for "
        "PBSD/FPR/TSC capacity responsibility. Shuffled control preserves all "
        "budgets and permutes only landmark depth-to-projection correspondence."
    )
    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "protocol": protocol,
        "tag": tag,
        "base_config": str(args.config),
        "frames": args.frames,
        "seed": args.seed,
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
                runner.run_variant,
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
        "protocol": protocol,
        "results": results,
    }
    if successful:
        baseline_result = results["A_stage35"]
        metric_result = results["B_landmark_metric_responsibility"]
        coordinate_result = results["C_landmark_original_responsibility"]
        shuffled_result = results["D_shuffled_original_responsibility"]
        summary["versus_baseline"] = {
            name: runner.metric_delta(row, baseline_result)
            for name, row in results.items()
            if name != "A_stage35"
        }
        summary["coordinate_decoupling"] = runner.metric_delta(
            coordinate_result, metric_result
        )
        summary["causal_correspondence"] = runner.metric_delta(
            coordinate_result, shuffled_result
        )
    runner.write_json(batch_root / "summary.json", summary)
    manifest["status"] = summary["status"]
    manifest["summary"] = str(batch_root / "summary.json")
    runner.write_json(batch_root / "manifest.json", manifest)
    print(f"Summary: {batch_root / 'summary.json'}", flush=True)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
