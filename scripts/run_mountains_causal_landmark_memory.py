#!/usr/bin/env python3
"""Run matched Mountains causal landmark-memory controls on GPUs 4-7."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

import run_mountains_crr_ablation as runner


BASE_CONFIG = (
    runner.ROOT
    / "Logs_mountains_adaptive_goal_8_12_8_13"
    / "final/stage35_full_765/batch_20260813_095909"
    / "runtime_configs/A_visible_residual_detail_real.yaml"
)
OUTPUT_ROOT = runner.ROOT / "Logs_mountains_far_depth_goal_8_13/landmark_memory_620"
FIXED_FAR_MASKS = (
    runner.ROOT
    / "Logs_mountains_adaptive_goal_8_12_8_13/evaluation"
    / "fixed_far_masks_q80_545_619_v2"
)


def memory(*, minimum_observations=1, shuffle=False):
    return {
        "enabled": True,
        "minimum_observations": int(minimum_observations),
        "maximum_conditioning_points": 500,
        "shuffle_depths": bool(shuffle),
        "shuffle_seed": 43,
    }


METHODS = {
    "A_stage35": {},
    "B_landmark_real_obs1": {"causal_landmark_memory": memory()},
    "C_landmark_shuffled_obs1": {
        "causal_landmark_memory": memory(shuffle=True)
    },
    "D_landmark_real_obs2": {
        "causal_landmark_memory": memory(minimum_observations=2)
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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
    args.gpus = ("4", "5", "6", "7")
    if args.frames != 620:
        raise ValueError("The fixed-far protocol is defined for exactly 620 frames")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    for path in (args.config, args.python, args.fixed_far_mask_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    baseline = runner.load_config(str(args.config))
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = args.save_dir / f"batch_{tag}"
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
        jobs.append((variant, gpu, config_path, config["Dataset"]["name"]))

    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "protocol": (
            "matched 620-frame mapping, identical 500-anchor DepthCov budget, "
            "real versus shuffled persistent landmark depths"
        ),
        "tag": tag,
        "base_config": str(args.config),
        "frames": args.frames,
        "seed": args.seed,
        "fixed_far_mask_dir": str(args.fixed_far_mask_dir),
        "fixed_far_range": [args.fixed_far_begin, args.fixed_far_end],
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
        "protocol": manifest["protocol"],
        "results": results,
    }
    if successful:
        baseline_result = results["A_stage35"]
        summary["versus_baseline"] = {
            name: runner.metric_delta(row, baseline_result)
            for name, row in results.items()
            if name != "A_stage35"
        }
        summary["causal_control"] = runner.metric_delta(
            results["B_landmark_real_obs1"],
            results["C_landmark_shuffled_obs1"],
        )
    runner.write_json(batch_root / "summary.json", summary)
    manifest["status"] = summary["status"]
    manifest["summary"] = str(batch_root / "summary.json")
    runner.write_json(batch_root / "manifest.json", manifest)
    print(f"Summary: {batch_root / 'summary.json'}", flush=True)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
