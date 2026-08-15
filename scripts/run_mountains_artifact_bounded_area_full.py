#!/usr/bin/env python3
"""Run the matched full-Mountains footprint-bound repair on GPUs 0 and 1."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

import run_mountains_causal_depth_posterior_paired as paired
import run_mountains_crr_ablation as runner
import run_mountains_observability_typed_episode_full_paired as episode
import run_mountains_observability_typed_ownership_paired as typed


OUTPUT_ROOT = runner.ROOT / "Logs_mountains_artifact_repair_8_14/bounded_area_full765"
CURRENT = episode.METHODS["B_typed_episode_ward_k24"]
METHODS = {
    "A_current_odds": CURRENT,
    "B_bounded_area": {
        **CURRENT,
        "far_field": {
            **CURRENT.get("far_field", {}),
            "footprint_trust_mode": (
                "certificate_residual_rd_visible_detail_bounded_area"
            ),
        },
    },
}
SCHEDULE = {
    "0": ("A_current_odds",),
    "1": ("B_bounded_area",),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=typed.BASE_CONFIG)
    parser.add_argument("--save-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=runner.DEFAULT_PYTHON)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_gpu(args, batch_root, gpu, job, tag):
    run_name, config_path, dataset_name = job
    return runner.run_variant(
        args, batch_root, run_name, gpu, config_path, dataset_name, tag
    )


def main():
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    args.frames = 0
    args.fixed_far_mask_dir = typed.FIXED_FAR_MASKS.resolve()
    args.fixed_far_begin = 545
    args.fixed_far_end = 619
    args.render_begin = None
    args.render_end = None
    args.fps = 24.0
    args.scale_cover_backend = "config"
    args.gpus = tuple(SCHEDULE)
    for path in (
        args.config,
        args.python,
        args.fixed_far_mask_dir,
        paired.GSPLAT_PREBUILT_EXTENSION,
        paired.GSPLAT_PREBUILT_HOOK,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    baseline = runner.load_config(str(args.config))
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = args.save_dir / f"batch_{tag}"
    config_dir = batch_root / "runtime_configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    jobs = {}
    manifest_jobs = []
    for gpu, methods in SCHEDULE.items():
        method = methods[0]
        run_name = f"gpu{gpu}_{method}"
        config = runner.build_config(
            baseline,
            run_name,
            METHODS[method],
            batch_root / run_name,
            0,
        )
        config_path = config_dir / f"{run_name}.yaml"
        with config_path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        jobs[gpu] = (run_name, config_path, config["Dataset"]["name"])
        manifest_jobs.append(
            {"gpu": gpu, "method": method, "config": str(config_path)}
        )

    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "protocol": (
            "matched full-765 concurrent control; only footprint certificate "
            "headroom law differs; PSNR/SSIM only"
        ),
        "base_commit": "1492e3d",
        "base_config": str(args.config),
        "seed": args.seed,
        "frames": 765,
        "jobs": manifest_jobs,
    }
    runner.write_json(batch_root / "manifest.json", manifest)
    print(f"Batch: {batch_root}", flush=True)
    if args.dry_run:
        return 0

    results = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_gpu, args, batch_root, gpu, job, tag): gpu
            for gpu, job in jobs.items()
        }
        for future in as_completed(futures):
            gpu = futures[future]
            results[gpu] = future.result()
            print(f"GPU {gpu}: {results[gpu]['status']}", flush=True)

    successful = all(row.get("status") == "success" for row in results.values())
    manifest["status"] = "success" if successful else "failed"
    manifest["results"] = results
    runner.write_json(batch_root / "summary.json", manifest)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
