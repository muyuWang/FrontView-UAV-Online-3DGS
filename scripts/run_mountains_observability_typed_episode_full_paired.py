#!/usr/bin/env python3
"""Run reversed full-Mountains pairs for episode-aware directional memory."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

import run_mountains_causal_depth_posterior_paired as paired
import run_mountains_crr_ablation as runner
import run_mountains_observability_typed_ownership_paired as typed


OUTPUT_ROOT = (
    runner.ROOT
    / "Logs_mountains_far_depth_goal_8_13/observability_typed_episode_full765"
)
METHODS = {
    "A_typed_bridge_k12": typed.METHODS["C_typed_abstain"],
    "B_typed_episode_ward_k24": {
        **typed.METHODS["C_typed_abstain"],
        "directional_layer": {
            **typed.METHODS["C_typed_abstain"]["directional_layer"],
            "anchor_selection_mode": "episode_ordered_ward",
            "max_anchors": 24,
        },
    },
}
SCHEDULE = {
    "6": ("A_typed_bridge_k12", "B_typed_episode_ward_k24"),
    "7": ("B_typed_episode_ward_k24", "A_typed_bridge_k12"),
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


def run_gpu_sequence(args, batch_root, gpu, jobs, tag):
    rows = []
    for index, (run_name, config_path, dataset_name) in enumerate(jobs):
        row = runner.run_variant(
            args, batch_root, run_name, gpu, config_path, dataset_name, tag
        )
        row["sequence_index"] = index
        rows.append(row)
        if row.get("status") != "success":
            break
    return gpu, rows


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
    jobs_by_gpu = {}
    manifest_jobs = []
    for gpu, methods in SCHEDULE.items():
        jobs = []
        for index, method in enumerate(methods):
            run_name = f"gpu{gpu}_{index}_{method}"
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
            jobs.append((run_name, config_path, config["Dataset"]["name"]))
            manifest_jobs.append(
                {
                    "gpu": gpu,
                    "sequence_index": index,
                    "method": method,
                    "config": str(config_path),
                }
            )
        jobs_by_gpu[gpu] = jobs

    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "protocol": (
            "reversed same-GPU full-765 pairs; typed metric abstention fixed; "
            "only directional episode-memory policy differs; PSNR/SSIM only"
        ),
        "base_commit": "23b17318235173e662f1c5b0c8245d2aede109d1",
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
            executor.submit(run_gpu_sequence, args, batch_root, gpu, jobs, tag): gpu
            for gpu, jobs in jobs_by_gpu.items()
        }
        for future in as_completed(futures):
            gpu, rows = future.result()
            results[gpu] = rows
            print(f"GPU {gpu}: {len(rows)} runs, {rows[-1]['status']}", flush=True)

    successful = all(
        len(results.get(gpu, ())) == len(methods)
        and all(row.get("status") == "success" for row in results[gpu])
        for gpu, methods in SCHEDULE.items()
    )
    manifest["status"] = "success" if successful else "failed"
    manifest["results"] = results
    runner.write_json(batch_root / "summary.json", manifest)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
