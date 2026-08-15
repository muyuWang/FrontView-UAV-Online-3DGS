#!/usr/bin/env python3
"""Run replicated same-GPU full Mountains Fisher metric-birth pairs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

import run_mountains_causal_depth_posterior_paired as paired
import run_mountains_fisher_metric_birth as experiment
import run_mountains_crr_ablation as runner


OUTPUT_ROOT = (
    runner.ROOT
    / "Logs_mountains_far_depth_goal_8_13/fisher_metric_birth_full765_paired"
)
SCHEDULE = {
    "6": ("A_real_observe", "B_real_fisher"),
    "7": ("A_real_observe", "B_real_fisher"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=experiment.runner.DEFAULT_CONFIG)
    parser.add_argument("--save-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=runner.DEFAULT_PYTHON)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    args.frames = 0
    args.fixed_far_mask_dir = paired.FIXED_FAR_MASKS.resolve()
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
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")

    baseline = runner.load_config(str(args.config))
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = args.save_dir / f"batch_{tag}"
    config_dir = batch_root / "runtime_configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    jobs_by_gpu = {}
    manifest_jobs = []
    for gpu, methods in SCHEDULE.items():
        jobs = []
        for sequence_index, method in enumerate(methods):
            run_name = f"gpu{gpu}_{sequence_index}_{method}"
            config = runner.build_config(
                baseline,
                run_name,
                experiment.runner.VARIANTS[method],
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
                    "sequence_index": sequence_index,
                    "method": method,
                    "config": str(config_path),
                }
            )
        jobs_by_gpu[gpu] = jobs

    protocol = (
        "two replicated same-GPU serial full-765 pairs; Fisher reference selection; "
        "real metric binding; fixed 545-619 far masks; no LPIPS"
    )
    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "protocol": protocol,
        "base_commit": "23b1731",
        "base_config": str(args.config),
        "frames": 765,
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
            executor.submit(
                paired.run_gpu_sequence, args, batch_root, gpu, jobs, tag
            ): gpu
            for gpu, jobs in jobs_by_gpu.items()
        }
        for future in as_completed(futures):
            gpu, results = future.result()
            results_by_gpu[gpu] = results
            print(
                f"GPU {gpu}: {len(results)} runs, {results[-1]['status']}",
                flush=True,
            )

    successful = all(
        len(rows) == len(SCHEDULE[gpu])
        and all(row.get("status") == "success" for row in rows)
        for gpu, rows in results_by_gpu.items()
    )
    summary = {
        "status": "success" if successful else "failed",
        "protocol": protocol,
        "results_by_gpu": results_by_gpu,
        "paired_deltas_fisher_minus_observe": {
            gpu: paired.paired_delta(rows[0], rows[1])
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
