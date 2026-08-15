#!/usr/bin/env python3
"""Run reversed same-GPU full Mountains typed-ownership pairs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

import run_mountains_causal_depth_posterior_paired as paired
import run_mountains_crr_ablation as runner
import run_mountains_observability_typed_ownership_paired as experiment


OUTPUT_ROOT = (
    runner.ROOT
    / "Logs_mountains_far_depth_goal_8_13/observability_typed_ownership_full765"
)
SCHEDULE = {
    "6": ("A_stage35", "C_typed_abstain"),
    "7": ("C_typed_abstain", "A_stage35"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=experiment.BASE_CONFIG)
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
    args.fixed_far_mask_dir = experiment.FIXED_FAR_MASKS.resolve()
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
                experiment.METHODS[method],
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

    protocol = (
        "reversed same-GPU serial full-765 pairs; equal PBSD/TSC/FPR and "
        "optimization budgets; fixed 545-619 far masks; PSNR/SSIM only"
    )
    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "protocol": protocol,
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
            executor.submit(
                experiment.run_gpu_sequence,
                args,
                batch_root,
                gpu,
                jobs,
                tag,
            ): gpu
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
    deltas = {}
    if successful:
        for gpu, rows in results.items():
            by_method = {
                SCHEDULE[gpu][row["sequence_index"]]: row for row in rows
            }
            deltas[gpu] = experiment.metric_delta(
                by_method["C_typed_abstain"], by_method["A_stage35"]
            )
    summary = {
        "status": "success" if successful else "failed",
        "protocol": protocol,
        "results_by_gpu": results,
        "typed_abstain_minus_stage35": deltas,
    }
    runner.write_json(batch_root / "summary.json", summary)
    manifest["status"] = summary["status"]
    manifest["summary"] = str(batch_root / "summary.json")
    runner.write_json(batch_root / "manifest.json", manifest)
    print(f"Summary: {batch_root / 'summary.json'}", flush=True)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
