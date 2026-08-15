#!/usr/bin/env python3
"""Run reversed same-GPU Mountains observability-typed ownership pairs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

import run_mountains_causal_depth_posterior_paired as paired
import run_mountains_crr_ablation as runner


BASE_CONFIG = (
    runner.ROOT
    / "Logs_mountains_adaptive_goal_8_12_8_13/final/stage35_full_765"
    / "batch_20260813_095909/runtime_configs/A_visible_residual_detail_real.yaml"
)
OUTPUT_ROOT = (
    runner.ROOT
    / "Logs_mountains_far_depth_goal_8_13/observability_typed_ownership_paired_620"
)
FIXED_FAR_MASKS = paired.FIXED_FAR_MASKS

DUAL = {
    "enabled": True,
    "depthcov_confidence_mode": "posterior",
    "directional_use_metric_depth": True,
    "geometry_use_metric_depth": True,
    "export_metric_depth": True,
    "minimum_metric_opacity": 1.0e-6,
}
METHODS = {
    "A_stage35": {},
    "B_typed_proxy": {
        "causal_dual_responsibility": DUAL,
        "directional_layer": {
            "geometry_gate_mode": "metric_transmittance",
            "blend_weight": 1.0,
        },
    },
    "C_typed_abstain": {
        "coverage_recovery": {"depth_fallback_enabled": False},
        "causal_dual_responsibility": DUAL,
        "directional_layer": {
            "geometry_gate_mode": "metric_transmittance",
            "blend_weight": 1.0,
        },
    },
}
SCHEDULE = {
    "6": ("A_stage35", "B_typed_proxy", "C_typed_abstain"),
    "7": ("C_typed_abstain", "B_typed_proxy", "A_stage35"),
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


def metric_delta(method, baseline):
    return {
        key: float(method["metrics"][key]) - float(baseline["metrics"][key])
        for key in method["metrics"].keys() & baseline["metrics"].keys()
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
    args.gpus = tuple(SCHEDULE)
    if args.frames != 620:
        raise ValueError("The fixed-far protocol requires exactly 620 frames")
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
                args.frames,
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
        "reversed same-GPU serial 620-frame pairs; equal PBSD/TSC/FPR and optimization "
        "budgets; fixed 545-619 far masks; PSNR/SSIM only"
    )
    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "protocol": protocol,
        "base_commit": "23b17318235173e662f1c5b0c8245d2aede109d1",
        "base_config": str(args.config),
        "seed": args.seed,
        "frames": args.frames,
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
    deltas = {}
    if successful:
        for gpu, rows in results.items():
            by_method = {
                SCHEDULE[gpu][row["sequence_index"]]: row for row in rows
            }
            deltas[gpu] = {
                method: metric_delta(by_method[method], by_method["A_stage35"])
                for method in ("B_typed_proxy", "C_typed_abstain")
            }
    summary = {
        "status": "success" if successful else "failed",
        "protocol": protocol,
        "results_by_gpu": results,
        "deltas_vs_stage35": deltas,
    }
    runner.write_json(batch_root / "summary.json", summary)
    manifest["status"] = summary["status"]
    manifest["summary"] = str(batch_root / "summary.json")
    runner.write_json(batch_root / "manifest.json", manifest)
    print(f"Summary: {batch_root / 'summary.json'}", flush=True)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
