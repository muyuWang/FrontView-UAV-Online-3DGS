#!/usr/bin/env python3
"""Run same-GPU order-balanced Stage 8 replications on Mountains."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

import run_mountains_crr_ablation as runner


ROOT = runner.ROOT
DEFAULT_OUTPUT = ROOT / "Logs_mountains_crr_stage8_replay_8_12"
COMMON = {
    "routing_mode": "causal_observability",
    "map_redundancy_gate": False,
    "projective_nms_mode": "budget_cells",
    "projective_covariance_mode": "isotropic",
}
METHODS = {
    "isotropic": {
        "far_field": {**COMMON, "fallback_support_mode": "budget_isotropic"},
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "certificate": {
        "far_field": {
            **COMMON,
            "fallback_support_mode": "budget_certificate_structure",
        },
        "sampling": {"selection_mode": "depth_stratified"},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=runner.DEFAULT_CONFIG)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=runner.DEFAULT_PYTHON)
    parser.add_argument("--gpus", nargs=2, default=("4", "5"))
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def paired_delta(certificate, isotropic):
    return {
        "psnr": certificate["metrics"]["psnr"] - isotropic["metrics"]["psnr"],
        "ssim": certificate["metrics"]["ssim"] - isotropic["metrics"]["ssim"],
        "num_gaussians": certificate["num_gaussians"] - isotropic["num_gaussians"],
        "online_recon_time": (
            certificate["online_recon_time"] - isotropic["online_recon_time"]
        ),
    }


def main():
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    args.scale_cover_backend = "config"
    if len(set(args.gpus)) != 2:
        raise ValueError("Two distinct GPUs are required")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if not args.config.is_file() or not args.python.is_file():
        raise FileNotFoundError(f"Missing config or Python: {args.config}, {args.python}")

    baseline = runner.load_config(str(args.config))
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = args.save_dir / f"batch_{tag}"
    config_root = batch_root / "runtime_configs"
    config_root.mkdir(parents=True, exist_ok=False)
    orders = {
        str(args.gpus[0]): ("isotropic", "certificate"),
        str(args.gpus[1]): ("certificate", "isotropic"),
    }
    jobs = {}
    for gpu, methods in orders.items():
        jobs[gpu] = []
        for order_index, method in enumerate(methods):
            variant = f"gpu{gpu}_{order_index + 1}_{method}"
            variant_root = batch_root / variant
            config = runner.build_config(
                baseline, variant, METHODS[method], variant_root, 0
            )
            path = config_root / f"{variant}.yaml"
            with path.open("x", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            jobs[gpu].append((variant, method, path, config["Dataset"]["name"]))

    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "protocol": "two same-GPU serial pairs with order reversal",
        "seed": args.seed,
        "cpu_threads_per_process": args.cpu_threads,
        "orders": orders,
        "jobs": {
            gpu: [
                {"variant": variant, "method": method, "config": str(path)}
                for variant, method, path, _ in rows
            ]
            for gpu, rows in jobs.items()
        },
    }
    runner.write_json(batch_root / "manifest.json", manifest)
    print(f"Batch: {batch_root}", flush=True)
    if args.dry_run:
        return 0

    def run_sequence(gpu, rows):
        output = []
        for variant, method, path, dataset_name in rows:
            result = runner.run_variant(
                args, batch_root, variant, gpu, path, dataset_name, tag
            )
            result["method"] = method
            output.append(result)
            print(f"{variant}: {result['status']}", flush=True)
            if result["status"] != "success":
                break
        return output

    results = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_sequence, gpu, rows): gpu
            for gpu, rows in jobs.items()
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    success = all(
        len(rows) == 2 and all(row["status"] == "success" for row in rows)
        for rows in results.values()
    )
    summary = {
        "status": "success" if success else "failed",
        "protocol": manifest["protocol"],
        "results": results,
    }
    if success:
        pairs = {}
        method_rows = {"isotropic": [], "certificate": []}
        for gpu, rows in results.items():
            by_method = {row["method"]: row for row in rows}
            pairs[gpu] = paired_delta(
                by_method["certificate"], by_method["isotropic"]
            )
            for method, row in by_method.items():
                method_rows[method].append(row)
        summary["paired_certificate_minus_isotropic"] = pairs
        summary["method_means"] = {
            method: {
                "psnr": sum(row["metrics"]["psnr"] for row in rows) / len(rows),
                "ssim": sum(row["metrics"]["ssim"] for row in rows) / len(rows),
                "num_gaussians": sum(row["num_gaussians"] for row in rows)
                / len(rows),
                "online_recon_time": sum(
                    row["online_recon_time"] for row in rows
                )
                / len(rows),
            }
            for method, rows in method_rows.items()
        }
        certificate = summary["method_means"]["certificate"]
        isotropic = summary["method_means"]["isotropic"]
        summary["mean_certificate_minus_isotropic"] = {
            "psnr": certificate["psnr"] - isotropic["psnr"],
            "ssim": certificate["ssim"] - isotropic["ssim"],
            "num_gaussians": (
                certificate["num_gaussians"] - isotropic["num_gaussians"]
            ),
            "online_recon_time": (
                certificate["online_recon_time"] - isotropic["online_recon_time"]
            ),
        }
    runner.write_json(batch_root / "summary.json", summary)
    manifest["status"] = summary["status"]
    manifest["summary"] = str(batch_root / "summary.json")
    runner.write_json(batch_root / "manifest.json", manifest)
    print(f"Summary: {batch_root / 'summary.json'}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
