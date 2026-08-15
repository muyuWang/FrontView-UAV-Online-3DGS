#!/usr/bin/env python3
"""Aggregate the three CPU-uncontended PanoAir K384 pairs."""

from __future__ import annotations

import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
FIRST = (
    ROOT
    / "Logs_frontview_uav/benchmarks/"
    "tgbr_spectral_residency_k384_pair_20260810/manifest.json"
)
REPEATS = (
    ROOT
    / "Logs_frontview_uav/benchmarks/"
    "tgbr_spectral_residency_k384_repeats_20260810/manifest.json"
)
OUTPUT = REPEATS.parent / "three_pair_summary.json"


def load(path):
    with path.open() as handle:
        return json.load(handle)


def method(results, name):
    return next(row for row in results if row["name"] == name)


def paired_record(name, results):
    baseline = method(results, "baseline")
    candidate = method(results, "dynamic_k384")
    baseline_peak = baseline["cuda_memory_profile"]["overall_peak_allocated_bytes"]
    candidate_peak = candidate["cuda_memory_profile"][
        "overall_peak_allocated_bytes"
    ]
    baseline_reserved = baseline["cuda_memory_profile"][
        "overall_peak_reserved_bytes"
    ]
    candidate_reserved = candidate["cuda_memory_profile"][
        "overall_peak_reserved_bytes"
    ]
    return {
        "name": name,
        "baseline_order": baseline.get("pair_order", 0),
        "candidate_order": candidate.get("pair_order", 1),
        "baseline": {
            "online_recon_time_s": baseline["online_recon_time"],
            "peak_allocated_bytes": baseline_peak,
            "peak_reserved_bytes": baseline_reserved,
            "psnr": baseline["render_metrics"]["mean"]["psnr"],
            "ssim": baseline["render_metrics"]["mean"]["ssim"],
            "num_gaussians": baseline["num_gaussians"],
        },
        "candidate": {
            "online_recon_time_s": candidate["online_recon_time"],
            "peak_allocated_bytes": candidate_peak,
            "peak_reserved_bytes": candidate_reserved,
            "psnr": candidate["render_metrics"]["mean"]["psnr"],
            "ssim": candidate["render_metrics"]["mean"]["ssim"],
            "num_gaussians": candidate["num_gaussians"],
            "spectral_residency": candidate["tgbr_spectral_residency"],
        },
        "delta": {
            "allocated_reduction_fraction": 1.0
            - candidate_peak / baseline_peak,
            "reserved_reduction_fraction": 1.0
            - candidate_reserved / baseline_reserved,
            "online_time_fraction": candidate["online_recon_time"]
            / baseline["online_recon_time"]
            - 1.0,
            "psnr": candidate["render_metrics"]["mean"]["psnr"]
            - baseline["render_metrics"]["mean"]["psnr"],
            "ssim": candidate["render_metrics"]["mean"]["ssim"]
            - baseline["render_metrics"]["mean"]["ssim"],
        },
    }


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    first = load(FIRST)
    repeats = load(REPEATS)
    pairs = [paired_record("repeat1_baseline_first", first["results"])]
    pairs.extend(
        paired_record(pair["name"], pair["results"])
        for pair in repeats["pairs"]
    )
    keys = (
        "allocated_reduction_fraction",
        "reserved_reduction_fraction",
        "online_time_fraction",
        "psnr",
        "ssim",
    )
    aggregate = {
        key: summarize([pair["delta"][key] for pair in pairs]) for key in keys
    }
    result = {
        "protocol": {
            "scene": "PanoAir full 2230 frames",
            "seed": 43,
            "gpu": 4,
            "execution": "same-GPU serial pairs with no concurrent reconstruction",
            "lpips_evaluated": False,
            "pair_count": len(pairs),
            "source_manifests": [str(FIRST), str(REPEATS)],
        },
        "method": {
            "name": "TGBR spectral evidence working set K384",
            "active_basis_dimension": "D_t = 9 + 7 r_t",
            "resident_view_limit": "K_t = min(512, floor(5472 / D_t))",
            "budget_invariant": "D_t K_t <= 5472",
        },
        "pairs": pairs,
        "aggregate": aggregate,
        "requirements": {
            "all_allocated_reductions_exceed_10_percent": all(
                pair["delta"]["allocated_reduction_fraction"] > 0.10
                for pair in pairs
            ),
            "mean_allocated_reduction_exceeds_10_percent": aggregate[
                "allocated_reduction_fraction"
            ]["mean"]
            > 0.10,
            "mean_online_time_overhead_below_5_percent": aggregate[
                "online_time_fraction"
            ]["mean"]
            < 0.05,
        },
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with OUTPUT.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(OUTPUT)


if __name__ == "__main__":
    main()
