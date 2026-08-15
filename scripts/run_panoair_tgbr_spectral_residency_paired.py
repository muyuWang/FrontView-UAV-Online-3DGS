#!/usr/bin/env python3
"""Run counterbalanced same-GPU PanoAir residency pairs on GPUs 4-7."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

from run_panoair_tgbr_spectral_residency_full import run_one


ROOT = Path(__file__).resolve().parents[1]
TAG = "tgbr_spectral_residency_paired_20260810"
BASELINE = (
    "baseline",
    None,
    "configs/frontview_uav/panoair_tgbr_spectral_baseline_full.yaml",
    "PanoAir-tgbr-spectral-baseline-full",
)
DYNAMIC = (
    "dynamic_25to16",
    None,
    "configs/frontview_uav/panoair_tgbr_residency_dynamic_full.yaml",
    "PanoAir-tgbr-residency-dynamic-full",
)
PAIRS = (
    (4, (BASELINE, DYNAMIC)),
    (5, (DYNAMIC, BASELINE)),
    (6, (BASELINE, DYNAMIC)),
    (7, (DYNAMIC, BASELINE)),
)


def run_pair(gpu, ordered_specs, output_root):
    pair_root = output_root / "gpu{}".format(gpu)
    pair_root.mkdir(parents=True, exist_ok=False)
    results = []
    for order, spec in enumerate(ordered_specs):
        name, _, config, dataset_name = spec
        result = run_one((name, gpu, config, dataset_name), pair_root)
        result["pair_order"] = order
        results.append(result)
        print("completed gpu{} {}".format(gpu, name), flush=True)
    return {"gpu": gpu, "results": results}


def main():
    output_root = ROOT / "Logs_frontview_uav" / "benchmarks" / TAG
    output_root.mkdir(parents=True, exist_ok=False)
    pairs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_pair, gpu, specs, output_root): gpu
            for gpu, specs in PAIRS
        }
        for future in concurrent.futures.as_completed(futures):
            pairs.append(future.result())
    pairs.sort(key=lambda pair: pair["gpu"])
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump({"tag": TAG, "seed": 43, "pairs": pairs}, handle, indent=2)
    print(manifest_path)


if __name__ == "__main__":
    main()
