#!/usr/bin/env python3
"""Run two additional CPU-uncontended K384 pairs on GPU 4."""

from __future__ import annotations

import json
from pathlib import Path

from run_panoair_tgbr_spectral_residency_full import run_one


ROOT = Path(__file__).resolve().parents[1]
TAG = "tgbr_spectral_residency_k384_repeats_20260810"
BASELINE = (
    "baseline",
    4,
    "configs/frontview_uav/panoair_tgbr_spectral_baseline_full.yaml",
    "PanoAir-tgbr-spectral-baseline-full",
)
K384 = (
    "dynamic_k384",
    4,
    "configs/frontview_uav/panoair_tgbr_residency_k384_full.yaml",
    "PanoAir-tgbr-residency-k384-full",
)
PAIRS = (
    ("repeat2_candidate_first", (K384, BASELINE)),
    ("repeat3_baseline_first", (BASELINE, K384)),
)


def main():
    output_root = ROOT / "Logs_frontview_uav" / "benchmarks" / TAG
    output_root.mkdir(parents=True, exist_ok=False)
    pairs = []
    for pair_name, specs in PAIRS:
        pair_root = output_root / pair_name
        pair_root.mkdir(parents=True, exist_ok=False)
        results = []
        for order, spec in enumerate(specs):
            result = run_one(spec, pair_root)
            result["pair_order"] = order
            results.append(result)
            print("completed {} {}".format(pair_name, result["name"]), flush=True)
        pairs.append({"name": pair_name, "gpu": 4, "results": results})
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump({"tag": TAG, "seed": 43, "pairs": pairs}, handle, indent=2)
    print(manifest_path)


if __name__ == "__main__":
    main()
