#!/usr/bin/env python3
"""Run a CPU-uncontended baseline/K384 PanoAir pair on GPU 4."""

from __future__ import annotations

import json
from pathlib import Path

from run_panoair_tgbr_spectral_residency_full import run_one


ROOT = Path(__file__).resolve().parents[1]
TAG = "tgbr_spectral_residency_k384_pair_20260810"
RUNS = (
    (
        "baseline",
        4,
        "configs/frontview_uav/panoair_tgbr_spectral_baseline_full.yaml",
        "PanoAir-tgbr-spectral-baseline-full",
    ),
    (
        "dynamic_k384",
        4,
        "configs/frontview_uav/panoair_tgbr_residency_k384_full.yaml",
        "PanoAir-tgbr-residency-k384-full",
    ),
)


def main():
    output_root = ROOT / "Logs_frontview_uav" / "benchmarks" / TAG
    output_root.mkdir(parents=True, exist_ok=False)
    results = []
    for spec in RUNS:
        result = run_one(spec, output_root)
        results.append(result)
        print("completed {}".format(result["name"]), flush=True)
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump({"tag": TAG, "seed": 43, "results": results}, handle, indent=2)
    print(manifest_path)


if __name__ == "__main__":
    main()
