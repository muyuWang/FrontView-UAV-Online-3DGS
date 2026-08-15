#!/usr/bin/env python3
"""Run the full PanoAir TGBR spectral-residency budget frontier."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

from run_panoair_tgbr_spectral_residency_full import run_one


ROOT = Path(__file__).resolve().parents[1]
TAG = "tgbr_spectral_residency_frontier_20260810"
RUNS = (
    (
        "dynamic_k48",
        4,
        "configs/frontview_uav/panoair_tgbr_residency_k48_full.yaml",
        "PanoAir-tgbr-residency-k48-full",
    ),
    (
        "dynamic_k96",
        5,
        "configs/frontview_uav/panoair_tgbr_residency_k96_full.yaml",
        "PanoAir-tgbr-residency-k96-full",
    ),
    (
        "dynamic_k192",
        6,
        "configs/frontview_uav/panoair_tgbr_residency_k192_full.yaml",
        "PanoAir-tgbr-residency-k192-full",
    ),
    (
        "dynamic_k384",
        7,
        "configs/frontview_uav/panoair_tgbr_residency_k384_full.yaml",
        "PanoAir-tgbr-residency-k384-full",
    ),
)


def main():
    output_root = ROOT / "Logs_frontview_uav" / "benchmarks" / TAG
    output_root.mkdir(parents=True, exist_ok=False)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_one, spec, output_root): spec[0] for spec in RUNS
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print("completed {}".format(result["name"]), flush=True)
    order = [spec[0] for spec in RUNS]
    results.sort(key=lambda row: order.index(row["name"]))
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump({"tag": TAG, "seed": 43, "results": results}, handle, indent=2)
    print(manifest_path)


if __name__ == "__main__":
    main()
