#!/usr/bin/env python3
"""Run counterbalanced same-GPU TGBR tail-certificate pairs."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import time

import run_panoair_tgbr_tail_certificate_screen as screen


ROOT = Path(__file__).resolve().parents[1]
STAMP = time.strftime("%Y%m%d_%H%M%S")
OUTPUT = ROOT / "Logs_frontview_uav" / "benchmarks" / (
    "tgbr_tail_certificate_paired_200_{}".format(STAMP)
)
BASELINE = (
    "configs/frontview_uav/panoair_tgbr_compute_dense_200.yaml",
    "PanoAir-tgbr-compute-dense-200",
)
CANDIDATE = (
    "configs/frontview_uav/panoair_tgbr_tailcert5_gain150_tail02_200.yaml",
    "PanoAir-tgbr-tailcert5-gain150-tail02-200",
)
PAIRS = (
    (4, 43, ("baseline", "candidate")),
    (5, 44, ("candidate", "baseline")),
)


def run_pair(gpu, seed, order):
    results = []
    for method in order:
        config, dataset = BASELINE if method == "baseline" else CANDIDATE
        name = "{}_seed{}_gpu{}".format(method, seed, gpu)
        results.append(
            screen.run_one(name, gpu, config, dataset, seed=seed)
        )
    return results


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    screen.OUTPUT = OUTPUT
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PAIRS)) as pool:
        futures = [pool.submit(run_pair, *spec) for spec in PAIRS]
        results = [row for future in futures for row in future.result()]
    manifest = {
        "protocol": (
            "counterbalanced same-GPU PanoAir 200-frame pairs; "
            "PSNR/SSIM only"
        ),
        "pairs": [
            {"gpu": gpu, "seed": seed, "order": list(order)}
            for gpu, seed, order in PAIRS
        ],
        "results": results,
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for result in sorted(results, key=lambda row: (row["seed"], row["name"])):
        print(
            "{name}: online={online_recon_time:.3f}s PSNR={psnr:.6f} "
            "SSIM={ssim:.6f}".format(
                psnr=result["mean"]["psnr"],
                ssim=result["mean"]["ssim"],
                **result,
            )
        )
    print(manifest_path)


if __name__ == "__main__":
    main()
