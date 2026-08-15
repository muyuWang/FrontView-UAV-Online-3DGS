#!/usr/bin/env python3
"""Run two-seed full-PanoAir TGBR tail-certificate validation on GPUs 4-7."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import time

import run_panoair_tgbr_tail_certificate_screen as screen


ROOT = Path(__file__).resolve().parents[1]
STAMP = time.strftime("%Y%m%d_%H%M%S")
OUTPUT = ROOT / "Logs_frontview_uav" / "benchmarks" / (
    "tgbr_tail_certificate_full_{}".format(STAMP)
)
RUNS = (
    (
        "baseline_seed43",
        4,
        "configs/frontview_uav/panoair_tgbr_compute_dense_full.yaml",
        "PanoAir-tgbr-compute-dense-full",
        43,
    ),
    (
        "candidate_seed43",
        5,
        "configs/frontview_uav/panoair_tgbr_tailcert5_gain150_tail02_full.yaml",
        "PanoAir-tgbr-tailcert5-gain150-tail02-full",
        43,
    ),
    (
        "candidate_seed44",
        6,
        "configs/frontview_uav/panoair_tgbr_tailcert5_gain150_tail02_full.yaml",
        "PanoAir-tgbr-tailcert5-gain150-tail02-full",
        44,
    ),
    (
        "baseline_seed44",
        7,
        "configs/frontview_uav/panoair_tgbr_compute_dense_full.yaml",
        "PanoAir-tgbr-compute-dense-full",
        44,
    ),
)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    screen.OUTPUT = OUTPUT
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(RUNS)) as pool:
        futures = [pool.submit(screen.run_one, *spec) for spec in RUNS]
        results = [future.result() for future in futures]
    manifest = {
        "protocol": (
            "full PanoAir, two matched seeds, method/GPU placement crossed, "
            "PSNR/SSIM only"
        ),
        "results": results,
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for result in results:
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
