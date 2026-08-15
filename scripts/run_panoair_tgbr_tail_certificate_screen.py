#!/usr/bin/env python3
"""Run matched 200-frame TGBR tail-certificate screening on GPUs 4-7."""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/wmy/anaconda3/envs/worldvln/bin/python")
CUDA_HOME = Path("/home/wmy/.local/cuda-12.1")
STAMP = time.strftime("%Y%m%d_%H%M%S")
OUTPUT = ROOT / "Logs_frontview_uav" / "benchmarks" / (
    "tgbr_tail_certificate_200_seed43_{}".format(STAMP)
)
RUNS = (
    (
        "baseline10",
        4,
        "configs/frontview_uav/panoair_tgbr_compute_dense_200.yaml",
        "PanoAir-tgbr-compute-dense-200",
    ),
    (
        "cert8_strict",
        5,
        "configs/frontview_uav/panoair_tgbr_tailcert8_strict_200.yaml",
        "PanoAir-tgbr-tailcert8-strict-200",
    ),
    (
        "cert8_balanced",
        6,
        "configs/frontview_uav/panoair_tgbr_tailcert8_balanced_200.yaml",
        "PanoAir-tgbr-tailcert8-balanced-200",
    ),
    (
        "cert7_balanced",
        7,
        "configs/frontview_uav/panoair_tgbr_tailcert7_strict_200.yaml",
        "PanoAir-tgbr-tailcert7-balanced-200",
    ),
)


def environment(gpu):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_HOME": str(CUDA_HOME),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_CUDA_ARCH_LIST": "8.9",
        }
    )
    env["PATH"] = str(CUDA_HOME / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = (
        str(CUDA_HOME / "lib64")
        + os.pathsep
        + env.get("LD_LIBRARY_PATH", "")
    )
    return env


def newest_run(dataset_name, experiment_name):
    matches = sorted(
        (ROOT / "Logs_frontview_uav" / dataset_name).glob(
            "*_{}".format(experiment_name)
        )
    )
    if not matches:
        raise RuntimeError("No output found for {}".format(experiment_name))
    return matches[-1]


def run_one(name, gpu, config, dataset_name, seed=43):
    experiment = "{}_{}_seed{}_gpu{}".format(name, OUTPUT.name, seed, gpu)
    slam_log = OUTPUT / "{}.slam.log".format(name)
    command = [
        str(PYTHON),
        "slam_new.py",
        "--config",
        str(ROOT / config),
        "--exp_name",
        experiment,
        "--seed",
        str(seed),
    ]
    started = time.monotonic()
    with slam_log.open("w") as console:
        code = subprocess.run(
            command,
            cwd=ROOT,
            env=environment(gpu),
            stdout=console,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if code != 0:
        raise RuntimeError("{} failed; see {}".format(name, slam_log))
    run_dir = newest_run(dataset_name, experiment)

    render_dir = OUTPUT / "{}_render".format(name)
    render_log = OUTPUT / "{}.render.log".format(name)
    render_command = [
        str(PYTHON),
        "render.py",
        "--run_dir",
        str(run_dir),
        "--output_dir",
        str(render_dir),
        "--device",
        "cuda:0",
        "--skip_lpips",
        "--skip_novel",
        "--skip_depth",
        "--skip_primitives",
    ]
    with render_log.open("w") as console:
        code = subprocess.run(
            render_command,
            cwd=ROOT,
            env=environment(gpu),
            stdout=console,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if code != 0:
        raise RuntimeError("{} render failed; see {}".format(name, render_log))

    reconstruction = json.loads((run_dir / "results.json").read_text())
    metrics = json.loads((render_dir / "render_metrics.json").read_text())
    return {
        "name": name,
        "gpu": gpu,
        "seed": seed,
        "config": config,
        "run_dir": str(run_dir),
        "render_dir": str(render_dir),
        "wall_time_s": time.monotonic() - started,
        "online_recon_time": reconstruction.get("online_recon_time"),
        "num_processed_frames": reconstruction.get("num_processed_frames"),
        "num_gaussians": reconstruction.get("num_gaussians"),
        "tgbr_optimization_budget": reconstruction.get(
            "tgbr_optimization_budget"
        ),
        "mean": metrics["mean"],
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(RUNS)) as pool:
        futures = [pool.submit(run_one, *spec) for spec in RUNS]
        results = [future.result() for future in futures]
    manifest = {
        "protocol": "matched PanoAir 200 frames, seed43, PSNR/SSIM only",
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
