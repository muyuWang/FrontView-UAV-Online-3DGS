#!/usr/bin/env python3
"""Run matched PanoAir TGBR optimization-budget controls on GPUs 4-7."""

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
    "tgbr_budget_frontier_200_seed43_{}".format(STAMP)
)
RUNS = (
    (
        "static10",
        4,
        "configs/frontview_uav/panoair_tgbr_compute_static_sh3_200.yaml",
        "PanoAir-tgbr-compute-static-sh3-200",
    ),
    (
        "static8",
        5,
        "configs/frontview_uav/panoair_tgbr_budget_static8_200.yaml",
        "PanoAir-tgbr-budget-static8-200",
    ),
    (
        "evidence8",
        6,
        "configs/frontview_uav/panoair_tgbr_budget_evidence8_200.yaml",
        "PanoAir-tgbr-budget-evidence8-200",
    ),
    (
        "shuffled8",
        7,
        "configs/frontview_uav/panoair_tgbr_budget_shuffled8_200.yaml",
        "PanoAir-tgbr-budget-shuffled8-200",
    ),
)


def environment(gpu):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_HOME": str(CUDA_HOME),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
    root = ROOT / "Logs_frontview_uav" / dataset_name
    matches = sorted(root.glob("*_{}".format(experiment_name)))
    if not matches:
        raise RuntimeError("No output found for {}".format(experiment_name))
    return matches[-1]


def run_one(name, gpu, config, dataset_name):
    experiment = "{}_{}_seed43_gpu{}".format(name, OUTPUT.name, gpu)
    slam_log = OUTPUT / "{}.slam.log".format(name)
    command = [
        str(PYTHON),
        "slam_new.py",
        "--config",
        str(ROOT / config),
        "--exp_name",
        experiment,
        "--seed",
        "43",
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
        render_code = subprocess.run(
            render_command,
            cwd=ROOT,
            env=environment(gpu),
            stdout=console,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if render_code != 0:
        raise RuntimeError("{} render failed; see {}".format(name, render_log))

    reconstruction = json.loads((run_dir / "results.json").read_text())
    metrics = json.loads((render_dir / "render_metrics.json").read_text())
    return {
        "name": name,
        "config": config,
        "run_dir": str(run_dir),
        "render_dir": str(render_dir),
        "wall_time_s": time.monotonic() - started,
        "online_recon_time": reconstruction.get("online_recon_time"),
        "num_gaussians": reconstruction.get("num_gaussians"),
        "streaming_appearance_lod": reconstruction.get(
            "streaming_appearance_lod"
        ),
        "mean": metrics["mean"],
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(RUNS)) as pool:
        futures = [pool.submit(run_one, *spec) for spec in RUNS]
        results = [future.result() for future in futures]
    manifest = {"protocol": "200-frame matched seed43, no LPIPS", "results": results}
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for result in results:
        print(
            "{name}: {online_recon_time:.3f}s PSNR={psnr:.6f} SSIM={ssim:.6f}".format(
                psnr=result["mean"]["psnr"],
                ssim=result["mean"]["ssim"],
                **result,
            )
        )
    print(OUTPUT)


if __name__ == "__main__":
    main()
