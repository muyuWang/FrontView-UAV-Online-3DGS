#!/usr/bin/env python3
"""Run matched PanoAir TGBR spectral-replay experiments on GPUs 4-7."""

from __future__ import annotations

import concurrent.futures
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/wmy/anaconda3/envs/worldvln/bin/python")
CUDA_HOME = Path("/home/wmy/.local/cuda-12.1")
RUNS_200 = (
    (
        "baseline",
        4,
        "configs/frontview_uav/panoair_tgbr_spectral_baseline_200.yaml",
        "PanoAir-tgbr-spectral-baseline-200",
    ),
    (
        "spectral_b54",
        5,
        "configs/frontview_uav/panoair_tgbr_spectral_b54_200.yaml",
        "PanoAir-tgbr-spectral-b54-200",
    ),
    (
        "spectral_b43",
        6,
        "configs/frontview_uav/panoair_tgbr_spectral_b43_200.yaml",
        "PanoAir-tgbr-spectral-b43-200",
    ),
    (
        "fixed3_control",
        7,
        "configs/frontview_uav/panoair_tgbr_spectral_fixed3_200.yaml",
        "PanoAir-tgbr-spectral-fixed3-200",
    ),
)
RUNS_FULL = (
    (
        "baseline",
        4,
        "configs/frontview_uav/panoair_tgbr_spectral_baseline_full.yaml",
        "PanoAir-tgbr-spectral-baseline-full",
    ),
    (
        "spectral_b54",
        5,
        "configs/frontview_uav/panoair_tgbr_spectral_b54_full.yaml",
        "PanoAir-tgbr-spectral-b54-full",
    ),
    (
        "spectral_b43",
        6,
        "configs/frontview_uav/panoair_tgbr_spectral_b43_full.yaml",
        "PanoAir-tgbr-spectral-b43-full",
    ),
    (
        "fixed3_control",
        7,
        "configs/frontview_uav/panoair_tgbr_spectral_fixed3_full.yaml",
        "PanoAir-tgbr-spectral-fixed3-full",
    ),
)


def environment(gpu):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_HOME": str(CUDA_HOME),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TORCH_CUDA_ARCH_LIST": "8.9",
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
    runs = sorted(
        (ROOT / "Logs_frontview_uav" / dataset_name).glob(
            "*_{}".format(experiment_name)
        )
    )
    if not runs:
        raise RuntimeError("Missing run directory for {}".format(experiment_name))
    return runs[-1]


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def run_one(spec, output_root, tag):
    name, gpu, config, dataset_name = spec
    experiment_name = "{}_{}_seed43_gpu{}".format(name, tag, gpu)
    slam_log = output_root / "{}.slam.log".format(name)
    command = [
        str(PYTHON),
        "slam_new.py",
        "--config",
        str(ROOT / config),
        "--exp_name",
        experiment_name,
        "--seed",
        "43",
    ]
    started = time.perf_counter()
    with slam_log.open("w") as console:
        code = subprocess.run(
            command,
            cwd=ROOT,
            env=environment(gpu),
            stdout=console,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if code:
        raise RuntimeError("{} failed; see {}".format(name, slam_log))

    run_dir = newest_run(dataset_name, experiment_name)
    render_dir = output_root / "{}_render".format(name)
    render_log = output_root / "{}.render.log".format(name)
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
    if code:
        raise RuntimeError("{} render failed; see {}".format(name, render_log))

    reconstruction = load_json(run_dir / "results.json")
    metric_paths = sorted(render_dir.rglob("render_metrics.json"))
    if not metric_paths:
        raise RuntimeError("Missing metrics for {}".format(name))
    metrics = load_json(metric_paths[-1])
    return {
        "name": name,
        "gpu": gpu,
        "config": str(ROOT / config),
        "run_dir": str(run_dir),
        "render_dir": str(render_dir),
        "slam_log": str(slam_log),
        "render_log": str(render_log),
        "wall_time_s": time.perf_counter() - started,
        "online_recon_time": reconstruction["online_recon_time"],
        "num_gaussians": reconstruction["num_gaussians"],
        "cuda_memory_profile": reconstruction.get("cuda_memory_profile"),
        "streaming_appearance_lod": reconstruction.get(
            "streaming_appearance_lod"
        ),
        "tgbr_exact_replay": reconstruction.get("tgbr_exact_replay"),
        "tgbr_spectral_replay": reconstruction.get("tgbr_spectral_replay"),
        "render_metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("200", "full"), default="200")
    args = parser.parse_args()
    tag = "tgbr_spectral_replay_{}_20260810".format(args.suite)
    runs = RUNS_200 if args.suite == "200" else RUNS_FULL
    output_root = ROOT / "Logs_frontview_uav" / "benchmarks" / tag
    output_root.mkdir(parents=True, exist_ok=False)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_one, spec, output_root, tag): spec[0] for spec in runs
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print("completed {}".format(result["name"]), flush=True)
    order = [spec[0] for spec in runs]
    results.sort(key=lambda row: order.index(row["name"]))
    manifest = {"tag": tag, "seed": 43, "results": results}
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(manifest_path)


if __name__ == "__main__":
    main()
