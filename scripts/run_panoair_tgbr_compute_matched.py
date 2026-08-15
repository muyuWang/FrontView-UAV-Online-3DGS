#!/usr/bin/env python3
"""Run the four-way PanoAir TGBR compute-routing comparison on GPUs 4-7."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/wmy/anaconda3/envs/worldvln/bin/python")
CUDA_HOME = Path("/home/wmy/.local/cuda-12.1")
RUNS = (
    (
        "static_sh3",
        4,
        "configs/frontview_uav/panoair_tgbr_compute_static_sh3_200.yaml",
        "PanoAir-tgbr-compute-static-sh3-200",
    ),
    (
        "dense_tgbr",
        5,
        "configs/frontview_uav/panoair_tgbr_compute_dense_200.yaml",
        "PanoAir-tgbr-compute-dense-200",
    ),
    (
        "routed_fp32",
        6,
        "configs/frontview_uav/panoair_tgbr_compute_routed_fp32_200.yaml",
        "PanoAir-tgbr-compute-routed-fp32-200",
    ),
    (
        "routed_fp16",
        7,
        "configs/frontview_uav/panoair_tgbr_compute_routed_fp16_200.yaml",
        "PanoAir-tgbr-compute-routed-fp16-200",
    ),
)
WARMUP_RUNS = (
    (
        "dense_tgbr_repeat",
        4,
        "configs/frontview_uav/panoair_tgbr_compute_dense_200.yaml",
        "PanoAir-tgbr-compute-dense-200",
    ),
    (
        "routed_warmup10_fp32",
        5,
        "configs/frontview_uav/panoair_tgbr_compute_routed_warmup10_fp32_200.yaml",
        "PanoAir-tgbr-compute-routed-warmup10-fp32-200",
    ),
    (
        "routed_warmup20_fp32",
        6,
        "configs/frontview_uav/panoair_tgbr_compute_routed_warmup20_fp32_200.yaml",
        "PanoAir-tgbr-compute-routed-warmup20-fp32-200",
    ),
    (
        "routed_warmup10_fp16",
        7,
        "configs/frontview_uav/panoair_tgbr_compute_routed_warmup10_fp16_200.yaml",
        "PanoAir-tgbr-compute-routed-warmup10-fp16-200",
    ),
)
CAPACITY_RUNS = (
    (
        "dense_tgbr_repeat2",
        4,
        "configs/frontview_uav/panoair_tgbr_compute_dense_200.yaml",
        "PanoAir-tgbr-compute-dense-200",
    ),
    (
        "routed_cap75",
        5,
        "configs/frontview_uav/panoair_tgbr_compute_routed_warmup10_fp16_200.yaml",
        "PanoAir-tgbr-compute-routed-warmup10-fp16-200",
    ),
    (
        "routed_cap60",
        6,
        "configs/frontview_uav/panoair_tgbr_compute_routed_warmup10_fp16_cap60_200.yaml",
        "PanoAir-tgbr-compute-routed-warmup10-fp16-cap60-200",
    ),
    (
        "routed_cap50",
        7,
        "configs/frontview_uav/panoair_tgbr_compute_routed_warmup10_fp16_cap50_200.yaml",
        "PanoAir-tgbr-compute-routed-warmup10-fp16-cap50-200",
    ),
)
FULL_RUNS = (
    (
        "static_sh3_full",
        4,
        "configs/frontview_uav/panoair_tgbr_compute_static_sh3_full.yaml",
        "PanoAir-tgbr-compute-static-sh3-full",
    ),
    (
        "dense_tgbr75_full",
        5,
        "configs/frontview_uav/panoair_tgbr_compute_dense_full.yaml",
        "PanoAir-tgbr-compute-dense-full",
    ),
    (
        "routed_tgbr60_full",
        6,
        "configs/frontview_uav/panoair_tgbr_compute_routed_cap60_full.yaml",
        "PanoAir-tgbr-compute-routed-cap60-full",
    ),
    (
        "routed_shuffled60_full",
        7,
        "configs/frontview_uav/panoair_tgbr_compute_routed_cap60_shuffled_full.yaml",
        "PanoAir-tgbr-compute-routed-cap60-shuffled-full",
    ),
)
FRONTIER_RUNS = (
    (
        "routed_tgbr60_repeat_full",
        4,
        "configs/frontview_uav/panoair_tgbr_compute_routed_cap60_full.yaml",
        "PanoAir-tgbr-compute-routed-cap60-full",
    ),
    (
        "routed_tgbr65_full",
        5,
        "configs/frontview_uav/panoair_tgbr_compute_routed_cap65_full.yaml",
        "PanoAir-tgbr-compute-routed-cap65-full",
    ),
    (
        "routed_tgbr70_full",
        6,
        "configs/frontview_uav/panoair_tgbr_compute_routed_cap70_full.yaml",
        "PanoAir-tgbr-compute-routed-cap70-full",
    ),
    (
        "routed_tgbr75_full",
        7,
        "configs/frontview_uav/panoair_tgbr_compute_routed_cap75_full.yaml",
        "PanoAir-tgbr-compute-routed-cap75-full",
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
    candidates = sorted(
        (ROOT / "Logs_frontview_uav" / dataset_name).glob(
            "*_{}".format(experiment_name)
        )
    )
    if not candidates:
        raise RuntimeError("No run directory found for {}".format(experiment_name))
    return candidates[-1]


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def run_one(spec, output_root, seed, tag):
    name, gpu, config, dataset_name = spec
    experiment_name = "{}_{}_seed{}_gpu{}".format(name, tag, seed, gpu)
    console_path = output_root / "{}.slam.log".format(name)
    started = time.perf_counter()
    command = [
        str(PYTHON),
        "slam_new.py",
        "--config",
        str(ROOT / config),
        "--exp_name",
        experiment_name,
        "--seed",
        str(seed),
    ]
    with console_path.open("w") as console:
        code = subprocess.run(
            command,
            cwd=ROOT,
            env=environment(gpu),
            stdout=console,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if code != 0:
        raise RuntimeError("{} reconstruction failed; see {}".format(name, console_path))
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
        render_code = subprocess.run(
            render_command,
            cwd=ROOT,
            env=environment(gpu),
            stdout=console,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if render_code != 0:
        raise RuntimeError("{} rendering failed; see {}".format(name, render_log))

    metric_files = sorted(render_dir.rglob("render_metrics.json"))
    if not metric_files:
        raise RuntimeError("No render metrics found for {}".format(name))
    metrics = load_json(metric_files[-1])
    reconstruction = load_json(run_dir / "results.json")
    return {
        "name": name,
        "gpu": gpu,
        "config": str(ROOT / config),
        "command": command,
        "render_command": render_command,
        "run_dir": str(run_dir),
        "render_dir": str(render_dir),
        "console_log": str(console_path),
        "render_log": str(render_log),
        "wall_time_s": time.perf_counter() - started,
        "online_recon_time": reconstruction.get("online_recon_time"),
        "num_processed_frames": reconstruction.get("num_processed_frames"),
        "num_gaussians": reconstruction.get("num_gaussians"),
        "streaming_appearance_lod": reconstruction.get(
            "streaming_appearance_lod"
        ),
        "render_metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--tag", default="tgbr_compute_matched_200_20260806"
    )
    parser.add_argument(
        "--suite",
        choices=("initial", "warmup", "capacity", "full", "frontier"),
        default="initial",
    )
    args = parser.parse_args()
    run_specs = {
        "initial": RUNS,
        "warmup": WARMUP_RUNS,
        "capacity": CAPACITY_RUNS,
        "full": FULL_RUNS,
        "frontier": FRONTIER_RUNS,
    }[args.suite]

    output_root = ROOT / "Logs_frontview_uav" / "benchmarks" / args.tag
    output_root.mkdir(parents=True, exist_ok=False)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(run_specs)) as executor:
        futures = {
            executor.submit(run_one, spec, output_root, args.seed, args.tag): spec[0]
            for spec in run_specs
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                "completed {}: online={:.3f}s".format(
                    result["name"], result["online_recon_time"]
                ),
                flush=True,
            )

    results.sort(
        key=lambda row: [spec[0] for spec in run_specs].index(row["name"])
    )
    manifest = {"tag": args.tag, "seed": args.seed, "results": results}
    with (output_root / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(output_root / "manifest.json")


if __name__ == "__main__":
    main()
