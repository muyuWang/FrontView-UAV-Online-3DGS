#!/usr/bin/env python3
"""Run matched PanoAir TGBR frequency-schedule experiments."""

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
        "routed_cap60_200",
        4,
        "configs/frontview_uav/panoair_tgbr_compute_routed_warmup10_fp16_cap60_200.yaml",
        "PanoAir-tgbr-compute-routed-warmup10-fp16-cap60-200",
    ),
    (
        "static_random",
        4,
        "configs/frontview_uav/panoair_tgbr_compute_static_sh3_200.yaml",
        "PanoAir-tgbr-compute-static-sh3-200",
    ),
    (
        "evidence_schedule",
        5,
        "configs/frontview_uav/panoair_tgbr_frequency_schedule_evidence_200.yaml",
        "PanoAir-tgbr-frequency-schedule-evidence-200",
    ),
    (
        "shuffled_schedule",
        6,
        "configs/frontview_uav/panoair_tgbr_frequency_schedule_shuffled_200.yaml",
        "PanoAir-tgbr-frequency-schedule-shuffled-200",
    ),
    (
        "static_schedule",
        7,
        "configs/frontview_uav/panoair_tgbr_frequency_schedule_static_200.yaml",
        "PanoAir-tgbr-frequency-schedule-static-200",
    ),
    (
        "evidence_schedule_d4",
        5,
        "configs/frontview_uav/panoair_tgbr_frequency_schedule_d4_200.yaml",
        "PanoAir-tgbr-frequency-schedule-d4-200",
    ),
    (
        "evidence_schedule_d5",
        6,
        "configs/frontview_uav/panoair_tgbr_frequency_schedule_d5_200.yaml",
        "PanoAir-tgbr-frequency-schedule-d5-200",
    ),
    (
        "evidence_schedule_d6",
        7,
        "configs/frontview_uav/panoair_tgbr_frequency_schedule_d6_200.yaml",
        "PanoAir-tgbr-frequency-schedule-d6-200",
    ),
    (
        "directional_budget_a",
        5,
        "configs/frontview_uav/panoair_tgbr_directional_budget_a_200.yaml",
        "PanoAir-tgbr-directional-budget-a-200",
    ),
    (
        "directional_budget_b",
        6,
        "configs/frontview_uav/panoair_tgbr_directional_budget_b_200.yaml",
        "PanoAir-tgbr-directional-budget-b-200",
    ),
    (
        "directional_budget_c",
        7,
        "configs/frontview_uav/panoair_tgbr_directional_budget_c_200.yaml",
        "PanoAir-tgbr-directional-budget-c-200",
    ),
    (
        "directional_views_a",
        5,
        "configs/frontview_uav/panoair_tgbr_directional_views_a_200.yaml",
        "PanoAir-tgbr-directional-views-a-200",
    ),
    (
        "directional_views_b",
        6,
        "configs/frontview_uav/panoair_tgbr_directional_views_b_200.yaml",
        "PanoAir-tgbr-directional-views-b-200",
    ),
    (
        "directional_views_c",
        7,
        "configs/frontview_uav/panoair_tgbr_directional_views_c_200.yaml",
        "PanoAir-tgbr-directional-views-c-200",
    ),
    (
        "directional_views_d",
        5,
        "configs/frontview_uav/panoair_tgbr_directional_views_d_200.yaml",
        "PanoAir-tgbr-directional-views-d-200",
    ),
    (
        "directional_views_e",
        6,
        "configs/frontview_uav/panoair_tgbr_directional_views_e_200.yaml",
        "PanoAir-tgbr-directional-views-e-200",
    ),
    (
        "directional_views_f",
        7,
        "configs/frontview_uav/panoair_tgbr_directional_views_f_200.yaml",
        "PanoAir-tgbr-directional-views-f-200",
    ),
    (
        "routed_cap60_full",
        4,
        "configs/frontview_uav/panoair_tgbr_compute_routed_cap60_full.yaml",
        "PanoAir-tgbr-compute-routed-cap60-full",
    ),
    (
        "directional_views_a_full",
        4,
        "configs/frontview_uav/panoair_tgbr_directional_views_a_full.yaml",
        "PanoAir-tgbr-directional-views-a-full",
    ),
    (
        "directional_views_percentile_200",
        5,
        "configs/frontview_uav/panoair_tgbr_directional_views_percentile_200.yaml",
        "PanoAir-tgbr-directional-views-percentile-200",
    ),
    (
        "directional_views_percentile_full",
        4,
        "configs/frontview_uav/panoair_tgbr_directional_views_percentile_full.yaml",
        "PanoAir-tgbr-directional-views-percentile-full",
    ),
    (
        "directional_views_shuffled_full",
        4,
        "configs/frontview_uav/panoair_tgbr_directional_views_shuffled_full.yaml",
        "PanoAir-tgbr-directional-views-shuffled-full",
    ),
    (
        "directional_views_percentile_aggressive_full",
        4,
        "configs/frontview_uav/panoair_tgbr_directional_views_percentile_aggressive_full.yaml",
        "PanoAir-tgbr-directional-views-percentile-aggressive-full",
    ),
    (
        "directional_views_percentile_balanced_200",
        4,
        "configs/frontview_uav/panoair_tgbr_directional_views_percentile_balanced_200.yaml",
        "PanoAir-tgbr-directional-views-percentile-balanced-200",
    ),
    (
        "directional_views_percentile_balanced_full",
        4,
        "configs/frontview_uav/panoair_tgbr_directional_views_percentile_balanced_full.yaml",
        "PanoAir-tgbr-directional-views-percentile-balanced-full",
    ),
    (
        "directional_step_percentile_200",
        4,
        "configs/frontview_uav/panoair_tgbr_directional_step_percentile_200.yaml",
        "PanoAir-tgbr-directional-step-percentile-200",
    ),
    (
        "directional_step_percentile_full",
        4,
        "configs/frontview_uav/panoair_tgbr_directional_step_percentile_full.yaml",
        "PanoAir-tgbr-directional-step-percentile-full",
    ),
    (
        "directional_step_shuffled_full",
        4,
        "configs/frontview_uav/panoair_tgbr_directional_step_shuffled_full.yaml",
        "PanoAir-tgbr-directional-step-shuffled-full",
    ),
    (
        "exact_replay_mb3_200",
        4,
        "configs/frontview_uav/panoair_tgbr_exact_replay_mb3_200.yaml",
        "PanoAir-tgbr-exact-replay-mb3-200",
    ),
    (
        "exact_replay_mb3_full",
        4,
        "configs/frontview_uav/panoair_tgbr_exact_replay_mb3_full.yaml",
        "PanoAir-tgbr-exact-replay-mb3-full",
    ),
    (
        "exact_replay_adaptive_full",
        4,
        "configs/frontview_uav/panoair_tgbr_exact_replay_adaptive_full.yaml",
        "PanoAir-tgbr-exact-replay-adaptive-full",
    ),
    (
        "bounded_residency_200",
        4,
        "configs/frontview_uav/panoair_tgbr_bounded_residency_200.yaml",
        "PanoAir-tgbr-bounded-residency-200",
    ),
    (
        "bounded_residency_full",
        4,
        "configs/frontview_uav/panoair_tgbr_bounded_residency_full.yaml",
        "PanoAir-tgbr-bounded-residency-full",
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
        raise RuntimeError("Missing run for {}".format(experiment_name))
    return runs[-1]


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def run_one(spec, output_root, seed, tag):
    name, gpu, config, dataset_name = spec
    experiment = "{}_{}_seed{}_gpu{}".format(name, tag, seed, gpu)
    slam_log = output_root / "{}.slam.log".format(name)
    started = time.perf_counter()
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
    with slam_log.open("w") as output:
        code = subprocess.run(
            command,
            cwd=ROOT,
            env=environment(gpu),
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if code:
        raise RuntimeError("{} failed; see {}".format(name, slam_log))
    run_dir = newest_run(dataset_name, experiment)

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
    with render_log.open("w") as output:
        code = subprocess.run(
            render_command,
            cwd=ROOT,
            env=environment(gpu),
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if code:
        raise RuntimeError("{} render failed; see {}".format(name, render_log))
    metric_paths = sorted(render_dir.rglob("render_metrics.json"))
    if not metric_paths:
        raise RuntimeError("Missing metrics for {}".format(name))
    reconstruction = load_json(run_dir / "results.json")
    return {
        "name": name,
        "gpu": gpu,
        "config": str(ROOT / config),
        "run_dir": str(run_dir),
        "render_dir": str(render_dir),
        "online_recon_time": reconstruction["online_recon_time"],
        "num_processed_frames": reconstruction["num_processed_frames"],
        "num_gaussians": reconstruction["num_gaussians"],
        "streaming_appearance_lod": reconstruction.get(
            "streaming_appearance_lod"
        ),
        "tgbr_frequency_schedule": reconstruction.get(
            "tgbr_frequency_schedule"
        ),
        "tgbr_directional_step_budget": reconstruction.get(
            "tgbr_directional_step_budget"
        ),
        "tgbr_directional_view_budget": reconstruction.get(
            "tgbr_directional_view_budget"
        ),
        "tgbr_exact_replay": reconstruction.get("tgbr_exact_replay"),
        "tgbr_replay_residency": reconstruction.get("tgbr_replay_residency"),
        "render_metrics": load_json(metric_paths[-1]),
        "wall_time_s": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--tag", default="tgbr_frequency_schedule_200_20260807"
    )
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument(
        "--names",
        nargs="+",
        choices=tuple(spec[0] for spec in RUNS),
        default=None,
    )
    args = parser.parse_args()
    run_specs = tuple(
        spec for spec in RUNS if args.names is None or spec[0] in args.names
    )

    output_root = ROOT / "Logs_frontview_uav" / "benchmarks" / args.tag
    output_root.mkdir(parents=True, exist_ok=False)
    results = []
    if args.sequential:
        for spec in run_specs:
            result = run_one((spec[0], 4, spec[2], spec[3]), output_root, args.seed, args.tag)
            results.append(result)
            print("completed {}".format(result["name"]), flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(run_one, spec, output_root, args.seed, args.tag)
                for spec in run_specs
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print("completed {}".format(result["name"]), flush=True)
    order = [spec[0] for spec in run_specs]
    results.sort(key=lambda result: order.index(result["name"]))
    manifest = {"tag": args.tag, "seed": args.seed, "results": results}
    with (output_root / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(output_root / "manifest.json")


if __name__ == "__main__":
    main()
