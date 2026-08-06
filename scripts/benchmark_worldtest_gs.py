#!/usr/bin/env python3
"""Materialize and run reproducible Canonical WorldTest-GS experiment suites."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
COORDINATE_CONFIGS = {
    "hybrid": REPO / "configs/worldtest_gs/hybrid_diagnostic_200.yaml",
    "colmap": REPO / "configs/worldtest_gs/colmap_canonical_200.yaml",
    "rtk": REPO / "configs/worldtest_gs/rtk_canonical_200.yaml",
}
QG_CONFIGS = {
    "colmap_true_qg": REPO / "configs/worldtest_gs/colmap_true_qg_200.yaml",
    "hybrid_true_qg": REPO / "configs/worldtest_gs/hybrid_true_qg_200.yaml",
}
CONTROL_MODES = (
    "matched_delay",
    "equal_count_random",
    "shuffled_qg",
    "npo_lite",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("coordinate", "qg_true", "controls"), default="coordinate"
    )
    parser.add_argument("--frames", type=int, choices=(40, 200, 500), required=True)
    parser.add_argument("--gpu-ids", default="0,1,2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="")
    parser.add_argument("--input", choices=("colmap", "hybrid"), default=None)
    parser.add_argument("--schedule", type=Path, default=None)
    parser.add_argument("--mean-trust-radius", type=float, default=None)
    parser.add_argument("--post-refinement-steps", type=int, default=None)
    parser.add_argument("--min-uniform-replays", type=int, default=None)
    parser.add_argument("--pose-prepass-steps", type=int, default=None)
    parser.add_argument("--pose-prepass-lr", type=float, default=1.0e-3)
    return parser.parse_args()


def materialize(source: Path, frames: int, output: Path) -> dict:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    dataset_path = Path(config["Dataset"]["dataset_path"])
    trajectory = json.loads(
        (dataset_path / "trajectory.json").read_text(encoding="utf-8")
    )
    total = len(trajectory["cameras"])
    if frames > total:
        raise ValueError(f"Requested {frames} frames from {total}-frame dataset")
    end_cutoff = total - frames
    for section in ("Dataset", "Testset"):
        config[section]["end_cutoff"] = end_cutoff
        base_name = config[section]["name"].removesuffix("-200")
        config[section]["name"] = f"{base_name}-{frames}"
    config["Mapper"]["force_keyframes_through_frame"] = frames - 1
    if config.get("WorldTestGS", {}).get("offline_sparse_track_cache", False):
        config["WorldTestGS"]["offline_cache_frames"] = max(
            int(config["WorldTestGS"].get("offline_cache_frames", frames)), frames
        )
    config["Results"]["save_dir"] = str((REPO / "Logs_worldtest_gs").resolve())
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def run_one(name: str, config: Path, gpu: int, seed: int, tag: str) -> dict:
    log_path = config.with_suffix(".console.log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("CUDA_HOME", "/usr/local/cuda-11.8")
    env["PATH"] = f"{env['CUDA_HOME']}/bin:{env.get('PATH', '')}"
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    experiment = f"worldtest_{name}_{tag}_seed{seed}_gpu{gpu}".strip("_")
    command = [
        sys.executable,
        "slam_new.py",
        "--config",
        str(config),
        "--exp_name",
        experiment,
        "--seed",
        str(seed),
    ]
    wall_start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_time_s = time.perf_counter() - wall_start
    resolved = yaml.safe_load(config.read_text(encoding="utf-8"))
    run_root = Path(resolved["Results"]["save_dir"]) / resolved["Dataset"]["name"]
    candidates = sorted(run_root.glob(f"*_{experiment}"), key=lambda path: path.stat().st_mtime)
    return {
        "name": name,
        "gpu": gpu,
        "command": command,
        "returncode": process.returncode,
        "wall_time_s": wall_time_s,
        "console_log": str(log_path),
        "run_dir": str(candidates[-1]) if candidates else None,
    }


def main() -> int:
    args = parse_args()
    gpu_ids = [int(item) for item in args.gpu_ids.split(",") if item.strip()]
    if args.suite == "coordinate":
        sources = COORDINATE_CONFIGS
    elif args.suite == "qg_true":
        sources = QG_CONFIGS
    else:
        if args.input is None or args.schedule is None:
            raise ValueError("Control suite requires --input and --schedule")
        source = QG_CONFIGS[f"{args.input}_true_qg"]
        sources = {mode: source for mode in CONTROL_MODES}
    if args.input is not None and args.suite != "controls":
        key = args.input if args.suite == "coordinate" else f"{args.input}_true_qg"
        sources = {key: sources[key]}
    if len(gpu_ids) < len(sources):
        raise ValueError(f"{args.suite} suite needs {len(sources)} GPU IDs")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    tag = args.tag or f"coordinate{args.frames}_{timestamp}"
    manifest_dir = REPO / "Logs_worldtest_gs" / "benchmarks" / tag
    manifest_dir.mkdir(parents=True, exist_ok=False)
    jobs = []
    for index, (name, source) in enumerate(sources.items()):
        runtime_config = manifest_dir / f"{name}_{args.frames}.yaml"
        config = materialize(source, args.frames, runtime_config)
        if args.mean_trust_radius is not None:
            config["WorldTestGS"]["freeze_committed_means"] = False
            config["WorldTestGS"]["committed_mean_trust_radius_m"] = float(
                args.mean_trust_radius
            )
        if args.post_refinement_steps is not None:
            config["Mapper"]["post_refinement"]["max_steps"] = int(
                args.post_refinement_steps
            )
        if args.min_uniform_replays is not None:
            config["Mapper"]["post_refinement"]["all_frame_min_uniform_replays"] = int(
                args.min_uniform_replays
            )
        if args.pose_prepass_steps is not None:
            camera = config["Mapper"]["CameraOptimizer"]
            camera["use_camera_opt"] = int(args.pose_prepass_steps) > 0
            post = config["Mapper"]["post_refinement"]
            post["pose_prepass_enabled"] = int(args.pose_prepass_steps) > 0
            post["pose_prepass_steps"] = int(args.pose_prepass_steps)
            post["pose_prepass_lr"] = float(args.pose_prepass_lr)
            post.setdefault("pose_prepass_level", 2)
            post.setdefault("pose_prepass_batch_size", 8)
            post.setdefault("pose_prepass_prior_weight", 0.05)
            post.setdefault("pose_prepass_max_translation", 0.05)
            post.setdefault("pose_prepass_frequency_weight", 0.25)
        runtime_config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        if args.suite == "controls":
            config["WorldTestGS"]["admission_mode"] = name
            config["WorldTestGS"]["schedule_path"] = str(args.schedule.resolve())
            for section in ("Dataset", "Testset"):
                config[section]["name"] = config[section]["name"] + "-" + name
            runtime_config.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
        jobs.append((name, runtime_config, gpu_ids[index]))
    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(
                run_one, name, config, gpu, args.seed, tag
            ): name
            for name, config, gpu in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)
    manifest = {
        "suite": args.suite,
        "frames": args.frames,
        "seed": args.seed,
        "tag": tag,
        "results": sorted(results, key=lambda item: item["name"]),
    }
    (manifest_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    failed = [result for result in results if result["returncode"] != 0]
    if failed:
        print(json.dumps({"failed": failed}, indent=2), file=sys.stderr)
        return 1
    print(str(manifest_dir / "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
