#!/usr/bin/env python3
"""Run paired baseline/PBSD full-sequence cross-scene experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CONFIGS = {
    "hku_baseline": REPO
    / "configs/frontview_uav_crossscene/hku_baseline.yaml",
    "hku_pbsd": REPO / "configs/frontview_uav_crossscene/hku_pbsd.yaml",
    "red_baseline": REPO
    / "configs/frontview_uav_crossscene/red_sculpture_baseline.yaml",
    "red_pbsd": REPO
    / "configs/frontview_uav_crossscene/red_sculpture_pbsd.yaml",
    "road_baseline": REPO
    / "configs/frontview_uav_crossscene/road_street1_baseline.yaml",
    "road_pbsd": REPO
    / "configs/frontview_uav_crossscene/road_street1_pbsd.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default=",".join(CONFIGS))
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def run_one(
    name: str,
    config: Path,
    gpu: int,
    seed: int,
    tag: str,
    console_log: Path,
) -> dict:
    from utils_new.tool_utils import load_config

    resolved = load_config(str(config))
    experiment = f"crossscene_{name}_{tag}_seed{seed}_gpu{gpu}"
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
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    wall_start = time.perf_counter()
    with console_log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_time_s = time.perf_counter() - wall_start

    run_root = Path(resolved["Results"]["save_dir"]) / resolved["Dataset"]["name"]
    candidates = sorted(
        run_root.glob(f"*_{experiment}"), key=lambda path: path.stat().st_mtime
    )
    run_dir = candidates[-1] if candidates else None
    metrics = None
    if run_dir is not None and (run_dir / "results.json").is_file():
        payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        metrics = {
            "online_recon_time": payload.get("online_recon_time"),
            "eval_time": payload.get("eval_time"),
            "num_processed_frames": payload.get("num_processed_frames"),
            "num_gaussians": payload.get("num_gaussians"),
            "num_keyframes": payload.get("num_keyframes"),
            "eval_res": payload.get("eval_res"),
            "frontview_sampling": payload.get("frontview_sampling"),
        }

    return {
        "name": name,
        "gpu": gpu,
        "config": str(config),
        "command": command,
        "returncode": process.returncode,
        "wall_time_s": wall_time_s,
        "console_log": str(console_log),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "metrics": metrics,
    }


def main() -> int:
    args = parse_args()
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    gpu_ids = [int(item) for item in args.gpu_ids.split(",") if item.strip()]
    unknown = set(arms) - set(CONFIGS)
    if unknown:
        raise ValueError(f"Unknown arms: {sorted(unknown)}")
    if len(gpu_ids) < len(arms):
        raise ValueError("One independent GPU ID is required per arm")

    tag = args.tag or datetime.now().strftime("full_%Y-%m-%d-%H-%M-%S")
    output = REPO / "Logs_frontview_uav_crossscene" / "benchmarks" / tag
    output.mkdir(parents=True, exist_ok=False)

    results = []
    with ThreadPoolExecutor(max_workers=len(arms)) as executor:
        futures = {
            executor.submit(
                run_one,
                arm,
                CONFIGS[arm],
                gpu,
                args.seed,
                tag,
                output / f"{arm}.console.log",
            ): arm
            for arm, gpu in zip(arms, gpu_ids)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)

    manifest = {
        "tag": tag,
        "seed": args.seed,
        "rad_interpretation": "road_street1",
        "results": sorted(results, key=lambda item: item["name"]),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 1 if any(item["returncode"] != 0 for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
