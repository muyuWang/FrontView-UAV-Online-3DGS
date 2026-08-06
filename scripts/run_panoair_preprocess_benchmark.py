#!/usr/bin/env python3
"""Run the causal PanoAir initialization benchmark on independent GPUs."""

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


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CONFIGS = {
    "colmap_start0": REPO / "configs/panoair_preprocess/colmap_start0_200.yaml",
    "colmap_start110": REPO / "configs/panoair_preprocess/colmap_start110_200.yaml",
    "colmap_start130": REPO / "configs/panoair_preprocess/colmap_start130_200.yaml",
    "learned4096_start110": REPO
    / "configs/panoair_preprocess/learned4096_start110_200.yaml",
    "fused_start110": REPO / "configs/panoair_preprocess/fused_start110_200.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--arms",
        default="colmap_start110,colmap_start130,learned4096_start110,fused_start110",
        help="Comma-separated benchmark arms. colmap_start0 is an optional prefix control.",
    )
    return parser.parse_args()


def run_one(name: str, config: Path, gpu: int, seed: int, tag: str, log: Path):
    from utils_new.tool_utils import load_config

    resolved = load_config(str(config))
    experiment = f"panoair_preprocess_{name}_{tag}_seed{seed}_gpu{gpu}"
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
    env.setdefault("CUDA_HOME", "/usr/local/cuda-11.8")
    env["PATH"] = f"{env['CUDA_HOME']}/bin:{env.get('PATH', '')}"
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    start = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    run_root = Path(resolved["Results"]["save_dir"]) / resolved["Dataset"]["name"]
    candidates = sorted(
        run_root.glob(f"*_{experiment}"), key=lambda path: path.stat().st_mtime
    )
    return {
        "name": name,
        "gpu": gpu,
        "config": str(config),
        "command": command,
        "returncode": process.returncode,
        "wall_time_s": time.perf_counter() - start,
        "console_log": str(log),
        "run_dir": str(candidates[-1]) if candidates else None,
    }


def main() -> int:
    args = parse_args()
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
    arm_names = [value for value in args.arms.split(",") if value.strip()]
    unknown = set(arm_names) - set(CONFIGS)
    if unknown:
        raise ValueError(f"Unknown benchmark arms: {sorted(unknown)}")
    selected = [(name, CONFIGS[name]) for name in arm_names]
    if len(gpu_ids) < len(selected):
        raise ValueError("One independent GPU ID is required per benchmark arm")
    tag = args.tag or datetime.now().strftime("preprocess200_%Y-%m-%d-%H-%M-%S")
    output = REPO / "Logs_panoair_preprocess" / "benchmarks" / tag
    output.mkdir(parents=True, exist_ok=False)

    results = []
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        futures = {}
        for gpu, (name, config) in zip(gpu_ids, selected):
            future = executor.submit(
                run_one,
                name,
                config,
                gpu,
                args.seed,
                tag,
                output / f"{name}.console.log",
            )
            futures[future] = name
        for future in as_completed(futures):
            results.append(future.result())

    manifest = {
        "tag": tag,
        "seed": args.seed,
        "results": sorted(results, key=lambda item: item["name"]),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    failed = [item for item in results if item["returncode"] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
