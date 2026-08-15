#!/usr/bin/env python3
"""Re-render TGBR benchmark manifests from saved PLYs without LPIPS."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/wmy/anaconda3/envs/worldvln/bin/python")
CUDA_HOME = Path("/home/wmy/.local/cuda-12.1")


def environment(gpu):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_HOME": str(CUDA_HOME),
            "CUDA_VISIBLE_DEVICES": str(gpu),
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


def rerender(row, gpu):
    output_dir = Path(row["render_dir"])
    command = [
        str(PYTHON),
        "render.py",
        "--run_dir",
        row["run_dir"],
        "--output_dir",
        str(output_dir),
        "--device",
        "cuda:0",
        "--skip_lpips",
        "--skip_novel",
        "--skip_depth",
        "--skip_primitives",
        "--ignore_cached_renders",
    ]
    log_path = output_dir.parent / (row["name"] + ".rerender.log")
    with log_path.open("w") as console:
        code = subprocess.run(
            command,
            cwd=ROOT,
            env=environment(gpu),
            stdout=console,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if code != 0:
        raise RuntimeError("{} failed; see {}".format(row["name"], log_path))
    metrics_path = output_dir / "render_metrics.json"
    with metrics_path.open() as handle:
        metrics = json.load(handle)
    return row["name"], metrics, command, str(log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()

    for manifest_path in args.manifests:
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        rows = manifest["results"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(rerender, row, 4 + index % 4): row
                for index, row in enumerate(rows)
            }
            updates = {}
            for future in concurrent.futures.as_completed(futures):
                name, metrics, command, log_path = future.result()
                updates[name] = (metrics, command, log_path)
                print("rerendered {}".format(name), flush=True)
        for row in rows:
            metrics, command, log_path = updates[row["name"]]
            row["render_metrics"] = metrics
            row["render_command"] = command
            row["render_log"] = log_path
        with manifest_path.open("w") as handle:
            json.dump(manifest, handle, indent=2)
        print(manifest_path)


if __name__ == "__main__":
    main()
