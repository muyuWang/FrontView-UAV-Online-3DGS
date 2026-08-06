#!/usr/bin/env python3
"""Render and score PanoAir preprocessing benchmark runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_command(command, env, log_path):
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ {}\n".format(" ".join(str(item) for item in command)))
        log.flush()
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Command failed with status {result.returncode}: {log_path}")


def contact_sheet(video_path, output_path, source_offset, samples=12):
    capture = cv2.VideoCapture(str(video_path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, count - 1, min(samples, count)).astype(int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        width = 480
        height = max(1, int(round(frame.shape[0] * width / frame.shape[1])))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        cv2.putText(
            frame,
            f"source {int(index) + int(source_offset):03d}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode {video_path}")
    columns = 3
    rows = (len(frames) + columns - 1) // columns
    height, width = frames[0].shape[:2]
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not write {output_path}")


def render_command(run_dir, output_dir, begin=None, end=None):
    command = [
        sys.executable,
        "render.py",
        "--run_dir",
        str(run_dir),
        "--output_dir",
        str(output_dir),
        "--device",
        "cuda:0",
        "--skip_novel",
        "--skip_primitives",
        "--save_opacity",
        "--skip_view_detail",
        "--skip_frequency_cache",
    ]
    if begin is not None:
        command.extend(["--render_begin", str(begin), "--render_end", str(end)])
    return command


def process_run(run_dir, gpu_id, force):
    run_dir = Path(run_dir).resolve()
    start = time.perf_counter()
    log_path = run_dir / "panoair_preprocess_postprocess.console.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TORCH_EXTENSIONS_DIR"] = f"/tmp/torch_extensions_panoair_gpu{gpu_id}"
    env.setdefault("CUDA_HOME", "/usr/local/cuda-11.8")
    env["PATH"] = f"{env['CUDA_HOME']}/bin:{env.get('PATH', '')}"
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    dataset = config.get("Testset", config["Dataset"])
    trajectory_name = "trajectory_orb.json" if dataset.get("data_source") == "orb" else "trajectory.json"
    cameras = json.loads(
        (Path(dataset["dataset_path"]) / trajectory_name).read_text(encoding="utf-8")
    )["cameras"]
    source_start = int(cameras[0].get("source_frame_index", 0))
    processed = int(json.loads((run_dir / "results.json").read_text())["num_processed_frames"])

    full_dir = run_dir / "videos_full"
    takeoff_dir = run_dir / "takeoff_source_120_160"
    expected = ("render_vs_gt.mp4", "render_depth.mp4", "render_opacity.mp4")
    full_dir.mkdir(exist_ok=True)
    takeoff_dir.mkdir(exist_ok=True)
    if force or not all((full_dir / name).is_file() for name in expected):
        run_command(render_command(run_dir, full_dir), env, log_path)

    source_begin = max(120, source_start)
    source_end = min(160, source_start + processed - 1)
    local_begin = source_begin - source_start
    local_end = source_end - source_start
    if force or not all((takeoff_dir / name).is_file() for name in expected):
        run_command(
            render_command(run_dir, takeoff_dir, local_begin, local_end), env, log_path
        )

    metrics_path = run_dir / "validation_metrics.json"
    if force or not metrics_path.is_file():
        run_command(
            [
                sys.executable,
                "scripts/evaluate_render_vs_gt.py",
                "--run_dir",
                str(run_dir),
                "--video",
                str(full_dir / "render_vs_gt.mp4"),
                "--output",
                str(metrics_path),
            ],
            env,
            log_path,
        )

    geometry_path = run_dir / "worldtest_evaluation.json"
    lawn_reference = min(max(150 - source_start, 0), processed - 1)
    if force or not geometry_path.is_file():
        run_command(
            [
                sys.executable,
                "scripts/evaluate_worldtest_run.py",
                "--run-dir",
                str(run_dir),
                "--metrics",
                str(metrics_path),
                "--lawn-reference-frame",
                str(lawn_reference),
                "--output",
                str(geometry_path),
            ],
            env,
            log_path,
        )

    for directory, offset in ((full_dir, source_start), (takeoff_dir, source_begin)):
        for name in expected:
            contact_sheet(
                directory / name,
                directory / "contact_sheets" / f"{Path(name).stem}.jpg",
                offset,
            )
    return {
        "run_dir": str(run_dir),
        "gpu": gpu_id,
        "source_start": source_start,
        "takeoff_local_range": [local_begin, local_end],
        "takeoff_source_range": [source_begin, source_end],
        "metrics": str(metrics_path),
        "geometry": str(geometry_path),
        "wall_time_s": time.perf_counter() - start,
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    runs = [item["run_dir"] for item in manifest["results"] if item["returncode"] == 0]
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
    workers = min(args.workers, len(gpu_ids), len(runs))
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_run, run, gpu_ids[index % workers], args.force): run
            for index, run in enumerate(runs)
        }
        for future in as_completed(futures):
            results.append(future.result())
    output = args.manifest.with_name("postprocess_manifest.json")
    output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
