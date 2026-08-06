#!/usr/bin/env python3
"""Render, score, and summarize WorldTest-GS benchmark manifests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[1]
VIDEO_NAMES = (
    "render_vs_gt.mp4",
    "render_depth.mp4",
    "render_opacity.mp4",
    "render_primitives.mp4",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
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
        raise RuntimeError(
            "Command failed with status {} (see {})".format(
                result.returncode, log_path
            )
        )


def contact_sheet(video_path, output_path, label_offset=0, samples=12):
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError("No frames in {}".format(video_path))
    indices = np.linspace(0, frame_count - 1, min(samples, frame_count)).astype(int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        target_width = 480
        scale = target_width / frame.shape[1]
        resized = cv2.resize(
            frame,
            (target_width, max(1, int(round(frame.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        cv2.putText(
            resized,
            "frame {:03d}".format(int(index) + int(label_offset)),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(resized)
    capture.release()
    if not frames:
        raise RuntimeError("Could not decode {}".format(video_path))
    columns = 3
    rows = (len(frames) + columns - 1) // columns
    height, width = frames[0].shape[:2]
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError("Could not write {}".format(output_path))


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
        "--save_opacity",
        "--skip_view_detail",
        "--skip_frequency_cache",
    ]
    if begin is not None:
        command.extend(["--render_begin", str(begin), "--render_end", str(end)])
    return command


def postprocess_one(run_dir, gpu_id, force):
    start = time.perf_counter()
    run_dir = Path(run_dir).resolve()
    log_path = run_dir / "worldtest_postprocess.console.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("CUDA_HOME", "/usr/local/cuda-11.8")
    env["PATH"] = "{}/bin:{}".format(env["CUDA_HOME"], env.get("PATH", ""))
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

    full_dir = run_dir / "videos_full"
    diagnostic_windows = (
        (run_dir / "takeoff_120_160", 120, 160),
        (run_dir / "identity_boundary_180_260", 180, 260),
        (run_dir / "late_400_499", 400, 499),
    )
    results_path = run_dir / "results.json"
    result_metadata = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.is_file()
        else {}
    )
    frame_count = int(result_metadata.get("num_processed_frames", 0))
    diagnostic_windows = tuple(
        (directory, begin, min(end, frame_count - 1))
        for directory, begin, end in diagnostic_windows
        if begin < frame_count
    )
    full_dir.mkdir(exist_ok=True)
    for directory, _, _ in diagnostic_windows:
        directory.mkdir(exist_ok=True)
    if force or not all((full_dir / name).is_file() for name in VIDEO_NAMES):
        run_command(render_command(run_dir, full_dir), env, log_path)
    for directory, begin, end in diagnostic_windows:
        if force or not all((directory / name).is_file() for name in VIDEO_NAMES):
            run_command(render_command(run_dir, directory, begin, end), env, log_path)

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
    evaluation_path = run_dir / "worldtest_evaluation.json"
    if force or not evaluation_path.is_file():
        run_command(
            [
                sys.executable,
                "scripts/evaluate_worldtest_run.py",
                "--run-dir",
                str(run_dir),
                "--metrics",
                str(metrics_path),
                "--output",
                str(evaluation_path),
            ],
            env,
            log_path,
        )

    sheet_sources = [(full_dir, 0)] + [
        (directory, begin) for directory, begin, _ in diagnostic_windows
    ]
    for directory, offset in sheet_sources:
        sheet_dir = directory / "contact_sheets"
        for video_name in VIDEO_NAMES:
            contact_sheet(
                directory / video_name,
                sheet_dir / (Path(video_name).stem + ".jpg"),
                label_offset=offset,
            )
    return {
        "run_dir": str(run_dir),
        "gpu": int(gpu_id),
        "wall_time_s": time.perf_counter() - start,
        "metrics": str(metrics_path),
        "evaluation": str(evaluation_path),
    }


def main():
    args = parse_args()
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
    runs = []
    for manifest_path in args.manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for result in manifest["results"]:
            if result.get("returncode") == 0 and result.get("run_dir"):
                runs.append(result["run_dir"])
    runs = list(dict.fromkeys(runs))
    if not runs:
        raise ValueError("No successful run directories found")
    workers = min(int(args.workers), len(gpu_ids), len(runs))
    if workers <= 0:
        raise ValueError("At least one worker and GPU are required")

    results = []
    available_gpus = Queue()
    for gpu_id in gpu_ids[:workers]:
        available_gpus.put(gpu_id)

    def dispatch(run_dir):
        gpu_id = available_gpus.get()
        try:
            return postprocess_one(run_dir, gpu_id, args.force)
        finally:
            available_gpus.put(gpu_id)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(dispatch, run_dir): run_dir for run_dir in runs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)
    payload = {"runs": sorted(results, key=lambda item: item["run_dir"])}
    summary_path = args.manifest[0].resolve().parent / "postprocess_manifest.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
