#!/usr/bin/env python3
"""Measure right-edge temporal changes after subtracting the paired GT video."""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Paired render-vs-GT H.264 video. May be specified repeatedly.",
    )
    parser.add_argument("--begin", type=int, default=620)
    parser.add_argument("--end", type=int, default=675)
    parser.add_argument("--frames", default="632,640,650,665")
    parser.add_argument("--right-fraction", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_video(value):
    if "=" not in value:
        raise ValueError("--video must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise ValueError("Video label cannot be empty")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return label, path


def temporal_changes(path, right_fraction):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    changes = []
    previous_error = None
    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height, paired_width = frame.shape[:2]
        if paired_width % 2:
            raise ValueError(f"Expected side-by-side render and GT: {path}")
        width = paired_width // 2
        edge_begin = width - max(1, int(round(width * right_fraction)))
        render = frame[:, edge_begin:width].astype(np.float32) / 255.0
        gt = frame[:, width + edge_begin :].astype(np.float32) / 255.0
        error = render - gt
        changes.append(
            None
            if previous_error is None
            else float(np.mean(np.abs(error - previous_error)))
        )
        previous_error = error
        frame_count += 1
    capture.release()
    return changes, frame_count


def main():
    args = parse_args()
    if not 0.0 < args.right_fraction <= 1.0:
        raise ValueError("--right-fraction must be in (0, 1]")
    if args.begin < 1 or args.end < args.begin:
        raise ValueError("Expected 1 <= begin <= end")
    selected_frames = [
        int(value) for value in args.frames.split(",") if value.strip()
    ]
    videos = dict(parse_video(value) for value in args.video)
    if len(videos) != len(args.video):
        raise ValueError("Video labels must be unique")

    results = {}
    for label, path in videos.items():
        changes, frame_count = temporal_changes(path, args.right_fraction)
        end = min(args.end, frame_count - 1)
        window = [changes[index] for index in range(args.begin, end + 1)]
        results[label] = {
            "video": str(path),
            "frame_count": frame_count,
            "window_mean": float(np.mean(window)),
            "window_max": float(np.max(window)),
            "selected_frames": {
                str(index): changes[index]
                for index in selected_frames
                if 0 < index < frame_count
            },
        }

    payload = {
        "metric": (
            "mean absolute temporal change of render-minus-GT error over the "
            "rightmost image fraction"
        ),
        "window": [args.begin, args.end],
        "right_fraction": args.right_fraction,
        "results": results,
    }
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
