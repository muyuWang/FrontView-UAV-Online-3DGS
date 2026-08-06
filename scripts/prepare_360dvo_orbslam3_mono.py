#!/usr/bin/env python3
"""Prepare a derived Online3DGS scene for ORB-SLAM3 mono_tum_vi."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")

    trajectory_path = source / "trajectory_orb.json"
    all_cameras = json.loads(trajectory_path.read_text(encoding="utf-8"))["cameras"]
    if args.start_frame < 0 or args.start_frame >= len(all_cameras):
        raise ValueError("--start-frame is outside the source trajectory")
    cameras = all_cameras[args.start_frame :]
    if args.num_frames > 0:
        cameras = cameras[: args.num_frames]
    if not cameras:
        raise RuntimeError("No frames selected")

    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists: {staging}")
    image_dir = staging / "images"
    image_dir.mkdir(parents=True)
    start_ns = 1_000_000_000
    step_ns = int(round(1.0e9 / args.fps))
    timestamps_ns = []
    source_indices = []
    for output_index, camera in enumerate(cameras):
        timestamp_ns = start_ns + output_index * step_ns
        image_path = source / "rectified" / camera["image"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        (image_dir / f"{timestamp_ns}.png").symlink_to(image_path.resolve())
        timestamps_ns.append(timestamp_ns)
        source_indices.append(
            int(
                camera.get(
                    "source_frame_index",
                    camera.get("frame_index", args.start_frame + output_index),
                )
            )
        )

    (staging / "times.txt").write_text(
        "\n".join(str(value) for value in timestamps_ns) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "start_frame": args.start_frame,
        "frame_count": len(cameras),
        "fps": args.fps,
        "timestamps_ns": timestamps_ns,
        "source_frame_indices": source_indices,
        "image_storage": "absolute symbolic links",
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    staging.replace(output)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
