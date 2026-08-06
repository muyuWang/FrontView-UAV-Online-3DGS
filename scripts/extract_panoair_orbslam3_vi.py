#!/usr/bin/env python3
"""Export a source-indexed PanoAir sequence in ORB-SLAM3 EuRoC layout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import rosbag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=110)
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Zero exports all retained frames."
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_frames(source: Path) -> list[tuple[int, str]]:
    rows = []
    for line in (source / "frame_sequences" / "rgb.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, filename = line.split()[:2]
        rows.append((int(timestamp), filename))
    if not rows:
        raise RuntimeError("PanoAir rgb.txt does not contain any frames")
    return rows


def write_imu(source: Path, destination: Path, first_ns: int, last_ns: int) -> int:
    # Keep one sample before the first image so ORB-SLAM3 can integrate the first interval.
    rows = []
    for line in (source / "frame_sequences" / "imu0.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp = int(line.split(",", 1)[0])
        rows.append((timestamp, line))
    timestamps = np.asarray([row[0] for row in rows], dtype=np.int64)
    begin = max(0, int(np.searchsorted(timestamps, first_ns, side="left")) - 1)
    end = int(np.searchsorted(timestamps, last_ns, side="right"))
    selected = rows[begin:end]
    destination.write_text(
        "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n"
        + "\n".join(row[1] for row in selected)
        + "\n",
        encoding="utf-8",
    )
    return len(selected)


def ensure_symlink(destination: Path, source: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(destination)
    destination.symlink_to(source.resolve())


def export_back_camera(bag_path: Path, output: Path, wanted: set[int]) -> int:
    exported = 0
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, message, _ in bag.read_messages(topics=["/cam1/image_raw"]):
            timestamp = message.header.stamp.to_nsec()
            if timestamp not in wanted:
                continue
            destination = output / f"{timestamp}.png"
            if not destination.is_file():
                if message.encoding not in ("mono8", "8UC1"):
                    raise RuntimeError(
                        f"Unsupported cam1 encoding {message.encoding!r} at {timestamp}"
                    )
                image = np.frombuffer(message.data, dtype=np.uint8).reshape(
                    message.height, message.step
                )[:, : message.width]
                if not cv2.imwrite(
                    str(destination), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]
                ):
                    raise RuntimeError(f"Could not write {destination}")
            exported += 1
            if exported == 1 or exported % 250 == 0 or exported == len(wanted):
                print(f"cam1: {exported}/{len(wanted)}", flush=True)
    return exported


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    frames = load_frames(source)
    if not 0 <= args.start_frame < len(frames):
        raise ValueError(f"start-frame {args.start_frame} is outside {len(frames)} frames")
    retained = frames[args.start_frame :]
    if args.max_frames > 0:
        retained = retained[: args.max_frames]
    if not retained:
        raise RuntimeError("No frames remain after slicing")

    metadata_path = output / "extraction.json"
    if metadata_path.is_file() and not args.force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = [row[0] for row in retained]
        if metadata.get("timestamps_ns") == expected:
            print(f"Reusing complete extraction: {output}")
            return 0
        raise RuntimeError(f"Existing extraction has different parameters: {output}")
    if output.exists() and args.force:
        raise RuntimeError(
            "Refusing to delete an existing extraction. Choose a new output directory."
        )

    cam0 = output / "mav0" / "cam0" / "data"
    cam1 = output / "mav0" / "cam1" / "data"
    imu = output / "mav0" / "imu0"
    cam0.mkdir(parents=True, exist_ok=True)
    cam1.mkdir(parents=True, exist_ok=True)
    imu.mkdir(parents=True, exist_ok=True)

    for timestamp, filename in retained:
        ensure_symlink(
            cam0 / f"{timestamp}.png",
            source / "frame_sequences" / "fisheyecam" / filename,
        )
    timestamps = [row[0] for row in retained]
    (output / "times.txt").write_text(
        "\n".join(str(timestamp) for timestamp in timestamps) + "\n",
        encoding="utf-8",
    )
    imu_count = write_imu(
        source, imu / "data.csv", timestamps[0], timestamps[-1]
    )
    cam1_count = export_back_camera(
        source / "seq1_dual_fisheye.bag", cam1, set(timestamps)
    )
    if cam1_count != len(retained):
        raise RuntimeError(
            f"cam1 synchronization failed: expected {len(retained)}, got {cam1_count}"
        )

    metadata = {
        "source": str(source),
        "start_source_frame": args.start_frame,
        "end_source_frame_inclusive": args.start_frame + len(retained) - 1,
        "frame_count": len(retained),
        "imu_count": imu_count,
        "cam0_storage": "absolute symbolic links to frame_sequences/fisheyecam",
        "cam1_storage": "PNG extracted from /cam1/image_raw",
        "timestamps_ns": timestamps,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "timestamps_ns"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
