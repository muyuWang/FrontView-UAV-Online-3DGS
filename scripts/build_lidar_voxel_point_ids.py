#!/usr/bin/env python3
"""Assign stable collision-free voxel identities to canonical LiDAR points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BITS_PER_AXIS = 21
AXIS_MASK = (1 << BITS_PER_AXIS) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def voxel_ids(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size must be finite and positive")
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must be an Nx3-or-wider array")
    if not np.isfinite(points[:, :3]).all():
        raise ValueError("points contain non-finite coordinates")
    quantized = np.rint(points[:, :3] / float(voxel_size)).astype(np.int64)
    zigzag = (quantized << 1) ^ (quantized >> 63)
    if np.any(zigzag > AXIS_MASK):
        raise ValueError("voxel coordinate exceeds the collision-free 21-bit range")
    return (
        zigzag[:, 0]
        | (zigzag[:, 1] << BITS_PER_AXIS)
        | (zigzag[:, 2] << (2 * BITS_PER_AXIS))
    ).astype(np.int64)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    point_dir = dataset / "orb_point_clouds"
    output_dir = dataset / "orb_point_ids"
    point_paths = sorted(
        point_dir.glob("point_cloud_*.npy"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    if not point_paths:
        raise ValueError(f"No LiDAR point clouds found in {point_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = []
    unique_ratios = []
    for index, point_path in enumerate(point_paths):
        frame_id = int(point_path.stem.rsplit("_", 1)[1])
        output = output_dir / f"point_ids_{frame_id}.npy"
        points = np.load(point_path, mmap_mode="r")
        identities = voxel_ids(points, args.voxel_size)
        if output.exists() and not args.force:
            existing = np.load(output)
            if not np.array_equal(existing, identities):
                raise FileExistsError(f"Existing identity sidecar differs: {output}")
        else:
            np.save(output, identities)
        counts.append(len(identities))
        unique_ratios.append(len(np.unique(identities)) / max(len(identities), 1))
        if index % 100 == 0 or index + 1 == len(point_paths):
            print(f"[{index + 1}/{len(point_paths)}] {point_path.name}", flush=True)
    stats = {
        "schema_version": 1,
        "method": "collision-free signed voxel coordinate packing",
        "dataset": str(dataset),
        "voxel_size_m": float(args.voxel_size),
        "frame_count": len(point_paths),
        "point_count_min": int(np.min(counts)),
        "point_count_median": float(np.median(counts)),
        "point_count_max": int(np.max(counts)),
        "within_frame_unique_ratio_median": float(np.median(unique_ratios)),
    }
    (output_dir / "voxel_identity_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
