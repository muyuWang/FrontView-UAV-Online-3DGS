#!/usr/bin/env python3
"""Measure reconstructed Gaussian geometry against canonical LiDAR samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ply", type=Path, default=None)
    parser.add_argument("--lidar-points-per-frame", type=int, default=250)
    parser.add_argument("--map-query-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def summarize(distances: np.ndarray) -> dict:
    return {
        "count": int(len(distances)),
        "median_m": float(np.median(distances)),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "p99_m": float(np.percentile(distances, 99)),
        "within_0_05m_percent": float(np.mean(distances <= 0.05) * 100.0),
        "within_0_10m_percent": float(np.mean(distances <= 0.10) * 100.0),
        "within_0_20m_percent": float(np.mean(distances <= 0.20) * 100.0),
    }


def load_map_points(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"].data
    return np.column_stack([vertex[axis] for axis in ("x", "y", "z")]).astype(
        np.float32
    )


def load_lidar_samples(dataset: Path, points_per_frame: int, rng) -> np.ndarray:
    paths = sorted(
        (dataset / "orb_point_clouds").glob("point_cloud_*.npy"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    samples = []
    for path in paths:
        points = np.load(path, mmap_mode="r")[:, :3]
        count = min(int(points_per_frame), len(points))
        indices = rng.choice(len(points), size=count, replace=False)
        samples.append(np.asarray(points[indices], dtype=np.float32))
    if not samples:
        raise ValueError(f"No LiDAR point clouds found in {dataset}")
    return np.concatenate(samples, axis=0)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    dataset = Path(config["Dataset"]["dataset_path"]).resolve()
    if args.ply is not None:
        ply_path = args.ply.resolve()
    elif (run_dir / "point_cloud_aerocommit_full.ply").is_file():
        ply_path = run_dir / "point_cloud_aerocommit_full.ply"
    else:
        ply_path = run_dir / "point_cloud.ply"
    rng = np.random.default_rng(args.seed)
    map_points = load_map_points(ply_path)
    lidar_points = load_lidar_samples(dataset, args.lidar_points_per_frame, rng)
    finite_map = map_points[np.isfinite(map_points).all(axis=1)]
    finite_lidar = lidar_points[np.isfinite(lidar_points).all(axis=1)]

    map_count = min(int(args.map_query_points), len(finite_map))
    map_query = finite_map[
        rng.choice(len(finite_map), size=map_count, replace=False)
    ]
    map_to_lidar, _ = cKDTree(finite_lidar).query(map_query, workers=-1)
    lidar_to_map, _ = cKDTree(finite_map).query(finite_lidar, workers=-1)
    payload = {
        "run_dir": str(run_dir),
        "ply": str(ply_path),
        "dataset": str(dataset),
        "map_point_count": int(len(finite_map)),
        "lidar_sample_count": int(len(finite_lidar)),
        "map_to_lidar": summarize(map_to_lidar),
        "lidar_to_map": summarize(lidar_to_map),
        "warning": "Nearest-neighbor proxy; adaptive or split GS need not equal a raw LiDAR return.",
    }
    output = args.output or run_dir / "lidar_geometry_metrics.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
