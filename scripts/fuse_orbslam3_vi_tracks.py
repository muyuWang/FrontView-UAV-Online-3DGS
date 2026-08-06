#!/usr/bin/env python3
"""Fuse native ORB-SLAM3 landmarks with learned tracks in one VI world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orb-source", type=Path, required=True)
    parser.add_argument("--learned-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dedup-radius-m", type=float, default=0.03)
    parser.add_argument("--max-points-per-frame", type=int, default=10000)
    return parser.parse_args()


def ply_colors(path: Path) -> np.ndarray:
    vertices = PlyData.read(path)["vertex"].data
    return np.column_stack(
        (vertices["red"], vertices["green"], vertices["blue"])
    ).astype(np.uint8)


def write_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    sources: np.ndarray,
    track_lengths: np.ndarray,
) -> None:
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("point_id", "u4"), ("source", "u1"), ("track_length", "u2"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    vertices["point_id"] = np.arange(len(points), dtype=np.uint32)
    vertices["source"] = sources
    vertices["track_length"] = np.minimum(track_lengths, 65535).astype(np.uint16)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def main() -> int:
    args = parse_args()
    orb = args.orb_source.resolve()
    learned = args.learned_source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    orb_points = np.load(orb / "preprocess" / "global_sparse_points.npy").astype(
        np.float32
    )
    learned_points = np.load(
        learned / "preprocess" / "global_sparse_points.npy"
    ).astype(np.float32)
    orb_colors = ply_colors(orb / "initialization_orbslam3_vi_total.ply")
    learned_colors = ply_colors(learned / "initialization_multiview_tracks.ply")
    orb_tracks = np.load(orb / "preprocess" / "point_observations.npy").astype(
        np.int32
    )
    learned_ply = PlyData.read(learned / "initialization_multiview_tracks.ply")[
        "vertex"
    ].data
    learned_tracks = np.asarray(learned_ply["track_length"], dtype=np.int32)
    if len(orb_colors) != len(orb_points) or len(learned_colors) != len(learned_points):
        raise RuntimeError("PLY row order does not match global sparse points")

    if len(learned_points):
        distances, _ = cKDTree(learned_points).query(orb_points, k=1, workers=-1)
        keep_orb = distances > args.dedup_radius_m
    else:
        keep_orb = np.ones(len(orb_points), dtype=bool)
    orb_remap = np.full(len(orb_points), -1, dtype=np.int64)
    orb_remap[keep_orb] = len(learned_points) + np.arange(keep_orb.sum())
    points = np.concatenate((learned_points, orb_points[keep_orb]), axis=0)
    colors = np.concatenate((learned_colors, orb_colors[keep_orb]), axis=0)
    sources = np.concatenate(
        (
            np.ones(len(learned_points), dtype=np.uint8),
            np.zeros(keep_orb.sum(), dtype=np.uint8),
        )
    )
    track_lengths = np.concatenate((learned_tracks, orb_tracks[keep_orb]))

    trajectory_text = (learned / "trajectory_orb.json").read_text(encoding="utf-8")
    cameras = json.loads(trajectory_text)["cameras"]
    for directory in (
        output / "rectified",
        output / "orb_point_clouds",
        output / "orb_point_ids",
        output / "preprocess",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for frame, camera in enumerate(cameras):
        image = learned / "rectified" / camera["image"]
        (output / "rectified" / f"aria_{frame:05d}.png").symlink_to(
            image.resolve()
        )
        learned_ids = np.load(
            learned / "orb_point_ids" / f"point_ids_{frame}.npy"
        ).astype(np.int64)
        orb_ids = np.load(orb / "orb_point_ids" / f"point_ids_{frame}.npy").astype(
            np.int64
        )
        remapped_orb = orb_remap[orb_ids]
        remapped_orb = remapped_orb[remapped_orb >= 0]
        ids = np.unique(np.concatenate((learned_ids, remapped_orb)))
        if len(ids) > args.max_points_per_frame:
            # Points are ordered learned-first, so preserve the verified tracks before ORB fill.
            ids = ids[: args.max_points_per_frame]
        np.save(output / "orb_point_ids" / f"point_ids_{frame}.npy", ids)
        np.save(
            output / "orb_point_clouds" / f"point_cloud_{frame}.npy",
            points[ids],
        )

    (output / "trajectory.json").write_text(trajectory_text, encoding="utf-8")
    (output / "trajectory_orb.json").write_text(trajectory_text, encoding="utf-8")
    np.save(output / "preprocess" / "global_sparse_points.npy", points)
    np.save(output / "preprocess" / "global_orbslam3_vi_fused_points.npy", points)
    source_indices = np.load(
        learned / "preprocess" / "source_frame_indices.npy"
    )
    np.save(output / "preprocess" / "source_frame_indices.npy", source_indices)
    observations = learned / "preprocess" / "track_observations.npz"
    if observations.is_file():
        (output / "preprocess" / "track_observations.npz").symlink_to(
            observations.resolve()
        )
    write_ply(
        output / "initialization_orbslam3_vi_fused_total.ply",
        points,
        colors,
        sources,
        track_lengths,
    )

    counts = [
        len(np.load(output / "orb_point_ids" / f"point_ids_{frame}.npy"))
        for frame in range(len(cameras))
    ]
    learned_stats = json.loads(
        (learned / "conversion_stats.json").read_text(encoding="utf-8")
    )
    orb_stats = json.loads(
        (orb / "conversion_stats.json").read_text(encoding="utf-8")
    )
    stats = {
        "schema_version": 1,
        "method": (
            "ORB-SLAM3 visual-inertial BA poses; strict DISK-LightGlue tracks plus "
            "deduplicated native ORB landmarks"
        ),
        "pose_source": "orbslam3_vi",
        "sparse_world_geometry": "persistent",
        "coordinate_contract": orb_stats["coordinate_contract"],
        "frame_count": len(cameras),
        "source_frame_start": int(source_indices[0]),
        "source_frame_end_inclusive": int(source_indices[-1]),
        "learned_point_count": int(len(learned_points)),
        "orb_point_count_input": int(len(orb_points)),
        "orb_point_count_added": int(keep_orb.sum()),
        "orb_points_deduplicated": int((~keep_orb).sum()),
        "total_point_count": int(len(points)),
        "dedup_radius_m": args.dedup_radius_m,
        "per_frame_points_min": int(np.min(counts)),
        "per_frame_points_median": float(np.median(counts)),
        "per_frame_points_mean": float(np.mean(counts)),
        "per_frame_points_max": int(np.max(counts)),
        "rtk_alignment": orb_stats["rtk_alignment"],
        "learned_track_quality": learned_stats["quality"],
    }
    (output / "conversion_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# PanoAir ORB-SLAM3 VI fused initialization\n\n"
        "All poses and points share one visual-inertial world and one global RTK Sim(3). "
        "Strict learned tracks provide accurate detail; native ORB landmarks fill frames "
        "with weak learned-track coverage. Inspect `initialization_orbslam3_vi_fused_total.ply`.\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
