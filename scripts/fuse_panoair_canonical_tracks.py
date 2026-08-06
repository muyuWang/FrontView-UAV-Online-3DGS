#!/usr/bin/env python3
"""Fuse filtered COLMAP tracks with learned tracks in one canonical world."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pycolmap
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

try:
    from scripts.build_panoair_learned_tracks import (
        HEIGHT,
        K,
        WIDTH,
        camera_centers,
        maximum_triangulation_angle_deg,
        percentile_summary,
        point_position_std,
        projection_errors,
    )
except ModuleNotFoundError:
    from build_panoair_learned_tracks import (
        HEIGHT,
        K,
        WIDTH,
        camera_centers,
        maximum_triangulation_angle_deg,
        percentile_summary,
        point_position_std,
        projection_errors,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap-source", type=Path, required=True)
    parser.add_argument("--learned-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=110)
    parser.add_argument("--num-frames", type=int, default=500)
    parser.add_argument("--min-colmap-observations", type=int, default=3)
    parser.add_argument("--max-colmap-error-px", type=float, default=1.5)
    parser.add_argument("--min-triangulation-angle-deg", type=float, default=1.5)
    parser.add_argument("--max-nearest-camera-distance-m", type=float, default=60.0)
    parser.add_argument("--max-position-std-m", type=float, default=1.0)
    parser.add_argument("--max-relative-position-std", type=float, default=0.05)
    parser.add_argument("--learned-dedup-radius-m", type=float, default=0.04)
    parser.add_argument("--max-points-per-frame", type=int, default=5000)
    parser.add_argument("--grid-size-px", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def source_frame_from_name(name: str) -> int:
    return int(Path(name).stem.rsplit("_", 1)[-1])


def load_learned(learned_source: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    points = np.load(learned_source / "preprocess" / "global_sparse_points.npy").astype(
        np.float32
    )
    ply = PlyData.read(learned_source / "initialization_multiview_tracks.ply")
    vertex = ply["vertex"].data
    colors = np.column_stack((vertex["red"], vertex["green"], vertex["blue"])).astype(
        np.uint8
    )
    if len(colors) != len(points):
        raise RuntimeError("Learned PLY and point table have different rows")
    with np.load(learned_source / "preprocess" / "track_observations.npz") as data:
        observations = {name: data[name] for name in data.files}
    quality = {
        "track_length": np.asarray(vertex["track_length"], dtype=np.int32),
        "reprojection_error_px": np.asarray(vertex["reprojection_error_px"], dtype=np.float32),
        "triangulation_angle_deg": np.asarray(vertex["triangulation_angle_deg"], dtype=np.float32),
        "position_std_m": np.asarray(vertex["position_std_m"], dtype=np.float32),
        "observations": observations,
    }
    return points, colors, quality


def filter_colmap_points(
    reconstruction: pycolmap.Reconstruction,
    canonical_points: np.ndarray,
    poses: np.ndarray,
    start_frame: int,
    end_frame: int,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    point_ids = np.asarray(list(reconstruction.point3D_ids()), dtype=np.int64)
    if canonical_points.shape != (len(point_ids), 3):
        raise RuntimeError("COLMAP point IDs do not align with the canonical point table")
    centers = camera_centers(poses)
    records = []
    outcomes: dict[str, int] = {}

    def reject(reason: str) -> None:
        outcomes[reason] = outcomes.get(reason, 0) + 1

    for row, point_id in enumerate(point_ids.tolist()):
        point3d = reconstruction.point3D(int(point_id))
        if float(point3d.error) > float(args.max_colmap_error_px):
            reject("reprojection_error")
            continue
        observations = []
        raw_centers = []
        pixels = []
        for element in point3d.track.elements:
            image = reconstruction.image(element.image_id)
            source_frame = source_frame_from_name(image.name)
            if not start_frame <= source_frame < end_frame:
                continue
            observations.append(source_frame - start_frame)
            raw_centers.append(np.asarray(image.projection_center(), dtype=np.float64))
            pixels.append(
                np.asarray(image.point2D(element.point2D_idx).xy, dtype=np.float64)
            )
        if len(observations) < int(args.min_colmap_observations):
            reject("insufficient_observations")
            continue
        angle = maximum_triangulation_angle_deg(
            np.asarray(point3d.xyz, dtype=np.float64), np.asarray(raw_centers)
        )
        if angle < float(args.min_triangulation_angle_deg):
            reject("triangulation_angle")
            continue
        point = canonical_points[row].astype(np.float64)
        nearest = float(np.linalg.norm(centers - point[None], axis=1).min())
        if nearest > float(args.max_nearest_camera_distance_m):
            reject("nearest_camera_distance")
            continue
        observation_poses = [poses[frame_id] for frame_id in observations]
        errors, depths = projection_errors(point, observation_poses, pixels)
        positive = np.isfinite(errors) & (depths > 0.3) & (depths < 120.0)
        if int(positive.sum()) < int(args.min_colmap_observations):
            reject("cheirality_or_depth")
            continue
        positive_poses = [pose for pose, keep in zip(observation_poses, positive) if keep]
        pixel_errors = np.full((len(positive_poses),), max(float(point3d.error), 0.25))
        position_std = point_position_std(point, positive_poses, pixel_errors)
        median_depth = float(np.median(depths[positive]))
        relative_std = position_std / max(median_depth, 1.0e-8)
        if position_std > float(args.max_position_std_m):
            reject("position_std")
            continue
        if relative_std > float(args.max_relative_position_std):
            reject("relative_position_std")
            continue
        kept_observations = [
            (frame_id, pixel.astype(np.float32))
            for frame_id, pixel, keep in zip(observations, pixels, positive)
            if keep
        ]
        records.append(
            {
                "point": point.astype(np.float32),
                "color": np.asarray(point3d.color, dtype=np.uint8),
                "source": 0,
                "source_point_id": int(point_id),
                "track_length": len(kept_observations),
                "reprojection_error_px": float(point3d.error),
                "triangulation_angle_deg": angle,
                "position_std_m": position_std,
                "observations": kept_observations,
            }
        )
    return records, dict(sorted(outcomes.items()))


def append_learned_points(
    records: list[dict],
    points: np.ndarray,
    colors: np.ndarray,
    quality: dict,
    dedup_radius: float,
) -> tuple[list[dict], dict[int, int], int]:
    original_points = np.asarray([record["point"] for record in records], dtype=np.float64)
    tree = cKDTree(original_points) if len(original_points) else None
    keep = np.ones((len(points),), dtype=bool)
    if tree is not None and len(points):
        distances, _ = tree.query(points, k=1)
        keep &= distances > float(dedup_radius)
    remap = {}
    observations = quality["observations"]
    observations_by_point: dict[int, list[tuple[int, np.ndarray]]] = {}
    for point_id, frame_id, uv in zip(
        observations["point_ids"], observations["frame_ids"], observations["uv"]
    ):
        observations_by_point.setdefault(int(point_id), []).append(
            (int(frame_id), np.asarray(uv, dtype=np.float32))
        )
    for learned_id in np.flatnonzero(keep).tolist():
        fused_id = len(records)
        remap[learned_id] = fused_id
        records.append(
            {
                "point": points[learned_id].astype(np.float32),
                "color": colors[learned_id],
                "source": 1,
                "source_point_id": learned_id,
                "track_length": int(quality["track_length"][learned_id]),
                "reprojection_error_px": float(
                    quality["reprojection_error_px"][learned_id]
                ),
                "triangulation_angle_deg": float(
                    quality["triangulation_angle_deg"][learned_id]
                ),
                "position_std_m": float(quality["position_std_m"][learned_id]),
                "observations": observations_by_point.get(learned_id, []),
            }
        )
    return records, remap, int((~keep).sum())


def visible_point_ids(
    pose: np.ndarray,
    points: np.ndarray,
    max_points: int,
    grid_size: int,
) -> np.ndarray:
    camera = (pose[:3, :3] @ points.T).T + pose[:3, 3]
    depth = camera[:, 2]
    screen = (K @ camera.T).T
    uv = screen[:, :2] / np.maximum(screen[:, 2:3], 1.0e-12)
    valid = (
        np.isfinite(uv).all(axis=1)
        & (depth > 0.3)
        & (depth < 120.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < WIDTH)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < HEIGHT)
    )
    ids = np.flatnonzero(valid)
    if len(ids) <= max_points:
        return ids.astype(np.int64)
    grid_width = math.ceil(WIDTH / max(grid_size, 1))
    cells = (
        (uv[ids, 1].astype(np.int64) // grid_size) * grid_width
        + uv[ids, 0].astype(np.int64) // grid_size
    )
    order = np.argsort(depth[ids])
    _, first = np.unique(cells[order], return_index=True)
    selected_positions = order[np.sort(first)]
    selected = ids[selected_positions]
    if len(selected) < max_points:
        chosen = np.zeros((len(ids),), dtype=bool)
        chosen[selected_positions] = True
        selected = np.concatenate((selected, ids[order[~chosen[order]]]))
    return np.sort(selected[:max_points].astype(np.int64))


def write_ply(path: Path, records: list[dict]) -> None:
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("point_id", "u4"), ("source", "u1"), ("track_length", "u2"),
        ("reprojection_error_px", "f4"), ("triangulation_angle_deg", "f4"),
        ("position_std_m", "f4"),
    ]
    vertices = np.empty((len(records),), dtype=dtype)
    points = np.asarray([record["point"] for record in records])
    colors = np.asarray([record["color"] for record in records])
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    vertices["point_id"] = np.arange(len(records), dtype=np.uint32)
    for name in (
        "source",
        "track_length",
        "reprojection_error_px",
        "triangulation_angle_deg",
        "position_std_m",
    ):
        vertices[name] = [record[name] for record in records]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def main() -> None:
    args = parse_args()
    start_time = time.monotonic()
    learned_stats = json.loads(
        (args.learned_source / "conversion_stats.json").read_text(encoding="utf-8")
    )
    cameras = json.loads(
        (args.learned_source / "trajectory_orb.json").read_text(encoding="utf-8")
    )["cameras"]
    if len(cameras) != int(args.num_frames):
        raise RuntimeError("Learned source frame count does not match --num-frames")
    poses = np.asarray([camera["T_camera_world"] for camera in cameras], dtype=np.float64)
    reconstruction = pycolmap.Reconstruction(
        args.colmap_source / "preprocess" / "colmap_model"
    )
    canonical = np.load(
        args.colmap_source / "preprocess" / "global_colmap_points_sim3.npy"
    ).astype(np.float64)
    records, filter_outcomes = filter_colmap_points(
        reconstruction,
        canonical,
        poses,
        int(args.start_frame),
        int(args.start_frame + args.num_frames),
        args,
    )
    original_count = len(records)
    learned_points, learned_colors, learned_quality = load_learned(args.learned_source)
    records, _, learned_duplicates = append_learned_points(
        records,
        learned_points,
        learned_colors,
        learned_quality,
        args.learned_dedup_radius_m,
    )
    learned_count = len(records) - original_count
    print(
        f"Fused points: COLMAP={original_count} learned={learned_count} "
        f"deduplicated={learned_duplicates}",
        flush=True,
    )

    if args.output.exists():
        if not args.force:
            raise FileExistsError(args.output)
        shutil.rmtree(args.output)
    staging = args.output.with_name(args.output.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    for relative in ("rectified", "orb_point_clouds", "orb_point_ids", "preprocess"):
        (staging / relative).mkdir(parents=True, exist_ok=True)
    for frame_id, camera in enumerate(cameras):
        source_image = args.learned_source / "rectified" / camera["image"]
        (staging / "rectified" / camera["image"]).symlink_to(source_image.resolve())
    trajectory = json.dumps({"cameras": cameras}, indent=2) + "\n"
    (staging / "trajectory.json").write_text(trajectory, encoding="utf-8")
    (staging / "trajectory_orb.json").write_text(trajectory, encoding="utf-8")

    points = np.asarray([record["point"] for record in records], dtype=np.float32)
    np.save(staging / "preprocess" / "global_sparse_points.npy", points)
    np.save(staging / "preprocess" / "global_colmap_points_sim3.npy", points)
    np.save(
        staging / "preprocess" / "point_sources.npy",
        np.asarray([record["source"] for record in records], dtype=np.uint8),
    )
    observation_point_ids = []
    observation_frame_ids = []
    observation_uv = []
    for point_id, record in enumerate(records):
        for frame_id, uv in record["observations"]:
            observation_point_ids.append(point_id)
            observation_frame_ids.append(frame_id)
            observation_uv.append(uv)
    np.savez(
        staging / "preprocess" / "track_observations.npz",
        point_ids=np.asarray(observation_point_ids, dtype=np.int64),
        frame_ids=np.asarray(observation_frame_ids, dtype=np.int32),
        uv=np.asarray(observation_uv, dtype=np.float32),
    )

    per_frame_counts = []
    for frame_id, pose in enumerate(poses):
        ids = visible_point_ids(
            pose,
            points,
            int(args.max_points_per_frame),
            int(args.grid_size_px),
        )
        np.save(staging / "orb_point_clouds" / f"point_cloud_{frame_id}.npy", points[ids])
        np.save(staging / "orb_point_ids" / f"point_ids_{frame_id}.npy", ids)
        per_frame_counts.append(len(ids))
    write_ply(staging / "initialization_fused_canonical.ply", records)

    centers = camera_centers(poses)
    elapsed = time.monotonic() - start_time
    stats = {
        "schema_version": 1,
        "method": "COLMAP visual poses with filtered COLMAP track triangulation and DISK-LightGlue multi-view triangulation",
        "pose_source": "colmap",
        "sparse_world_geometry": "persistent",
        "colmap_source": str(args.colmap_source.resolve()),
        "learned_source": str(args.learned_source.resolve()),
        "removed_prefix_frames": int(args.start_frame),
        "frame_count": len(cameras),
        "world_frame": "one COLMAP canonical world shared by poses and all points",
        "per_frame_export_mode": "canonical visibility with real observations retained in preprocess/track_observations.npz",
        "trajectory_span_m": np.ptp(centers, axis=0).tolist(),
        "trajectory_length_m": float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()),
        "colmap_filter_outcomes": filter_outcomes,
        "filtered_colmap_points": original_count,
        "learned_input_points": int(learned_stats["accepted_point_count"]),
        "learned_deduplicated_points": learned_duplicates,
        "learned_appended_points": learned_count,
        "global_point_count": len(records),
        "point_bounds_min": points.min(axis=0).tolist(),
        "point_bounds_max": points.max(axis=0).tolist(),
        "track_length": percentile_summary(
            np.asarray([record["track_length"] for record in records], dtype=np.float64)
        ),
        "reprojection_error_px": percentile_summary(
            np.asarray(
                [record["reprojection_error_px"] for record in records], dtype=np.float64
            )
        ),
        "triangulation_angle_deg": percentile_summary(
            np.asarray(
                [record["triangulation_angle_deg"] for record in records], dtype=np.float64
            )
        ),
        "position_std_m": percentile_summary(
            np.asarray([record["position_std_m"] for record in records], dtype=np.float64)
        ),
        "per_frame_points": percentile_summary(np.asarray(per_frame_counts, dtype=np.float64)),
        "frames_below_100_points": int(np.sum(np.asarray(per_frame_counts) < 100)),
        "thresholds": {
            "min_colmap_observations": args.min_colmap_observations,
            "max_colmap_error_px": args.max_colmap_error_px,
            "min_triangulation_angle_deg": args.min_triangulation_angle_deg,
            "max_nearest_camera_distance_m": args.max_nearest_camera_distance_m,
            "max_position_std_m": args.max_position_std_m,
            "max_relative_position_std": args.max_relative_position_std,
            "learned_dedup_radius_m": args.learned_dedup_radius_m,
        },
        "preprocess_wall_time_s": elapsed,
    }
    (staging / "conversion_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    (staging / "README.md").write_text(
        "# PanoAir filtered canonical fusion\n\n"
        f"The first {args.start_frame} source frames are omitted. High-confidence real "
        "COLMAP tracks and deduplicated DISK-LightGlue tracks share one immutable "
        "COLMAP world. MODP input files preserve the existing dataset contract.\n",
        encoding="utf-8",
    )
    staging.replace(args.output)
    print(f"Wrote {args.output} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
