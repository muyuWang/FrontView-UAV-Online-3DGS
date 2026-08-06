#!/usr/bin/env python3
"""Align ORB-SLAM3 VI poses/points once and export a canonical MODP dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


K = np.asarray([[320.0, 0.0, 480.0], [0.0, 320.0, 320.0], [0.0, 0.0, 1.0]])
WIDTH = 960
HEIGHT = 640


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--orb-output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Limit the extracted prefix before tracking-edge trimming."
    )
    parser.add_argument("--max-points-per-frame", type=int, default=10000)
    parser.add_argument("--min-orb-observations", type=int, default=3)
    parser.add_argument("--min-orb-found-ratio", type=float, default=0.25)
    parser.add_argument("--max-nearest-camera-distance-m", type=float, default=60.0)
    parser.add_argument("--alignment-threshold-m", type=float, default=0.35)
    parser.add_argument("--max-interpolation-gap-s", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_frame_rows(source: Path, extraction: Path) -> tuple[np.ndarray, list[Path], np.ndarray]:
    metadata = json.loads((extraction / "extraction.json").read_text(encoding="utf-8"))
    source_rows = []
    for line in (source / "frame_sequences" / "rgb.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, filename = line.split()[:2]
        source_rows.append((int(timestamp), filename))
    start = int(metadata["start_source_frame"])
    count = int(metadata["frame_count"])
    selected = source_rows[start : start + count]
    timestamps_ns = np.asarray([row[0] for row in selected], dtype=np.int64)
    if timestamps_ns.tolist() != metadata["timestamps_ns"]:
        raise RuntimeError("Extraction timestamps do not match the source rgb.txt slice")
    images = [source / "frame_sequences" / "pinhole" / row[1] for row in selected]
    return timestamps_ns, images, np.arange(start, start + count, dtype=np.int32)


def load_gt(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for line in (source / "Seq1_GT.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append([float(value) for value in line.split()])
    values = np.asarray(rows, dtype=np.float64)
    return values[:, 0], values[:, 1:4], values[:, 4:8]


def load_orb_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None]
    if rows.shape[1] != 8:
        raise RuntimeError(f"Expected 8 trajectory columns, got {rows.shape}")
    order = np.argsort(rows[:, 0])
    rows = rows[order]
    unique = np.concatenate(([True], np.diff(rows[:, 0]) > 0))
    rows = rows[unique]
    return rows[:, 0].astype(np.int64), rows[:, 1:4], rows[:, 4:8]


def load_map_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None]
    if rows.shape[1] < 6:
        raise RuntimeError(f"Expected point_id xyz observations found_ratio, got {rows.shape}")
    return (
        rows[:, 1:4],
        rows[:, 4].astype(np.int32),
        rows[:, 5].astype(np.float32),
        rows[:, 0].astype(np.int64),
    )


def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    variance = np.mean(np.sum(source_centered**2, axis=1))
    if variance <= 1.0e-12:
        raise RuntimeError("Trajectory has insufficient translation for Sim(3) alignment")
    scale = float(np.sum(singular * sign) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, similarity: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = similarity
    return (scale * (rotation @ points.T)).T + translation


def robust_similarity(
    source: np.ndarray, target: np.ndarray, threshold: float, seed: int
) -> tuple[tuple[float, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    if len(source) < 8:
        raise RuntimeError("At least eight tracked poses are required for alignment")
    rng = np.random.default_rng(seed)
    best = np.zeros(len(source), dtype=bool)
    iterations = min(4000, max(500, len(source) * 4))
    for _ in range(iterations):
        indices = rng.choice(len(source), 4, replace=False)
        try:
            candidate = fit_similarity(source[indices], target[indices])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        residuals = np.linalg.norm(apply_similarity(source, candidate) - target, axis=1)
        inliers = residuals <= threshold
        if inliers.sum() > best.sum() or (
            inliers.sum() == best.sum()
            and inliers.any()
            and np.median(residuals[inliers])
            < np.median(residuals[best])
        ):
            best = inliers
    if best.sum() < max(8, int(0.3 * len(source))):
        raise RuntimeError(
            f"Robust alignment retained only {best.sum()}/{len(source)} poses"
        )
    similarity = fit_similarity(source[best], target[best])
    residuals = np.linalg.norm(apply_similarity(source, similarity) - target, axis=1)
    best = residuals <= threshold
    similarity = fit_similarity(source[best], target[best])
    residuals = np.linalg.norm(apply_similarity(source, similarity) - target, axis=1)
    return similarity, best, residuals


def interpolate_orb_poses(
    frame_ns: np.ndarray,
    pose_ns: np.ndarray,
    translations: np.ndarray,
    quaternions: np.ndarray,
    max_gap: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    frame_s = frame_ns.astype(np.float64) / 1.0e9
    pose_s = pose_ns.astype(np.float64) / 1.0e9
    if frame_s[0] < pose_s[0] or frame_s[-1] > pose_s[-1]:
        raise RuntimeError(
            "ORB trajectory does not cover the retained sequence endpoints: "
            f"frames=[{frame_s[0]:.6f},{frame_s[-1]:.6f}], "
            f"poses=[{pose_s[0]:.6f},{pose_s[-1]:.6f}]"
        )
    insertion = np.searchsorted(pose_s, frame_s, side="left")
    left = np.maximum(insertion - 1, 0)
    right = np.minimum(insertion, len(pose_s) - 1)
    gaps = pose_s[right] - pose_s[left]
    if gaps.max(initial=0.0) > max_gap:
        raise RuntimeError(
            f"Largest ORB tracking gap {gaps.max():.3f}s exceeds {max_gap:.3f}s"
        )
    interpolated_t = np.column_stack(
        [np.interp(frame_s, pose_s, translations[:, axis]) for axis in range(3)]
    )
    interpolated_r = Slerp(pose_s, Rotation.from_quat(quaternions))(frame_s).as_matrix()
    exact = np.isin(frame_ns, pose_ns)
    return interpolated_t, interpolated_r, {
        "tracked_pose_count": int(len(pose_ns)),
        "exact_frame_pose_count": int(exact.sum()),
        "interpolated_frame_pose_count": int((~exact).sum()),
        "largest_pose_gap_s": float(np.diff(pose_s).max(initial=0.0)),
    }


def load_t_cam_imu(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    payload = None
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError:
        # ORB-SLAM3 uses OpenCV's non-standard %YAML:1.0 dialect.
        pass
    if isinstance(payload, dict) and "cam0" in payload:
        value = np.asarray(payload["cam0"]["T_cam_imu"], dtype=np.float64)
    else:
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            if not storage.isOpened():
                raise RuntimeError(f"Could not parse camera-IMU calibration {path}")
            t_body_camera = storage.getNode("IMU.T_b_c1").mat()
        finally:
            storage.release()
        if t_body_camera is None:
            raise RuntimeError(
                "Calibration must contain cam0.T_cam_imu or IMU.T_b_c1"
            )
        # ORB-SLAM3 stores body_from_camera, while make_camera_poses needs
        # camera_from_body to turn the EuRoC body trajectory into T_camera_world.
        value = np.linalg.inv(np.asarray(t_body_camera, dtype=np.float64))
    if value.shape != (4, 4):
        raise RuntimeError("Camera-IMU transform must be 4x4")
    if not np.isfinite(value).all():
        raise RuntimeError("Camera-IMU transform contains non-finite values")
    return value


def make_camera_poses(
    positions: np.ndarray,
    body_rotations: np.ndarray,
    t_cam_imu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t_world_body = np.repeat(np.eye(4)[None], len(positions), axis=0)
    t_world_body[:, :3, :3] = body_rotations
    t_world_body[:, :3, 3] = positions
    poses = np.asarray([t_cam_imu @ np.linalg.inv(pose) for pose in t_world_body])
    old_to_new = poses[0].copy()
    poses = np.asarray([pose @ np.linalg.inv(old_to_new) for pose in poses])
    return poses, old_to_new


def visible_ids(pose: np.ndarray, points: np.ndarray, maximum: int) -> np.ndarray:
    camera = (pose[:3, :3] @ points.T).T + pose[:3, 3]
    depth = camera[:, 2]
    projected = (K @ camera.T).T
    uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1.0e-12)
    valid = (
        np.isfinite(uv).all(axis=1)
        & (depth > 0.15)
        & (depth < 120.0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < WIDTH)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < HEIGHT)
    )
    ids = np.flatnonzero(valid)
    if len(ids) <= maximum:
        return ids.astype(np.int64)
    # Preserve image coverage first, then fill with the nearest remaining points.
    cells = (uv[ids, 1].astype(int) // 16) * math.ceil(WIDTH / 16) + uv[
        ids, 0
    ].astype(int) // 16
    order = np.argsort(depth[ids])
    _, first = np.unique(cells[order], return_index=True)
    selected_positions = order[np.sort(first)]
    selected = ids[selected_positions]
    if len(selected) < maximum:
        used = np.zeros(len(ids), dtype=bool)
        used[selected_positions] = True
        selected = np.concatenate((selected, ids[order[~used[order]]]))
    return np.sort(selected[:maximum].astype(np.int64))


def color_points(points: np.ndarray, poses: np.ndarray, images: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((len(points), 3), dtype=np.float64)
    counts = np.zeros(len(points), dtype=np.int32)
    # Sampling every third frame is enough for robust median-like averaging and limits I/O.
    for frame in range(0, len(poses), 3):
        ids = visible_ids(poses[frame], points, len(points))
        if not len(ids):
            continue
        image = cv2.imread(str(images[frame]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {images[frame]}")
        camera = (poses[frame, :3, :3] @ points[ids].T).T + poses[frame, :3, 3]
        projected = (K @ camera.T).T
        uv = np.rint(projected[:, :2] / projected[:, 2:3]).astype(int)
        uv[:, 0] = np.clip(uv[:, 0], 0, WIDTH - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, HEIGHT - 1)
        rgb = image[uv[:, 1], uv[:, 0], ::-1]
        sums[ids] += rgb
        counts[ids] += 1
    colors = np.full((len(points), 3), 128, dtype=np.uint8)
    observed = counts > 0
    colors[observed] = np.rint(sums[observed] / counts[observed, None]).clip(0, 255)
    return colors, counts


def write_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    ids: np.ndarray,
    observations: np.ndarray,
    front_view_count: np.ndarray,
) -> None:
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("orb_point_id", "u4"), ("orb_observations", "u2"),
            ("front_view_count", "u2"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    if ids.max(initial=0) > np.iinfo(np.uint32).max:
        raise RuntimeError("ORB point IDs exceed the PLY uint32 limit")
    vertices["orb_point_id"] = ids.astype(np.uint32)
    vertices["orb_observations"] = np.minimum(observations, 65535)
    vertices["front_view_count"] = np.minimum(front_view_count, 65535)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    extraction = args.extraction.resolve()
    orb_output = args.orb_output.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    frame_ns, images, source_indices = load_frame_rows(source, extraction)
    if args.max_frames > 0:
        frame_ns = frame_ns[: args.max_frames]
        images = images[: args.max_frames]
        source_indices = source_indices[: args.max_frames]
    pose_ns, orb_positions, orb_quaternions = load_orb_trajectory(
        orb_output / "frame_trajectory.txt"
    )
    points, point_observations, point_found_ratios, point_ids = load_map_points(
        orb_output / "map_points.txt"
    )
    covered = (frame_ns >= pose_ns[0]) & (frame_ns <= pose_ns[-1])
    if not covered.any():
        raise RuntimeError("ORB trajectory does not overlap the requested image slice")
    first_covered = int(np.flatnonzero(covered)[0])
    last_covered = int(np.flatnonzero(covered)[-1]) + 1
    trimmed_leading = first_covered
    trimmed_trailing = len(frame_ns) - last_covered
    frame_ns = frame_ns[first_covered:last_covered]
    images = images[first_covered:last_covered]
    source_indices = source_indices[first_covered:last_covered]
    frame_positions, frame_rotations, tracking_stats = interpolate_orb_poses(
        frame_ns,
        pose_ns,
        orb_positions,
        orb_quaternions,
        args.max_interpolation_gap_s,
    )
    tracking_stats["trimmed_untracked_leading_frames"] = trimmed_leading
    tracking_stats["trimmed_untracked_trailing_frames"] = trimmed_trailing

    gt_times, gt_positions, _ = load_gt(source)
    pose_times = pose_ns.astype(np.float64) / 1.0e9
    gt_at_tracked = np.column_stack(
        [np.interp(pose_times, gt_times, gt_positions[:, axis]) for axis in range(3)]
    )
    similarity, inliers, residuals = robust_similarity(
        orb_positions,
        gt_at_tracked,
        args.alignment_threshold_m,
        args.seed,
    )
    scale, alignment_rotation, translation = similarity
    aligned_positions = apply_similarity(frame_positions, similarity)
    aligned_rotations = alignment_rotation[None] @ frame_rotations
    aligned_points = apply_similarity(points, similarity)

    poses, old_to_new = make_camera_poses(
        aligned_positions,
        aligned_rotations,
        load_t_cam_imu(args.calibration),
    )
    homogeneous = np.concatenate(
        (aligned_points, np.ones((len(aligned_points), 1))), axis=1
    )
    normalized_points = (old_to_new @ homogeneous.T).T[:, :3]
    camera_centers = np.linalg.inv(poses)[:, :3, 3]
    nearest_camera_distances = cKDTree(camera_centers).query(
        normalized_points, k=1, workers=-1
    )[0]
    valid = (
        np.isfinite(normalized_points).all(axis=1)
        & (point_observations >= args.min_orb_observations)
        & (point_found_ratios >= args.min_orb_found_ratio)
        & (nearest_camera_distances <= args.max_nearest_camera_distance_m)
    )
    normalized_points = normalized_points[valid]
    point_observations = point_observations[valid]
    point_found_ratios = point_found_ratios[valid]
    point_ids = point_ids[valid]
    nearest_camera_distances = nearest_camera_distances[valid]
    colors, front_counts = color_points(normalized_points, poses, images)
    front_visible = front_counts > 0
    normalized_points = normalized_points[front_visible]
    point_observations = point_observations[front_visible]
    point_found_ratios = point_found_ratios[front_visible]
    point_ids = point_ids[front_visible]
    nearest_camera_distances = nearest_camera_distances[front_visible]
    colors = colors[front_visible]
    front_counts = front_counts[front_visible]

    for directory in (
        output / "rectified",
        output / "orb_point_clouds",
        output / "orb_point_ids",
        output / "preprocess",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    cameras = []
    per_frame_counts = []
    for frame, (pose, image, source_index) in enumerate(
        zip(poses, images, source_indices)
    ):
        link = output / "rectified" / f"aria_{frame:05d}.png"
        link.symlink_to(image.resolve())
        cameras.append(
            {
                "T_camera_world": pose.tolist(),
                "image": link.name,
                "timestamp": float(frame_ns[frame]) / 1.0e9,
                "frame_index": frame,
                "source_frame_index": int(source_index),
                "intrinsic": {"fx": 320.0, "fy": 320.0, "cx": 480.0, "cy": 320.0},
                "width": WIDTH,
                "height": HEIGHT,
                "focal": 320.0,
            }
        )
        ids = visible_ids(pose, normalized_points, args.max_points_per_frame)
        np.save(output / "orb_point_clouds" / f"point_cloud_{frame}.npy", normalized_points[ids].astype(np.float32))
        np.save(output / "orb_point_ids" / f"point_ids_{frame}.npy", ids)
        per_frame_counts.append(len(ids))

    trajectory = json.dumps({"cameras": cameras}, indent=2) + "\n"
    (output / "trajectory.json").write_text(trajectory, encoding="utf-8")
    (output / "trajectory_orb.json").write_text(trajectory, encoding="utf-8")
    np.save(output / "preprocess" / "source_frame_indices.npy", source_indices)
    np.save(output / "preprocess" / "global_sparse_points.npy", normalized_points.astype(np.float32))
    np.save(output / "preprocess" / "global_orbslam3_points.npy", normalized_points.astype(np.float32))
    np.save(output / "preprocess" / "point_observations.npy", point_observations)
    np.save(output / "preprocess" / "point_found_ratios.npy", point_found_ratios)
    np.save(output / "preprocess" / "nearest_camera_distances.npy", nearest_camera_distances)
    write_ply(
        output / "initialization_orbslam3_vi_total.ply",
        normalized_points.astype(np.float32),
        colors,
        point_ids,
        point_observations,
        front_counts,
    )
    write_ply(
        output / "initialization_orbslam3_vi_front_visible.ply",
        normalized_points.astype(np.float32),
        colors,
        point_ids,
        point_observations,
        front_counts,
    )

    centers = np.linalg.inv(poses)[:, :3, 3]
    stats = {
        "schema_version": 1,
        "method": "ORB-SLAM3 visual-inertial BA, one robust global Sim(3) to RTK",
        "pose_source": "orbslam3_vi",
        "sparse_world_geometry": "persistent",
        "source": str(source),
        "requested_removed_prefix_frames": 110,
        "removed_prefix_frames": int(source_indices[0]),
        "source_frame_end_inclusive": int(source_indices[-1]),
        "frame_count": len(cameras),
        "tracking": tracking_stats,
        "rtk_alignment": {
            "scale": scale,
            "rotation": alignment_rotation.tolist(),
            "translation": translation.tolist(),
            "tracked_pose_count": len(pose_ns),
            "inlier_count": int(inliers.sum()),
            "rmse_all_m": float(np.sqrt(np.mean(residuals**2))),
            "median_all_m": float(np.median(residuals)),
            "p90_all_m": float(np.percentile(residuals, 90)),
            "threshold_m": args.alignment_threshold_m,
        },
        "coordinate_contract": (
            "one Sim(3) is applied jointly to all ORB body poses and all ORB map points; "
            "the result is then normalized once to the first retained front camera"
        ),
        "point_count_total": int(len(normalized_points)),
        "point_count_front_visible": int(len(normalized_points)),
        "point_observations_median": float(np.median(point_observations)),
        "orb_point_filter": {
            "min_observations": args.min_orb_observations,
            "min_found_ratio": args.min_orb_found_ratio,
            "max_nearest_camera_distance_m": args.max_nearest_camera_distance_m,
            "require_front_pinhole_visibility": True,
        },
        "per_frame_points_min": int(np.min(per_frame_counts)),
        "per_frame_points_median": float(np.median(per_frame_counts)),
        "per_frame_points_mean": float(np.mean(per_frame_counts)),
        "per_frame_points_max": int(np.max(per_frame_counts)),
        "trajectory_span_m": np.ptp(centers, axis=0).tolist(),
        "trajectory_length_m": float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()),
        "intrinsics": {"fx": 320.0, "fy": 320.0, "cx": 480.0, "cy": 320.0},
    }
    (output / "conversion_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# PanoAir seq1 ORB-SLAM3 VI canonical input\n\n"
        "Source frames 0-109 are omitted. ORB-SLAM3 jointly estimates the visual-inertial "
        "trajectory and persistent sparse map. One robust global Sim(3) aligns poses and "
        "points together to RTK; no point is transformed with a per-frame pose. The total "
        "initialization cloud is `initialization_orbslam3_vi_total.ply`.\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
