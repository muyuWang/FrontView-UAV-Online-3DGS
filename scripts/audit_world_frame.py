#!/usr/bin/env python3
"""Audit whether per-frame sparse geometry has a persistent world identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument(
        "--canonical-reference",
        type=Path,
        default=Path("data/Online3DGS_PanoAir/seq1_colmap_consistent"),
    )
    parser.add_argument("--write-track-sidecars", action="store_true")
    return parser.parse_args()


def load_poses(dataset: Path, frames: int) -> list[np.ndarray]:
    trajectory = json.loads((dataset / "trajectory.json").read_text(encoding="utf-8"))
    # The historical key is misnamed: every repository consumer uses it as world-to-camera.
    return [
        np.asarray(camera["T_camera_world"], dtype=np.float64)
        for camera in trajectory["cameras"][:frames]
    ]


def camera_centers(poses: list[np.ndarray]) -> np.ndarray:
    return np.asarray([np.linalg.inv(pose)[:3, 3] for pose in poses])


def classify_contract(stats: dict[str, Any]) -> dict[str, Any]:
    pose_source = str(stats.get("pose_source", "unknown"))
    sparse_geometry = str(stats.get("sparse_world_geometry", "frame_local_reprojected"))
    method = str(stats.get("method", ""))
    if pose_source == "colmap" and sparse_geometry == "persistent":
        mode = "colmap_canonical"
        valid = True
        world_frame_id = "panoair_seq1_colmap_sim3_v1"
        depth_source = "colmap_track_world_point"
    elif pose_source == "gt" and sparse_geometry == "persistent" and "triangulation" in method:
        mode = "rtk_canonical"
        valid = True
        world_frame_id = "panoair_seq1_rtk_triangulated_v1"
        depth_source = "rtk_pose_orb_triangulation"
    else:
        mode = "hybrid_frame_local"
        valid = False
        world_frame_id = "invalid_frame_local_reprojection"
        depth_source = "colmap_camera_depth_reprojected_by_rtk_pose"
    return {
        "geometry_mode": mode,
        "contract_valid_for_permanent_birth": valid,
        "world_frame_id": world_frame_id,
        "pose_source": pose_source,
        "depth_source": depth_source,
        "sparse_world_geometry": sparse_geometry,
    }


def exact_row_lookup(global_points: np.ndarray) -> dict[bytes, int]:
    points = np.ascontiguousarray(global_points.astype(np.float32, copy=False))
    lookup: dict[bytes, int] = {}
    for index, point in enumerate(points):
        key = point.tobytes()
        # Exact duplicate coordinates are one geometric world identity for this audit.
        # Fresh conversions preserve distinct source point IDs in sidecars.
        lookup.setdefault(key, index)
    return lookup


def exact_ids(points: np.ndarray, lookup: dict[bytes, int]) -> np.ndarray:
    ids = np.empty(points.shape[0], dtype=np.int64)
    for index, point in enumerate(np.ascontiguousarray(points.astype(np.float32, copy=False))):
        key = point.tobytes()
        if key not in lookup:
            raise ValueError(f"Point row {index} has no exact global identity")
        ids[index] = lookup[key]
    return ids


def global_sparse_map(dataset: Path, mode: str) -> np.ndarray | None:
    name = (
        "global_colmap_points_sim3.npy"
        if mode == "colmap_canonical"
        else "global_sparse_points.npy"
    )
    path = dataset / "preprocess" / name
    return np.load(path) if path.exists() else None


def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((points.shape[0], 1))], axis=1
    )
    return (pose @ homogeneous.T).T[:, :3]


def recover_frame_ids(
    dataset: Path,
    mode: str,
    frame: int,
    points: np.ndarray,
    lookup: dict[bytes, int] | None,
    canonical_reference: Path,
    pose: np.ndarray,
    reference_pose: np.ndarray | None,
) -> tuple[np.ndarray, str, dict[str, float] | None]:
    sidecar = dataset / "orb_point_ids" / f"point_ids_{frame}.npy"
    if sidecar.exists():
        ids = np.load(sidecar).astype(np.int64)
        if ids.shape != (points.shape[0],):
            raise ValueError(f"Identity sidecar shape mismatch at frame {frame}")
        return ids, "saved_point_id_sidecar", None
    if mode in {"colmap_canonical", "rtk_canonical"}:
        if lookup is None:
            raise ValueError("Persistent dataset has no recoverable global sparse map")
        return exact_ids(points, lookup), "exact_global_float32_coordinate_identity", None

    reference_points = np.load(
        canonical_reference / "orb_point_clouds" / f"point_cloud_{frame}.npy"
    )
    if reference_points.shape != points.shape:
        raise ValueError(
            f"Hybrid/canonical row count differs at frame {frame}: "
            f"{points.shape[0]} vs {reference_points.shape[0]}"
        )
    if lookup is None:
        raise ValueError("Hybrid identity recovery requires the canonical global point map")
    reference_ids = exact_ids(reference_points, lookup)
    validation = None
    if reference_pose is not None and points.size:
        source_camera = transform_points(pose, points)
        reference_camera = transform_points(reference_pose, reference_points)
        errors = np.linalg.norm(source_camera - reference_camera, axis=1)
        validation = {
            "camera_coordinate_row_validation_median_m": float(np.median(errors)),
            "camera_coordinate_row_validation_p95_m": float(np.percentile(errors, 95)),
        }
        if not np.isfinite(errors).all() or np.percentile(errors, 95) > 1e-3:
            raise ValueError(
                f"Cannot validate row-aligned hybrid identities at frame {frame}: "
                f"camera-coordinate p95={np.percentile(errors, 95):.6g} m"
            )
    return reference_ids, "validated_row_aligned_colmap_identity", validation


def percentile_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "median_m": None, "p95_m": None}
    return {
        "count": int(values.size),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95)),
    }


def track_dispersion(
    ids: np.ndarray, points: np.ndarray, frame_ids: np.ndarray, start: int, end: int
) -> dict[str, float | int | None]:
    selected = (frame_ids >= start) & (frame_ids < end)
    selected_ids = ids[selected]
    selected_points = points[selected]
    if selected_ids.size == 0:
        return percentile_summary(np.empty(0))
    order = np.argsort(selected_ids, kind="stable")
    selected_ids = selected_ids[order]
    selected_points = selected_points[order]
    boundaries = np.flatnonzero(np.diff(selected_ids)) + 1
    groups = np.split(np.arange(selected_ids.size), boundaries)
    distances = []
    track_count = 0
    for group in groups:
        if group.size < 2:
            continue
        center = np.median(selected_points[group], axis=0)
        distances.append(np.linalg.norm(selected_points[group] - center, axis=1))
        track_count += 1
    result = percentile_summary(
        np.concatenate(distances) if distances else np.empty(0)
    )
    result["multi_observation_track_count"] = track_count
    return result


def adjacent_identity_distance(
    frame_points: list[np.ndarray], frame_ids: list[np.ndarray], start: int, end: int
) -> dict[str, float | int | None]:
    distances = []
    for frame in range(max(start + 1, 1), min(end, len(frame_points))):
        previous = {int(identity): point for identity, point in zip(frame_ids[frame - 1], frame_points[frame - 1])}
        current = {int(identity): point for identity, point in zip(frame_ids[frame], frame_points[frame])}
        common = previous.keys() & current.keys()
        if common:
            distances.extend(np.linalg.norm(previous[key] - current[key]) for key in common)
    return percentile_summary(np.asarray(distances))


def auxiliary_nearest_neighbor(
    frame_points: list[np.ndarray], start: int, end: int
) -> dict[str, float | int | None]:
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return {"count": 0, "median_m": None, "p95_m": None, "status": "scipy_unavailable"}
    distances = []
    for frame in range(max(start + 1, 1), min(end, len(frame_points))):
        previous = frame_points[frame - 1]
        current = frame_points[frame]
        if previous.size and current.size:
            values, _ = cKDTree(previous).query(current, k=1, workers=-1)
            distances.append(values)
    result = percentile_summary(np.concatenate(distances) if distances else np.empty(0))
    result["warning"] = "auxiliary_only_not_point_identity"
    return result


def umeyama_alignment(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    count = min(len(source), len(target))
    source = source[:count]
    target = target[:count]
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / count
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    variance = np.mean(np.sum(source_centered**2, axis=1))
    scale = float(np.sum(singular * sign) / max(variance, 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = scale * (rotation @ source.T).T + translation
    errors = np.linalg.norm(aligned - target, axis=1)
    return {
        "reference": "COLMAP canonical camera centers",
        "count": count,
        "scale": scale,
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    contract = report["contract"]
    lines = [
        "# World-frame audit",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Geometry mode: `{contract['geometry_mode']}`",
        f"- Pose source: `{contract['pose_source']}`",
        f"- Depth source: `{contract['depth_source']}`",
        f"- Permanent-birth contract: **{'valid' if contract['contract_valid_for_permanent_birth'] else 'invalid'}**",
        f"- Pose convention: `{report['pose_convention']}`",
        f"- Identity recovery: `{report['identity_recovery']}`",
        "",
        "| Window | Track dispersion median/p95 (m) | Adjacent identity median/p95 (m) |",
        "|---|---:|---:|",
    ]
    for name, metrics in report["windows"].items():
        track = metrics["track_world_dispersion"]
        adjacent = metrics["adjacent_common_identity_world_distance"]
        lines.append(
            f"| {name} | {track['median_m']} / {track['p95_m']} | "
            f"{adjacent['median_m']} / {adjacent['p95_m']} |"
        )
    alignment = report["global_sim3_trajectory_alignment"]
    lines.extend(
        [
            "",
            "## Global Sim(3) trajectory alignment",
            "",
            f"RMSE/median/p90: {alignment['rmse_m']:.6f} / "
            f"{alignment['median_m']:.6f} / {alignment['p90_m']:.6f} m.",
            "",
            "Nearest-neighbor statistics in JSON are auxiliary only and are not used as identity evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    reference = args.canonical_reference.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stats = json.loads((dataset / "conversion_stats.json").read_text(encoding="utf-8"))
    contract = classify_contract(stats)
    poses = load_poses(dataset, args.frames)
    reference_poses = load_poses(reference, args.frames)
    frame_count = min(args.frames, len(poses))
    poses = poses[:frame_count]
    reference_poses = reference_poses[:frame_count]
    global_points = global_sparse_map(dataset, contract["geometry_mode"])
    if contract["geometry_mode"] == "hybrid_frame_local":
        global_points = np.load(
            reference / "preprocess" / "global_colmap_points_sim3.npy"
        )
    lookup = exact_row_lookup(global_points) if global_points is not None else None
    frame_points: list[np.ndarray] = []
    frame_identities: list[np.ndarray] = []
    recovery_methods: set[str] = set()
    validation_values = []
    sidecar_dir = dataset / "orb_point_ids"
    if args.write_track_sidecars:
        sidecar_dir.mkdir(exist_ok=True)
    for frame in range(frame_count):
        points = np.load(dataset / "orb_point_clouds" / f"point_cloud_{frame}.npy").astype(np.float64)
        identities, method, validation = recover_frame_ids(
            dataset,
            contract["geometry_mode"],
            frame,
            points,
            lookup,
            reference,
            poses[frame],
            reference_poses[frame],
        )
        if args.write_track_sidecars:
            sidecar_path = sidecar_dir / f"point_ids_{frame}.npy"
            if sidecar_path.exists():
                existing = np.load(sidecar_path)
                if not np.array_equal(existing, identities):
                    raise ValueError(f"Refusing to overwrite conflicting sidecar {sidecar_path}")
            else:
                np.save(sidecar_path, identities)
        frame_points.append(points)
        frame_identities.append(identities)
        recovery_methods.add(method)
        if validation is not None:
            validation_values.append(validation)

    observation_points = np.concatenate(frame_points)
    observation_ids = np.concatenate(frame_identities)
    observation_frames = np.concatenate(
        [np.full(ids.shape, frame, dtype=np.int32) for frame, ids in enumerate(frame_identities)]
    )
    windows = {
        "all": (0, frame_count),
        "frames_120_150": (min(120, frame_count), min(151, frame_count)),
    }
    window_metrics = {}
    for name, (start, end) in windows.items():
        window_metrics[name] = {
            "frame_range_half_open": [start, end],
            "track_world_dispersion": track_dispersion(
                observation_ids, observation_points, observation_frames, start, end
            ),
            "adjacent_common_identity_world_distance": adjacent_identity_distance(
                frame_points, frame_identities, start, end
            ),
            "auxiliary_nearest_neighbor_not_identity": auxiliary_nearest_neighbor(
                frame_points, start, end
            ),
        }
    report = {
        "schema_version": 1,
        "dataset": str(dataset),
        "frame_count": frame_count,
        "contract": contract,
        "pose_convention": (
            "trajectory key T_camera_world is historical/misnamed; matrices are world_to_camera, "
            "camera center = inverse(matrix)[:3,3]"
        ),
        "identity_recovery": sorted(recovery_methods),
        "identity_validation": {
            "method": "same exporter selection order plus row-wise camera-coordinate equality",
            "frames_checked": len(validation_values),
            "maximum_p95_m": max(
                (item["camera_coordinate_row_validation_p95_m"] for item in validation_values),
                default=0.0,
            ),
        },
        "metadata_alignment": stats.get("rtk_alignment"),
        "global_sim3_trajectory_alignment": umeyama_alignment(
            camera_centers(poses), camera_centers(reference_poses)
        ),
        "windows": window_metrics,
    }
    (output / "world_frame_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output / "world_frame_audit.md").write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "contract": report["contract"],
                "global_sim3_trajectory_alignment": report[
                    "global_sim3_trajectory_alignment"
                ],
                "windows": report["windows"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
