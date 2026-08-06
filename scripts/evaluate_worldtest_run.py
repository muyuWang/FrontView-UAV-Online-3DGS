#!/usr/bin/env python3
"""Aggregate takeoff image metrics and robust lawn-plane geometry diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData
from sklearn.linear_model import LinearRegression, RANSACRegressor
from scipy.spatial import cKDTree
import yaml


WINDOWS = {
    "frames_0_80": (0, 80),
    "frames_80_120": (80, 120),
    "frames_120_130": (120, 130),
    "frames_130_140": (130, 140),
    "frames_140_150": (140, 150),
    "frames_150_200": (150, 200),
    "frames_0_200": (0, 200),
    "frames_180_260": (180, 260),
    "frames_200_300": (200, 300),
    "frames_300_400": (300, 400),
    "frames_400_500": (400, 500),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--ply", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ground-threshold-m", type=float, default=0.03)
    parser.add_argument("--lawn-reference-frame", type=int, default=120)
    parser.add_argument(
        "--lawn-image-roi",
        default="0,320,960,640",
        help="Fixed x0,y0,x1,y1 image ROI used to select visible lawn points.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mean_fields(rows):
    if not rows:
        return {"frame_count": 0, "psnr": None, "ssim_downscaled": None, "coverage": None}
    return {
        "frame_count": len(rows),
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "ssim_downscaled": float(
            np.mean([row["ssim_downscaled"] for row in rows])
        ),
        "coverage": float(
            np.mean([row["nonblack_coverage_percent"] for row in rows])
        ),
        "near_side_psnr": float(np.mean([row["near_side_psnr"] for row in rows])),
    }


def image_windows(metrics):
    frames = metrics.get("frames", [])
    if not frames:
        frames = metrics.get("trajectory_metrics", {}).get("frames", [])
    if not frames and isinstance(metrics.get("video"), dict):
        frames = metrics["video"].get("frames", [])
    output = {}
    for name, (start, end) in WINDOWS.items():
        rows = [row for row in frames if start <= int(row["frame_index"]) < end]
        output[name] = mean_fields(rows)
    return output


def load_points_and_colors(path):
    vertex = PlyData.read(str(path))["vertex"].data
    points = np.stack([vertex[axis] for axis in ("x", "y", "z")], axis=1).astype(
        np.float64
    )
    dc = np.stack(
        [vertex[f"f_dc_{index}"] for index in range(3)], axis=1
    ).astype(np.float64)
    colors = np.clip(dc * 0.28209479177387814 + 0.5, 0.0, 1.0)
    return points, colors


def canonical_nearest_distance(run_dir, points):
    """Measure auxiliary map drift to the immutable canonical COLMAP cloud."""
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    dataset_path = Path(config["Dataset"]["dataset_path"])
    canonical_path = dataset_path / "preprocess" / "global_colmap_points_sim3.npy"
    if not canonical_path.is_file():
        return {
            "status": "canonical_cloud_unavailable",
            "warning": "auxiliary_only_not_point_identity",
        }
    canonical = np.asarray(np.load(canonical_path), dtype=np.float64)
    if canonical.ndim != 2 or canonical.shape[1] < 3:
        raise ValueError("Canonical COLMAP cloud must be an Nx3-or-wider array")
    canonical = canonical[:, :3]
    finite_points = points[np.isfinite(points).all(axis=1)]
    finite_canonical = canonical[np.isfinite(canonical).all(axis=1)]
    distances, _ = cKDTree(finite_canonical).query(
        finite_points, k=1, workers=-1
    )
    return {
        "status": "ok",
        "map_point_count": int(len(finite_points)),
        "canonical_point_count": int(len(finite_canonical)),
        "median_m": float(np.median(distances)),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "max_m": float(np.max(distances)),
        "warning": "auxiliary_only_not_point_identity_and_includes_split_children",
    }


def select_lawn_image_roi(run_dir, points, colors, frame_id, roi_text):
    roi = tuple(float(value) for value in roi_text.split(","))
    if len(roi) != 4 or roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise ValueError("lawn-image-roi must be x0,y0,x1,y1")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    dataset = config.get("Testset", config["Dataset"])
    trajectory_name = (
        "trajectory_orb.json"
        if dataset.get("data_source", "aria") == "orb"
        else "trajectory.json"
    )
    trajectory = json.loads(
        (Path(dataset["dataset_path"]) / trajectory_name).read_text(encoding="utf-8")
    )["cameras"]
    if not 0 <= int(frame_id) < len(trajectory):
        raise ValueError("lawn-reference-frame is outside the trajectory")
    camera = trajectory[int(frame_id)]
    pose = np.asarray(camera["T_camera_world"], dtype=np.float64)
    intrinsic = camera["intrinsic"]
    intrinsics = np.asarray(
        [
            [intrinsic["fx"], 0.0, intrinsic["cx"]],
            [0.0, intrinsic["fy"], intrinsic["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    camera_points = homogeneous @ pose.T
    screen = camera_points[:, :3] @ intrinsics.T
    uv = screen[:, :2] / np.maximum(camera_points[:, 2:3], 1.0e-12)
    x0, y0, x1, y1 = roi
    mask = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(camera_points[:, 2])
        & (camera_points[:, 2] > 0.0)
        & (uv[:, 0] >= x0)
        & (uv[:, 0] < x1)
        & (uv[:, 1] >= y0)
        & (uv[:, 1] < y1)
    )
    return points[mask], colors[mask], {
        "reference_frame": int(frame_id),
        "image_bbox_xyxy": list(roi),
        "projected_point_count": int(np.count_nonzero(mask)),
        "selection": "projected fixed image ROI, followed by green SH0 mask",
    }


def fit_ground(points, colors, threshold, seed):
    finite = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    green = (
        (colors[:, 1] > colors[:, 0] * 1.08)
        & (colors[:, 1] > colors[:, 2] * 1.08)
        & (colors[:, 1] > 0.18)
    )
    indices = np.flatnonzero(finite & green)
    if len(indices) < 100:
        indices = np.flatnonzero(finite)
    rng = np.random.default_rng(seed)
    if len(indices) > 50000:
        indices = rng.choice(indices, size=50000, replace=False)
    selected = points[indices]
    best = None
    for dependent in range(3):
        independent = [axis for axis in range(3) if axis != dependent]
        estimator = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=float(threshold),
            max_trials=500,
            min_samples=3,
            random_state=seed,
        )
        estimator.fit(selected[:, independent], selected[:, dependent])
        coefficients = np.zeros((3,), dtype=np.float64)
        coefficients[independent] = estimator.estimator_.coef_
        coefficients[dependent] = -1.0
        offset = float(estimator.estimator_.intercept_)
        norm = float(np.linalg.norm(coefficients))
        distances = np.abs(selected @ coefficients + offset) / max(norm, 1.0e-12)
        inliers = distances <= float(threshold)
        candidate = {
            "coefficients": coefficients / norm,
            "offset": offset / norm,
            "distances": distances,
            "inliers": inliers,
            "count": int(np.count_nonzero(inliers)),
            "selected": selected,
        }
        if best is None or candidate["count"] > best["count"]:
            best = candidate
    inlier_distances = best["distances"][best["inliers"]]
    all_distances = best["distances"]
    return {
        "candidate_green_points": int(len(best["selected"])),
        "ransac_inlier_points": int(best["count"]),
        "ransac_inlier_ratio": float(best["count"] / max(len(best["selected"]), 1)),
        "point_to_plane_median_m": float(np.median(all_distances)),
        "point_to_plane_p95_m": float(np.percentile(all_distances, 95)),
        "ransac_inlier_distance_median_m": float(np.median(inlier_distances)),
        "ransac_inlier_distance_p95_m": float(np.percentile(inlier_distances, 95)),
        "plane_normal": best["coefficients"].tolist(),
        "plane_offset": float(best["offset"]),
        "threshold_m": float(threshold),
    }, best


def save_side_view(path, fit):
    points = fit["selected"]
    normal = fit["coefficients"]
    tangent = np.asarray([1.0, 0.0, 0.0])
    if abs(float(tangent @ normal)) > 0.9:
        tangent = np.asarray([0.0, 1.0, 0.0])
    tangent -= float(tangent @ normal) * normal
    tangent /= np.linalg.norm(tangent)
    horizontal = points @ tangent
    vertical = points @ normal + fit["offset"]
    if len(points) > 40000:
        selection = np.linspace(0, len(points) - 1, 40000).astype(np.int64)
        horizontal, vertical = horizontal[selection], vertical[selection]
    figure, axis = plt.subplots(figsize=(12, 4), dpi=150)
    axis.scatter(horizontal, vertical, s=0.5, alpha=0.35, c="#238b45")
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.axhline(0.03, color="#d7301f", linewidth=0.8, linestyle="--")
    axis.axhline(-0.03, color="#d7301f", linewidth=0.8, linestyle="--")
    axis.set_xlabel("coordinate along fitted lawn plane (m)")
    axis.set_ylabel("signed plane distance (m)")
    axis.set_title("Committed lawn GS side view")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metrics_path = args.metrics or run_dir / "validation_metrics.json"
    ply_path = args.ply or run_dir / "point_cloud_aerocommit_full.ply"
    output_path = args.output or run_dir / "worldtest_evaluation.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else {}
    )
    points, colors = load_points_and_colors(ply_path)
    lawn_points, lawn_colors, lawn_roi = select_lawn_image_roi(
        run_dir,
        points,
        colors,
        args.lawn_reference_frame,
        args.lawn_image_roi,
    )
    ground, fit = fit_ground(
        lawn_points, lawn_colors, args.ground_threshold_m, args.seed
    )
    side_view = output_path.with_name("lawn_ground_side_view.png")
    save_side_view(side_view, fit)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    payload = {
        "run_dir": str(run_dir),
        "ply": str(ply_path),
        "gaussian_count": int(len(points)),
        "lawn_roi": lawn_roi,
        "image_windows": image_windows(metrics),
        "lawn_ground_geometry": ground,
        "canonical_nearest_distance": canonical_nearest_distance(run_dir, points),
        "lawn_side_view": str(side_view),
        "future_view_invalid_commit_rate": results.get("worldtest_summary", {}).get(
            "future_view_invalid_commit_rate"
        ),
        "bypass_count": results.get("worldtest_summary", {})
        .get("certificate_authority", {})
        .get("bypass_count"),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
