#!/usr/bin/env python3
"""Convert extracted AirVLN RGB+poses into this repo's Aria/ORB dataset layout.

The Online-3DGS-Monocular dataset loader expects:
  rectified/aria_XXXX.png
  trajectory_orb.json with cameras[*].T_camera_world
  orb_point_clouds/point_cloud_<idx>.txt

The AirVLN extractor saves front-camera images and AirSim body poses. This
script converts images to RGB, converts body poses to world-to-camera matrices,
and triangulates sparse ORB points from RGB pairs using the known poses.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "data"
    / "airvln_extracted"
    / "aerialvln_s_scene10_val_seen_0_rgb"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "Online3DGS_AirVLN"
    / "aerialvln_s_scene10_val_seen_0"
)
DEFAULT_CONFIG = REPO_ROOT / "configs" / "airvln"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--name", default="AerialVLNS-scene10-val_seen-0")
    parser.add_argument("--config-prefix", default="AerialVLNS_scene10_val_seen_0")
    parser.add_argument("--smoke-frames", type=int, default=20)
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--orb-features", type=int, default=4000)
    parser.add_argument("--pair-gaps", default="1,2,4,8")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--max-reproj-error", type=float, default=8.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=1000.0)
    parser.add_argument("--voxel-size", type=float, default=0.25)
    parser.add_argument("--max-points-per-frame", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def camera_matrix(width: int, height: int, fov_deg: float) -> tuple[np.ndarray, float, float, float, float]:
    fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return k, fx, fy, cx, cy


def body_to_camera_world(frame: dict) -> np.ndarray:
    """Return T_camera_world from an AirVLN T_world_body frame.

    AirSim body convention is x-forward, y-right, z-down. The image camera
    convention used by this project is x-right, y-down, z-forward. The default
    front camera is assumed to be aligned with the vehicle body.
    """
    t_world_body = np.array(frame["T_world_body"], dtype=np.float64)
    r_world_body = t_world_body[:3, :3]
    t_world_body_vec = t_world_body[:3, 3]

    r_body_world = r_world_body.T
    r_camera_body = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    r_camera_world = r_camera_body @ r_body_world
    t_camera_world = -r_camera_world @ t_world_body_vec

    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = r_camera_world
    t[:3, 3] = t_camera_world
    return t


def project_points(k: np.ndarray, t_camera_world: np.ndarray, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_h = np.concatenate(
        [points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1
    )
    points_cam = (t_camera_world @ points_h.T).T[:, :3]
    uvw = (k @ points_cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, points_cam[:, 2]


def triangulate_pair(
    k: np.ndarray,
    t_i: np.ndarray,
    t_j: np.ndarray,
    kp_i: list[cv2.KeyPoint],
    kp_j: list[cv2.KeyPoint],
    des_i: np.ndarray | None,
    des_j: np.ndarray | None,
    matcher: cv2.BFMatcher,
    ratio: float,
    max_reproj_error: float,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    if des_i is None or des_j is None or len(kp_i) < 8 or len(kp_j) < 8:
        return np.empty((0, 3), dtype=np.float64)

    matches = matcher.knnMatch(des_i, des_j, k=2)
    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 8:
        return np.empty((0, 3), dtype=np.float64)

    pts_i = np.float32([kp_i[m.queryIdx].pt for m in good])
    pts_j = np.float32([kp_j[m.trainIdx].pt for m in good])

    p_i = k @ t_i[:3, :]
    p_j = k @ t_j[:3, :]
    x_h = cv2.triangulatePoints(p_i, p_j, pts_i.T, pts_j.T).T
    finite_w = np.abs(x_h[:, 3]) > 1e-8
    x = np.empty((0, 3), dtype=np.float64)
    if finite_w.any():
        x = x_h[finite_w, :3] / x_h[finite_w, 3:4]
        pts_i = pts_i[finite_w]
        pts_j = pts_j[finite_w]
    if x.shape[0] == 0:
        return x

    uv_i, depth_i = project_points(k, t_i, x)
    uv_j, depth_j = project_points(k, t_j, x)
    reproj_error = np.linalg.norm(uv_i - pts_i, axis=1) + np.linalg.norm(uv_j - pts_j, axis=1)
    valid = (
        np.isfinite(x).all(axis=1)
        & (depth_i > min_depth)
        & (depth_j > min_depth)
        & (depth_i < max_depth)
        & (depth_j < max_depth)
        & (reproj_error < max_reproj_error)
    )
    return x[valid]


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.shape[0] == 0 or voxel_size <= 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def visible_points(
    k: np.ndarray,
    t_camera_world: np.ndarray,
    points_world: np.ndarray,
    width: int,
    height: int,
    min_depth: float,
    max_depth: float,
    max_points: int,
) -> np.ndarray:
    if points_world.shape[0] == 0:
        return points_world
    uv, depth = project_points(k, t_camera_world, points_world)
    valid = (
        (depth > min_depth)
        & (depth < max_depth)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] <= width - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= height - 1)
    )
    pts = points_world[valid]
    if max_points > 0 and pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(pts.shape[0], max_points, replace=False)]
    return pts


def write_points(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if points.shape[0] == 0:
        path.write_text("", encoding="utf-8")
        return
    np.savetxt(path, points.astype(np.float32), fmt="%.6f")


def write_config(
    config_dir: Path,
    dataset_dir: Path,
    name: str,
    config_prefix: str,
    frame_count: int,
    smoke_frames: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> tuple[Path, Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = str(dataset_dir)

    common_dataset = f"""  name: "{name}"
  type: "aria"
  data_source: "orb"
  dataset_path: "{dataset_path}"
  num_threads: 0
  begin_cutoff: 0
  end_cutoff: 0
  max_pts_num: 5000
  vignette: False
  use_vignette_type: "post-render"
  Calibration:
    fx: {fx:.6f}
    fy: {fy:.6f}
    cx: {cx:.6f}
    cy: {cy:.6f}
    width: {width}
    height: {height}
    near: 0.01
    far: 1000
"""
    full_config = config_dir / f"{config_prefix}_orb.yaml"
    smoke_config = config_dir / f"{config_prefix}_orb_smoke.yaml"

    full_config.write_text(
        f"""inherit_from: "configs/aria/orb_tracking/aria_base.yaml"

Dataset:
{common_dataset}

Testset:
{common_dataset}

Results:
  save_dir: "./Logs_airvln"
  save_gt: False
  save_exr: False
  save_mesh: False
  skip_eval: True

Mapper:
  use_multi_reso: False
  initialization_frames: 4
  optimization_iters: 10
  initialization_iters: 10
  post_refinement:
    max_steps: 100
    opt_cam: False
  KFGraph:
    kf_interval: 3
    global_window_size: 2
  CameraOptimizer:
    pose_refine_init_steps: 0
    pose_opt_steps: 2

Model:
  camera_scale_rescalar: 0.25
  scene_scale: 1.0
""",
        encoding="utf-8",
    )
    if smoke_frames <= 0:
        smoke_end_cutoff = 0
    else:
        smoke_end_cutoff = max(frame_count - smoke_frames, 0)
    smoke_dataset = common_dataset.replace(
        "  end_cutoff: 0", f"  end_cutoff: {smoke_end_cutoff}"
    ).replace("  max_pts_num: 5000", "  max_pts_num: 2000")
    smoke_config.write_text(
        f"""inherit_from: "configs/local_smoke/BikeShop_orb_smoke.yaml"

Dataset:
{smoke_dataset}

Testset:
{smoke_dataset}

Results:
  save_dir: "./Logs_airvln_smoke"
  save_gt: False
  save_exr: False
  save_mesh: False
  skip_eval: True

Mapper:
  use_multi_reso: False
  initialization_frames: 2
  optimization_iters: 1
  initialization_iters: 1
  post_refinement:
    max_steps: 0
    opt_cam: False
  KFGraph:
    kf_interval: 1
    global_window_size: 1
  CameraOptimizer:
    pose_refine_init_steps: 0
    pose_opt_steps: 0

Model:
  extra_pts_num: 64
  camera_scale_rescalar: 0.25
  scene_scale: 1.0
""",
        encoding="utf-8",
    )
    return full_config, smoke_config


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output exists: {output_dir}. Use --force to overwrite.")
        shutil.rmtree(output_dir)

    poses_path = input_dir / "poses_airvln.json"
    if not poses_path.exists():
        raise FileNotFoundError(poses_path)
    frames = json.load(open(poses_path, "r", encoding="utf-8"))["frames"]
    if not frames:
        raise ValueError("No frames found in poses_airvln.json")
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames must be >= 1")
        frames = frames[: args.max_frames]

    first_image = Image.open(input_dir / frames[0]["image_path"])
    width, height = first_image.size
    k, fx, fy, cx, cy = camera_matrix(width, height, args.fov_deg)
    pair_gaps = [int(x) for x in args.pair_gaps.split(",") if x.strip()]

    rectified_dir = output_dir / "rectified"
    point_dir = output_dir / "orb_point_clouds"
    rectified_dir.mkdir(parents=True, exist_ok=True)
    point_dir.mkdir(parents=True, exist_ok=True)

    t_camera_worlds: list[np.ndarray] = []
    image_names: list[str] = []
    gray_images: list[np.ndarray] = []

    for i, frame in enumerate(frames):
        src = input_dir / frame["image_path"]
        image_name = f"aria_{i:04d}.png"
        rgb = Image.open(src).convert("RGB")
        rgb.save(rectified_dir / image_name)
        image_names.append(image_name)
        gray_images.append(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY))
        t_camera_worlds.append(body_to_camera_world(frame))
        if i == 0 or (i + 1) % 200 == 0 or i + 1 == len(frames):
            print(f"Prepared RGB/poses: {i + 1}/{len(frames)}")

    cameras = []
    for image_name, t_camera_world in zip(image_names, t_camera_worlds):
        cameras.append(
            {
                "T_camera_world": t_camera_world.astype(float).tolist(),
                "image": image_name,
                "intrinsic": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                "width": float(width),
                "height": float(height),
                "focal": float(fx),
            }
        )
    trajectory = {"cameras": cameras}
    for name in ["trajectory_orb.json", "trajectory.json"]:
        with open(output_dir / name, "w", encoding="utf-8") as f:
            json.dump(trajectory, f)

    with open(output_dir / "image_list.txt", "w", encoding="utf-8") as f:
        for image_name in image_names:
            f.write(f"rectified/{image_name}\n")

    orb = cv2.ORB_create(nfeatures=args.orb_features)
    keypoints: list[list[cv2.KeyPoint]] = []
    descriptors: list[np.ndarray | None] = []
    for i, gray in enumerate(gray_images):
        kp, des = orb.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)
        if i == 0 or (i + 1) % 200 == 0 or i + 1 == len(gray_images):
            print(f"Detected ORB features: {i + 1}/{len(gray_images)}")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    global_points = []
    pair_stats = []
    for i in range(len(frames)):
        for gap in pair_gaps:
            j = i + gap
            if j >= len(frames):
                continue
            center_i = np.linalg.inv(t_camera_worlds[i])[:3, 3]
            center_j = np.linalg.inv(t_camera_worlds[j])[:3, 3]
            baseline = float(np.linalg.norm(center_i - center_j))
            if baseline < 1e-4:
                continue
            pts = triangulate_pair(
                k,
                t_camera_worlds[i],
                t_camera_worlds[j],
                keypoints[i],
                keypoints[j],
                descriptors[i],
                descriptors[j],
                matcher,
                args.ratio,
                args.max_reproj_error,
                args.min_depth,
                args.max_depth,
            )
            if pts.shape[0] > 0:
                global_points.append(pts)
            pair_stats.append(
                {"i": i, "j": j, "baseline": baseline, "triangulated_valid": int(pts.shape[0])}
            )
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(frames):
            point_count = sum(points.shape[0] for points in global_points)
            print(f"Triangulated pairs through frame {i + 1}/{len(frames)}; raw points={point_count}")

    if global_points:
        global_points_arr = np.concatenate(global_points, axis=0)
        global_points_arr = voxel_downsample(global_points_arr, args.voxel_size)
    else:
        global_points_arr = np.empty((0, 3), dtype=np.float64)

    per_frame_counts = {}
    for i, t_camera_world in enumerate(t_camera_worlds):
        pts = visible_points(
            k,
            t_camera_world,
            global_points_arr,
            width,
            height,
            args.min_depth,
            args.max_depth,
            args.max_points_per_frame,
        )
        write_points(point_dir / f"point_cloud_{i}.txt", pts)
        per_frame_counts[str(i)] = int(pts.shape[0])
        if i == 0 or (i + 1) % 200 == 0 or i + 1 == len(t_camera_worlds):
            print(f"Wrote per-frame point clouds: {i + 1}/{len(t_camera_worlds)}")

    stats = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "frame_count": len(frames),
        "width": width,
        "height": height,
        "fov_deg": args.fov_deg,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "pair_gaps": pair_gaps,
        "global_point_count": int(global_points_arr.shape[0]),
        "per_frame_point_count_min": int(min(per_frame_counts.values())) if per_frame_counts else 0,
        "per_frame_point_count_max": int(max(per_frame_counts.values())) if per_frame_counts else 0,
        "per_frame_point_count_mean": float(np.mean(list(per_frame_counts.values()))) if per_frame_counts else 0.0,
        "per_frame_counts": per_frame_counts,
        "pair_stats": pair_stats,
        "pose_note": (
            "T_camera_world is world-to-camera. Converted from AirVLN T_world_body "
            "assuming AirSim default front camera aligned with vehicle body."
        ),
    }
    with open(output_dir / "conversion_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    full_config, smoke_config = write_config(
        args.config_dir.resolve(),
        output_dir,
        args.name,
        args.config_prefix,
        len(frames),
        args.smoke_frames,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
    )

    print(f"Converted dataset: {output_dir}")
    print(f"Frames: {len(frames)}")
    print(f"Global sparse points: {global_points_arr.shape[0]}")
    print(
        "Per-frame points: min={} mean={:.1f} max={}".format(
            stats["per_frame_point_count_min"],
            stats["per_frame_point_count_mean"],
            stats["per_frame_point_count_max"],
        )
    )
    print(f"Full config: {full_config}")
    print(f"Smoke config: {smoke_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
