#!/usr/bin/env python3
"""Build canonical MODP inputs from fixed poses and learned multi-view tracks."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement


K = np.asarray(
    [[320.0, 0.0, 480.0], [0.0, 320.0, 320.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
WIDTH = 960
HEIGHT = 640
CACHE_FRAME_OFFSET = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--trajectory-file", default="trajectory.json")
    parser.add_argument(
        "--pose-source-label",
        choices=("gt", "colmap", "orbslam3_vi", "orbslam3_mono"),
        default="gt",
    )
    parser.add_argument("--start-frame", type=int, default=110)
    parser.add_argument("--num-frames", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=2048)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument(
        "--cache-frame-offset",
        type=int,
        default=0,
        help="Map requested frame 0 to this frame in an equivalent image-only cache.",
    )
    parser.add_argument("--pair-gaps", default="1,2,4,8,16,32,64")
    parser.add_argument("--min-pair-baseline-m", type=float, default=0.03)
    parser.add_argument("--min-match-score", type=float, default=0.15)
    parser.add_argument("--max-epipolar-error-px", type=float, default=1.5)
    parser.add_argument("--min-track-length", type=int, default=3)
    parser.add_argument("--max-reprojection-error-px", type=float, default=2.0)
    parser.add_argument("--max-median-reprojection-error-px", type=float, default=1.0)
    parser.add_argument("--min-triangulation-angle-deg", type=float, default=1.5)
    parser.add_argument("--min-depth-m", type=float, default=0.3)
    parser.add_argument("--max-depth-m", type=float, default=120.0)
    parser.add_argument("--max-nearest-camera-distance-m", type=float, default=60.0)
    parser.add_argument("--max-position-std-m", type=float, default=1.0)
    parser.add_argument("--max-relative-position-std", type=float, default=0.05)
    parser.add_argument("--max-points-per-frame", type=int, default=10000)
    parser.add_argument(
        "--camera-yaw-deg",
        type=float,
        default=0.0,
        help="Fixed source-camera to virtual-camera yaw applied to every w2c pose.",
    )
    parser.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=0.0,
        help="Fixed source-camera to virtual-camera pitch applied to every w2c pose.",
    )
    parser.add_argument(
        "--camera-roll-deg",
        type=float,
        default=0.0,
        help="Fixed source-camera to virtual-camera roll applied to every w2c pose.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Extract image features and matches without pose-dependent filtering.",
    )
    return parser.parse_args()


def virtual_camera_rotation(
    yaw_deg: float, pitch_deg: float, roll_deg: float
) -> np.ndarray:
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    ry = np.asarray(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rz = np.asarray(
        [
            [np.cos(roll), -np.sin(roll), 0.0],
            [np.sin(roll), np.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ry @ rx @ rz


def apply_virtual_camera_extrinsic(
    pose: np.ndarray, yaw_deg: float, pitch_deg: float, roll_deg: float
) -> np.ndarray:
    correction = np.eye(4, dtype=np.float64)
    correction[:3, :3] = virtual_camera_rotation(yaw_deg, pitch_deg, roll_deg).T
    return correction @ np.asarray(pose, dtype=np.float64)


def configure_camera_model(cameras: list[dict]) -> None:
    global K, WIDTH, HEIGHT
    if not cameras:
        raise ValueError("At least one camera is required")
    first = cameras[0]
    intrinsic = first.get("intrinsic", {})
    required = ("fx", "fy", "cx", "cy")
    missing = [name for name in required if name not in intrinsic]
    if missing:
        raise KeyError(f"Camera intrinsic is missing keys: {missing}")
    width = int(first.get("width", round(2.0 * float(intrinsic["cx"]))))
    height = int(first.get("height", round(2.0 * float(intrinsic["cy"]))))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid camera dimensions: {width}x{height}")
    reference = np.asarray(
        [
            float(intrinsic["fx"]),
            float(intrinsic["fy"]),
            float(intrinsic["cx"]),
            float(intrinsic["cy"]),
            width,
            height,
        ],
        dtype=np.float64,
    )
    for camera in cameras[1:]:
        current_intrinsic = camera.get("intrinsic", {})
        current = np.asarray(
            [
                float(current_intrinsic[name]) for name in required
            ]
            + [
                int(camera.get("width", width)),
                int(camera.get("height", height)),
            ],
            dtype=np.float64,
        )
        if not np.allclose(current, reference, rtol=0.0, atol=1.0e-6):
            raise ValueError("Per-frame camera intrinsics are not constant")
    K = np.asarray(
        [
            [reference[0], 0.0, reference[2]],
            [0.0, reference[1], reference[3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    WIDTH = width
    HEIGHT = height


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def fundamental_from_poses(t_i: np.ndarray, t_j: np.ndarray) -> np.ndarray:
    t_j_i = t_j @ np.linalg.inv(t_i)
    essential = skew(t_j_i[:3, 3]) @ t_j_i[:3, :3]
    k_inv = np.linalg.inv(K)
    return k_inv.T @ essential @ k_inv


def sampson_errors(
    fundamental: np.ndarray, points_i: np.ndarray, points_j: np.ndarray
) -> np.ndarray:
    ones = np.ones((len(points_i), 1), dtype=np.float64)
    x_i = np.concatenate((points_i, ones), axis=1)
    x_j = np.concatenate((points_j, ones), axis=1)
    f_x_i = (fundamental @ x_i.T).T
    ft_x_j = (fundamental.T @ x_j.T).T
    numerator = np.sum(x_j * f_x_i, axis=1) ** 2
    denominator = (
        f_x_i[:, 0] ** 2
        + f_x_i[:, 1] ** 2
        + ft_x_j[:, 0] ** 2
        + ft_x_j[:, 1] ** 2
        + 1.0e-12
    )
    return np.sqrt(numerator / denominator)


def camera_centers(poses: np.ndarray) -> np.ndarray:
    return np.linalg.inv(poses)[:, :3, 3]


def project_point(pose: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, float]:
    camera = pose[:3, :3] @ point + pose[:3, 3]
    if not np.isfinite(camera).all() or camera[2] <= 1.0e-12:
        return np.full((2,), np.nan), float(camera[2])
    screen = K @ camera
    return screen[:2] / screen[2], float(camera[2])


def linear_triangulate(poses: list[np.ndarray], pixels: list[np.ndarray]) -> np.ndarray:
    rows = []
    k_inv = np.linalg.inv(K)
    for pose, pixel in zip(poses, pixels):
        normalized = k_inv @ np.asarray([pixel[0], pixel[1], 1.0])
        projection = pose[:3]
        rows.append(normalized[0] * projection[2] - projection[0])
        rows.append(normalized[1] * projection[2] - projection[1])
    _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=False)
    homogeneous = vh[-1]
    if abs(homogeneous[3]) <= 1.0e-12:
        return np.full((3,), np.nan)
    return homogeneous[:3] / homogeneous[3]


def projection_errors(
    point: np.ndarray, poses: list[np.ndarray], pixels: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    errors = []
    depths = []
    for pose, pixel in zip(poses, pixels):
        projected, depth = project_point(pose, point)
        errors.append(np.linalg.norm(projected - pixel))
        depths.append(depth)
    return np.asarray(errors), np.asarray(depths)


def maximum_triangulation_angle_deg(
    point: np.ndarray, centers: np.ndarray
) -> float:
    rays = point[None] - centers
    rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1.0e-12
    cosine = np.clip(rays @ rays.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine.min())))


def point_position_std(
    point: np.ndarray, poses: list[np.ndarray], errors: np.ndarray
) -> float:
    jacobians = []
    for pose in poses:
        rotation = pose[:3, :3]
        camera = rotation @ point + pose[:3, 3]
        x, y, z = camera
        if z <= 1.0e-8:
            return math.inf
        local = np.asarray(
            [[K[0, 0] / z, 0.0, -K[0, 0] * x / (z * z)],
             [0.0, K[1, 1] / z, -K[1, 1] * y / (z * z)]],
            dtype=np.float64,
        )
        jacobians.append(local @ rotation)
    jacobian = np.concatenate(jacobians, axis=0)
    sigma2 = max(float(np.median(errors) ** 2), 0.25)
    information = jacobian.T @ jacobian
    try:
        covariance = np.linalg.inv(information) * sigma2
        return float(np.sqrt(max(np.linalg.eigvalsh(covariance).max(), 0.0)))
    except np.linalg.LinAlgError:
        return math.inf


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def add(self, value: int) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        self.add(left)
        self.add(right)
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def encode_node(frame_id: int, feature_id: int) -> int:
    return (int(frame_id) << 32) | int(feature_id)


def decode_node(node: int) -> tuple[int, int]:
    return int(node >> 32), int(node & 0xFFFFFFFF)


def load_camera_slice(
    source: Path,
    trajectory_file: str,
    start_frame: int,
    num_frames: int,
    camera_yaw_deg: float = 0.0,
    camera_pitch_deg: float = 0.0,
    camera_roll_deg: float = 0.0,
) -> tuple[list[dict], list[Path], np.ndarray, list[int]]:
    trajectory = json.loads((source / trajectory_file).read_text(encoding="utf-8"))
    all_cameras = trajectory["cameras"]
    end = len(all_cameras) if num_frames <= 0 else min(len(all_cameras), start_frame + num_frames)
    if start_frame < 0 or start_frame >= end:
        raise ValueError("Requested frame slice is empty")
    selected = all_cameras[start_frame:end]
    cameras = []
    image_paths = []
    source_indices = []
    for output_index, camera in enumerate(selected):
        source_index = int(
            camera.get(
                "source_frame_index",
                camera.get("frame_index", start_frame + output_index),
            )
        )
        source_path = source / "rectified" / camera["image"]
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        updated = dict(camera)
        updated["image"] = f"aria_{output_index:05d}.png"
        updated["frame_index"] = output_index
        updated["source_frame_index"] = source_index
        updated["T_camera_world"] = apply_virtual_camera_extrinsic(
            np.asarray(camera["T_camera_world"], dtype=np.float64),
            camera_yaw_deg,
            camera_pitch_deg,
            camera_roll_deg,
        ).tolist()
        cameras.append(updated)
        image_paths.append(source_path)
        source_indices.append(source_index)
    poses = np.asarray([camera["T_camera_world"] for camera in cameras], dtype=np.float64)
    return cameras, image_paths, poses, source_indices


def feature_path(cache: Path, frame_id: int) -> Path:
    return cache / "features" / f"{frame_id + CACHE_FRAME_OFFSET:05d}.npz"


def match_path(cache: Path, frame_i: int, frame_j: int) -> Path:
    return cache / "matches" / (
        f"{frame_i + CACHE_FRAME_OFFSET:05d}_{frame_j + CACHE_FRAME_OFFSET:05d}.npz"
    )


def extract_features(
    image_paths: list[Path], cache: Path, args: argparse.Namespace
) -> None:
    from kornia.feature import DISK

    feature_dir = cache / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    missing = [i for i in range(len(image_paths)) if not feature_path(cache, i).is_file()]
    if not missing:
        print("Feature cache complete", flush=True)
        return
    device = torch.device(args.device)
    model = DISK.from_pretrained("depth", device=device).eval()
    batch_size = max(int(args.feature_batch_size), 1)
    for batch_start in range(0, len(missing), batch_size):
        frame_ids = missing[batch_start : batch_start + batch_size]
        images = []
        colors = []
        for frame_id in frame_ids:
            rgb = np.asarray(Image.open(image_paths[frame_id]).convert("RGB"))
            colors.append(rgb)
            images.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0)
        batch = torch.stack(images).to(device)
        with torch.inference_mode():
            outputs = model(
                batch,
                n=int(args.max_features),
                window_size=5,
                score_threshold=0.0,
                pad_if_not_divisible=True,
            )
        for frame_id, output, rgb in zip(frame_ids, outputs, colors):
            keypoints = output.keypoints.detach().cpu().numpy().astype(np.float32)
            descriptors = output.descriptors.detach().cpu().numpy().astype(np.float16)
            scores = output.detection_scores.detach().cpu().numpy().astype(np.float16)
            x = np.clip(np.rint(keypoints[:, 0]).astype(int), 0, rgb.shape[1] - 1)
            y = np.clip(np.rint(keypoints[:, 1]).astype(int), 0, rgb.shape[0] - 1)
            np.savez(
                feature_path(cache, frame_id),
                keypoints=keypoints,
                descriptors=descriptors,
                scores=scores,
                colors=rgb[y, x].astype(np.uint8),
            )
        done = min(batch_start + len(frame_ids), len(missing))
        if done == len(missing) or done % 50 == 0:
            print(f"DISK features: {done}/{len(missing)}", flush=True)
    del model
    torch.cuda.empty_cache()


def load_features(cache: Path, frame_count: int) -> list[dict[str, np.ndarray]]:
    features = []
    for frame_id in range(frame_count):
        with np.load(feature_path(cache, frame_id)) as data:
            features.append({name: data[name] for name in data.files})
    return features


def build_pairs(frame_count: int, gaps: list[int]) -> list[tuple[int, int]]:
    return [
        (frame_i, frame_i + gap)
        for frame_i in range(frame_count)
        for gap in gaps
        if frame_i + gap < frame_count
    ]


def compute_matches(
    pairs: list[tuple[int, int]],
    features: list[dict[str, np.ndarray]],
    cache: Path,
    args: argparse.Namespace,
) -> None:
    from kornia.feature import LightGlueMatcher, laf_from_center_scale_ori

    match_dir = cache / "matches"
    match_dir.mkdir(parents=True, exist_ok=True)
    missing = [(i, j) for i, j in pairs if not match_path(cache, i, j).is_file()]
    if not missing:
        print("Match cache complete", flush=True)
        return
    device = torch.device(args.device)
    matcher = LightGlueMatcher("disk").to(device).eval()
    for number, (frame_i, frame_j) in enumerate(missing, start=1):
        feature_i = features[frame_i]
        feature_j = features[frame_j]
        descriptor_i = torch.from_numpy(feature_i["descriptors"].astype(np.float32)).to(device)
        descriptor_j = torch.from_numpy(feature_j["descriptors"].astype(np.float32)).to(device)
        keypoint_i = torch.from_numpy(feature_i["keypoints"]).to(device)
        keypoint_j = torch.from_numpy(feature_j["keypoints"]).to(device)
        laf_i = laf_from_center_scale_ori(keypoint_i[None])
        laf_j = laf_from_center_scale_ori(keypoint_j[None])
        with torch.inference_mode():
            scores, indices = matcher(
                descriptor_i,
                descriptor_j,
                laf_i,
                laf_j,
                (HEIGHT, WIDTH),
                (HEIGHT, WIDTH),
            )
        np.savez(
            match_path(cache, frame_i, frame_j),
            indices=indices.detach().cpu().numpy().astype(np.int32),
            scores=scores.detach().cpu().numpy().reshape(-1).astype(np.float16),
        )
        if number == len(missing) or number % 250 == 0:
            print(f"LightGlue pairs: {number}/{len(missing)}", flush=True)
    del matcher
    torch.cuda.empty_cache()


def build_track_components(
    pairs: list[tuple[int, int]],
    features: list[dict[str, np.ndarray]],
    poses: np.ndarray,
    cache: Path,
    args: argparse.Namespace,
) -> tuple[list[list[tuple[int, int]]], dict]:
    centers = camera_centers(poses)
    union_find = UnionFind()
    raw_matches = 0
    score_matches = 0
    geometry_matches = 0
    eligible_pairs = 0
    for number, (frame_i, frame_j) in enumerate(pairs, start=1):
        baseline = float(np.linalg.norm(centers[frame_i] - centers[frame_j]))
        if baseline < float(args.min_pair_baseline_m):
            continue
        eligible_pairs += 1
        with np.load(match_path(cache, frame_i, frame_j)) as data:
            indices = data["indices"]
            scores = data["scores"].astype(np.float32)
        raw_matches += len(indices)
        selected = scores >= float(args.min_match_score)
        indices = indices[selected]
        score_matches += len(indices)
        if not len(indices):
            continue
        pixels_i = features[frame_i]["keypoints"][indices[:, 0]].astype(np.float64)
        pixels_j = features[frame_j]["keypoints"][indices[:, 1]].astype(np.float64)
        errors = sampson_errors(
            fundamental_from_poses(poses[frame_i], poses[frame_j]), pixels_i, pixels_j
        )
        indices = indices[errors <= float(args.max_epipolar_error_px)]
        geometry_matches += len(indices)
        for feature_i, feature_j in indices.tolist():
            union_find.union(
                encode_node(frame_i, feature_i), encode_node(frame_j, feature_j)
            )
        if number == len(pairs) or number % 500 == 0:
            print(f"Track graph pairs: {number}/{len(pairs)}", flush=True)

    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for node in union_find.parent:
        groups[union_find.find(node)].append(decode_node(node))
    components = []
    duplicate_frame_components = 0
    for observations in groups.values():
        frames = [frame_id for frame_id, _ in observations]
        if len(set(frames)) != len(frames):
            duplicate_frame_components += 1
            continue
        if len(observations) >= int(args.min_track_length):
            components.append(sorted(observations))
    components.sort(key=lambda item: (-len(item), item[0]))
    stats = {
        "pair_count": len(pairs),
        "eligible_pair_count": eligible_pairs,
        "raw_lightglue_matches": raw_matches,
        "score_filtered_matches": score_matches,
        "geometry_filtered_matches": geometry_matches,
        "connected_components": len(groups),
        "duplicate_frame_components": duplicate_frame_components,
        "candidate_tracks": len(components),
    }
    return components, stats


def triangulate_component(
    component: list[tuple[int, int]],
    features: list[dict[str, np.ndarray]],
    poses: np.ndarray,
    centers: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict | None, str]:
    observation_poses = [poses[frame_id] for frame_id, _ in component]
    pixels = [
        features[frame_id]["keypoints"][feature_id].astype(np.float64)
        for frame_id, feature_id in component
    ]
    observation_centers = centers[[frame_id for frame_id, _ in component]]
    center_distances = np.linalg.norm(
        observation_centers[:, None] - observation_centers[None], axis=2
    )
    support_flat = int(np.argmax(center_distances))
    support_i, support_j = np.unravel_index(support_flat, center_distances.shape)
    if support_i == support_j:
        return None, "zero_support_baseline"
    point = linear_triangulate(
        [observation_poses[support_i], observation_poses[support_j]],
        [pixels[support_i], pixels[support_j]],
    )
    if not np.isfinite(point).all():
        return None, "initial_triangulation_nonfinite"
    errors, depths = projection_errors(point, observation_poses, pixels)
    inlier = (
        np.isfinite(errors)
        & (errors <= float(args.max_reprojection_error_px))
        & (depths >= float(args.min_depth_m))
        & (depths <= float(args.max_depth_m))
    )
    if inlier.sum() < int(args.min_track_length):
        return None, "insufficient_initial_inliers"
    support_mask = np.ones(len(component), dtype=bool)
    support_mask[[support_i, support_j]] = False
    if not np.any(inlier & support_mask):
        return None, "heldout_observation_failed"

    for _ in range(2):
        point = linear_triangulate(
            [pose for pose, keep in zip(observation_poses, inlier) if keep],
            [pixel for pixel, keep in zip(pixels, inlier) if keep],
        )
        if not np.isfinite(point).all():
            return None, "refined_triangulation_nonfinite"
        errors, depths = projection_errors(point, observation_poses, pixels)
        inlier = (
            np.isfinite(errors)
            & (errors <= float(args.max_reprojection_error_px))
            & (depths >= float(args.min_depth_m))
            & (depths <= float(args.max_depth_m))
        )
        if inlier.sum() < int(args.min_track_length):
            return None, "insufficient_refined_inliers"

    inlier_errors = errors[inlier]
    if np.median(inlier_errors) > float(args.max_median_reprojection_error_px):
        return None, "median_reprojection_error"
    inlier_centers = observation_centers[inlier]
    angle = maximum_triangulation_angle_deg(point, inlier_centers)
    if angle < float(args.min_triangulation_angle_deg):
        return None, "triangulation_angle"
    nearest_camera_distance = float(np.linalg.norm(centers - point[None], axis=1).min())
    if nearest_camera_distance > float(args.max_nearest_camera_distance_m):
        return None, "nearest_camera_distance"
    inlier_poses = [pose for pose, keep in zip(observation_poses, inlier) if keep]
    position_std = point_position_std(point, inlier_poses, inlier_errors)
    median_depth = float(np.median(depths[inlier]))
    relative_std = position_std / max(median_depth, 1.0e-8)
    if position_std > float(args.max_position_std_m):
        return None, "position_std"
    if relative_std > float(args.max_relative_position_std):
        return None, "relative_position_std"
    inlier_component = [item for item, keep in zip(component, inlier) if keep]
    colors = np.asarray(
        [features[frame]["colors"][feature] for frame, feature in inlier_component],
        dtype=np.float64,
    )
    return {
        "point": point.astype(np.float32),
        "color": np.rint(colors.mean(axis=0)).clip(0, 255).astype(np.uint8),
        "observations": inlier_component,
        "pixels": np.asarray([pixel for pixel, keep in zip(pixels, inlier) if keep], dtype=np.float32),
        "track_length": int(inlier.sum()),
        "median_reprojection_error_px": float(np.median(inlier_errors)),
        "p95_reprojection_error_px": float(np.percentile(inlier_errors, 95)),
        "max_triangulation_angle_deg": angle,
        "nearest_camera_distance_m": nearest_camera_distance,
        "position_std_m": position_std,
        "relative_position_std": relative_std,
    }, "accepted"


def percentile_summary(values: np.ndarray) -> dict:
    if not len(values):
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def write_ply(path: Path, records: list[dict]) -> None:
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("point_id", "u4"), ("track_length", "u2"),
        ("reprojection_error_px", "f4"), ("triangulation_angle_deg", "f4"),
        ("position_std_m", "f4"),
    ]
    vertices = np.empty((len(records),), dtype=dtype)
    if records:
        points = np.asarray([record["point"] for record in records])
        colors = np.asarray([record["color"] for record in records])
        vertices["x"], vertices["y"], vertices["z"] = points.T
        vertices["red"], vertices["green"], vertices["blue"] = colors.T
        vertices["point_id"] = np.arange(len(records), dtype=np.uint32)
        vertices["track_length"] = [record["track_length"] for record in records]
        vertices["reprojection_error_px"] = [
            record["median_reprojection_error_px"] for record in records
        ]
        vertices["triangulation_angle_deg"] = [
            record["max_triangulation_angle_deg"] for record in records
        ]
        vertices["position_std_m"] = [record["position_std_m"] for record in records]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def write_dataset(
    output: Path,
    cameras: list[dict],
    image_paths: list[Path],
    source_indices: list[int],
    poses: np.ndarray,
    records: list[dict],
    graph_stats: dict,
    args: argparse.Namespace,
    elapsed_seconds: float,
) -> dict:
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    rectified = staging / "rectified"
    point_dir = staging / "orb_point_clouds"
    id_dir = staging / "orb_point_ids"
    preprocess = staging / "preprocess"
    for directory in (rectified, point_dir, id_dir, preprocess):
        directory.mkdir(parents=True, exist_ok=True)

    for frame_id, image_path in enumerate(image_paths):
        (rectified / f"aria_{frame_id:05d}.png").symlink_to(image_path.resolve())
    trajectory = {"cameras": cameras}
    for name in ("trajectory.json", "trajectory_orb.json"):
        (staging / name).write_text(json.dumps(trajectory, indent=2) + "\n", encoding="utf-8")
    np.save(preprocess / "source_frame_indices.npy", np.asarray(source_indices, dtype=np.int32))

    points = np.asarray([record["point"] for record in records], dtype=np.float32)
    colors = np.asarray([record["color"] for record in records], dtype=np.uint8)
    np.save(preprocess / "global_sparse_points.npy", points)
    np.save(preprocess / "global_colmap_points_sim3.npy", points)

    per_frame_ids: list[list[int]] = [[] for _ in cameras]
    observation_point_ids = []
    observation_frame_ids = []
    observation_uv = []
    for point_id, record in enumerate(records):
        for (frame_id, _), pixel in zip(record["observations"], record["pixels"]):
            per_frame_ids[frame_id].append(point_id)
            observation_point_ids.append(point_id)
            observation_frame_ids.append(frame_id)
            observation_uv.append(pixel)
    per_frame_counts = []
    rng = np.random.default_rng(args.seed)
    for frame_id, ids in enumerate(per_frame_ids):
        ids_array = np.asarray(sorted(set(ids)), dtype=np.int64)
        if len(ids_array) > int(args.max_points_per_frame):
            ids_array = np.sort(
                rng.choice(ids_array, int(args.max_points_per_frame), replace=False)
            )
        np.save(point_dir / f"point_cloud_{frame_id}.npy", points[ids_array])
        np.save(id_dir / f"point_ids_{frame_id}.npy", ids_array)
        per_frame_counts.append(len(ids_array))
    np.savez(
        preprocess / "track_observations.npz",
        point_ids=np.asarray(observation_point_ids, dtype=np.int64),
        frame_ids=np.asarray(observation_frame_ids, dtype=np.int32),
        uv=np.asarray(observation_uv, dtype=np.float32),
    )
    write_ply(staging / "initialization_multiview_tracks.ply", records)

    centers = camera_centers(poses)
    metrics = {
        "track_length": percentile_summary(
            np.asarray([record["track_length"] for record in records], dtype=np.float64)
        ),
        "median_reprojection_error_px": percentile_summary(
            np.asarray(
                [record["median_reprojection_error_px"] for record in records],
                dtype=np.float64,
            )
        ),
        "max_triangulation_angle_deg": percentile_summary(
            np.asarray(
                [record["max_triangulation_angle_deg"] for record in records],
                dtype=np.float64,
            )
        ),
        "nearest_camera_distance_m": percentile_summary(
            np.asarray(
                [record["nearest_camera_distance_m"] for record in records],
                dtype=np.float64,
            )
        ),
        "position_std_m": percentile_summary(
            np.asarray([record["position_std_m"] for record in records], dtype=np.float64)
        ),
    }
    method_by_pose = {
        "colmap": "COLMAP visual poses plus DISK-LightGlue multi-view track triangulation",
        "gt": "RTK fixed poses plus DISK-LightGlue multi-view track triangulation",
        "orbslam3_vi": (
            "ORB-SLAM3 visual-inertial BA poses plus DISK-LightGlue multi-view "
            "track triangulation"
        ),
        "orbslam3_mono": (
            "ORB-SLAM3 monocular BA poses plus DISK-LightGlue multi-view track "
            "triangulation"
        ),
    }
    method = method_by_pose[args.pose_source_label]
    stats = {
        "schema_version": 1,
        "method": method,
        "pose_source": args.pose_source_label,
        "sparse_world_geometry": "persistent",
        "source": str(args.source.resolve()),
        "trajectory_file": args.trajectory_file,
        "removed_prefix_frames": int(args.start_frame),
        "source_frame_start": int(source_indices[0]),
        "source_frame_end_inclusive": int(source_indices[-1]),
        "frame_count": len(cameras),
        "image_storage": "absolute symbolic links",
        "intrinsics": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "width": WIDTH,
            "height": HEIGHT,
        },
        "virtual_camera_extrinsic_deg": {
            "yaw": float(args.camera_yaw_deg),
            "pitch": float(args.camera_pitch_deg),
            "roll": float(args.camera_roll_deg),
        },
        "world_frame": "source camera world; points triangulated under the same poses",
        "trajectory_span_m": np.ptp(centers, axis=0).tolist(),
        "trajectory_length_m": float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()),
        "feature_model": "Kornia DISK depth checkpoint",
        "matcher": "Kornia LightGlue disk checkpoint",
        "max_features_per_frame": int(args.max_features),
        "pair_gaps": [int(value) for value in args.pair_gaps.split(",") if value],
        "graph": graph_stats,
        "accepted_point_count": len(records),
        "point_bounds_min": points.min(axis=0).tolist() if len(points) else None,
        "point_bounds_max": points.max(axis=0).tolist() if len(points) else None,
        "quality": metrics,
        "per_frame_points": percentile_summary(np.asarray(per_frame_counts, dtype=np.float64)),
        "frames_below_100_points": int(np.sum(np.asarray(per_frame_counts) < 100)),
        "thresholds": {
            "min_pair_baseline_m": args.min_pair_baseline_m,
            "min_match_score": args.min_match_score,
            "max_epipolar_error_px": args.max_epipolar_error_px,
            "min_track_length": args.min_track_length,
            "max_reprojection_error_px": args.max_reprojection_error_px,
            "max_median_reprojection_error_px": args.max_median_reprojection_error_px,
            "min_triangulation_angle_deg": args.min_triangulation_angle_deg,
            "max_nearest_camera_distance_m": args.max_nearest_camera_distance_m,
            "max_position_std_m": args.max_position_std_m,
            "max_relative_position_std": args.max_relative_position_std,
        },
        "preprocess_wall_time_s": elapsed_seconds,
    }
    (staging / "conversion_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    (staging / "README.md").write_text(
        "# Canonical learned-track MODP input\n\n"
        f"The first {args.start_frame} source frames are omitted. Poses come from the "
        f"{args.pose_source_label} camera trajectory. Persistent sparse points are triangulated once "
        "per DISK-LightGlue multi-view track and exported only to frames containing a "
        "real inlier observation. The fixed virtual-camera extrinsic is recorded in "
        "`conversion_stats.json`.\n",
        encoding="utf-8",
    )
    staging.replace(output)
    return stats


def main() -> None:
    global CACHE_FRAME_OFFSET
    args = parse_args()
    CACHE_FRAME_OFFSET = int(args.cache_frame_offset)
    if CACHE_FRAME_OFFSET < 0:
        raise ValueError("--cache-frame-offset must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start_time = time.monotonic()
    cameras, image_paths, poses, source_indices = load_camera_slice(
        args.source,
        args.trajectory_file,
        args.start_frame,
        args.num_frames,
        args.camera_yaw_deg,
        args.camera_pitch_deg,
        args.camera_roll_deg,
    )
    configure_camera_model(cameras)
    cache_metadata = {
        "source": str(args.source.resolve()),
        "source_indices": source_indices,
        "max_features": int(args.max_features),
        "model": "disk-depth-lightglue",
    }
    args.feature_cache.mkdir(parents=True, exist_ok=True)
    metadata_path = args.feature_cache / "metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparable_existing = {key: value for key, value in existing.items() if key != "source"}
        existing_indices = comparable_existing.get("source_indices", [])
        comparable_existing["source_indices"] = existing_indices[
            CACHE_FRAME_OFFSET : CACHE_FRAME_OFFSET + len(source_indices)
        ]
        comparable_requested = {
            key: value for key, value in cache_metadata.items() if key != "source"
        }
        if comparable_existing != comparable_requested:
            raise RuntimeError("Feature cache metadata does not match this experiment")
        if existing != cache_metadata:
            print(
                "Reusing image-only feature/match cache from an equivalent derived dataset",
                flush=True,
            )
    else:
        metadata_path.write_text(json.dumps(cache_metadata, indent=2) + "\n", encoding="utf-8")

    print(
        f"Frames: source {source_indices[0]}..{source_indices[-1]} -> 0..{len(cameras)-1}",
        flush=True,
    )
    extract_features(image_paths, args.feature_cache, args)
    features = load_features(args.feature_cache, len(cameras))
    gaps = [int(value) for value in args.pair_gaps.split(",") if value.strip()]
    pairs = build_pairs(len(cameras), gaps)
    compute_matches(pairs, features, args.feature_cache, args)
    if args.cache_only:
        print(
            f"Image-only cache ready: frames={len(cameras)} pairs={len(pairs)}",
            flush=True,
        )
        return
    components, graph_stats = build_track_components(
        pairs, features, poses, args.feature_cache, args
    )
    print(f"Candidate multi-view tracks: {len(components)}", flush=True)
    centers = camera_centers(poses)
    records = []
    rejection_reasons: dict[str, int] = defaultdict(int)
    for number, component in enumerate(components, start=1):
        record, reason = triangulate_component(component, features, poses, centers, args)
        rejection_reasons[reason] += 1
        if record is not None:
            records.append(record)
        if number == len(components) or number % 10000 == 0:
            print(
                f"Triangulated tracks: {number}/{len(components)} accepted={len(records)}",
                flush=True,
            )
    graph_stats["triangulation_outcomes"] = dict(sorted(rejection_reasons.items()))
    elapsed = time.monotonic() - start_time
    stats = write_dataset(
        args.output,
        cameras,
        image_paths,
        source_indices,
        poses,
        records,
        graph_stats,
        args,
        elapsed,
    )
    print(
        f"Wrote {args.output}: points={stats['accepted_point_count']} "
        f"wall={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
