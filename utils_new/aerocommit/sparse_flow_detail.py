"""Sparse RGB correspondence triangulation for side-view detail geometry."""

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class SparseFlowTriangulation:
    world_points: np.ndarray
    colors: np.ndarray
    depths: np.ndarray
    scores: np.ndarray


def _empty_result() -> SparseFlowTriangulation:
    return SparseFlowTriangulation(
        world_points=np.empty((0, 3), dtype=np.float32),
        colors=np.empty((0, 3), dtype=np.float32),
        depths=np.empty((0,), dtype=np.float32),
        scores=np.empty((0,), dtype=np.float32),
    )


def triangulate_sparse_flow_detail(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    previous_world_to_camera: np.ndarray,
    current_world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    config: Mapping[str, object],
) -> SparseFlowTriangulation:
    """Track side-image corners and triangulate only consistent static rays."""
    previous = np.asarray(previous_rgb, dtype=np.float32)
    current = np.asarray(current_rgb, dtype=np.float32)
    if previous.shape != current.shape or previous.ndim != 3 or previous.shape[2] != 3:
        raise ValueError("Sparse flow RGB inputs must have equal HxWx3 shapes")
    height, width = previous.shape[:2]
    previous_u8 = np.clip(previous * 255.0, 0.0, 255.0).astype(np.uint8)
    current_u8 = np.clip(current * 255.0, 0.0, 255.0).astype(np.uint8)
    previous_gray = cv2.cvtColor(previous_u8, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_u8, cv2.COLOR_RGB2GRAY)

    side_start = float(config["side_start"])
    vertical_start = int(round(float(config["vertical_start"]) * height))
    side_width = int(round(0.5 * (1.0 - side_start) * width))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[vertical_start:, :side_width] = 255
    mask[vertical_start:, width - side_width :] = 255
    previous_points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=int(config["max_corners"]),
        qualityLevel=float(config["quality_level"]),
        minDistance=float(config["min_corner_distance_px"]),
        mask=mask,
        blockSize=int(config["corner_block_size"]),
    )
    if previous_points is None or len(previous_points) == 0:
        return _empty_result()

    window = int(config["lk_window_size"])
    levels = int(config["lk_max_level"])
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(config["lk_iterations"]),
        float(config["lk_epsilon"]),
    )
    current_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        winSize=(window, window),
        maxLevel=levels,
        criteria=criteria,
    )
    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        current_points,
        None,
        winSize=(window, window),
        maxLevel=levels,
        criteria=criteria,
    )
    forward_backward = np.linalg.norm(
        backward_points - previous_points, axis=2
    ).reshape(-1)
    valid = (
        (forward_status.reshape(-1) > 0)
        & (backward_status.reshape(-1) > 0)
        & np.isfinite(current_points).all(axis=(1, 2))
        & (forward_backward <= float(config["forward_backward_threshold_px"]))
    )
    points0 = previous_points[:, 0][valid]
    points1 = current_points[:, 0][valid]
    flow_error = forward_error.reshape(-1)[valid]
    if len(points0) == 0:
        return _empty_result()
    inside = (
        (points1[:, 0] >= 0.0)
        & (points1[:, 0] <= width - 1.0)
        & (points1[:, 1] >= vertical_start)
        & (points1[:, 1] <= height - 1.0)
    )
    points0, points1, flow_error = (
        points0[inside],
        points1[inside],
        flow_error[inside],
    )
    if len(points0) == 0:
        return _empty_result()

    pose0 = np.asarray(previous_world_to_camera, dtype=np.float64)
    pose1 = np.asarray(current_world_to_camera, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    homogeneous = cv2.triangulatePoints(
        k @ pose0[:3], k @ pose1[:3], points0.T, points1.T
    )
    valid_w = np.abs(homogeneous[3]) > 1.0e-8
    world = np.full((len(points0), 3), np.nan, dtype=np.float64)
    world[valid_w] = (
        homogeneous[:3, valid_w] / homogeneous[3:4, valid_w]
    ).T
    world_h = np.concatenate((world, np.ones((len(world), 1))), axis=1)
    depth0 = (world_h @ pose0.T)[:, 2]
    depth1 = (world_h @ pose1.T)[:, 2]
    center0 = np.linalg.inv(pose0)[:3, 3]
    center1 = np.linalg.inv(pose1)[:3, 3]
    ray0 = world - center0
    ray1 = world - center1
    denominator = np.maximum(
        np.linalg.norm(ray0, axis=1) * np.linalg.norm(ray1, axis=1), 1.0e-8
    )
    cosine = np.sum(ray0 * ray1, axis=1) / denominator
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    pixel0 = np.rint(points0).astype(np.int64)
    pixel1 = np.rint(points1).astype(np.int64)
    color0 = previous[pixel0[:, 1], pixel0[:, 0]]
    color1 = current[pixel1[:, 1], pixel1[:, 0]]
    color_error = np.mean(np.abs(color0 - color1), axis=1)
    near = float(config["near_depth_m"])
    geometric = (
        np.isfinite(world).all(axis=1)
        & np.isfinite(angle)
        & (depth0 > 0.1)
        & (depth1 > 0.1)
        & (depth0 <= near)
        & (depth1 <= near)
        & (angle >= float(config["min_parallax_deg"]))
        & (angle <= float(config["max_parallax_deg"]))
        & (color_error <= float(config["color_consistency_threshold"]))
    )
    if not np.any(geometric):
        return _empty_result()
    score = angle / (
        1.0
        + forward_backward[valid][inside]
        + 0.05 * flow_error
        + 4.0 * color_error
    )
    return SparseFlowTriangulation(
        world_points=world[geometric].astype(np.float32),
        colors=(0.5 * (color0 + color1))[geometric].astype(np.float32),
        depths=np.minimum(depth0, depth1)[geometric].astype(np.float32),
        scores=score[geometric].astype(np.float32),
    )


def triangulate_multiview_flow_detail(
    rgb_images,
    world_to_camera_poses,
    intrinsics: np.ndarray,
    config: Mapping[str, object],
) -> SparseFlowTriangulation:
    """Triangulate tracklets jointly and require every view to reproject."""
    images = [np.asarray(image, dtype=np.float32) for image in rgb_images]
    poses = [np.asarray(pose, dtype=np.float64) for pose in world_to_camera_poses]
    if len(images) < 3 or len(images) != len(poses):
        raise ValueError("Multiview flow detail requires equal image/pose lists of length >=3")
    if any(image.shape != images[0].shape for image in images):
        raise ValueError("Multiview flow detail images must have equal shapes")
    height, width = images[0].shape[:2]
    grays = [
        cv2.cvtColor(
            np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        for image in images
    ]
    side_start = float(config["side_start"])
    vertical_start = int(round(float(config["vertical_start"]) * height))
    side_width = int(round(0.5 * (1.0 - side_start) * width))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[vertical_start:, :side_width] = 255
    mask[vertical_start:, width - side_width :] = 255
    first = cv2.goodFeaturesToTrack(
        grays[0],
        maxCorners=int(config["max_corners"]),
        qualityLevel=float(config["quality_level"]),
        minDistance=float(config["min_corner_distance_px"]),
        mask=mask,
        blockSize=int(config["corner_block_size"]),
    )
    if first is None or len(first) == 0:
        return _empty_result()

    window = int(config["lk_window_size"])
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(config["lk_iterations"]),
        float(config["lk_epsilon"]),
    )
    tracks = [first]
    valid = np.ones((len(first),), dtype=np.bool_)
    consistency = np.zeros((len(first),), dtype=np.float32)
    current_points = first
    for index in range(1, len(grays)):
        next_points, status, flow_error = cv2.calcOpticalFlowPyrLK(
            grays[index - 1],
            grays[index],
            current_points,
            None,
            winSize=(window, window),
            maxLevel=int(config["lk_max_level"]),
            criteria=criteria,
        )
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            grays[index],
            grays[index - 1],
            next_points,
            None,
            winSize=(window, window),
            maxLevel=int(config["lk_max_level"]),
            criteria=criteria,
        )
        forward_backward = np.linalg.norm(
            backward - current_points, axis=2
        ).reshape(-1)
        valid &= (
            (status.reshape(-1) > 0)
            & (backward_status.reshape(-1) > 0)
            & np.isfinite(next_points).all(axis=(1, 2))
            & (
                forward_backward
                <= float(config["forward_backward_threshold_px"])
            )
        )
        consistency = np.maximum(
            consistency, forward_backward + 0.05 * flow_error.reshape(-1)
        )
        tracks.append(next_points)
        current_points = next_points

    points = [track[:, 0][valid] for track in tracks]
    consistency = consistency[valid]
    if len(points[0]) == 0:
        return _empty_result()
    inside = np.ones((len(points[0]),), dtype=np.bool_)
    for values in points:
        inside &= (
            (values[:, 0] >= 0.0)
            & (values[:, 0] <= width - 1.0)
            & (values[:, 1] >= vertical_start)
            & (values[:, 1] <= height - 1.0)
        )
    points = [values[inside] for values in points]
    consistency = consistency[inside]
    if len(points[0]) == 0:
        return _empty_result()

    k = np.asarray(intrinsics, dtype=np.float64)
    projections = [k @ pose[:3] for pose in poses]
    world = np.empty((len(points[0]), 3), dtype=np.float64)
    for row in range(len(world)):
        equations = []
        for projection, values in zip(projections, points):
            u, v = values[row]
            equations.extend(
                (u * projection[2] - projection[0], v * projection[2] - projection[1])
            )
        _, _, vh = np.linalg.svd(np.asarray(equations))
        homogeneous = vh[-1]
        world[row] = homogeneous[:3] / homogeneous[3]

    world_h = np.concatenate((world, np.ones((len(world), 1))), axis=1)
    depths = np.stack([(world_h @ pose.T)[:, 2] for pose in poses], axis=1)
    reprojection_errors = []
    for projection, values in zip(projections, points):
        projected = world_h @ projection.T
        projected = projected[:, :2] / np.maximum(projected[:, 2:3], 1.0e-8)
        reprojection_errors.append(np.linalg.norm(projected - values, axis=1))
    reprojection_error = np.max(np.stack(reprojection_errors, axis=1), axis=1)
    center0 = np.linalg.inv(poses[0])[:3, 3]
    center1 = np.linalg.inv(poses[-1])[:3, 3]
    ray0, ray1 = world - center0, world - center1
    cosine = np.sum(ray0 * ray1, axis=1) / np.maximum(
        np.linalg.norm(ray0, axis=1) * np.linalg.norm(ray1, axis=1), 1.0e-8
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    sampled_colors = []
    for image, values in zip(images, points):
        pixels = np.rint(values).astype(np.int64)
        sampled_colors.append(image[pixels[:, 1], pixels[:, 0]])
    sampled_colors = np.stack(sampled_colors, axis=1)
    median_color = np.median(sampled_colors, axis=1)
    color_error = np.max(
        np.mean(np.abs(sampled_colors - median_color[:, None]), axis=2), axis=1
    )
    near = float(config["near_depth_m"])
    valid_geometry = (
        np.isfinite(world).all(axis=1)
        & np.isfinite(depths).all(axis=1)
        & (depths.min(axis=1) > 0.1)
        & (depths.max(axis=1) <= near)
        & (angle >= float(config["min_parallax_deg"]))
        & (angle <= float(config["max_parallax_deg"]))
        & (reprojection_error <= float(config["reprojection_threshold_px"]))
        & (color_error <= float(config["color_consistency_threshold"]))
    )
    if not np.any(valid_geometry):
        return _empty_result()
    score = angle / (
        1.0 + consistency + reprojection_error + 4.0 * color_error
    )
    return SparseFlowTriangulation(
        world_points=world[valid_geometry].astype(np.float32),
        colors=median_color[valid_geometry].astype(np.float32),
        depths=depths.min(axis=1)[valid_geometry].astype(np.float32),
        scores=score[valid_geometry].astype(np.float32),
    )
