"""Geometry-preserving projection for repeated sparse map points."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SparseTrackObservation:
    world_points: np.ndarray
    uv: np.ndarray
    depths: np.ndarray
    pixel_indices: np.ndarray
    source_indices: np.ndarray


def conditional_scale_expansion_limits(
    frequency_scores: np.ndarray, config: Mapping[str, object]
) -> np.ndarray:
    """Assign a finite footprint trust region only to attributed detail rows."""
    scores = np.asarray(frequency_scores, dtype=np.float32).reshape(-1)
    limits = np.full(scores.shape, np.inf, dtype=np.float32)
    control = config.get("conditional_scale_control", {})
    if not isinstance(control, Mapping) or not bool(control.get("enabled", False)):
        return limits
    threshold = float(control.get("frequency_score_threshold", 0.65))
    maximum = float(control.get("max_scale_expansion", 0.0))
    if maximum > 0.0:
        limits[scores >= threshold] = maximum
    return limits


def zbuffer_sparse_tracks(
    world_points: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> SparseTrackObservation:
    """Project persistent sparse points and retain the nearest point per pixel."""
    points = np.asarray(world_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Sparse world points must have shape Nx3")
    if len(points) == 0:
        return SparseTrackObservation(
            world_points=np.empty((0, 3), dtype=np.float32),
            uv=np.empty((0, 2), dtype=np.float32),
            depths=np.empty((0,), dtype=np.float32),
            pixel_indices=np.empty((0,), dtype=np.int64),
            source_indices=np.empty((0,), dtype=np.int64),
        )

    pose = np.asarray(world_to_camera, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    homogeneous = np.concatenate(
        (points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)),
        axis=1,
    )
    camera_points = homogeneous @ pose.T
    depths = camera_points[:, 2]
    screen = camera_points[:, :3] @ k.T
    uv = screen[:, :2] / np.maximum(screen[:, 2:3], 1.0e-8)
    pixel_x = np.floor(uv[:, 0]).astype(np.int64)
    pixel_y = np.floor(uv[:, 1]).astype(np.int64)
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(depths)
        & (depths > 0.0)
        & (pixel_x >= 0)
        & (pixel_x < int(width))
        & (pixel_y >= 0)
        & (pixel_y < int(height))
    )
    source_indices = np.flatnonzero(valid)
    if len(source_indices) == 0:
        return SparseTrackObservation(
            world_points=np.empty((0, 3), dtype=np.float32),
            uv=np.empty((0, 2), dtype=np.float32),
            depths=np.empty((0,), dtype=np.float32),
            pixel_indices=np.empty((0,), dtype=np.int64),
            source_indices=np.empty((0,), dtype=np.int64),
        )

    valid_x = pixel_x[source_indices]
    valid_y = pixel_y[source_indices]
    valid_depth = depths[source_indices]
    flat_pixels = valid_y * int(width) + valid_x
    order = np.lexsort((source_indices, valid_depth, flat_pixels))
    ordered_pixels = flat_pixels[order]
    nearest = np.concatenate(
        (np.ones((1,), dtype=np.bool_), ordered_pixels[1:] != ordered_pixels[:-1])
    )
    selected = source_indices[order[nearest]]
    selected_pixels = flat_pixels[order[nearest]]
    return SparseTrackObservation(
        world_points=points[selected].astype(np.float32, copy=False),
        uv=uv[selected].astype(np.float32, copy=False),
        depths=depths[selected].astype(np.float32, copy=False),
        pixel_indices=selected_pixels.astype(np.int64, copy=False),
        source_indices=selected.astype(np.int64, copy=False),
    )
