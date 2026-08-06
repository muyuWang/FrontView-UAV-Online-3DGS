"""Bounded projective candidate association helpers."""

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Tuple

import numpy as np

from .types import CandidateRecord


def camera_center(world_to_camera: np.ndarray) -> np.ndarray:
    return np.linalg.inv(world_to_camera)[:3, 3]


def project_world(points: np.ndarray, world_to_camera: np.ndarray, K: np.ndarray):
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float32)), axis=1
    )
    camera = homogeneous @ world_to_camera.T
    screen = camera[:, :3] @ K.T
    uv = screen[:, :2] / np.maximum(screen[:, 2:3], 1.0e-8)
    return uv.astype(np.float32), camera[:, 2].astype(np.float32)


def candidate_world_point(candidate: CandidateRecord) -> np.ndarray:
    if candidate.representative_world_point is not None:
        return np.asarray(candidate.representative_world_point, dtype=np.float32)
    return np.median(candidate.proposal_batch.world_points, axis=0).astype(np.float32)


def build_projected_bins(
    candidates: Iterable[CandidateRecord], world_to_camera, K, width, height, cell_size
):
    candidates = list(candidates)
    if not candidates:
        return {}, {}, {}
    points = np.stack([candidate_world_point(candidate) for candidate in candidates])
    uv, depth = project_world(points, world_to_camera, K)
    bins: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    projected: Dict[int, np.ndarray] = {}
    projected_depth: Dict[int, float] = {}
    for candidate, point_uv, point_depth in zip(candidates, uv, depth):
        if (
            point_depth <= 0.0
            or point_uv[0] < 0.0
            or point_uv[0] >= width
            or point_uv[1] < 0.0
            or point_uv[1] >= height
        ):
            continue
        key = (int(point_uv[0] // cell_size), int(point_uv[1] // cell_size))
        bins[key].append(candidate.candidate_id)
        projected[candidate.candidate_id] = point_uv
        projected_depth[candidate.candidate_id] = float(point_depth)
    return bins, projected, projected_depth


def neighboring_candidate_ids(bins, uv, cell_size):
    x = int(uv[0] // cell_size)
    y = int(uv[1] // cell_size)
    result = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            result.extend(bins.get((x + dx, y + dy), ()))
    return result


def parallax_angle(world_point, reference_pose, support_pose):
    ray_a = world_point - camera_center(reference_pose)
    ray_b = world_point - camera_center(support_pose)
    ray_a /= max(np.linalg.norm(ray_a), 1.0e-8)
    ray_b /= max(np.linalg.norm(ray_b), 1.0e-8)
    return float(np.arccos(np.clip(np.dot(ray_a, ray_b), -1.0, 1.0)))
