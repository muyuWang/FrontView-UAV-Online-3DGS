"""Causal metric-depth certificates from tracked image bearings."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ParallaxDepthEstimate:
    world_point: np.ndarray
    depths: np.ndarray
    reprojection_errors_px: np.ndarray
    current_depth: float
    log_depth_std: float
    position_std: float
    maximum_parallax_deg: float
    minimum_positive_depth: float

    @property
    def finite(self) -> bool:
        return bool(
            np.isfinite(self.world_point).all()
            and np.isfinite(self.depths).all()
            and np.isfinite(self.reprojection_errors_px).all()
            and math.isfinite(self.current_depth)
            and math.isfinite(self.log_depth_std)
            and math.isfinite(self.position_std)
            and math.isfinite(self.maximum_parallax_deg)
        )


@dataclass(frozen=True)
class ConsensusParallaxDepthEstimate:
    estimate: ParallaxDepthEstimate
    reference_indices: tuple[int, ...]
    pair_depths: np.ndarray
    pair_log_depth_stds: np.ndarray
    maximum_pairwise_chi2: float

    @property
    def support(self) -> int:
        return len(self.reference_indices)


def camera_center(world_to_camera: np.ndarray) -> np.ndarray:
    pose = np.asarray(world_to_camera, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("World-to-camera pose must be 4x4")
    return -pose[:3, :3].T @ pose[:3, 3]


def project_world_point(
    world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    world_point: np.ndarray,
) -> tuple[np.ndarray, float]:
    pose = np.asarray(world_to_camera, dtype=np.float64)
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    point = np.asarray(world_point, dtype=np.float64).reshape(3)
    camera_point = pose[:3, :3] @ point + pose[:3, 3]
    depth = float(camera_point[2])
    if not math.isfinite(depth) or abs(depth) <= 1.0e-12:
        return np.full((2,), np.nan, dtype=np.float64), depth
    screen = intrinsic @ camera_point
    return screen[:2] / screen[2], depth


def maximum_parallax_angle_deg(
    world_point: np.ndarray, world_to_camera_poses: Sequence[np.ndarray]
) -> float:
    point = np.asarray(world_point, dtype=np.float64).reshape(3)
    centers = np.stack([camera_center(pose) for pose in world_to_camera_poses])
    rays = point[None] - centers
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        return math.nan
    rays /= norms
    cosine = np.clip(rays @ rays.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(np.min(cosine))))


def _linear_triangulate(
    world_to_camera_poses: Sequence[np.ndarray],
    intrinsics: np.ndarray,
    pixels: Sequence[np.ndarray],
) -> np.ndarray:
    intrinsic_inverse = np.linalg.inv(np.asarray(intrinsics, dtype=np.float64))
    equations = []
    for pose, pixel in zip(world_to_camera_poses, pixels):
        projection = np.asarray(pose, dtype=np.float64)[:3]
        uv = np.asarray(pixel, dtype=np.float64).reshape(2)
        normalized = intrinsic_inverse @ np.asarray([uv[0], uv[1], 1.0])
        equations.append(normalized[0] * projection[2] - projection[0])
        equations.append(normalized[1] * projection[2] - projection[1])
    _, _, right = np.linalg.svd(np.asarray(equations), full_matrices=False)
    homogeneous = right[-1]
    if abs(homogeneous[3]) <= 1.0e-12:
        return np.full((3,), np.nan, dtype=np.float64)
    return homogeneous[:3] / homogeneous[3]


def _point_covariance(
    world_point: np.ndarray,
    world_to_camera_poses: Sequence[np.ndarray],
    intrinsics: np.ndarray,
    pixel_sigma_px: float,
) -> np.ndarray:
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    point = np.asarray(world_point, dtype=np.float64).reshape(3)
    jacobians = []
    for pose in world_to_camera_poses:
        pose = np.asarray(pose, dtype=np.float64)
        rotation = pose[:3, :3]
        camera_point = rotation @ point + pose[:3, 3]
        x, y, z = camera_point
        if z <= 1.0e-8:
            return np.full((3, 3), np.nan, dtype=np.float64)
        local = np.asarray(
            [
                [intrinsic[0, 0] / z, 0.0, -intrinsic[0, 0] * x / (z * z)],
                [0.0, intrinsic[1, 1] / z, -intrinsic[1, 1] * y / (z * z)],
            ],
            dtype=np.float64,
        )
        jacobians.append(local @ rotation)
    jacobian = np.concatenate(jacobians, axis=0)
    information = jacobian.T @ jacobian
    try:
        return np.linalg.inv(information) * float(pixel_sigma_px) ** 2
    except np.linalg.LinAlgError:
        return np.full((3, 3), np.nan, dtype=np.float64)


def triangulate_parallax_depth(
    world_to_camera_poses: Sequence[np.ndarray],
    intrinsics: np.ndarray,
    pixels: Sequence[np.ndarray],
    *,
    current_view_index: int = -1,
    pixel_sigma_px: float = 0.5,
) -> ParallaxDepthEstimate:
    """Triangulate bearings and propagate pixel noise to current log depth."""

    poses = [np.asarray(pose, dtype=np.float64) for pose in world_to_camera_poses]
    observations = [np.asarray(pixel, dtype=np.float64).reshape(2) for pixel in pixels]
    if len(poses) < 2 or len(poses) != len(observations):
        raise ValueError("Triangulation requires equal pose/pixel lists of length >= 2")
    if pixel_sigma_px <= 0.0:
        raise ValueError("Pixel uncertainty must be positive")
    current_index = int(current_view_index) % len(poses)
    point = _linear_triangulate(poses, intrinsics, observations)
    projections = [
        project_world_point(pose, intrinsics, point) for pose in poses
    ]
    errors = np.asarray(
        [
            np.linalg.norm(projected - observed)
            for (projected, _), observed in zip(projections, observations)
        ],
        dtype=np.float64,
    )
    depths = np.asarray([depth for _, depth in projections], dtype=np.float64)
    covariance = _point_covariance(
        point, poses, intrinsics, float(pixel_sigma_px)
    )
    current_rotation = poses[current_index][:3, :3]
    current_depth = float(depths[current_index])
    if np.isfinite(covariance).all() and current_depth > 0.0:
        depth_variance = float(
            current_rotation[2] @ covariance @ current_rotation[2].T
        )
        log_depth_std = math.sqrt(max(depth_variance, 0.0)) / current_depth
        position_std = math.sqrt(max(float(np.linalg.eigvalsh(covariance).max()), 0.0))
    else:
        log_depth_std = math.inf
        position_std = math.inf
    return ParallaxDepthEstimate(
        world_point=point,
        depths=depths,
        reprojection_errors_px=errors,
        current_depth=current_depth,
        log_depth_std=float(log_depth_std),
        position_std=float(position_std),
        maximum_parallax_deg=maximum_parallax_angle_deg(point, poses),
        minimum_positive_depth=float(np.min(depths)) if len(depths) else math.nan,
    )


def certificate_information_gain(
    prior_log_depth_std: float, geometric_log_depth_std: float
) -> float:
    """Return the Gaussian entropy reduction supplied by tracked geometry."""

    prior = float(prior_log_depth_std)
    geometric = float(geometric_log_depth_std)
    if prior <= 0.0 or geometric <= 0.0:
        return 0.0
    if not math.isfinite(prior) or not math.isfinite(geometric):
        return 0.0
    posterior = 1.0 / math.sqrt(1.0 / (prior * prior) + 1.0 / (geometric * geometric))
    return max(0.0, math.log(prior / posterior))


def consensus_triangulate_parallax_depth(
    current_world_to_camera: np.ndarray,
    current_pixel: np.ndarray,
    reference_world_to_camera_poses: Sequence[np.ndarray],
    reference_pixels: Sequence[np.ndarray],
    intrinsics: np.ndarray,
    *,
    pixel_sigma_px: float = 0.5,
    consistency_chi2: float = 3.841458820694124,
    minimum_references: int = 2,
) -> ConsensusParallaxDepthEstimate | None:
    """Find a statistically consistent inverse-depth set and triangulate jointly.

    Each history/current pair independently estimates log depth and its propagated
    uncertainty. The largest all-pairs-compatible subset is selected using the
    one-degree-of-freedom chi-square statistic. This rejects a single accidental
    epipolar match without introducing scene-specific metric thresholds.
    """

    references = [
        np.asarray(pose, dtype=np.float64)
        for pose in reference_world_to_camera_poses
    ]
    pixels = [np.asarray(pixel, dtype=np.float64).reshape(2) for pixel in reference_pixels]
    if len(references) != len(pixels):
        raise ValueError("Reference pose and pixel lists must align")
    if minimum_references < 2:
        raise ValueError("Consensus depth requires at least two reference views")
    if consistency_chi2 <= 0.0:
        raise ValueError("Consistency chi-square threshold must be positive")

    pair_estimates: list[tuple[int, ParallaxDepthEstimate]] = []
    for index, (reference_pose, reference_pixel) in enumerate(zip(references, pixels)):
        estimate = triangulate_parallax_depth(
            [reference_pose, current_world_to_camera],
            intrinsics,
            [reference_pixel, current_pixel],
            current_view_index=1,
            pixel_sigma_px=pixel_sigma_px,
        )
        if (
            estimate.finite
            and estimate.minimum_positive_depth > 0.0
            and estimate.current_depth > 0.0
            and estimate.log_depth_std > 0.0
        ):
            pair_estimates.append((index, estimate))
    if len(pair_estimates) < minimum_references:
        return None

    def pairwise_chi2(subset) -> tuple[bool, float]:
        maximum = 0.0
        for left, right in combinations(subset, 2):
            left_estimate = pair_estimates[left][1]
            right_estimate = pair_estimates[right][1]
            difference = math.log(left_estimate.current_depth) - math.log(
                right_estimate.current_depth
            )
            variance = (
                left_estimate.log_depth_std**2
                + right_estimate.log_depth_std**2
            )
            statistic = difference * difference / max(variance, 1.0e-12)
            maximum = max(maximum, statistic)
            if statistic > consistency_chi2:
                return False, maximum
        return True, maximum

    candidates = []
    for size in range(len(pair_estimates), minimum_references - 1, -1):
        for subset in combinations(range(len(pair_estimates)), size):
            compatible, maximum_chi2 = pairwise_chi2(subset)
            if not compatible:
                continue
            precision = sum(
                1.0 / max(pair_estimates[index][1].log_depth_std**2, 1.0e-12)
                for index in subset
            )
            candidates.append((size, precision, -maximum_chi2, subset, maximum_chi2))
        if candidates:
            break
    if not candidates:
        return None
    _, _, _, subset, maximum_chi2 = max(candidates)
    selected_indices = tuple(pair_estimates[index][0] for index in subset)
    selected_estimates = [pair_estimates[index][1] for index in subset]
    joint_poses = [references[index] for index in selected_indices] + [
        np.asarray(current_world_to_camera, dtype=np.float64)
    ]
    joint_pixels = [pixels[index] for index in selected_indices] + [
        np.asarray(current_pixel, dtype=np.float64).reshape(2)
    ]
    joint = triangulate_parallax_depth(
        joint_poses,
        intrinsics,
        joint_pixels,
        current_view_index=-1,
        pixel_sigma_px=pixel_sigma_px,
    )
    if not joint.finite or joint.minimum_positive_depth <= 0.0:
        return None
    return ConsensusParallaxDepthEstimate(
        estimate=joint,
        reference_indices=selected_indices,
        pair_depths=np.asarray(
            [estimate.current_depth for estimate in selected_estimates],
            dtype=np.float64,
        ),
        pair_log_depth_stds=np.asarray(
            [estimate.log_depth_std for estimate in selected_estimates],
            dtype=np.float64,
        ),
        maximum_pairwise_chi2=float(maximum_chi2),
    )
