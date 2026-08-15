"""Causal metric Gaussian births from statistically certified image tracks."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import math
import time
from typing import Sequence

import numpy as np
import torch

from utils_new.frontview_parallax_depth_certificate import (
    certificate_information_gain,
    consensus_triangulate_parallax_depth,
)


DEFAULT_CAUSAL_METRIC_BIRTH_CONFIG = {
    "enabled": False,
    "history_frames": 4,
    "max_features": 2048,
    "minimum_match_score": 0.15,
    "maximum_sampson_error_px": 1.5,
    "pixel_sigma_px": 0.5,
    "consistency_chi2": 3.841458820694124,
    "minimum_references": 2,
    "minimum_information_gain": math.log(math.sqrt(2.0)),
    "maximum_reprojection_error_px": 1.5,
    "shuffle_evidence": False,
    "shuffle_seed": 43,
    "cache_frames": 8,
    "support_mode": "point",
    "birth_mode": "tracked_features",
    "field_neighbors": 6,
    "field_confidence_probability": 0.95,
    "shuffle_field_binding": False,
    "posterior_innovation_chi2": 3.841458820694124,
    "posterior_invalid_prior_policy": "track_only",
    "posterior_action": "fuse",
    "reference_selection": "recent",
    "reference_grid_size": 3,
}


def validate_causal_metric_birth_config(config=None):
    result = deepcopy(DEFAULT_CAUSAL_METRIC_BIRTH_CONFIG)
    if config is not None:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(
                "Unknown CausalMetricBirth options: {}".format(sorted(unknown))
            )
        result.update(config)
    for key in ("enabled", "shuffle_evidence", "shuffle_field_binding"):
        if not isinstance(result[key], bool):
            raise TypeError("CausalMetricBirth.{} must be boolean".format(key))
    for key in (
        "history_frames",
        "max_features",
        "minimum_references",
        "cache_frames",
        "field_neighbors",
        "reference_grid_size",
    ):
        if not isinstance(result[key], int) or result[key] < 1:
            raise ValueError("CausalMetricBirth.{} must be positive".format(key))
    if result["minimum_references"] > result["history_frames"]:
        raise ValueError(
            "CausalMetricBirth.minimum_references cannot exceed history_frames"
        )
    if int(result["field_neighbors"]) < 4:
        raise ValueError("CausalMetricBirth.field_neighbors must be at least four")
    for key in (
        "minimum_match_score",
        "maximum_sampson_error_px",
        "pixel_sigma_px",
        "consistency_chi2",
        "minimum_information_gain",
        "maximum_reprojection_error_px",
        "posterior_innovation_chi2",
    ):
        value = float(result[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("CausalMetricBirth.{} must be positive".format(key))
    if not isinstance(result["shuffle_seed"], int) or result["shuffle_seed"] < 0:
        raise ValueError("CausalMetricBirth.shuffle_seed must be nonnegative")
    if result["support_mode"] not in (
        "point",
        "budget_isotropic",
        "budget_structure",
    ):
        raise ValueError("CausalMetricBirth.support_mode is invalid")
    if result["birth_mode"] not in (
        "tracked_features",
        "local_affine_field",
        "depthcov_recondition",
        "footprint_reanchor",
        "posterior_proxy",
        "cross_fitted_gauge",
    ):
        raise ValueError("CausalMetricBirth.birth_mode is invalid")
    probability = float(result["field_confidence_probability"])
    if not 0.5 < probability < 1.0:
        raise ValueError(
            "CausalMetricBirth.field_confidence_probability must lie in (0.5, 1)"
        )
    if result["posterior_invalid_prior_policy"] not in ("track_only", "abstain"):
        raise ValueError(
            "CausalMetricBirth.posterior_invalid_prior_policy must be "
            "track_only or abstain"
        )
    if result["posterior_action"] not in ("fuse", "observe_only", "abstain"):
        raise ValueError(
            "CausalMetricBirth.posterior_action must be fuse, observe_only, or abstain"
        )
    if result["reference_selection"] not in ("recent", "inverse_depth_fisher"):
        raise ValueError(
            "CausalMetricBirth.reference_selection must be recent or "
            "inverse_depth_fisher"
        )
    return result


def causal_birth_replaces_depth_fallback(config, *, dual_responsibility_enabled=False):
    """Return whether a causal birth mode owns the failed DepthCov rows.

    Posterior-proxy modes preserve the original candidate rows and only update
    certified depths, while footprint reanchoring needs those rows as slots.
    The remaining modes produce an independent metric candidate set and may
    therefore replace the sparse-prior fallback when they are triggered.
    """

    config = validate_causal_metric_birth_config(config)
    if (
        not config["enabled"]
        or bool(dual_responsibility_enabled)
        or config["posterior_action"] == "observe_only"
    ):
        return False
    return config["birth_mode"] in (
        "tracked_features",
        "local_affine_field",
        "depthcov_recondition",
    )


@dataclass(frozen=True)
class CausalMetricBirthBatch:
    uv: np.ndarray
    depths: np.ndarray
    world_points: np.ndarray
    log_depth_stds: np.ndarray
    information_gains: np.ndarray
    supports: np.ndarray

    def __len__(self):
        return int(len(self.depths))


@dataclass(frozen=True)
class CausalDepthPosterior:
    """Candidate-aligned result of a local track/depth posterior update."""

    depths: torch.Tensor
    log_depth_stds: torch.Tensor
    valid: torch.Tensor
    certified: torch.Tensor
    conflicted: torch.Tensor
    bound: torch.Tensor
    track_only: torch.Tensor
    innovation_chi2: torch.Tensor
    information_gain: torch.Tensor
    binding_distances_px: torch.Tensor


@dataclass(frozen=True)
class FisherReferenceSelection:
    indices: tuple[int, ...]
    objective_gain: float
    used_recent_fallback: bool
    candidate_information: np.ndarray


def _camera_center(world_to_camera):
    pose = np.asarray(world_to_camera, dtype=np.float64)
    return -pose[:3, :3].T @ pose[:3, 3]


def far_inverse_depth_fisher(
    current_pose,
    reference_pose,
    intrinsics,
    image_size,
    *,
    grid_size=3,
    pixel_sigma_px=0.5,
):
    """Evaluate far-field inverse-depth information over image bearings.

    A point on current bearing ``d`` is ``C + d / rho``.  After multiplying
    reference-camera coordinates by ``rho``, perspective projection is
    ``pi(q + rho * a)``.  Its analytic derivative at ``rho=0`` gives a
    depth-free Fisher score for distant structure.
    """

    current = np.asarray(current_pose, dtype=np.float64)
    reference = np.asarray(reference_pose, dtype=np.float64)
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    width, height = (int(image_size[0]), int(image_size[1]))
    grid_size = int(grid_size)
    sigma = float(pixel_sigma_px)
    if current.shape != (4, 4) or reference.shape != (4, 4):
        raise ValueError("Camera poses must be 4x4")
    if intrinsic.shape != (3, 3):
        raise ValueError("Intrinsics must be 3x3")
    if width <= 0 or height <= 0 or grid_size < 1 or sigma <= 0.0:
        raise ValueError("Image, grid, and pixel uncertainty must be positive")

    us = (np.arange(grid_size, dtype=np.float64) + 0.5) * width / grid_size
    vs = (np.arange(grid_size, dtype=np.float64) + 0.5) * height / grid_size
    pixels = np.stack(np.meshgrid(us, vs, indexing="xy"), axis=-1).reshape(-1, 2)
    inverse_intrinsic = np.linalg.inv(intrinsic)
    bearings_current = np.concatenate(
        (pixels, np.ones((len(pixels), 1), dtype=np.float64)), axis=1
    ) @ inverse_intrinsic.T
    world_rays = bearings_current @ current[:3, :3]
    q = world_rays @ reference[:3, :3].T
    baseline = reference[:3, :3] @ (
        _camera_center(current) - _camera_center(reference)
    )

    qz = q[:, 2]
    visible = qz > 1.0e-9
    projected = np.full((len(q), 2), np.nan, dtype=np.float64)
    projected[visible, 0] = intrinsic[0, 0] * q[visible, 0] / qz[visible] + intrinsic[0, 2]
    projected[visible, 1] = intrinsic[1, 1] * q[visible, 1] / qz[visible] + intrinsic[1, 2]
    visible &= (
        (projected[:, 0] >= 0.0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < height)
    )

    fisher = np.zeros((len(q),), dtype=np.float64)
    if np.any(visible):
        denominator = qz[visible] ** 2
        du = intrinsic[0, 0] * (
            baseline[0] * qz[visible] - q[visible, 0] * baseline[2]
        ) / denominator
        dv = intrinsic[1, 1] * (
            baseline[1] * qz[visible] - q[visible, 1] * baseline[2]
        ) / denominator
        fisher[visible] = (du * du + dv * dv) / (sigma * sigma)
    return fisher


def select_inverse_depth_fisher_references(
    current_pose,
    candidate_poses,
    intrinsics,
    image_size,
    budget,
    *,
    grid_size=3,
    pixel_sigma_px=0.5,
):
    """Greedily maximize spatially saturated inverse-depth information."""

    poses = [np.asarray(pose, dtype=np.float64) for pose in candidate_poses]
    budget = min(max(0, int(budget)), len(poses))
    if budget == 0:
        return FisherReferenceSelection((), 0.0, False, np.empty((0, 0)))
    information = np.stack(
        [
            far_inverse_depth_fisher(
                current_pose,
                pose,
                intrinsics,
                image_size,
                grid_size=grid_size,
                pixel_sigma_px=pixel_sigma_px,
            )
            for pose in poses
        ],
        axis=0,
    )
    positive = information[information > np.finfo(np.float64).eps]
    if not len(positive):
        indices = tuple(range(len(poses) - budget, len(poses)))
        return FisherReferenceSelection(indices, 0.0, True, information)

    scale = float(np.median(positive))
    normalized = information / max(scale, np.finfo(np.float64).tiny)
    accumulated = np.zeros((information.shape[1],), dtype=np.float64)
    remaining = set(range(len(poses)))
    selected = []
    objective = 0.0
    for _ in range(budget):
        gains = {
            index: float(
                np.sum(np.log1p(accumulated + normalized[index]) - np.log1p(accumulated))
            )
            for index in remaining
        }
        best = max(remaining, key=lambda index: (gains[index], index))
        if gains[best] <= np.finfo(np.float64).eps:
            break
        selected.append(best)
        remaining.remove(best)
        accumulated += normalized[best]
        objective += gains[best]
    if len(selected) < budget:
        for index in range(len(poses) - 1, -1, -1):
            if index not in selected:
                selected.append(index)
            if len(selected) == budget:
                break
    return FisherReferenceSelection(
        tuple(sorted(selected)), objective, False, information
    )


def _empty_candidate_posterior(depths, log_depth_stds, valid):
    depths = torch.as_tensor(depths)
    log_depth_stds = torch.as_tensor(
        log_depth_stds, device=depths.device, dtype=depths.dtype
    )
    valid = torch.as_tensor(valid, device=depths.device, dtype=torch.bool)
    row_count = int(depths.numel())
    zeros_bool = torch.zeros(row_count, device=depths.device, dtype=torch.bool)
    return CausalDepthPosterior(
        depths=depths.clone(),
        log_depth_stds=log_depth_stds.clone(),
        valid=valid.clone(),
        certified=zeros_bool.clone(),
        conflicted=zeros_bool.clone(),
        bound=zeros_bool.clone(),
        track_only=zeros_bool.clone(),
        innovation_chi2=torch.full_like(depths, float("nan")),
        information_gain=torch.zeros_like(depths),
        binding_distances_px=torch.full_like(depths, float("nan")),
    )


def _empty_batch() -> CausalMetricBirthBatch:
    return CausalMetricBirthBatch(
        uv=np.empty((0, 2), dtype=np.float32),
        depths=np.empty((0,), dtype=np.float32),
        world_points=np.empty((0, 3), dtype=np.float32),
        log_depth_stds=np.empty((0,), dtype=np.float32),
        information_gains=np.empty((0,), dtype=np.float32),
        supports=np.empty((0,), dtype=np.float32),
    )


def shuffle_metric_depth_binding(
    batch: CausalMetricBirthBatch,
    current_pose,
    intrinsics,
    *,
    seed,
) -> CausalMetricBirthBatch:
    """Keep track pixels/count fixed while breaking their metric association."""

    if len(batch) < 2:
        return batch
    permutation = np.random.default_rng(int(seed)).permutation(len(batch))
    uv = np.asarray(batch.uv, dtype=np.float64)
    depths = np.asarray(batch.depths, dtype=np.float64)[permutation]
    pose = np.asarray(current_pose, dtype=np.float64)
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    homogeneous = np.concatenate((uv, np.ones((len(uv), 1))), axis=1)
    camera_points = (homogeneous @ np.linalg.inv(intrinsic).T) * depths[:, None]
    world_points = (camera_points - pose[:3, 3]) @ pose[:3, :3]
    return CausalMetricBirthBatch(
        uv=np.asarray(batch.uv, dtype=np.float32).copy(),
        depths=depths.astype(np.float32),
        world_points=world_points.astype(np.float32),
        log_depth_stds=np.asarray(batch.log_depth_stds, dtype=np.float32)[
            permutation
        ].copy(),
        information_gains=np.asarray(batch.information_gains, dtype=np.float32)[
            permutation
        ].copy(),
        supports=np.asarray(batch.supports, dtype=np.float32)[permutation].copy(),
    )


def bind_tracks_to_proxy_slots(
    candidate_uv,
    fallback_positions,
    tracked_uv,
    *,
    support_radius_px,
):
    """Bind ranked metric tracks to unique nearby appearance-proxy slots.

    The admissible distance is the circumradius of the square projective cell
    represented by a proxy. This makes the gate a consequence of the online
    birth budget instead of a scene-specific metric-depth threshold.
    """

    candidate_uv = torch.as_tensor(candidate_uv)
    fallback_positions = torch.as_tensor(
        fallback_positions, device=candidate_uv.device, dtype=torch.long
    ).reshape(-1)
    tracked_uv = torch.as_tensor(
        tracked_uv, device=candidate_uv.device, dtype=candidate_uv.dtype
    ).reshape(-1, 2)
    if candidate_uv.ndim != 2 or candidate_uv.shape[1] != 2:
        raise ValueError("Candidate pixels must have shape [N, 2]")
    if fallback_positions.numel() and bool(
        torch.any(
            (fallback_positions < 0)
            | (fallback_positions >= candidate_uv.shape[0])
        )
    ):
        raise ValueError("Fallback positions are outside the candidate array")
    radius = float(support_radius_px)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("Proxy support radius must be positive")
    maximum_distance_sq = 2.0 * radius * radius
    available = fallback_positions.clone()
    assigned_positions = []
    assigned_tracks = []
    assigned_distances = []
    for track_index, tracked_row in enumerate(tracked_uv):
        if not available.numel():
            break
        distances_sq = torch.sum(
            (candidate_uv[available] - tracked_row.reshape(1, 2)) ** 2,
            dim=1,
        )
        nearest = int(torch.argmin(distances_sq).item())
        if float(distances_sq[nearest].item()) > maximum_distance_sq:
            continue
        assigned_positions.append(available[nearest])
        assigned_tracks.append(track_index)
        assigned_distances.append(torch.sqrt(distances_sq[nearest]))
        available = torch.cat((available[:nearest], available[nearest + 1 :]))
    if not assigned_positions:
        empty_long = torch.empty(0, device=candidate_uv.device, dtype=torch.long)
        empty_float = torch.empty(0, device=candidate_uv.device, dtype=candidate_uv.dtype)
        return empty_long, empty_long.clone(), empty_float
    return (
        torch.stack(assigned_positions),
        torch.as_tensor(assigned_tracks, device=candidate_uv.device, dtype=torch.long),
        torch.stack(assigned_distances),
    )


def bind_tracks_to_responsibility_cells(
    candidate_uv,
    tracked_uv,
    *,
    image_size,
    birth_budget,
):
    """Bind each track to one candidate in the same budget-induced image cell.

    A square cell has area ``image_area / birth_budget``. This makes the
    association scale a consequence of the online representation budget, while
    the same-cell constraint prevents a track from crossing a projective
    responsibility boundary merely because it is the nearest feature.
    """

    candidate_uv = torch.as_tensor(candidate_uv)
    tracked_uv = torch.as_tensor(
        tracked_uv, device=candidate_uv.device, dtype=candidate_uv.dtype
    )
    if candidate_uv.ndim != 2 or candidate_uv.shape[1] != 2:
        raise ValueError("Candidate pixels must have shape [N, 2]")
    if tracked_uv.ndim != 2 or tracked_uv.shape[1] != 2:
        raise ValueError("Tracked pixels must have shape [M, 2]")
    width, height = (int(value) for value in image_size)
    birth_budget = int(birth_budget)
    if width <= 0 or height <= 0 or birth_budget <= 0:
        raise ValueError("Responsibility cells require positive image and budget")
    if not len(candidate_uv) or not len(tracked_uv):
        empty_long = torch.empty(0, device=candidate_uv.device, dtype=torch.long)
        empty_float = torch.empty(
            0, device=candidate_uv.device, dtype=candidate_uv.dtype
        )
        return empty_long, empty_long.clone(), empty_float

    cell_size = math.sqrt(float(width * height) / float(birth_budget))
    candidate_cells = torch.floor(candidate_uv / cell_size).to(torch.long)
    track_cells = torch.floor(tracked_uv / cell_size).to(torch.long)
    available = torch.ones(len(candidate_uv), device=candidate_uv.device, dtype=torch.bool)
    assigned_positions = []
    assigned_tracks = []
    assigned_distances = []
    for track_index, (track_uv, track_cell) in enumerate(zip(tracked_uv, track_cells)):
        same_cell = torch.all(candidate_cells == track_cell.reshape(1, 2), dim=1)
        rows = torch.nonzero(same_cell & available, as_tuple=False).reshape(-1)
        if not rows.numel():
            continue
        distances_sq = torch.sum((candidate_uv[rows] - track_uv.reshape(1, 2)) ** 2, dim=1)
        nearest = rows[int(torch.argmin(distances_sq).item())]
        assigned_positions.append(nearest)
        assigned_tracks.append(track_index)
        assigned_distances.append(torch.sqrt(torch.min(distances_sq)))
        available[nearest] = False
    if not assigned_positions:
        empty_long = torch.empty(0, device=candidate_uv.device, dtype=torch.long)
        empty_float = torch.empty(
            0, device=candidate_uv.device, dtype=candidate_uv.dtype
        )
        return empty_long, empty_long.clone(), empty_float
    return (
        torch.stack(assigned_positions),
        torch.as_tensor(assigned_tracks, device=candidate_uv.device, dtype=torch.long),
        torch.stack(assigned_distances),
    )


def fuse_candidate_log_depth_posteriors(
    candidate_uv,
    prior_depths,
    prior_log_depth_stds,
    prior_valid,
    tracks: CausalMetricBirthBatch,
    *,
    image_size,
    birth_budget,
    config=None,
    seed=43,
):
    """Fuse causal track likelihoods into candidate log-depth priors.

    Candidate pixels and row count never change. A track may update only one
    candidate in its projective responsibility cell. Valid priors additionally
    pass a one-dimensional chi-square innovation test; invalid priors may use a
    certified track directly, or fail closed according to configuration.
    """

    config = validate_causal_metric_birth_config(config)
    candidate_uv = torch.as_tensor(candidate_uv)
    prior_depths = torch.as_tensor(
        prior_depths, device=candidate_uv.device, dtype=torch.float32
    ).reshape(-1)
    prior_log_depth_stds = torch.as_tensor(
        prior_log_depth_stds, device=candidate_uv.device, dtype=torch.float32
    ).reshape(-1)
    prior_valid = torch.as_tensor(
        prior_valid, device=candidate_uv.device, dtype=torch.bool
    ).reshape(-1)
    if not (
        len(candidate_uv)
        == len(prior_depths)
        == len(prior_log_depth_stds)
        == len(prior_valid)
    ):
        raise ValueError("Candidate posterior arrays must align")
    result = _empty_candidate_posterior(
        prior_depths, prior_log_depth_stds, prior_valid
    )
    if not len(candidate_uv) or not len(tracks):
        return result

    tracked_uv = torch.as_tensor(
        tracks.uv, device=candidate_uv.device, dtype=candidate_uv.dtype
    )
    tracked_depths = torch.as_tensor(
        tracks.depths, device=candidate_uv.device, dtype=torch.float32
    )
    tracked_stds = torch.as_tensor(
        tracks.log_depth_stds, device=candidate_uv.device, dtype=torch.float32
    )
    tracked_information = torch.as_tensor(
        tracks.information_gains, device=candidate_uv.device, dtype=torch.float32
    )
    if bool(config["shuffle_field_binding"]) and len(tracks) > 1:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        permutation = torch.randperm(len(tracks), generator=generator).to(
            device=candidate_uv.device
        )
        tracked_depths = tracked_depths[permutation]
        tracked_stds = tracked_stds[permutation]
        tracked_information = tracked_information[permutation]

    positions, track_rows, distances = bind_tracks_to_responsibility_cells(
        candidate_uv,
        tracked_uv,
        image_size=image_size,
        birth_budget=birth_budget,
    )
    if not positions.numel():
        return result

    depths = result.depths.clone()
    log_stds = result.log_depth_stds.clone()
    valid = result.valid.clone()
    certified = result.certified.clone()
    conflicted = result.conflicted.clone()
    bound = result.bound.clone()
    track_only = result.track_only.clone()
    innovation_chi2 = result.innovation_chi2.clone()
    information_gain = result.information_gain.clone()
    binding_distances = result.binding_distances_px.clone()
    bound[positions] = True
    binding_distances[positions] = distances.to(binding_distances)

    eps = torch.finfo(torch.float32).eps
    track_depth = torch.clamp(tracked_depths[track_rows], min=eps)
    track_std = torch.clamp(tracked_stds[track_rows], min=eps)
    rows_with_prior = prior_valid[positions] & torch.isfinite(prior_depths[positions])
    rows_with_prior &= prior_depths[positions] > 0.0
    if bool(rows_with_prior.any().item()):
        positions_prior = positions[rows_with_prior]
        track_depth_prior = track_depth[rows_with_prior]
        track_std_prior = track_std[rows_with_prior]
        prior_depth = torch.clamp(prior_depths[positions_prior], min=eps)
        prior_std = torch.clamp(prior_log_depth_stds[positions_prior], min=eps)
        prior_log = torch.log(prior_depth)
        track_log = torch.log(track_depth_prior)
        variance_sum = prior_std.square() + track_std_prior.square()
        statistic = (prior_log - track_log).square() / torch.clamp(
            variance_sum, min=eps
        )
        innovation_chi2[positions_prior] = statistic
        compatible = statistic <= float(config["posterior_innovation_chi2"])
        conflicted[positions_prior] = ~compatible

        prior_precision = prior_std.reciprocal().square()
        track_precision = track_std_prior.reciprocal().square()
        posterior_precision = prior_precision + track_precision
        posterior_std = torch.rsqrt(posterior_precision)
        gain = torch.log(prior_std / posterior_std)
        information_gain[positions_prior] = gain
        informative = gain >= float(config["minimum_information_gain"])
        accept = compatible & informative
        accepted_positions = positions_prior[accept]
        if accepted_positions.numel():
            posterior_log = (
                prior_precision[accept] * prior_log[accept]
                + track_precision[accept] * track_log[accept]
            ) / posterior_precision[accept]
            depths[accepted_positions] = torch.exp(posterior_log)
            log_stds[accepted_positions] = posterior_std[accept]
            certified[accepted_positions] = True

    rows_without_prior = ~rows_with_prior
    if (
        config["posterior_invalid_prior_policy"] == "track_only"
        and bool(rows_without_prior.any().item())
    ):
        accepted_positions = positions[rows_without_prior]
        depths[accepted_positions] = track_depth[rows_without_prior]
        log_stds[accepted_positions] = track_std[rows_without_prior]
        valid[accepted_positions] = True
        certified[accepted_positions] = True
        track_only[accepted_positions] = True
        information_gain[accepted_positions] = tracked_information[
            track_rows[rows_without_prior]
        ]

    if config["posterior_action"] == "observe_only":
        depths = result.depths
        log_stds = result.log_depth_stds
        valid = result.valid
    return CausalDepthPosterior(
        depths=depths,
        log_depth_stds=log_stds,
        valid=valid,
        certified=certified,
        conflicted=conflicted,
        bound=bound,
        track_only=track_only,
        innovation_chi2=innovation_chi2,
        information_gain=information_gain,
        binding_distances_px=binding_distances,
    )


def fundamental_from_poses(pose0, pose1, intrinsics):
    """Return the fundamental matrix for world-to-camera poses."""

    first = np.asarray(pose0, dtype=np.float64)
    second = np.asarray(pose1, dtype=np.float64)
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    relative = second @ np.linalg.inv(first)
    rotation, translation = relative[:3, :3], relative[:3, 3]
    skew = np.asarray(
        [
            [0.0, -translation[2], translation[1]],
            [translation[2], 0.0, -translation[0]],
            [-translation[1], translation[0], 0.0],
        ],
        dtype=np.float64,
    )
    inverse = np.linalg.inv(intrinsic)
    return inverse.T @ skew @ rotation @ inverse


def sampson_errors(fundamental, pixels0, pixels1):
    first = np.asarray(pixels0, dtype=np.float64).reshape(-1, 2)
    second = np.asarray(pixels1, dtype=np.float64).reshape(-1, 2)
    if len(first) != len(second):
        raise ValueError("Sampson pixel arrays must align")
    if not len(first):
        return np.empty((0,), dtype=np.float64)
    first_h = np.concatenate((first, np.ones((len(first), 1))), axis=1)
    second_h = np.concatenate((second, np.ones((len(second), 1))), axis=1)
    line1 = first_h @ np.asarray(fundamental, dtype=np.float64).T
    line0 = second_h @ np.asarray(fundamental, dtype=np.float64)
    numerator = np.sum(second_h * line1, axis=1) ** 2
    denominator = (
        line1[:, 0] ** 2
        + line1[:, 1] ** 2
        + line0[:, 0] ** 2
        + line0[:, 1] ** 2
    )
    return np.sqrt(numerator / np.maximum(denominator, 1.0e-12))


def certify_track_observations(
    current_pose,
    current_keypoints,
    reference_poses,
    observations_by_current,
    intrinsics,
    prior_log_depth_std,
    config,
    *,
    residual_scores=None,
    maximum_births=None,
):
    """Triangulate tracks and admit only statistically informative depths."""

    config = validate_causal_metric_birth_config(config)
    current_keypoints = np.asarray(current_keypoints, dtype=np.float64).reshape(-1, 2)
    reference_poses = [np.asarray(pose, dtype=np.float64) for pose in reference_poses]
    residual_scores = (
        np.ones((len(current_keypoints),), dtype=np.float64)
        if residual_scores is None
        else np.asarray(residual_scores, dtype=np.float64).reshape(-1)
    )
    if len(residual_scores) != len(current_keypoints):
        raise ValueError("Residual scores must align with current keypoints")
    accepted = []
    for current_index, observations in observations_by_current.items():
        current_index = int(current_index)
        if current_index < 0 or current_index >= len(current_keypoints):
            continue
        if len(observations) < int(config["minimum_references"]):
            continue
        reference_indices = [int(item[0]) for item in observations]
        reference_pixels = [np.asarray(item[1], dtype=np.float64) for item in observations]
        consensus = consensus_triangulate_parallax_depth(
            current_pose,
            current_keypoints[current_index],
            [reference_poses[index] for index in reference_indices],
            reference_pixels,
            intrinsics,
            pixel_sigma_px=float(config["pixel_sigma_px"]),
            consistency_chi2=float(config["consistency_chi2"]),
            minimum_references=int(config["minimum_references"]),
        )
        if consensus is None:
            continue
        estimate = consensus.estimate
        information_gain = certificate_information_gain(
            prior_log_depth_std, estimate.log_depth_std
        )
        if information_gain < float(config["minimum_information_gain"]):
            continue
        if float(np.max(estimate.reprojection_errors_px)) > float(
            config["maximum_reprojection_error_px"]
        ):
            continue
        accepted.append(
            (
                residual_scores[current_index] * information_gain,
                current_index,
                estimate,
                information_gain,
                consensus.support,
            )
        )
    accepted.sort(key=lambda item: (-item[0], item[1]))
    if maximum_births is not None:
        accepted = accepted[: max(0, int(maximum_births))]
    if not accepted:
        return _empty_batch()
    return CausalMetricBirthBatch(
        uv=np.asarray(
            [current_keypoints[item[1]] for item in accepted], dtype=np.float32
        ),
        depths=np.asarray(
            [item[2].current_depth for item in accepted], dtype=np.float32
        ),
        world_points=np.asarray(
            [item[2].world_point for item in accepted], dtype=np.float32
        ),
        log_depth_stds=np.asarray(
            [item[2].log_depth_std for item in accepted], dtype=np.float32
        ),
        information_gains=np.asarray(
            [item[3] for item in accepted], dtype=np.float32
        ),
        supports=np.asarray([item[4] for item in accepted], dtype=np.float32),
    )


def propagate_local_affine_inverse_depth(
    anchors: CausalMetricBirthBatch,
    query_uv,
    current_pose,
    intrinsics,
    prior_log_depth_std,
    config,
    *,
    residual_scores=None,
    maximum_births=None,
    seed=43,
):
    """Certify residual-grid depths with local affine inverse-depth models.

    Perspective projection makes inverse depth affine in image coordinates for
    a 3D plane. Each query is fitted only from its nearest tracked metric
    anchors, must lie inside their convex hull, and must pass a chi-square
    goodness-of-fit test. Propagated uncertainty must reduce depth entropy.
    """

    from scipy.spatial import Delaunay, QhullError
    from scipy.stats import chi2

    config = validate_causal_metric_birth_config(config)
    query_uv = np.asarray(query_uv, dtype=np.float64).reshape(-1, 2)
    if len(anchors) < int(config["field_neighbors"]) or not len(query_uv):
        return _empty_batch()
    anchor_uv = np.asarray(anchors.uv, dtype=np.float64)
    anchor_depth = np.asarray(anchors.depths, dtype=np.float64)
    anchor_log_std = np.asarray(anchors.log_depth_stds, dtype=np.float64)
    if bool(config["shuffle_field_binding"]) and len(anchors) > 1:
        permutation = np.random.default_rng(int(seed)).permutation(len(anchors))
        anchor_depth = anchor_depth[permutation]
        anchor_log_std = anchor_log_std[permutation]
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(current_pose, dtype=np.float64)
    image_scale = np.asarray(
        [max(2.0 * intrinsic[0, 2], 1.0), max(2.0 * intrinsic[1, 2], 1.0)],
        dtype=np.float64,
    )
    normalized_anchor = anchor_uv / image_scale
    normalized_query = query_uv / image_scale
    residual_scores = (
        np.ones((len(query_uv),), dtype=np.float64)
        if residual_scores is None
        else np.asarray(residual_scores, dtype=np.float64).reshape(-1)
    )
    if len(residual_scores) != len(query_uv):
        raise ValueError("Field residual scores must align with query pixels")
    neighbors = int(config["field_neighbors"])
    degrees_of_freedom = neighbors - 3
    chi2_limit = float(
        chi2.ppf(float(config["field_confidence_probability"]), degrees_of_freedom)
    )
    accepted = []
    for query_index, query in enumerate(normalized_query):
        distances = np.sum((normalized_anchor - query[None]) ** 2, axis=1)
        rows = np.argpartition(distances, neighbors - 1)[:neighbors]
        local_uv = normalized_anchor[rows]
        try:
            if Delaunay(local_uv).find_simplex(query.reshape(1, 2))[0] < 0:
                continue
        except QhullError:
            continue
        offsets = local_uv - query[None]
        design = np.column_stack((np.ones((neighbors,)), offsets))
        inverse_depth = 1.0 / anchor_depth[rows]
        inverse_std = inverse_depth * anchor_log_std[rows]
        if np.any(~np.isfinite(inverse_std)) or np.any(inverse_std <= 0.0):
            continue
        precision = 1.0 / np.square(inverse_std)
        information = design.T @ (precision[:, None] * design)
        try:
            covariance = np.linalg.inv(information)
        except np.linalg.LinAlgError:
            continue
        coefficients = covariance @ (
            design.T @ (precision * inverse_depth)
        )
        predicted_inverse = float(coefficients[0])
        if not math.isfinite(predicted_inverse) or predicted_inverse <= 0.0:
            continue
        normalized_residual = (
            inverse_depth - design @ coefficients
        ) / inverse_std
        statistic = float(np.sum(np.square(normalized_residual)))
        if not math.isfinite(statistic) or statistic > chi2_limit:
            continue
        inflation = max(1.0, statistic / max(degrees_of_freedom, 1))
        predicted_inverse_std = math.sqrt(
            max(float(covariance[0, 0]) * inflation, 0.0)
        )
        predicted_log_std = predicted_inverse_std / predicted_inverse
        information_gain = certificate_information_gain(
            prior_log_depth_std, predicted_log_std
        )
        if information_gain < float(config["minimum_information_gain"]):
            continue
        accepted.append(
            (
                residual_scores[query_index] * information_gain,
                query_index,
                1.0 / predicted_inverse,
                predicted_log_std,
                information_gain,
                neighbors,
            )
        )
    accepted.sort(key=lambda item: (-item[0], item[1]))
    if maximum_births is not None:
        accepted = accepted[: max(0, int(maximum_births))]
    if not accepted:
        return _empty_batch()
    selected_uv = np.asarray([query_uv[item[1]] for item in accepted])
    depths = np.asarray([item[2] for item in accepted], dtype=np.float64)
    homogeneous = np.concatenate(
        (selected_uv, np.ones((len(selected_uv), 1), dtype=np.float64)), axis=1
    )
    camera_points = (homogeneous @ np.linalg.inv(intrinsic).T) * depths[:, None]
    world_points = (camera_points - pose[:3, 3]) @ pose[:3, :3]
    return CausalMetricBirthBatch(
        uv=selected_uv.astype(np.float32),
        depths=depths.astype(np.float32),
        world_points=world_points.astype(np.float32),
        log_depth_stds=np.asarray([item[3] for item in accepted], dtype=np.float32),
        information_gains=np.asarray([item[4] for item in accepted], dtype=np.float32),
        supports=np.asarray([item[5] for item in accepted], dtype=np.float32),
    )


class CausalMetricBirth:
    """Lazy GPU DISK-LightGlue front end for causal metric birth certificates."""

    def __init__(self, config, device):
        self.config = validate_causal_metric_birth_config(config)
        self.device = torch.device(device)
        self._disk = None
        self._matcher = None
        self._features = OrderedDict()
        self.stats = {
            "calls": 0,
            "reference_frames": 0,
            "extracted_frames": 0,
            "raw_matches": 0,
            "score_matches": 0,
            "epipolar_matches": 0,
            "candidate_tracks": 0,
            "certified_track_anchors": 0,
            "field_calls": 0,
            "recondition_calls": 0,
            "recondition_valid_before": 0,
            "recondition_valid_after": 0,
            "certified_rows": 0,
            "depth_sum_m": 0.0,
            "log_depth_std_sum": 0.0,
            "information_gain_sum": 0.0,
            "support_sum": 0.0,
            "feature_seconds": 0.0,
            "matching_seconds": 0.0,
            "certificate_seconds": 0.0,
            "shuffle_calls": 0,
            "metric_binding_shuffle_calls": 0,
            "reanchor_proxy_rows": 0,
            "reanchor_bound_rows": 0,
            "reanchor_distance_sum_px": 0.0,
            "reanchor_absolute_log_shift_sum": 0.0,
            "posterior_calls": 0,
            "posterior_candidate_rows": 0,
            "posterior_bound_rows": 0,
            "posterior_certified_rows": 0,
            "posterior_track_only_rows": 0,
            "posterior_conflicted_rows": 0,
            "posterior_abstained_rows": 0,
            "posterior_information_gain_sum": 0.0,
            "posterior_binding_distance_sum_px": 0.0,
            "gauge_calls": 0,
            "gauge_field_calls": 0,
            "gauge_fallback_calls": 0,
            "gauge_abstained_calls": 0,
            "gauge_applied_rows": 0,
            "gauge_sample_rows": 0,
            "gauge_log_scale_sum": 0.0,
            "gauge_min_fold_gain_sum": 0.0,
            "gauge_selected_fallback_depth_sum_m": 0.0,
            "gauge_decisions": [],
            "reference_selection_calls": 0,
            "reference_candidate_rows": 0,
            "reference_selected_rows": 0,
            "reference_recent_fallback_calls": 0,
            "reference_frame_gap_sum": 0,
            "reference_frame_gap_max": 0,
            "reference_objective_gain_sum": 0.0,
            "last_frame": -1,
        }

    @property
    def enabled(self):
        return bool(self.config["enabled"])

    def _load_models(self):
        if self._disk is not None:
            return
        from kornia.feature import DISK, LightGlueMatcher

        self._disk = DISK.from_pretrained("depth", device=self.device).eval()
        self._matcher = LightGlueMatcher("disk").to(self.device).eval()

    @staticmethod
    def _frame_id(camera):
        return int(camera.cam_idx)

    def _rng_devices(self):
        if self.device.type != "cuda":
            return []
        index = self.device.index
        if index is None:
            index = torch.cuda.current_device()
        return [int(index)]

    def select_reference_cameras(self, camera, candidates):
        candidates = list(candidates or ())
        budget = min(int(self.config["history_frames"]), len(candidates))
        if self.config["reference_selection"] == "recent" or budget == 0:
            selected = candidates[-budget:] if budget else []
            result = None
        else:
            result = select_inverse_depth_fisher_references(
                camera.get_raw_pose().detach().cpu().numpy(),
                [candidate.get_raw_pose().detach().cpu().numpy() for candidate in candidates],
                camera.get_int_mat(0).detach().cpu().numpy(),
                (camera.get_width(0), camera.get_height(0)),
                budget,
                grid_size=int(self.config["reference_grid_size"]),
                pixel_sigma_px=float(self.config["pixel_sigma_px"]),
            )
            selected = [candidates[index] for index in result.indices]
        self.stats["reference_selection_calls"] += 1
        self.stats["reference_candidate_rows"] += len(candidates)
        self.stats["reference_selected_rows"] += len(selected)
        if result is not None:
            self.stats["reference_recent_fallback_calls"] += int(
                result.used_recent_fallback
            )
            self.stats["reference_objective_gain_sum"] += float(
                result.objective_gain
            )
        current_frame = self._frame_id(camera)
        gaps = [current_frame - self._frame_id(reference) for reference in selected]
        self.stats["reference_frame_gap_sum"] += sum(gaps)
        self.stats["reference_frame_gap_max"] = max(
            [self.stats["reference_frame_gap_max"], *gaps]
        )
        return selected

    def _extract(self, camera, level):
        frame_id = self._frame_id(camera)
        cache_key = (frame_id, int(level))
        cached = self._features.get(cache_key)
        if cached is not None:
            self._features.move_to_end(cache_key)
            return cached
        image = camera.get_gt_image(level).detach().to(self.device)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("CausalMetricBirth expects an HxWx3 image")
        tensor = image.permute(2, 0, 1).unsqueeze(0).float()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._disk(
                tensor,
                n=int(self.config["max_features"]),
                window_size=5,
                score_threshold=0.0,
                pad_if_not_divisible=True,
            )[0]
        feature = {
            "keypoints": output.keypoints.detach(),
            "descriptors": output.descriptors.detach(),
            "scores": output.detection_scores.detach(),
            "image_size": (int(image.shape[0]), int(image.shape[1])),
        }
        self.stats["feature_seconds"] += time.perf_counter() - started
        self.stats["extracted_frames"] += 1
        self._features[cache_key] = feature
        while len(self._features) > int(self.config["cache_frames"]):
            self._features.popitem(last=False)
        return feature

    def certify(
        self,
        camera,
        reference_cameras: Sequence,
        level,
        eligible_mask,
        residual_map,
        prior_log_depth_std,
        maximum_births,
        query_uv=None,
    ):
        """Run the visual certificate without advancing the mapper RNG stream."""

        if not self.enabled or maximum_births <= 0:
            return _empty_batch()
        seed = int(self.config["shuffle_seed"]) + self._frame_id(camera)
        with torch.random.fork_rng(devices=self._rng_devices(), enabled=True):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            return self._certify_impl(
                camera,
                reference_cameras,
                level,
                eligible_mask,
                residual_map,
                prior_log_depth_std,
                maximum_births,
                query_uv=query_uv,
            )

    def _certify_impl(
        self,
        camera,
        reference_cameras: Sequence,
        level,
        eligible_mask,
        residual_map,
        prior_log_depth_std,
        maximum_births,
        query_uv=None,
    ):
        references = list(reference_cameras or ())[-int(self.config["history_frames"]):]
        if len(references) < int(self.config["minimum_references"]):
            return _empty_batch()
        self._load_models()
        self.stats["calls"] += 1
        self.stats["reference_frames"] += len(references)
        self.stats["last_frame"] = self._frame_id(camera)
        current = self._extract(camera, level)
        reference_features = [self._extract(reference, level) for reference in references]
        current_keypoints = current["keypoints"]
        height, width = current["image_size"]
        rounded_x = torch.clamp(current_keypoints[:, 0].round().long(), 0, width - 1)
        rounded_y = torch.clamp(current_keypoints[:, 1].round().long(), 0, height - 1)
        eligible = torch.as_tensor(eligible_mask, device=self.device, dtype=torch.bool)
        residual = torch.as_tensor(residual_map, device=self.device, dtype=torch.float32)
        current_eligible = eligible[rounded_y, rounded_x]
        current_indices_allowed = set(
            torch.nonzero(current_eligible, as_tuple=False).flatten().cpu().tolist()
        )
        if not current_indices_allowed:
            return _empty_batch()

        from kornia.feature import laf_from_center_scale_ori

        current_pose = camera.get_raw_pose().detach().cpu().numpy()
        intrinsics = camera.get_int_mat(level).detach().cpu().numpy()
        reference_poses = [
            reference.get_raw_pose().detach().cpu().numpy() for reference in references
        ]
        observations = {}
        started = time.perf_counter()
        rng = np.random.default_rng(
            int(self.config["shuffle_seed"]) + self._frame_id(camera)
        )
        for reference_index, (reference, feature, reference_pose) in enumerate(
            zip(references, reference_features, reference_poses)
        ):
            with torch.inference_mode():
                scores, indices = self._matcher(
                    feature["descriptors"].float(),
                    current["descriptors"].float(),
                    laf_from_center_scale_ori(feature["keypoints"][None]),
                    laf_from_center_scale_ori(current["keypoints"][None]),
                    feature["image_size"],
                    current["image_size"],
                )
            scores = scores.reshape(-1).detach().cpu().numpy()
            indices = indices.detach().cpu().numpy().astype(np.int64, copy=False)
            self.stats["raw_matches"] += len(indices)
            keep = scores >= float(self.config["minimum_match_score"])
            indices = indices[keep]
            scores = scores[keep]
            self.stats["score_matches"] += len(indices)
            if not len(indices):
                continue
            reference_uv = feature["keypoints"][indices[:, 0]].detach().cpu().numpy()
            current_uv = current["keypoints"][indices[:, 1]].detach().cpu().numpy()
            if self.config["shuffle_evidence"] and len(reference_uv) > 1:
                reference_uv = reference_uv[rng.permutation(len(reference_uv))]
            fundamental = fundamental_from_poses(reference_pose, current_pose, intrinsics)
            epipolar = sampson_errors(fundamental, reference_uv, current_uv)
            keep = epipolar <= float(self.config["maximum_sampson_error_px"])
            indices = indices[keep]
            reference_uv = reference_uv[keep]
            self.stats["epipolar_matches"] += len(indices)
            for match, pixel in zip(indices, reference_uv):
                current_index = int(match[1])
                if current_index not in current_indices_allowed:
                    continue
                observations.setdefault(current_index, []).append(
                    (reference_index, pixel)
                )
        self.stats["matching_seconds"] += time.perf_counter() - started
        self.stats["candidate_tracks"] += len(observations)
        self.stats["shuffle_calls"] += int(self.config["shuffle_evidence"])

        current_keypoints_np = current_keypoints.detach().cpu().numpy()
        residual_scores = residual[rounded_y, rounded_x].detach().cpu().numpy()
        started = time.perf_counter()
        tracked = certify_track_observations(
            current_pose,
            current_keypoints_np,
            reference_poses,
            observations,
            intrinsics,
            prior_log_depth_std,
            self.config,
            residual_scores=residual_scores,
            maximum_births=(
                None
                if self.config["birth_mode"] == "local_affine_field"
                else maximum_births
            ),
        )
        self.stats["certified_track_anchors"] += len(tracked)
        result = tracked
        if (
            self.config["birth_mode"] == "tracked_features"
            and bool(self.config["shuffle_field_binding"])
            and len(result) > 1
        ):
            result = shuffle_metric_depth_binding(
                result,
                current_pose,
                intrinsics,
                seed=int(self.config["shuffle_seed"]) + self._frame_id(camera),
            )
            self.stats["metric_binding_shuffle_calls"] += 1
        if (
            self.config["birth_mode"] == "local_affine_field"
            and query_uv is not None
            and len(tracked)
        ):
            self.stats["field_calls"] += 1
            query_uv = torch.as_tensor(
                query_uv, device=self.device, dtype=torch.float32
            ).reshape(-1, 2)
            query_x = torch.clamp(query_uv[:, 0].round().long(), 0, width - 1)
            query_y = torch.clamp(query_uv[:, 1].round().long(), 0, height - 1)
            result = propagate_local_affine_inverse_depth(
                tracked,
                query_uv.detach().cpu().numpy(),
                current_pose,
                intrinsics,
                prior_log_depth_std,
                self.config,
                residual_scores=residual[query_y, query_x].detach().cpu().numpy(),
                maximum_births=maximum_births,
                seed=int(self.config["shuffle_seed"]) + self._frame_id(camera),
            )
        elif self.config["birth_mode"] in (
            "depthcov_recondition",
            "footprint_reanchor",
        ):
            self.stats["recondition_calls"] += 1
            if bool(self.config["shuffle_field_binding"]) and len(tracked) > 1:
                permutation = np.random.default_rng(
                    int(self.config["shuffle_seed"]) + self._frame_id(camera)
                ).permutation(len(tracked))
                result = CausalMetricBirthBatch(
                    uv=tracked.uv.copy(),
                    depths=tracked.depths[permutation].copy(),
                    world_points=tracked.world_points[permutation].copy(),
                    log_depth_stds=tracked.log_depth_stds[permutation].copy(),
                    information_gains=tracked.information_gains[permutation].copy(),
                    supports=tracked.supports[permutation].copy(),
                )
        self.stats["certificate_seconds"] += time.perf_counter() - started
        self.stats["certified_rows"] += len(result)
        self.stats["depth_sum_m"] += float(np.sum(result.depths, dtype=np.float64))
        self.stats["log_depth_std_sum"] += float(
            np.sum(result.log_depth_stds, dtype=np.float64)
        )
        self.stats["information_gain_sum"] += float(
            np.sum(result.information_gains, dtype=np.float64)
        )
        self.stats["support_sum"] += float(np.sum(result.supports, dtype=np.float64))
        return result

    def record_reanchoring(self, proxy_rows, old_depths, new_depths, distances):
        old_depths = torch.as_tensor(old_depths).reshape(-1)
        new_depths = torch.as_tensor(
            new_depths, device=old_depths.device, dtype=old_depths.dtype
        ).reshape(-1)
        distances = torch.as_tensor(distances).reshape(-1)
        if len(old_depths) != len(new_depths) or len(old_depths) != len(distances):
            raise ValueError("Reanchoring statistics must align")
        self.stats["reanchor_proxy_rows"] += int(proxy_rows)
        self.stats["reanchor_bound_rows"] += int(len(old_depths))
        self.stats["reanchor_distance_sum_px"] += float(distances.sum().item())
        if len(old_depths):
            eps = torch.finfo(old_depths.dtype).eps
            shift = torch.log(torch.clamp(new_depths, min=eps)) - torch.log(
                torch.clamp(old_depths, min=eps)
            )
            self.stats["reanchor_absolute_log_shift_sum"] += float(
                torch.abs(shift).sum().item()
            )

    def record_reconditioned_validity(self, valid_before, valid_after):
        self.stats["recondition_valid_before"] += int(valid_before)
        self.stats["recondition_valid_after"] += int(valid_after)

    def record_posterior(self, posterior):
        bound = torch.as_tensor(posterior.bound, dtype=torch.bool).reshape(-1)
        certified = torch.as_tensor(posterior.certified, dtype=torch.bool).reshape(-1)
        conflicted = torch.as_tensor(posterior.conflicted, dtype=torch.bool).reshape(-1)
        track_only = torch.as_tensor(posterior.track_only, dtype=torch.bool).reshape(-1)
        distances = torch.as_tensor(
            posterior.binding_distances_px, dtype=torch.float32
        ).reshape(-1)
        information = torch.as_tensor(
            posterior.information_gain, dtype=torch.float32
        ).reshape(-1)
        self.stats["posterior_calls"] += 1
        self.stats["posterior_candidate_rows"] += int(len(bound))
        self.stats["posterior_bound_rows"] += int(bound.sum().item())
        self.stats["posterior_certified_rows"] += int(certified.sum().item())
        self.stats["posterior_track_only_rows"] += int(track_only.sum().item())
        self.stats["posterior_conflicted_rows"] += int(conflicted.sum().item())
        self.stats["posterior_abstained_rows"] += int(
            (bound & ~certified).sum().item()
        )
        if bool(bound.any().item()):
            self.stats["posterior_binding_distance_sum_px"] += float(
                distances[bound].sum().item()
            )
            self.stats["posterior_information_gain_sum"] += float(
                information[bound].sum().item()
            )

    def record_depth_gauge(self, gauge, *, applied_rows, frame_id):
        self.stats["gauge_calls"] += 1
        self.stats["gauge_sample_rows"] += int(gauge.sample_count)
        self.stats["gauge_min_fold_gain_sum"] += float(
            min(gauge.fold_nll_gains)
        )
        if gauge.selected_model == "calibrated_field":
            self.stats["gauge_field_calls"] += 1
            self.stats["gauge_log_scale_sum"] += float(gauge.log_scale)
        elif gauge.selected_model == "fallback":
            self.stats["gauge_fallback_calls"] += 1
            self.stats["gauge_selected_fallback_depth_sum_m"] += float(
                gauge.selected_fallback_depth
            )
        else:
            self.stats["gauge_abstained_calls"] += 1
        self.stats["gauge_applied_rows"] += int(applied_rows)
        self.stats["gauge_decisions"].append(
            {
                "frame_id": int(frame_id),
                "model": str(gauge.selected_model),
                "sample_rows": int(gauge.sample_count),
                "applied_rows": int(applied_rows),
                "log_scale": float(gauge.log_scale),
                "selected_fallback_depth_m": float(
                    gauge.selected_fallback_depth
                ),
                "fold_nll_gains": [
                    float(value) for value in gauge.fold_nll_gains
                ],
            }
        )

    def summary(self):
        result = dict(self.stats)
        rows = max(int(result["certified_rows"]), 1)
        result.update(
            enabled=self.enabled,
            shuffle_evidence=bool(self.config["shuffle_evidence"]),
            support_mode=self.config["support_mode"],
            birth_mode=self.config["birth_mode"],
            shuffle_field_binding=bool(self.config["shuffle_field_binding"]),
            mean_depth_m=(result["depth_sum_m"] / rows),
            mean_log_depth_std=(result["log_depth_std_sum"] / rows),
            mean_information_gain=(result["information_gain_sum"] / rows),
            mean_support=(result["support_sum"] / rows),
            mean_reanchor_distance_px=(
                result["reanchor_distance_sum_px"]
                / max(int(result["reanchor_bound_rows"]), 1)
            ),
            mean_reanchor_absolute_log_shift=(
                result["reanchor_absolute_log_shift_sum"]
                / max(int(result["reanchor_bound_rows"]), 1)
            ),
            mean_posterior_binding_distance_px=(
                result["posterior_binding_distance_sum_px"]
                / max(int(result["posterior_bound_rows"]), 1)
            ),
            mean_posterior_information_gain=(
                result["posterior_information_gain_sum"]
                / max(int(result["posterior_bound_rows"]), 1)
            ),
            mean_gauge_log_scale=(
                result["gauge_log_scale_sum"]
                / max(int(result["gauge_field_calls"]), 1)
            ),
            mean_gauge_min_fold_gain=(
                result["gauge_min_fold_gain_sum"]
                / max(int(result["gauge_calls"]), 1)
            ),
            mean_gauge_selected_fallback_depth_m=(
                result["gauge_selected_fallback_depth_sum_m"]
                / max(int(result["gauge_fallback_calls"]), 1)
            ),
            reference_selection=self.config["reference_selection"],
            mean_reference_frame_gap=(
                result["reference_frame_gap_sum"]
                / max(int(result["reference_selected_rows"]), 1)
            ),
            mean_reference_objective_gain=(
                result["reference_objective_gain_sum"]
                / max(int(result["reference_selection_calls"]), 1)
            ),
            posterior_action=self.config["posterior_action"],
            posterior_invalid_prior_policy=self.config[
                "posterior_invalid_prior_policy"
            ],
        )
        return result
