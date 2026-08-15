"""Causal reprojection memory for persistent sparse world landmarks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import torch

from utils_new.aerocommit.sparse_track_geometry import zbuffer_sparse_tracks


DEFAULT_CAUSAL_LANDMARK_MEMORY_CONFIG = {
    "enabled": False,
    "conditioning_mode": "all_queries",
    "minimum_observations": 1,
    "maximum_conditioning_points": 500,
    "transport_rule": "full",
    "propagate_conditioned_uncertainty": False,
    "responsibility_coordinate": "metric",
    "shuffle_depths": False,
    "shuffle_seed": 43,
}


def information_gain_transport(
    original_depth,
    conditioned_depth,
    original_log_std,
    conditioned_log_std,
):
    """Transport log depth in proportion to posterior variance reduction."""

    tiny = torch.finfo(original_depth.dtype).tiny
    variance_ratio = conditioned_log_std.square() / torch.clamp(
        original_log_std.square(), min=tiny
    )
    weight = torch.clamp(1.0 - variance_ratio, min=0.0, max=1.0)
    transported = torch.exp(
        torch.log(torch.clamp(original_depth, min=tiny))
        + weight
        * (
            torch.log(torch.clamp(conditioned_depth, min=tiny))
            - torch.log(torch.clamp(original_depth, min=tiny))
        )
    )
    return transported, weight


def validate_causal_landmark_memory_config(config=None):
    result = deepcopy(DEFAULT_CAUSAL_LANDMARK_MEMORY_CONFIG)
    if config is not None:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(
                "Unknown CausalPersistentLandmarkMemory options: {}".format(
                    sorted(unknown)
                )
            )
        result.update(config)
    for key in (
        "enabled",
        "propagate_conditioned_uncertainty",
        "shuffle_depths",
    ):
        if not isinstance(result[key], bool):
            raise TypeError(
                "CausalPersistentLandmarkMemory.{} must be boolean".format(key)
            )
    if result["conditioning_mode"] not in (
        "all_queries",
        "admitted_mean",
        "fallback_repair",
    ):
        raise ValueError(
            "CausalPersistentLandmarkMemory.conditioning_mode is invalid"
        )
    if result["transport_rule"] not in ("full", "variance_gain"):
        raise ValueError("CausalPersistentLandmarkMemory.transport_rule is invalid")
    if result["responsibility_coordinate"] not in (
        "metric",
        "original_posterior",
    ):
        raise ValueError(
            "CausalPersistentLandmarkMemory.responsibility_coordinate is invalid"
        )
    for key in ("minimum_observations", "maximum_conditioning_points"):
        if not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(
                "CausalPersistentLandmarkMemory.{} must be positive".format(key)
            )
    if not isinstance(result["shuffle_seed"], int) or result["shuffle_seed"] < 0:
        raise ValueError(
            "CausalPersistentLandmarkMemory.shuffle_seed must be nonnegative"
        )
    return result


@dataclass(frozen=True)
class LandmarkConditionBatch:
    point_ids: np.ndarray
    world_points: np.ndarray
    uv: np.ndarray
    depths: np.ndarray
    pixel_indices: np.ndarray
    observation_counts: np.ndarray
    last_seen_frames: np.ndarray

    def __len__(self):
        return int(len(self.depths))


def empty_landmark_batch() -> LandmarkConditionBatch:
    return LandmarkConditionBatch(
        point_ids=np.empty((0,), dtype=np.int64),
        world_points=np.empty((0, 3), dtype=np.float32),
        uv=np.empty((0, 2), dtype=np.float32),
        depths=np.empty((0,), dtype=np.float32),
        pixel_indices=np.empty((0,), dtype=np.int64),
        observation_counts=np.empty((0,), dtype=np.int64),
        last_seen_frames=np.empty((0,), dtype=np.int64),
    )


def shuffle_landmark_depths(batch, seed):
    """Permute only metric depths while preserving UV, count, and compute budget."""

    if len(batch) < 2:
        return batch
    permutation = np.random.default_rng(int(seed)).permutation(len(batch))
    return LandmarkConditionBatch(
        point_ids=batch.point_ids.copy(),
        world_points=batch.world_points.copy(),
        uv=batch.uv.copy(),
        depths=batch.depths[permutation].copy(),
        pixel_indices=batch.pixel_indices.copy(),
        observation_counts=batch.observation_counts.copy(),
        last_seen_frames=batch.last_seen_frames.copy(),
    )


class CausalPersistentLandmarkMemory:
    """Accumulate only already-read persistent landmarks and reproject them.

    The memory never creates Gaussian candidates. It restores the conditioning
    support of the existing DepthCov query while preserving its candidate UV,
    RGB, count, and downstream admission rules.
    """

    def __init__(self, config=None):
        self.config = validate_causal_landmark_memory_config(config)
        self._points = {}
        self._counts = {}
        self._first_seen = {}
        self._last_seen = {}
        self._observed_frames = set()
        self.stats = {
            "observed_frames": 0,
            "observed_rows": 0,
            "unique_landmarks": 0,
            "repeated_observations": 0,
            "inconsistent_observations": 0,
            "query_calls": 0,
            "eligible_landmarks": 0,
            "projected_landmarks": 0,
            "conditioning_landmarks": 0,
            "occupied_pixel_rejections": 0,
            "budget_rejections": 0,
            "shuffled_calls": 0,
            "conditioning_depth_sum_m": 0.0,
            "last_conditioning_landmarks": 0,
            "repair_trigger_calls": 0,
            "repair_conditioned_calls": 0,
            "repair_skipped_no_landmarks": 0,
            "repair_preserved_valid_rows": 0,
            "repair_invalid_before_rows": 0,
            "repair_newly_valid_rows": 0,
            "repair_remaining_invalid_rows": 0,
            "admitted_mean_calls": 0,
            "admitted_mean_conditioned_calls": 0,
            "admitted_mean_rows": 0,
            "admitted_mean_absolute_log_shift_sum": 0.0,
            "admitted_mean_transport_weight_sum": 0.0,
            "admitted_mean_uncertainty_rows": 0,
            "admitted_mean_original_log_std_sum": 0.0,
            "admitted_mean_conditioned_log_std_sum": 0.0,
            "responsibility_coordinate_rows": 0,
            "responsibility_shifted_rows": 0,
            "responsibility_absolute_log_shift_sum": 0.0,
            "responsibility_registered_rows": 0,
        }

    @property
    def enabled(self):
        return bool(self.config["enabled"])

    @property
    def uses_original_posterior_responsibility(self):
        return bool(
            self.enabled
            and self.config["conditioning_mode"] == "admitted_mean"
            and self.config["responsibility_coordinate"]
            == "original_posterior"
        )

    def record_responsibility_coordinates(self, metric_depths, responsibility_depths):
        if not self.uses_original_posterior_responsibility:
            return
        metric = np.asarray(metric_depths, dtype=np.float64).reshape(-1)
        responsibility = np.asarray(
            responsibility_depths, dtype=np.float64
        ).reshape(-1)
        valid = (
            np.isfinite(metric)
            & np.isfinite(responsibility)
            & (metric > 0.0)
            & (responsibility > 0.0)
        )
        shift = np.zeros_like(metric)
        shift[valid] = np.abs(
            np.log(metric[valid]) - np.log(responsibility[valid])
        )
        self.stats["responsibility_coordinate_rows"] += int(np.sum(valid))
        self.stats["responsibility_shifted_rows"] += int(
            np.sum(shift > 1.0e-7)
        )
        self.stats["responsibility_absolute_log_shift_sum"] += float(
            np.sum(shift[valid], dtype=np.float64)
        )

    def record_responsibility_registration(self, metric_depths, responsibility_depths):
        if not self.uses_original_posterior_responsibility:
            return
        metric = np.asarray(metric_depths, dtype=np.float64).reshape(-1)
        responsibility = np.asarray(
            responsibility_depths, dtype=np.float64
        ).reshape(-1)
        valid = (
            np.isfinite(metric)
            & np.isfinite(responsibility)
            & (metric > 0.0)
            & (responsibility > 0.0)
        )
        log_shift = np.zeros_like(metric)
        log_shift[valid] = np.abs(
            np.log(metric[valid]) - np.log(responsibility[valid])
        )
        self.stats["responsibility_registered_rows"] += int(
            np.sum(valid & (log_shift > 1.0e-7))
        )

    def observe(self, camera):
        if not self.enabled:
            return 0
        frame_id = int(camera.cam_idx)
        if frame_id in self._observed_frames:
            return 0
        self._observed_frames.add(frame_id)
        rows = np.asarray(camera.get_color_pts_depth())
        point_ids = np.asarray(camera.get_point_ids(), dtype=np.int64).reshape(-1)
        if rows.ndim != 2 or rows.shape[1] < 3 or len(rows) != len(point_ids):
            raise ValueError("Persistent landmark IDs must align with sparse world rows")
        world_points = np.asarray(rows[:, :3], dtype=np.float64)
        valid = (point_ids >= 0) & np.isfinite(world_points).all(axis=1)
        point_ids = point_ids[valid]
        world_points = world_points[valid]
        if len(point_ids):
            _, unique_rows = np.unique(point_ids, return_index=True)
            unique_rows.sort()
            point_ids = point_ids[unique_rows]
            world_points = world_points[unique_rows]
        repeated = 0
        inconsistent = 0
        for point_id, point in zip(point_ids.tolist(), world_points):
            point_id = int(point_id)
            if point_id in self._points:
                repeated += 1
                previous = self._points[point_id]
                scale = max(float(np.linalg.norm(previous)), 1.0)
                if float(np.linalg.norm(point - previous)) > 1.0e-5 * scale:
                    inconsistent += 1
                count = int(self._counts[point_id])
                self._points[point_id] = previous + (point - previous) / (count + 1)
                self._counts[point_id] = count + 1
            else:
                self._points[point_id] = point.copy()
                self._counts[point_id] = 1
                self._first_seen[point_id] = frame_id
            self._last_seen[point_id] = frame_id
        self.stats["observed_frames"] += 1
        self.stats["observed_rows"] += int(len(point_ids))
        self.stats["unique_landmarks"] = int(len(self._points))
        self.stats["repeated_observations"] += repeated
        self.stats["inconsistent_observations"] += inconsistent
        return int(len(point_ids))

    def project(
        self,
        camera,
        level=0,
        *,
        exclude_ids=(),
        occupied_pixel_indices=(),
        maximum_points=None,
        shuffle_depths=None,
        shuffle_seed=None,
    ):
        if not self.enabled or not self._points:
            return empty_landmark_batch()
        frame_id = int(camera.cam_idx)
        excluded = set(np.asarray(list(exclude_ids), dtype=np.int64).reshape(-1).tolist())
        minimum = int(self.config["minimum_observations"])
        point_ids = np.asarray(
            [
                point_id
                for point_id in sorted(self._points)
                if point_id not in excluded
                and int(self._first_seen[point_id]) < frame_id
                and int(self._counts[point_id]) >= minimum
            ],
            dtype=np.int64,
        )
        self.stats["query_calls"] += 1
        self.stats["eligible_landmarks"] += int(len(point_ids))
        if not len(point_ids):
            self.stats["last_conditioning_landmarks"] = 0
            return empty_landmark_batch()
        world_points = np.asarray(
            [self._points[int(point_id)] for point_id in point_ids], dtype=np.float32
        )
        projected = zbuffer_sparse_tracks(
            world_points,
            camera.get_raw_pose().detach().cpu().numpy(),
            camera.get_int_mat(level).detach().cpu().numpy(),
            camera.get_width(level),
            camera.get_height(level),
        )
        self.stats["projected_landmarks"] += int(len(projected.depths))
        if not len(projected.depths):
            self.stats["last_conditioning_landmarks"] = 0
            return empty_landmark_batch()
        source = projected.source_indices
        batch = LandmarkConditionBatch(
            point_ids=point_ids[source],
            world_points=projected.world_points,
            uv=projected.uv,
            depths=projected.depths,
            pixel_indices=projected.pixel_indices,
            observation_counts=np.asarray(
                [self._counts[int(point_id)] for point_id in point_ids[source]],
                dtype=np.int64,
            ),
            last_seen_frames=np.asarray(
                [self._last_seen[int(point_id)] for point_id in point_ids[source]],
                dtype=np.int64,
            ),
        )
        occupied = np.asarray(list(occupied_pixel_indices), dtype=np.int64).reshape(-1)
        if len(occupied):
            keep = ~np.isin(batch.pixel_indices, occupied)
            rejected = int(np.sum(~keep))
            self.stats["occupied_pixel_rejections"] += rejected
            batch = LandmarkConditionBatch(
                **{field: getattr(batch, field)[keep] for field in batch.__dataclass_fields__}
            )
        limit = int(
            self.config["maximum_conditioning_points"]
            if maximum_points is None
            else maximum_points
        )
        if limit < 0:
            raise ValueError("Maximum landmark conditioning points must be nonnegative")
        if len(batch) > limit:
            # Recurrent support is preferred, then recent visibility. The sort is
            # deterministic and the same rows are retained in shuffled controls.
            order = np.lexsort(
                (
                    batch.point_ids,
                    -batch.last_seen_frames,
                    -batch.observation_counts,
                )
            )[:limit]
            self.stats["budget_rejections"] += int(len(batch) - limit)
            batch = LandmarkConditionBatch(
                **{field: getattr(batch, field)[order] for field in batch.__dataclass_fields__}
            )
        should_shuffle = (
            bool(self.config["shuffle_depths"])
            if shuffle_depths is None
            else bool(shuffle_depths)
        )
        if should_shuffle and len(batch) > 1:
            seed = (
                int(self.config["shuffle_seed"])
                if shuffle_seed is None
                else int(shuffle_seed)
            )
            batch = shuffle_landmark_depths(batch, seed + frame_id)
            self.stats["shuffled_calls"] += 1
        self.stats["conditioning_landmarks"] += int(len(batch))
        self.stats["conditioning_depth_sum_m"] += float(np.sum(batch.depths))
        self.stats["last_conditioning_landmarks"] = int(len(batch))
        return batch

    def record_repair(
        self,
        *,
        valid_before,
        invalid_before,
        newly_valid,
        conditioning_landmarks,
    ):
        """Record a fallback-repair attempt without changing its decisions."""

        valid_before = int(valid_before)
        invalid_before = int(invalid_before)
        newly_valid = int(newly_valid)
        conditioning_landmarks = int(conditioning_landmarks)
        if min(valid_before, invalid_before, newly_valid, conditioning_landmarks) < 0:
            raise ValueError("Causal landmark repair counts must be nonnegative")
        if newly_valid > invalid_before:
            raise ValueError("Newly valid rows cannot exceed invalid rows")
        self.stats["repair_trigger_calls"] += 1
        if conditioning_landmarks:
            self.stats["repair_conditioned_calls"] += 1
        else:
            self.stats["repair_skipped_no_landmarks"] += 1
        self.stats["repair_preserved_valid_rows"] += valid_before
        self.stats["repair_invalid_before_rows"] += invalid_before
        self.stats["repair_newly_valid_rows"] += newly_valid
        self.stats["repair_remaining_invalid_rows"] += invalid_before - newly_valid

    def record_admitted_mean(
        self,
        original_depths,
        conditioned_depths,
        landmarks,
        transport_weights=None,
        original_log_stds=None,
        conditioned_log_stds=None,
    ):
        """Record support-preserving posterior-mean replacement."""

        original_depths = np.asarray(original_depths, dtype=np.float64).reshape(-1)
        conditioned_depths = np.asarray(
            conditioned_depths, dtype=np.float64
        ).reshape(-1)
        if original_depths.shape != conditioned_depths.shape:
            raise ValueError("Admitted-mean depth arrays must align")
        valid = (
            np.isfinite(original_depths)
            & np.isfinite(conditioned_depths)
            & (original_depths > 0.0)
            & (conditioned_depths > 0.0)
        )
        if transport_weights is None:
            transport_weights = np.ones_like(original_depths)
        transport_weights = np.asarray(
            transport_weights, dtype=np.float64
        ).reshape(-1)
        if transport_weights.shape != original_depths.shape:
            raise ValueError("Admitted-mean transport weights must align")
        valid &= np.isfinite(transport_weights)
        self.stats["admitted_mean_calls"] += 1
        if int(landmarks) > 0:
            self.stats["admitted_mean_conditioned_calls"] += 1
        self.stats["admitted_mean_rows"] += int(np.sum(valid))
        self.stats["admitted_mean_transport_weight_sum"] += float(
            np.sum(transport_weights[valid])
        )
        if (original_log_stds is None) != (conditioned_log_stds is None):
            raise ValueError("Both admitted-mean uncertainty arrays are required")
        if original_log_stds is not None:
            original_log_stds = np.asarray(
                original_log_stds, dtype=np.float64
            ).reshape(-1)
            conditioned_log_stds = np.asarray(
                conditioned_log_stds, dtype=np.float64
            ).reshape(-1)
            if (
                original_log_stds.shape != original_depths.shape
                or conditioned_log_stds.shape != original_depths.shape
            ):
                raise ValueError("Admitted-mean uncertainty arrays must align")
            uncertainty_valid = (
                valid
                & np.isfinite(original_log_stds)
                & np.isfinite(conditioned_log_stds)
                & (original_log_stds >= 0.0)
                & (conditioned_log_stds >= 0.0)
            )
            self.stats["admitted_mean_uncertainty_rows"] += int(
                np.sum(uncertainty_valid)
            )
            self.stats["admitted_mean_original_log_std_sum"] += float(
                np.sum(original_log_stds[uncertainty_valid])
            )
            self.stats["admitted_mean_conditioned_log_std_sum"] += float(
                np.sum(conditioned_log_stds[uncertainty_valid])
            )
        if np.any(valid):
            self.stats["admitted_mean_absolute_log_shift_sum"] += float(
                np.sum(
                    np.abs(
                        np.log(conditioned_depths[valid])
                        - np.log(original_depths[valid])
                    )
                )
            )

    def summary(self):
        result = dict(self.stats)
        rows = int(result["conditioning_landmarks"])
        result.update(
            enabled=self.enabled,
            conditioning_mode=self.config["conditioning_mode"],
            minimum_observations=int(self.config["minimum_observations"]),
            maximum_conditioning_points=int(
                self.config["maximum_conditioning_points"]
            ),
            transport_rule=self.config["transport_rule"],
            responsibility_coordinate=self.config["responsibility_coordinate"],
            propagate_conditioned_uncertainty=bool(
                self.config["propagate_conditioned_uncertainty"]
            ),
            shuffle_depths=bool(self.config["shuffle_depths"]),
            mean_conditioning_depth_m=(
                float(result["conditioning_depth_sum_m"]) / rows if rows else None
            ),
            mean_admitted_absolute_log_shift=(
                float(result["admitted_mean_absolute_log_shift_sum"])
                / int(result["admitted_mean_rows"])
                if int(result["admitted_mean_rows"])
                else None
            ),
            mean_responsibility_absolute_log_shift=(
                float(result["responsibility_absolute_log_shift_sum"])
                / int(result["responsibility_coordinate_rows"])
                if int(result["responsibility_coordinate_rows"])
                else None
            ),
            mean_admitted_transport_weight=(
                float(result["admitted_mean_transport_weight_sum"])
                / int(result["admitted_mean_rows"])
                if int(result["admitted_mean_rows"])
                else None
            ),
            mean_admitted_original_log_std=(
                float(result["admitted_mean_original_log_std_sum"])
                / int(result["admitted_mean_uncertainty_rows"])
                if int(result["admitted_mean_uncertainty_rows"])
                else None
            ),
            mean_admitted_conditioned_log_std=(
                float(result["admitted_mean_conditioned_log_std_sum"])
                / int(result["admitted_mean_uncertainty_rows"])
                if int(result["admitted_mean_uncertainty_rows"])
                else None
            ),
        )
        return result
