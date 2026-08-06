"""Causal high-frequency appearance carriers on repeated sparse map tracks."""

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

import numpy as np


TrackKey = Tuple[int, int, int]


@dataclass
class _TrackEvidence:
    world_point: np.ndarray
    colors: list = field(default_factory=list)
    depths: list = field(default_factory=list)
    gradients: list = field(default_factory=list)
    side_scores: list = field(default_factory=list)
    last_frame_id: int = -1


@dataclass(frozen=True)
class TrackDetailBatch:
    world_points: np.ndarray
    colors: np.ndarray
    log_scales: np.ndarray
    scores: np.ndarray
    keys: Tuple[TrackKey, ...]

    def __len__(self) -> int:
        return int(self.world_points.shape[0])


def empty_track_detail_batch() -> TrackDetailBatch:
    return TrackDetailBatch(
        world_points=np.empty((0, 3), dtype=np.float32),
        colors=np.empty((0, 3), dtype=np.float32),
        log_scales=np.empty((0, 1), dtype=np.float32),
        scores=np.empty((0,), dtype=np.float32),
        keys=(),
    )


class SparseTrackDetailAccumulator:
    """Accumulate only causal, repeated side-detail observations.

    The sparse map coordinates are the track identity. A track becomes a detail
    carrier only after repeated high-gradient observations at the image sides.
    Geometry stays at the sparse anchor; only robust appearance evidence is
    fused into the returned batch.
    """

    def __init__(self, config: Mapping[str, object]):
        self.config = dict(config)
        self.quantization = float(self.config["track_quantization"])
        self.min_support = int(self.config["min_support_views"])
        self.max_samples = int(self.config["max_observations_per_track"])
        self.gradient_threshold = float(self.config["gradient_threshold"])
        self.side_start = float(self.config["side_start"])
        self.near_depth = float(self.config["near_depth_m"])
        self.color_mad_threshold = float(self.config["color_mad_threshold"])
        self.projected_scale = float(self.config["projected_scale_px"])
        self.max_per_frame = int(self.config["max_commits_per_frame"])
        self.max_total = int(self.config["max_total_gaussians"])
        self.side_boost = float(self.config["side_score_boost"])
        self.near_boost = float(self.config["near_score_boost"])
        self.evidence: Dict[TrackKey, _TrackEvidence] = {}
        self.pending = set()
        self.committed = set()

    def _key(self, point: np.ndarray) -> TrackKey:
        quantized = np.rint(point / self.quantization).astype(np.int64)
        return tuple(int(value) for value in quantized)

    def observe(
        self,
        frame_id: int,
        world_points: np.ndarray,
        colors: np.ndarray,
        depths: np.ndarray,
        gradients: np.ndarray,
        side_scores: np.ndarray,
        focal: float,
    ) -> TrackDetailBatch:
        arrays = [
            np.asarray(world_points, dtype=np.float32),
            np.asarray(colors, dtype=np.float32),
            np.asarray(depths, dtype=np.float32).reshape(-1),
            np.asarray(gradients, dtype=np.float32).reshape(-1),
            np.asarray(side_scores, dtype=np.float32).reshape(-1),
        ]
        count = len(arrays[0])
        if arrays[0].shape != (count, 3) or arrays[1].shape != (count, 3):
            raise ValueError("Track detail points and colors must have shape Nx3")
        if any(len(value) != count for value in arrays[2:]):
            raise ValueError("Track detail observation fields must have equal length")
        if focal <= 0.0:
            raise ValueError("Track detail focal length must be positive")

        eligible = (
            np.isfinite(arrays[0]).all(axis=1)
            & np.isfinite(arrays[1]).all(axis=1)
            & np.isfinite(arrays[2])
            & np.isfinite(arrays[3])
            & (arrays[2] > 0.0)
            & (arrays[2] <= self.near_depth)
            & (arrays[3] >= self.gradient_threshold)
            & (arrays[4] >= self.side_start)
        )
        touched = set()
        for point, color, depth, gradient, side in zip(
            arrays[0][eligible],
            arrays[1][eligible],
            arrays[2][eligible],
            arrays[3][eligible],
            arrays[4][eligible],
        ):
            key = self._key(point)
            if key in self.committed:
                continue
            record = self.evidence.get(key)
            if record is None:
                record = _TrackEvidence(world_point=point.copy())
                self.evidence[key] = record
            if record.last_frame_id == int(frame_id):
                continue
            record.last_frame_id = int(frame_id)
            record.colors.append(np.clip(color, 0.0, 1.0).copy())
            record.depths.append(float(depth))
            record.gradients.append(float(gradient))
            record.side_scores.append(float(side))
            if len(record.colors) > self.max_samples:
                record.colors.pop(0)
                record.depths.pop(0)
                record.gradients.pop(0)
                record.side_scores.pop(0)
            if len(record.colors) >= self.min_support:
                self.pending.add(key)
                touched.add(key)

        remaining = self.max_total - len(self.committed)
        limit = min(self.max_per_frame, remaining)
        if limit <= 0 or not touched:
            return empty_track_detail_batch()

        ranked = []
        for key in touched:
            record = self.evidence[key]
            colors_array = np.asarray(record.colors, dtype=np.float32)
            median_color = np.median(colors_array, axis=0)
            color_mad = float(
                np.median(np.mean(np.abs(colors_array - median_color), axis=1))
            )
            if color_mad > self.color_mad_threshold:
                continue
            gradient = float(np.mean(record.gradients))
            side = float(np.mean(record.side_scores))
            depth = max(float(np.min(record.depths)), 1.0e-6)
            side_weight = 1.0 + self.side_boost * max(
                0.0, (side - self.side_start) / max(1.0 - self.side_start, 1.0e-6)
            )
            near_weight = 1.0 + self.near_boost * max(
                0.0, 1.0 - depth / max(self.near_depth, 1.0e-6)
            )
            support_weight = np.log1p(len(record.colors))
            ranked.append(
                (
                    gradient * side_weight * near_weight * support_weight,
                    key,
                    median_color,
                    depth,
                )
            )

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = ranked[:limit]
        if not selected:
            return empty_track_detail_batch()

        keys = tuple(item[1] for item in selected)
        world_points_out = np.stack(
            [self.evidence[key].world_point for key in keys]
        ).astype(np.float32)
        colors_out = np.stack([item[2] for item in selected]).astype(np.float32)
        world_scales = np.asarray(
            [self.projected_scale * item[3] / focal for item in selected],
            dtype=np.float32,
        )
        log_scales = np.log(np.maximum(world_scales, 1.0e-8)).reshape(-1, 1)
        scores = np.asarray([item[0] for item in selected], dtype=np.float32)

        for key in keys:
            self.pending.discard(key)
            self.committed.add(key)
            self.evidence.pop(key, None)
        return TrackDetailBatch(
            world_points=world_points_out,
            colors=colors_out,
            log_scales=log_scales,
            scores=scores,
            keys=keys,
        )


class StableSurfaceDetailSampler:
    """Select unique small appearance carriers on an already rendered surface."""

    def __init__(self, config: Mapping[str, object]):
        self.voxel_size = float(config["voxel_size"])
        self.max_per_keyframe = int(config["max_commits_per_keyframe"])
        self.max_total = int(config["max_total_gaussians"])
        self.committed = set()

    @property
    def full(self) -> bool:
        return len(self.committed) >= self.max_total

    def select(
        self,
        world_points: np.ndarray,
        colors: np.ndarray,
        log_scales: np.ndarray,
        scores: np.ndarray,
    ) -> TrackDetailBatch:
        points = np.asarray(world_points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        scales = np.asarray(log_scales, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        count = len(points)
        if points.shape != (count, 3) or colors.shape != (count, 3):
            raise ValueError("Surface detail points and colors must have shape Nx3")
        if scales.shape != (count, 1) or len(scores) != count:
            raise ValueError("Surface detail scales and scores have the wrong shape")
        limit = min(
            self.max_per_keyframe,
            self.max_total - len(self.committed),
        )
        if count == 0 or limit <= 0:
            return empty_track_detail_batch()

        quantized = np.floor(points / self.voxel_size).astype(np.int64)
        order = np.argsort(scores)[::-1]
        selected = []
        keys = []
        frame_claimed = set()
        for index in order:
            key = tuple(int(value) for value in quantized[index])
            if key in self.committed or key in frame_claimed:
                continue
            selected.append(int(index))
            keys.append(key)
            frame_claimed.add(key)
            if len(selected) >= limit:
                break
        if not selected:
            return empty_track_detail_batch()
        self.committed.update(keys)
        selected = np.asarray(selected, dtype=np.int64)
        return TrackDetailBatch(
            world_points=points[selected].copy(),
            colors=np.clip(colors[selected], 0.0, 1.0).copy(),
            log_scales=scales[selected].copy(),
            scores=scores[selected].copy(),
            keys=tuple(keys),
        )
