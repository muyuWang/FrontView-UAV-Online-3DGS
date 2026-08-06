"""Ephemeral shadow observations; these objects never own torch parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ShadowObservation:
    frame_id: int
    uv: np.ndarray
    inverse_depth: float
    inverse_depth_variance: float
    pose_id: int
    world_to_camera: np.ndarray
    pose_covariance: np.ndarray
    source_kind: str
    track_confidence: float
    rgb: np.ndarray
    world_point: np.ndarray
    intrinsics: np.ndarray


@dataclass
class ShadowGroup:
    group_id: int
    track_id: int
    source_kind: str
    created_frame: int
    proposal_batch: object
    observations: list[ShadowObservation] = field(default_factory=list)
    last_frame: int = -1
    q_g: float = float("-inf")
    evidence: dict = field(default_factory=dict)
    offline_cached: bool = False
    cached_result: object = None

    def add(self, observation: ShadowObservation, proposal_batch, max_views=8):
        if self.observations and observation.frame_id == self.observations[-1].frame_id:
            self.observations[-1] = observation
        else:
            self.observations.append(observation)
        if len(self.observations) > int(max_views):
            self.observations = self.observations[-int(max_views) :]
        self.proposal_batch = proposal_batch
        self.last_frame = int(observation.frame_id)

    @property
    def age(self):
        return max(0, self.last_frame - self.created_frame)

    @property
    def distinct_view_count(self):
        return len({observation.frame_id for observation in self.observations})

    def byte_size(self):
        return sum(
            observation.uv.nbytes
            + observation.world_to_camera.nbytes
            + observation.pose_covariance.nbytes
            + observation.rgb.nbytes
            + observation.world_point.nbytes
            + observation.intrinsics.nbytes
            + 32
            for observation in self.observations
        )


def cap_shadow_alpha(alpha, cap=0.1):
    """Scale composited shadow alpha without changing its spatial proportions."""
    values = np.asarray(alpha, dtype=np.float32)
    if values.ndim < 3:
        return np.minimum(values, float(cap))
    accumulated = np.sum(values, axis=0, keepdims=True)
    scale = np.minimum(1.0, float(cap) / np.maximum(accumulated, 1.0e-8))
    return values * scale
