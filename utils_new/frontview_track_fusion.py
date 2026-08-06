"""Causal appearance fusion for persistent VI world-point identities."""

from copy import deepcopy

import torch


DEFAULT_FRONT_VIEW_TRACK_FUSION_CONFIG = {
    "enabled": False,
    "color_ema": 0.05,
    "max_color_step": 0.10,
    "shuffle_identity": False,
    "shuffle_seed": 42,
    "shuffle_depth_edges_m": [20.0, 50.0, 80.0],
}


def validate_front_view_track_fusion_config(config=None):
    merged = deepcopy(DEFAULT_FRONT_VIEW_TRACK_FUSION_CONFIG)
    if config is not None:
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError(
                "Unknown FrontViewTrackFusion options: {}".format(sorted(unknown))
            )
        merged.update(config)
    for key in ("enabled", "shuffle_identity"):
        if not isinstance(merged[key], bool):
            raise TypeError("FrontViewTrackFusion.{} must be boolean".format(key))
    if not 0.0 < float(merged["color_ema"]) <= 1.0:
        raise ValueError("FrontViewTrackFusion.color_ema must be in (0, 1]")
    if float(merged["max_color_step"]) <= 0.0:
        raise ValueError("FrontViewTrackFusion.max_color_step must be positive")
    if not isinstance(merged["shuffle_seed"], int):
        raise TypeError("FrontViewTrackFusion.shuffle_seed must be an integer")
    edges = [float(value) for value in merged["shuffle_depth_edges_m"]]
    if not edges or any(value <= 0.0 for value in edges):
        raise ValueError("FrontViewTrackFusion depth edges must be positive")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("FrontViewTrackFusion depth edges must increase")
    merged["shuffle_depth_edges_m"] = edges
    return merged


def robust_color_ema(current, observed, alpha, max_color_step):
    """Fuse RGB evidence while bounding the norm of each causal update."""

    if current.shape != observed.shape or current.ndim != 2 or current.shape[1] != 3:
        raise ValueError("Track-fusion colors must have matching [N, 3] shapes")
    alpha = float(alpha)
    max_color_step = float(max_color_step)
    if not 0.0 < alpha <= 1.0 or max_color_step <= 0.0:
        raise ValueError("Track-fusion EMA parameters are invalid")
    delta = observed - current
    norm = torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp_min(1.0e-8)
    bounded_delta = delta * torch.clamp(max_color_step / norm, max=1.0)
    return current + alpha * bounded_delta
