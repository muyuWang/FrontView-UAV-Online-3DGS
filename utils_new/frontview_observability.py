"""Causal raywise geometry control for forward-view Gaussian mapping."""

import math
from copy import deepcopy

import torch


DEFAULT_FRONT_VIEW_OBSERVABILITY_CONFIG = {
    "enabled": False,
    "min_ray_lr_scale": 0.10,
    "unlock_parallax_deg": 4.0,
    "visibility_margin_px": 16.0,
    "evidence_update_interval": 10,
    "apply_post_refinement": True,
    "shuffle_evidence": False,
    "shuffle_seed": 42,
}


def validate_front_view_observability_config(config=None):
    merged = deepcopy(DEFAULT_FRONT_VIEW_OBSERVABILITY_CONFIG)
    if config is not None:
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError(
                "Unknown FrontViewObservability options: {}".format(sorted(unknown))
            )
        merged.update(config)

    for key in ("enabled", "apply_post_refinement", "shuffle_evidence"):
        if not isinstance(merged[key], bool):
            raise TypeError("FrontViewObservability.{} must be boolean".format(key))
    min_scale = float(merged["min_ray_lr_scale"])
    if not 0.0 <= min_scale <= 1.0:
        raise ValueError("FrontViewObservability.min_ray_lr_scale must be in [0, 1]")
    unlock = float(merged["unlock_parallax_deg"])
    if not 0.0 < unlock <= 90.0:
        raise ValueError(
            "FrontViewObservability.unlock_parallax_deg must be in (0, 90]"
        )
    if float(merged["visibility_margin_px"]) < 0.0:
        raise ValueError(
            "FrontViewObservability.visibility_margin_px cannot be negative"
        )
    if not isinstance(merged["shuffle_seed"], int):
        raise TypeError("FrontViewObservability.shuffle_seed must be an integer")
    if (
        not isinstance(merged["evidence_update_interval"], int)
        or merged["evidence_update_interval"] <= 0
    ):
        raise ValueError(
            "FrontViewObservability.evidence_update_interval must be positive"
        )
    return merged


def parallax_learning_scale(
    max_parallax_sin2: torch.Tensor,
    min_scale: float,
    unlock_parallax_deg: float,
) -> torch.Tensor:
    """Map causal angular evidence to a smooth radial learning-rate scale."""

    threshold = math.sin(math.radians(float(unlock_parallax_deg))) ** 2
    normalized = torch.clamp(max_parallax_sin2 / max(threshold, 1.0e-12), 0.0, 1.0)
    smooth = normalized.square() * (3.0 - 2.0 * normalized)
    return float(min_scale) + (1.0 - float(min_scale)) * smooth


def precondition_raywise_gradient(
    gradients: torch.Tensor,
    reference_rays: torch.Tensor,
    radial_scales: torch.Tensor,
) -> torch.Tensor:
    """Scale only the weak birth-ray component and retain tangent updates."""

    if gradients.shape != reference_rays.shape or gradients.ndim != 2:
        raise ValueError("Gradients and reference rays must share shape [N, 3]")
    if radial_scales.shape != (gradients.shape[0],):
        raise ValueError("Radial scales must have one value per Gaussian")
    rays = torch.nn.functional.normalize(reference_rays, dim=1, eps=1.0e-8)
    radial = torch.sum(gradients * rays, dim=1, keepdim=True) * rays
    return gradients + (radial_scales.reshape(-1, 1) - 1.0) * radial
