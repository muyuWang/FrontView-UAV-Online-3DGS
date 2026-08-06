"""Budget-neutral image-frequency allocation for Gaussian proposals."""

from typing import Mapping, Tuple

import torch
import torch.nn.functional as F


def frequency_evidence_map(
    image: torch.Tensor,
    residual: torch.Tensor,
    opacity: torch.Tensor,
    config: Mapping[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sampling, footprint, and admission-frequency evidence."""

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Frequency sampling expects an HxWx3 image")
    height, width = image.shape[:2]
    if residual.shape != (height, width):
        raise ValueError("Residual must match the image height and width")

    rgb_weights = image.new_tensor((0.299, 0.587, 0.114))
    gray = (image * rgb_weights).sum(dim=-1)
    gradient = torch.zeros_like(gray)
    gradient[:, 1:] += torch.abs(gray[:, 1:] - gray[:, :-1])
    gradient[1:, :] += torch.abs(gray[1:, :] - gray[:-1, :])
    laplacian_kernel = image.new_tensor(
        ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
    ).view(1, 1, 3, 3)
    laplacian = torch.abs(
        F.conv2d(gray.view(1, 1, height, width), laplacian_kernel, padding=1)
    ).view(height, width)
    gradient_threshold = max(float(config.get("gradient_threshold", 0.04)), 1.0e-6)
    laplacian_threshold = max(
        float(config.get("laplacian_threshold", 0.08)), 1.0e-6
    )
    frequency = torch.maximum(
        gradient / gradient_threshold, laplacian / laplacian_threshold
    ).clamp_(0.0, 1.0)

    columns = torch.linspace(-1.0, 1.0, width, device=image.device).abs()
    side_start = min(max(float(config.get("side_start", 0.45)), 0.0), 1.0)
    side_ramp = torch.clamp(
        (columns - side_start) / max(1.0 - side_start, 1.0e-6), 0.0, 1.0
    ).view(1, width)
    side_boost = max(float(config.get("side_boost", 1.0)), 1.0)
    spatial_priority = 1.0 + (side_boost - 1.0) * side_ramp

    if opacity is None:
        stable = torch.zeros_like(gray)
    else:
        if opacity.ndim == 3 and opacity.shape[-1] == 1:
            opacity = opacity.squeeze(-1)
        if opacity.shape != (height, width):
            raise ValueError("Opacity must match the image height and width")
        stable = (
            opacity >= float(config.get("stable_opacity_threshold", 0.30))
        ).to(gray.dtype)

    residual_threshold = max(float(config.get("residual_threshold", 0.05)), 1.0e-6)
    residual_priority = torch.clamp(residual / residual_threshold, 1.0, 3.0)
    stable_boost = max(float(config.get("stable_opacity_boost", 1.0)), 1.0)
    evidence = frequency * spatial_priority * residual_priority
    evidence = evidence * (1.0 + (stable_boost - 1.0) * stable)
    fraction = min(max(float(config.get("frequency_fraction", 0.75)), 0.0), 1.0)
    importance = (1.0 - fraction) + fraction * (1.0 + evidence)

    # Footprints are reduced only where the permanent map already renders a
    # stable surface. Missing coverage is still sampled, but it is not sharpened.
    footprint_evidence = (frequency * stable * spatial_priority).clamp_(0.0, 1.0)
    admission_evidence = frequency * (0.5 + 0.5 * side_ramp)
    return importance, footprint_evidence, admission_evidence.clamp_(0.0, 1.0)


def sample_frequency_balanced_indices(
    valid_mask: torch.Tensor,
    importance: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Sample unique flat pixel indices without changing the proposal budget."""

    if valid_mask.shape != importance.shape or valid_mask.ndim != 2:
        raise ValueError("Valid mask and importance must be matching 2D tensors")
    requested = max(0, int(count))
    candidates = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).reshape(-1)
    if requested == 0 or candidates.numel() == 0:
        return candidates[:0]
    if candidates.numel() <= requested:
        return candidates
    weights = torch.clamp(importance.reshape(-1)[candidates], min=1.0e-8)
    selected = torch.multinomial(weights, requested, replacement=False)
    return candidates[selected]


def frequency_footprint_log_offset(
    evidence: torch.Tensor,
    max_shrink_fraction: float,
) -> torch.Tensor:
    """Convert stable frequency evidence to a bounded log-scale reduction."""

    shrink = min(max(float(max_shrink_fraction), 0.0), 0.49)
    factor = 1.0 - shrink * torch.clamp(evidence, 0.0, 1.0)
    return torch.log(torch.clamp(factor, min=0.5))
