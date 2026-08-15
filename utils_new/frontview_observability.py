"""Causal raywise geometry control for forward-view Gaussian mapping."""

import math
from copy import deepcopy

import torch


DEFAULT_FRONT_VIEW_OBSERVABILITY_CONFIG = {
    "enabled": False,
    "learning_scale_mode": "fixed_angle",
    "optimization_mode": "gradient_preconditioner",
    "responsibility_scope": "all_depthcov",
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
    if merged["learning_scale_mode"] not in (
        "fixed_angle",
        "resolution_information",
        "posterior_information",
    ):
        raise ValueError(
            "FrontViewObservability.learning_scale_mode must be fixed_angle, "
            "resolution_information, or posterior_information"
        )
    if merged["optimization_mode"] not in (
        "gradient_preconditioner",
        "post_step_projection",
    ):
        raise ValueError(
            "FrontViewObservability.optimization_mode must be "
            "gradient_preconditioner or post_step_projection"
        )
    if merged["responsibility_scope"] not in (
        "all_depthcov",
        "projective_only",
    ):
        raise ValueError(
            "FrontViewObservability.responsibility_scope must be all_depthcov or "
            "projective_only"
        )
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


def resolution_information_scale(
    max_parallax_sin2: torch.Tensor,
    gaussian_scales: torch.Tensor,
    reference_ranges: torch.Tensor,
) -> torch.Tensor:
    """Convert triangulation information into a threshold-free radial scale.

    Ray intersection supplies depth information proportional to ``sin(theta)^2``.
    A Gaussian with world-space standard deviation ``s`` subtends angular noise
    ``s / z``.  Their ratio is therefore a dimensionless Fisher information,
    and ``I / (1 + I)`` gives a bounded radial learning scale without a metric
    depth or hand-selected unlock angle.
    """

    if max_parallax_sin2.ndim != 1:
        raise ValueError("Parallax information must be one-dimensional")
    if gaussian_scales.ndim == 2:
        angular_scales = torch.amax(gaussian_scales, dim=1)
    elif gaussian_scales.ndim == 1:
        angular_scales = gaussian_scales
    else:
        raise ValueError("Gaussian scales must have shape [N] or [N, D]")
    reference_ranges = reference_ranges.reshape(-1)
    if not (
        angular_scales.shape == reference_ranges.shape == max_parallax_sin2.shape
    ):
        raise ValueError("Resolution-information arrays must align")
    if bool(torch.any(angular_scales <= 0.0)) or bool(
        torch.any(reference_ranges <= 0.0)
    ):
        raise ValueError("Gaussian scales and reference ranges must be positive")

    angular_variance = (angular_scales / reference_ranges).square()
    information = torch.clamp(max_parallax_sin2, min=0.0) / torch.clamp(
        angular_variance, min=torch.finfo(angular_variance.dtype).tiny
    )
    return information / (1.0 + information)


def posterior_information_scale(
    max_parallax_sin2: torch.Tensor,
    gaussian_scales: torch.Tensor,
    reference_ranges: torch.Tensor,
    birth_log_depth_stds: torch.Tensor,
) -> torch.Tensor:
    """Return the causal likelihood-to-posterior radial update fraction.

    The strongest observed baseline contributes log-depth Fisher information
    ``sin(theta)^2 / (s / z)^2`` at the Gaussian's angular resolution.  DepthCov
    contributes the birth prior precision ``1 / sigma_logz^2``.  Their ratio is
    the scalar Kalman gain for the weak birth-ray gauge.  It is similarity
    invariant and approaches one only when incoming view evidence dominates the
    birth-depth prior.
    """

    if max_parallax_sin2.ndim != 1:
        raise ValueError("Parallax information must be one-dimensional")
    if gaussian_scales.ndim == 2:
        angular_scales = torch.amax(gaussian_scales, dim=1)
    elif gaussian_scales.ndim == 1:
        angular_scales = gaussian_scales
    else:
        raise ValueError("Gaussian scales must have shape [N] or [N, D]")
    reference_ranges = reference_ranges.reshape(-1)
    birth_log_depth_stds = birth_log_depth_stds.reshape(-1)
    if not (
        angular_scales.shape
        == reference_ranges.shape
        == birth_log_depth_stds.shape
        == max_parallax_sin2.shape
    ):
        raise ValueError("Posterior-information arrays must align")
    if bool(torch.any(angular_scales <= 0.0)) or bool(
        torch.any(reference_ranges <= 0.0)
    ):
        raise ValueError("Gaussian scales and reference ranges must be positive")
    if bool(torch.any(~torch.isfinite(birth_log_depth_stds))) or bool(
        torch.any(birth_log_depth_stds < 0.0)
    ):
        raise ValueError("Birth log-depth standard deviations must be finite")

    angular_variance = (angular_scales / reference_ranges).square()
    likelihood_information = torch.clamp(max_parallax_sin2, min=0.0) / torch.clamp(
        angular_variance, min=torch.finfo(angular_variance.dtype).tiny
    )
    prior_variance = birth_log_depth_stds.square()
    gain = torch.where(
        prior_variance > 0.0,
        likelihood_information
        / (
            likelihood_information
            + torch.reciprocal(
                torch.clamp(
                    prior_variance,
                    min=torch.finfo(prior_variance.dtype).tiny,
                )
            )
        ),
        torch.zeros_like(likelihood_information),
    )
    return torch.clamp(gain, 0.0, 1.0)


def shuffle_within_log_depth_regimes(
    values: torch.Tensor,
    depths: torch.Tensor,
    seed: int,
    regime_count: int = 3,
) -> torch.Tensor:
    """Shuffle evidence within causal Lloyd log-depth regimes on-device."""

    values = values.reshape(-1)
    depths = depths.reshape(-1)
    if values.shape != depths.shape:
        raise ValueError("Evidence and depth rows must align")
    if int(regime_count) < 2:
        raise ValueError("At least two depth regimes are required")
    if values.numel() <= 1:
        return values.clone()
    if bool(torch.any(~torch.isfinite(values))) or bool(
        torch.any(~torch.isfinite(depths) | (depths <= 0.0))
    ):
        raise ValueError("Shuffled evidence and depths must be finite")

    log_depth = torch.log(depths)
    quantiles = (
        torch.arange(
            int(regime_count), device=depths.device, dtype=depths.dtype
        )
        + 0.5
    ) / float(regime_count)
    centers = torch.quantile(log_depth, quantiles)
    for _ in range(32):
        labels = torch.argmin(
            torch.abs(log_depth[:, None] - centers[None, :]), dim=1
        )
        updated = centers.clone()
        for regime in range(int(regime_count)):
            members = log_depth[labels == regime]
            if members.numel():
                updated[regime] = members.mean()
        updated, _ = torch.sort(updated)
        if torch.equal(updated, centers):
            centers = updated
            break
        centers = updated
    labels = torch.argmin(
        torch.abs(log_depth[:, None] - centers[None, :]), dim=1
    )
    result = values.clone()
    generator = torch.Generator(device=values.device)
    generator.manual_seed(int(seed))
    for regime in range(int(regime_count)):
        rows = torch.nonzero(labels == regime, as_tuple=False).flatten()
        if rows.numel() > 1:
            order = torch.randperm(
                rows.numel(), generator=generator, device=values.device
            )
            result[rows] = values[rows[order]]
    return result


def resolved_footprint_mask(
    max_parallax_sin2: torch.Tensor,
    target_scales: torch.Tensor,
    reference_ranges: torch.Tensor,
    birth_log_depth_stds: torch.Tensor,
) -> torch.Tensor:
    """Certify that later views resolve a birth-time responsibility footprint.

    The target footprint subtends ``a = s_target / z`` radians at birth. A
    later baseline resolves it when angular parallax ``p`` spans that support
    while propagated log-depth uncertainty ``p sigma`` remains inside it.
    """

    max_parallax_sin2 = max_parallax_sin2.reshape(-1)
    target_scales = target_scales.reshape(-1)
    reference_ranges = reference_ranges.reshape(-1)
    birth_log_depth_stds = birth_log_depth_stds.reshape(-1)
    if not (
        max_parallax_sin2.shape
        == target_scales.shape
        == reference_ranges.shape
        == birth_log_depth_stds.shape
    ):
        raise ValueError("Footprint-resolution arrays must align")
    if bool(torch.any(~torch.isfinite(max_parallax_sin2))) or bool(
        torch.any((max_parallax_sin2 < 0.0) | (max_parallax_sin2 > 1.0))
    ):
        raise ValueError("Parallax evidence must be finite and lie in [0, 1]")
    if bool(torch.any(~torch.isfinite(target_scales))) or bool(
        torch.any(target_scales <= 0.0)
    ):
        raise ValueError("Target footprint scales must be finite and positive")
    if bool(torch.any(~torch.isfinite(reference_ranges))) or bool(
        torch.any(reference_ranges <= 0.0)
    ):
        raise ValueError("Reference ranges must be finite and positive")
    if bool(torch.any(~torch.isfinite(birth_log_depth_stds))) or bool(
        torch.any(birth_log_depth_stds < 0.0)
    ):
        raise ValueError("Birth log-depth uncertainty must be finite")

    parallax = torch.sqrt(torch.clamp(max_parallax_sin2, 0.0, 1.0))
    target_angle = target_scales / reference_ranges
    return (parallax >= target_angle) & (
        parallax * birth_log_depth_stds <= target_angle
    )


def release_owned_scale_caps(
    current_limits: torch.Tensor,
    release_limits: torch.Tensor,
    ownership_mask: torch.Tensor,
    resolved_mask: torch.Tensor,
):
    """Remove resolved footprint caps without touching other scale owners."""

    current_limits = current_limits.reshape(-1)
    release_limits = release_limits.reshape(-1)
    ownership_mask = ownership_mask.reshape(-1)
    resolved_mask = resolved_mask.reshape(-1)
    if not (
        current_limits.shape
        == release_limits.shape
        == ownership_mask.shape
        == resolved_mask.shape
    ):
        raise ValueError("Scale-cap ownership arrays must align")
    if ownership_mask.dtype != torch.bool or resolved_mask.dtype != torch.bool:
        raise TypeError("Scale-cap ownership and resolution masks must be boolean")
    if bool(torch.any(torch.isnan(current_limits))) or bool(
        torch.any(torch.isnan(release_limits))
    ):
        raise ValueError("Scale limits cannot contain NaNs")
    if bool(torch.any(current_limits < 1.0)) or bool(torch.any(release_limits < 1.0)):
        raise ValueError("Scale limits must be at least one")

    release = ownership_mask & resolved_mask
    updated_limits = current_limits.clone()
    updated_ownership = ownership_mask.clone()
    updated_limits[release] = release_limits[release]
    updated_ownership[release] = False
    return updated_limits, updated_ownership


def matched_events_within_log_depth_regimes(
    events: torch.Tensor,
    eligible: torch.Tensor,
    depths: torch.Tensor,
    seed: int,
    regime_count: int = 3,
) -> torch.Tensor:
    """Randomly relocate binary events with exact per-regime counts."""

    events = events.reshape(-1)
    eligible = eligible.reshape(-1)
    depths = depths.reshape(-1)
    if events.dtype != torch.bool or eligible.dtype != torch.bool:
        raise TypeError("Events and eligibility must be boolean")
    if not (events.shape == eligible.shape == depths.shape):
        raise ValueError("Matched-event arrays must align")
    if events.numel() <= 1:
        return events.clone()

    log_depth = torch.log(depths)
    quantiles = (
        torch.arange(
            int(regime_count), device=depths.device, dtype=depths.dtype
        )
        + 0.5
    ) / float(regime_count)
    centers = torch.quantile(log_depth, quantiles)
    for _ in range(32):
        labels = torch.argmin(
            torch.abs(log_depth[:, None] - centers[None, :]), dim=1
        )
        updated = centers.clone()
        for regime in range(int(regime_count)):
            members = log_depth[labels == regime]
            if members.numel():
                updated[regime] = members.mean()
        updated, _ = torch.sort(updated)
        if torch.equal(updated, centers):
            centers = updated
            break
        centers = updated
    labels = torch.argmin(
        torch.abs(log_depth[:, None] - centers[None, :]), dim=1
    )

    result = torch.zeros_like(events)
    generator = torch.Generator(device=depths.device)
    generator.manual_seed(int(seed))
    for regime in range(int(regime_count)):
        requested = int(torch.count_nonzero(events & (labels == regime)).item())
        candidates = torch.nonzero(
            eligible & (labels == regime), as_tuple=False
        ).flatten()
        if requested > int(candidates.numel()):
            raise RuntimeError("Matched certificate regime has too few eligible rows")
        if requested:
            order = torch.randperm(
                candidates.numel(), generator=generator, device=depths.device
            )[:requested]
            result[candidates[order]] = True
    return result


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


def project_raywise_update(
    previous_means: torch.Tensor,
    updated_means: torch.Tensor,
    reference_rays: torch.Tensor,
    radial_scales: torch.Tensor,
) -> torch.Tensor:
    """Project an optimizer's realized update in the weak birth-ray gauge."""

    if previous_means.shape != updated_means.shape or previous_means.ndim != 2:
        raise ValueError("Previous and updated means must share shape [N, 3]")
    if reference_rays.shape != previous_means.shape:
        raise ValueError("Reference rays must align with Gaussian means")
    if radial_scales.shape != (previous_means.shape[0],):
        raise ValueError("Radial scales must have one value per Gaussian")
    if bool(torch.any((radial_scales < 0.0) | (radial_scales > 1.0))):
        raise ValueError("Radial scales must lie in [0, 1]")

    rays = torch.nn.functional.normalize(reference_rays, dim=1, eps=1.0e-8)
    update = updated_means - previous_means
    radial = torch.sum(update * rays, dim=1, keepdim=True) * rays
    return updated_means + (radial_scales.reshape(-1, 1) - 1.0) * radial
