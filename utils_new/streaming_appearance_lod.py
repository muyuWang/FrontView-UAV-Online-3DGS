"""Causal projective evidence for online Gaussian appearance capacity."""

from __future__ import annotations

import math

import torch


DEFAULT_CONFIG = {
    "enabled": False,
    "birth_degree": 1,
    "target_degree": 2,
    "min_views": 2,
    "min_mean_radius": 0.5,
    "min_angular_dispersion": 0.0,
    "max_target_fraction": 0.75,
    "promotion_interval": 10,
    "selection_mode": "evidence",
    "utility_ema_decay": 0.9,
    "shuffle_seed": 42,
}


def validate_streaming_appearance_lod_config(config=None):
    config = {} if config is None else dict(config)
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise TypeError(
            "Unknown StreamingAppearanceLOD options: {}".format(sorted(unknown))
        )
    result = dict(DEFAULT_CONFIG)
    result.update(config)
    for key in ("enabled",):
        if not isinstance(result[key], bool):
            raise TypeError("StreamingAppearanceLOD.{} must be boolean".format(key))
    for key in ("birth_degree", "target_degree", "min_views", "promotion_interval"):
        if not isinstance(result[key], int):
            raise TypeError("StreamingAppearanceLOD.{} must be an integer".format(key))
    if result["birth_degree"] < 0 or result["target_degree"] < 1:
        raise ValueError("Streaming appearance SH degrees must be non-negative")
    if result["target_degree"] > 3:
        raise ValueError("StreamingAppearanceLOD currently supports up to SH3")
    if result["birth_degree"] >= result["target_degree"]:
        raise ValueError("birth_degree must be lower than target_degree")
    if result["min_views"] < 1 or result["promotion_interval"] < 1:
        raise ValueError("Streaming appearance count and interval must be positive")
    for key in ("min_mean_radius", "min_angular_dispersion"):
        result[key] = float(result[key])
        if result[key] < 0.0:
            raise ValueError("Streaming appearance thresholds must be non-negative")
    result["max_target_fraction"] = float(result["max_target_fraction"])
    if not 0.0 <= result["max_target_fraction"] <= 1.0:
        raise ValueError("max_target_fraction must be in [0, 1]")
    result["utility_ema_decay"] = float(result["utility_ema_decay"])
    if not 0.0 <= result["utility_ema_decay"] < 1.0:
        raise ValueError("utility_ema_decay must be in [0, 1)")
    if result["selection_mode"] not in {
        "evidence",
        "shuffled",
        "shuffled_global",
        "gradient",
        "gradient_agreement",
        "gradient_shuffled",
    }:
        raise ValueError(
            "selection_mode must be evidence, shuffled, shuffled_global, "
            "gradient, gradient_agreement, or gradient_shuffled"
        )
    result["shuffle_seed"] = int(result["shuffle_seed"])
    return result


def sh_band_bounds(degree):
    """Return the coefficient interval for one non-DC SH degree."""

    degree = int(degree)
    if degree < 1:
        raise ValueError("SH band degree must be positive")
    return degree * degree - 1, (degree + 1) * (degree + 1) - 1


def bias_corrected_gradient_utility(utility_ema, observation_count, decay):
    """Remove the cold-start bias from a visibility-conditioned utility EMA."""

    utility = torch.as_tensor(utility_ema).float()
    count = torch.as_tensor(observation_count, device=utility.device).float()
    decay = float(decay)
    correction = 1.0 - torch.pow(
        torch.full_like(count, decay), count.clamp_min(0.0)
    )
    return torch.where(count > 0, utility / correction.clamp_min(1.0e-12), 0.0)


def persistent_gradient_utility(gradient_ema, observation_count, decay):
    """Estimate loss-reducing energy of one coefficient vector across views."""

    gradient = torch.as_tensor(gradient_ema).float()
    if gradient.ndim != 2:
        raise ValueError("Gradient agreement evidence must have shape [N, D]")
    count = torch.as_tensor(observation_count, device=gradient.device).float()
    if count.shape != gradient.shape[:1]:
        raise ValueError("Gradient agreement counts must align with rows")
    decay = float(decay)
    correction = 1.0 - torch.pow(
        torch.full_like(count, decay), count.clamp_min(0.0)
    )
    corrected = torch.where(
        (count > 0).reshape(-1, 1),
        gradient / correction.clamp_min(1.0e-12).reshape(-1, 1),
        0.0,
    )
    return torch.sum(corrected.square(), dim=1)


@torch.no_grad()
def select_gradient_agreement_promotions(
    degrees,
    gradient_ema,
    observation_count,
    config,
):
    """Allocate one SH band to rows with a persistent descent direction."""

    config = validate_streaming_appearance_lod_config(config)
    degrees = torch.as_tensor(degrees, dtype=torch.uint8)
    gradient_ema = torch.as_tensor(
        gradient_ema, device=degrees.device, dtype=torch.float32
    )
    observation_count = torch.as_tensor(
        observation_count, device=degrees.device
    )
    count = degrees.numel()
    if gradient_ema.ndim != 2 or gradient_ema.shape[0] != count:
        raise ValueError("Gradient agreement evidence must align with degrees")
    if observation_count.shape != (count,):
        raise ValueError("Gradient agreement counts must align with degrees")

    target = int(config["target_degree"])
    result = degrees.clone()
    result.clamp_(min=int(config["birth_degree"]), max=target)
    budget = int(math.floor(float(config["max_target_fraction"]) * count))
    promoted = int((result >= target).sum().item())
    available = max(0, budget - promoted)
    if available == 0 or count == 0:
        return result, torch.empty(0, device=degrees.device, dtype=torch.long)

    score = persistent_gradient_utility(
        gradient_ema,
        observation_count,
        config["utility_ema_decay"],
    )
    eligible = (
        (result < target)
        & (observation_count >= int(config["min_views"]))
        & torch.isfinite(score)
        & (score > 0)
    )
    eligible_indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    select_count = min(available, int(eligible_indices.numel()))
    if select_count == 0:
        return result, torch.empty(0, device=degrees.device, dtype=torch.long)
    order = torch.topk(score[eligible_indices], k=select_count, sorted=False).indices
    selected = eligible_indices[order]
    result[selected] = target
    return result, selected


@torch.no_grad()
def select_gradient_promotions(
    degrees,
    utility_ema,
    observation_count,
    config,
    update_index=0,
):
    """Allocate one SH band by counterfactual first-order loss reduction."""

    config = validate_streaming_appearance_lod_config(config)
    degrees = torch.as_tensor(degrees, dtype=torch.uint8)
    utility_ema = torch.as_tensor(
        utility_ema, device=degrees.device, dtype=torch.float32
    )
    observation_count = torch.as_tensor(
        observation_count, device=degrees.device
    )
    count = degrees.numel()
    if utility_ema.shape != (count,) or observation_count.shape != (count,):
        raise ValueError("Gradient appearance evidence must align with degrees")

    target = int(config["target_degree"])
    result = degrees.clone()
    result.clamp_(min=int(config["birth_degree"]), max=target)
    budget = int(math.floor(float(config["max_target_fraction"]) * count))
    promoted = int((result >= target).sum().item())
    available = max(0, budget - promoted)
    if available == 0 or count == 0:
        return result, torch.empty(0, device=degrees.device, dtype=torch.long)

    score = bias_corrected_gradient_utility(
        utility_ema,
        observation_count,
        config["utility_ema_decay"],
    )
    eligible = (
        (result < target)
        & (observation_count >= int(config["min_views"]))
        & torch.isfinite(score)
        & (score > 0)
    )
    eligible_indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    select_count = min(available, int(eligible_indices.numel()))
    if select_count == 0:
        return result, torch.empty(0, device=degrees.device, dtype=torch.long)

    if config["selection_mode"] == "gradient_shuffled":
        generator = torch.Generator(device=degrees.device)
        generator.manual_seed(int(config["shuffle_seed"]) + int(update_index))
        order = torch.randperm(
            eligible_indices.numel(), generator=generator, device=degrees.device
        )[:select_count]
        selected = eligible_indices[order]
    else:
        order = torch.topk(
            score[eligible_indices], k=select_count, sorted=False
        ).indices
        selected = eligible_indices[order]
    result[selected] = target
    return result, selected


def projective_evidence_measurements(view_count, radius_sum, direction_sum):
    """Return radius, angular dispersion, and a scale-aware evidence score."""

    count = torch.as_tensor(view_count).float()
    radius_sum = torch.as_tensor(radius_sum, device=count.device).float()
    direction_sum = torch.as_tensor(direction_sum, device=count.device).float()
    denominator = count.clamp_min(1.0)
    mean_radius = radius_sum / denominator
    mean_resultant = torch.linalg.norm(direction_sum, dim=-1) / denominator
    dispersion = torch.clamp(1.0 - mean_resultant, min=0.0, max=1.0)
    score = (
        torch.log1p(count)
        * torch.log1p(mean_radius)
        * torch.sqrt(dispersion + 1.0e-4)
    )
    return mean_radius, dispersion, score


@torch.no_grad()
def select_monotonic_promotions(
    degrees,
    view_count,
    radius_sum,
    direction_sum,
    config,
    update_index=0,
):
    """Select new target-degree rows under a growing fixed-fraction budget."""

    config = validate_streaming_appearance_lod_config(config)
    degrees = torch.as_tensor(degrees, dtype=torch.uint8)
    view_count = torch.as_tensor(view_count, device=degrees.device)
    radius_sum = torch.as_tensor(radius_sum, device=degrees.device)
    direction_sum = torch.as_tensor(direction_sum, device=degrees.device)
    count = degrees.numel()
    if view_count.shape != (count,) or radius_sum.shape != (count,):
        raise ValueError("Streaming appearance scalar evidence must align with degrees")
    if direction_sum.shape != (count, 3):
        raise ValueError("Streaming appearance directions must have shape [N, 3]")

    target = int(config["target_degree"])
    result = degrees.clone()
    result.clamp_(min=int(config["birth_degree"]), max=target)
    budget = int(math.floor(float(config["max_target_fraction"]) * count))
    promoted = int((result >= target).sum().item())
    available = max(0, budget - promoted)
    if available == 0 or count == 0:
        return result, torch.empty(0, device=degrees.device, dtype=torch.long)

    mean_radius, dispersion, score = projective_evidence_measurements(
        view_count, radius_sum, direction_sum
    )
    eligible = (
        (result < target)
        & (view_count >= int(config["min_views"]))
        & (mean_radius >= float(config["min_mean_radius"]))
        & (dispersion >= float(config["min_angular_dispersion"]))
    )
    eligible_indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    select_count = min(available, int(eligible_indices.numel()))
    if select_count == 0:
        return result, torch.empty(0, device=degrees.device, dtype=torch.long)

    if config["selection_mode"] == "shuffled_global":
        candidate_indices = torch.nonzero(
            result < target, as_tuple=False
        ).reshape(-1)
        generator = torch.Generator(device=degrees.device)
        generator.manual_seed(int(config["shuffle_seed"]) + int(update_index))
        order = torch.randperm(
            candidate_indices.numel(), generator=generator, device=degrees.device
        )[:select_count]
        selected = candidate_indices[order]
    elif config["selection_mode"] == "shuffled":
        generator = torch.Generator(device=degrees.device)
        generator.manual_seed(int(config["shuffle_seed"]) + int(update_index))
        order = torch.randperm(
            eligible_indices.numel(), generator=generator, device=degrees.device
        )[:select_count]
        selected = eligible_indices[order]
    else:
        order = torch.topk(
            score[eligible_indices], k=select_count, sorted=False
        ).indices
        selected = eligible_indices[order]
    result[selected] = target
    return result, selected
