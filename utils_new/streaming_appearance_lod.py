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
    "compute_routing": False,
    "compute_routing_warmup_evidence_updates": 0,
    "gradient_ema_dtype": "float32",
    "gradient_ema_scale": 1.0,
    "exact_replay_microbatch_size": 0,
    "exact_replay_gaussian_view_budget": 0,
    "spectral_replay_enabled": False,
    "spectral_replay_basis_budget": 48.0,
    "spectral_replay_max_views": 6,
    "spectral_residency_enabled": False,
    "spectral_residency_basis_budget": 5472.0,
    "spectral_residency_max_views": 512,
    "bounded_replay_residency_enabled": False,
    "sparse_model_export": False,
    "frequency_schedule_enabled": False,
    "frequency_schedule_warmup_evidence_updates": 10,
    "frequency_schedule_base_level_weights": [0.4, 0.3, 0.2, 0.1],
    "frequency_schedule_inactive_gain": 0.25,
    "frequency_schedule_min_full_fraction": 0.3,
    "frequency_schedule_reallocation_level": 2,
    "frequency_schedule_coarse_sh_degree": 2,
    "frequency_schedule_full_anchor_steps": 1,
    "frequency_schedule_full_tail_steps": 2,
    "directional_step_budget_enabled": False,
    "directional_step_budget_selection_mode": "novelty",
    "directional_step_budget_warmup_keyframes": 8,
    "directional_step_budget_very_low_degrees": 0.5,
    "directional_step_budget_low_degrees": 1.5,
    "directional_step_budget_very_low_steps": 6,
    "directional_step_budget_low_steps": 8,
    "directional_step_budget_shuffle_very_low_fraction": 0.2,
    "directional_step_budget_shuffle_low_fraction": 0.6,
    "directional_step_budget_percentile_window": 64,
    "directional_step_budget_percentile_very_low_fraction": 0.3,
    "directional_step_budget_percentile_low_fraction": 0.4,
    "directional_step_budget_lr_compensation": False,
    "directional_view_budget_enabled": False,
    "directional_view_budget_selection_mode": "novelty",
    "directional_view_budget_warmup_keyframes": 8,
    "directional_view_budget_very_low_degrees": 0.5,
    "directional_view_budget_low_degrees": 1.5,
    "directional_view_budget_very_low_compact_steps": 4,
    "directional_view_budget_low_compact_steps": 2,
    "directional_view_budget_compact_views": 2,
    "directional_view_budget_sampling_mode": "current_support",
    "directional_view_budget_full_anchor_steps": 1,
    "directional_view_budget_full_tail_steps": 2,
    "directional_view_budget_shuffle_very_low_fraction": 0.2,
    "directional_view_budget_shuffle_low_fraction": 0.6,
    "directional_view_budget_percentile_window": 64,
    "directional_view_budget_percentile_very_low_fraction": 0.3,
    "directional_view_budget_percentile_low_fraction": 0.4,
    "optimization_budget_routing_enabled": False,
    "optimization_budget_tail_collapse_mode": "gain",
    "optimization_budget_min_steps": 9,
    "optimization_budget_warmup_evidence_updates": 10,
    "optimization_budget_max_relative_tail": 1.0e-3,
    "optimization_budget_max_high_band_ratio": 1.25,
    "optimization_budget_decay_cap": 0.9,
    "optimization_budget_tail_gain_factor": 1.0,
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
    for key in (
        "enabled",
        "compute_routing",
        "spectral_replay_enabled",
        "spectral_residency_enabled",
        "bounded_replay_residency_enabled",
        "sparse_model_export",
        "frequency_schedule_enabled",
        "directional_step_budget_enabled",
        "directional_view_budget_enabled",
        "directional_step_budget_lr_compensation",
        "optimization_budget_routing_enabled",
    ):
        if not isinstance(result[key], bool):
            raise TypeError("StreamingAppearanceLOD.{} must be boolean".format(key))
    for key in (
        "birth_degree",
        "target_degree",
        "min_views",
        "promotion_interval",
        "compute_routing_warmup_evidence_updates",
        "exact_replay_microbatch_size",
        "exact_replay_gaussian_view_budget",
        "spectral_replay_max_views",
        "spectral_residency_max_views",
        "frequency_schedule_warmup_evidence_updates",
        "frequency_schedule_reallocation_level",
        "frequency_schedule_coarse_sh_degree",
        "frequency_schedule_full_anchor_steps",
        "frequency_schedule_full_tail_steps",
        "directional_step_budget_warmup_keyframes",
        "directional_step_budget_very_low_steps",
        "directional_step_budget_low_steps",
        "directional_step_budget_percentile_window",
        "directional_view_budget_warmup_keyframes",
        "directional_view_budget_very_low_compact_steps",
        "directional_view_budget_low_compact_steps",
        "directional_view_budget_compact_views",
        "directional_view_budget_full_anchor_steps",
        "directional_view_budget_full_tail_steps",
        "directional_view_budget_percentile_window",
        "optimization_budget_min_steps",
        "optimization_budget_warmup_evidence_updates",
    ):
        if not isinstance(result[key], int):
            raise TypeError("StreamingAppearanceLOD.{} must be an integer".format(key))
    if result["birth_degree"] < 0 or result["target_degree"] < 1:
        raise ValueError("Streaming appearance SH degrees must be non-negative")
    if result["target_degree"] > 3:
        raise ValueError("StreamingAppearanceLOD currently supports up to SH3")
    if result["birth_degree"] >= result["target_degree"]:
        raise ValueError("birth_degree must be lower than target_degree")
    if result["sparse_model_export"] and not result["enabled"]:
        raise ValueError("sparse_model_export requires StreamingAppearanceLOD.enabled")
    if result["bounded_replay_residency_enabled"] and not result["enabled"]:
        raise ValueError(
            "bounded_replay_residency_enabled requires StreamingAppearanceLOD.enabled"
        )
    if result["optimization_budget_routing_enabled"] and not result["enabled"]:
        raise ValueError(
            "optimization_budget_routing_enabled requires StreamingAppearanceLOD.enabled"
        )
    if result["optimization_budget_routing_enabled"] and result[
        "directional_step_budget_enabled"
    ]:
        raise ValueError(
            "optimization budget routing and directional step budgets cannot be combined"
        )
    if result["optimization_budget_routing_enabled"] and result[
        "directional_view_budget_enabled"
    ]:
        raise ValueError(
            "optimization budget routing and directional view budgets cannot be combined"
        )
    if result["optimization_budget_routing_enabled"] and result[
        "frequency_schedule_enabled"
    ]:
        raise ValueError(
            "optimization budget routing and frequency schedules cannot be combined"
        )
    if (
        result["directional_step_budget_enabled"]
        and result["directional_view_budget_enabled"]
    ):
        raise ValueError("directional step and view budgets cannot be enabled together")
    if result["min_views"] < 1 or result["promotion_interval"] < 1:
        raise ValueError("Streaming appearance count and interval must be positive")
    if result["compute_routing_warmup_evidence_updates"] < 0:
        raise ValueError("compute routing warmup must be non-negative")
    if result["exact_replay_microbatch_size"] < 0:
        raise ValueError("exact replay microbatch size must be non-negative")
    if result["exact_replay_gaussian_view_budget"] < 0:
        raise ValueError("exact replay Gaussian-view budget must be non-negative")
    if result["spectral_replay_enabled"] and not result["enabled"]:
        raise ValueError(
            "spectral_replay_enabled requires StreamingAppearanceLOD.enabled"
        )
    if result["spectral_replay_enabled"] and result[
        "exact_replay_microbatch_size"
    ] > 0:
        raise ValueError(
            "spectral replay and fixed exact replay cannot be enabled together"
        )
    if result["spectral_replay_max_views"] < 1:
        raise ValueError("spectral replay max views must be positive")
    if result["spectral_residency_enabled"] and not result["enabled"]:
        raise ValueError(
            "spectral_residency_enabled requires StreamingAppearanceLOD.enabled"
        )
    if result["spectral_residency_enabled"] and result[
        "bounded_replay_residency_enabled"
    ]:
        raise ValueError(
            "spectral residency and bounded replay residency cannot be combined"
        )
    if result["spectral_residency_max_views"] < 1:
        raise ValueError("spectral residency max views must be positive")
    if result["optimization_budget_min_steps"] < 3:
        raise ValueError("optimization budget routing requires at least three steps")
    if result["optimization_budget_warmup_evidence_updates"] < 0:
        raise ValueError("optimization budget warmup must be non-negative")
    if result["frequency_schedule_warmup_evidence_updates"] < 0:
        raise ValueError("frequency schedule warmup must be non-negative")
    if not 1 <= result["frequency_schedule_reallocation_level"] <= 3:
        raise ValueError("frequency schedule reallocation level must be in [1, 3]")
    if not 0 <= result["frequency_schedule_coarse_sh_degree"] <= 3:
        raise ValueError("frequency schedule coarse SH degree must be in [0, 3]")
    if (
        result["enabled"]
        and result["frequency_schedule_enabled"]
        and result["frequency_schedule_coarse_sh_degree"]
        != result["birth_degree"]
    ):
        raise ValueError("frequency schedule coarse SH degree must match birth_degree")
    if (
        result["frequency_schedule_full_anchor_steps"] < 0
        or result["frequency_schedule_full_tail_steps"] < 1
    ):
        raise ValueError("frequency schedule requires a non-negative anchor and a full tail")
    if result["directional_step_budget_warmup_keyframes"] < 0:
        raise ValueError("directional step budget warmup must be non-negative")
    if result["directional_step_budget_percentile_window"] < 2:
        raise ValueError("directional step percentile window must contain at least two keyframes")
    if (
        result["directional_step_budget_very_low_steps"] < 1
        or result["directional_step_budget_low_steps"]
        < result["directional_step_budget_very_low_steps"]
    ):
        raise ValueError("directional step budgets must be positive and ordered")
    if result["directional_view_budget_warmup_keyframes"] < 0:
        raise ValueError("directional view budget warmup must be non-negative")
    if result["directional_view_budget_percentile_window"] < 2:
        raise ValueError("directional percentile window must contain at least two keyframes")
    if (
        result["directional_view_budget_very_low_compact_steps"]
        < result["directional_view_budget_low_compact_steps"]
        or result["directional_view_budget_low_compact_steps"] < 0
    ):
        raise ValueError("directional compact-step budgets must be non-negative and ordered")
    if result["directional_view_budget_compact_views"] < 1:
        raise ValueError("directional compact view count must be positive")
    if (
        result["directional_view_budget_full_anchor_steps"] < 0
        or result["directional_view_budget_full_tail_steps"] < 1
    ):
        raise ValueError("directional view budget requires a full-view tail")
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
    if result["directional_step_budget_selection_mode"] not in {
        "novelty",
        "percentile",
        "shuffled",
    }:
        raise ValueError("directional step budget selection must be novelty, percentile, or shuffled")
    if result["directional_view_budget_selection_mode"] not in {
        "novelty",
        "percentile",
        "shuffled",
    }:
        raise ValueError("directional view budget selection must be novelty, percentile, or shuffled")
    if result["directional_view_budget_sampling_mode"] not in {
        "current_support",
        "cyclic_balanced",
    }:
        raise ValueError("directional view sampling must be current_support or cyclic_balanced")
    if result["optimization_budget_tail_collapse_mode"] not in {
        "none",
        "replay",
        "gain",
    }:
        raise ValueError(
            "optimization budget tail collapse mode must be none, replay, or gain"
        )
    result["shuffle_seed"] = int(result["shuffle_seed"])
    if result["gradient_ema_dtype"] not in {"float16", "float32"}:
        raise ValueError("gradient_ema_dtype must be float16 or float32")
    result["gradient_ema_scale"] = float(result["gradient_ema_scale"])
    if not math.isfinite(result["gradient_ema_scale"]) or result[
        "gradient_ema_scale"
    ] <= 0.0:
        raise ValueError("gradient_ema_scale must be finite and positive")
    result["spectral_replay_basis_budget"] = float(
        result["spectral_replay_basis_budget"]
    )
    if (
        not math.isfinite(result["spectral_replay_basis_budget"])
        or result["spectral_replay_basis_budget"] <= 0.0
    ):
        raise ValueError("spectral replay basis budget must be finite and positive")
    result["spectral_residency_basis_budget"] = float(
        result["spectral_residency_basis_budget"]
    )
    if (
        not math.isfinite(result["spectral_residency_basis_budget"])
        or result["spectral_residency_basis_budget"] <= 0.0
    ):
        raise ValueError(
            "spectral residency basis budget must be finite and positive"
        )
    weights = result["frequency_schedule_base_level_weights"]
    if not isinstance(weights, (list, tuple)) or len(weights) != 4:
        raise TypeError("frequency schedule level weights must contain four values")
    weights = [float(value) for value in weights]
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("frequency schedule level weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("frequency schedule level weights must sum to one")
    if weights[0] <= 0.0:
        raise ValueError("frequency schedule requires non-zero full-resolution weight")
    result["frequency_schedule_base_level_weights"] = weights
    for key in (
        "frequency_schedule_inactive_gain",
        "frequency_schedule_min_full_fraction",
    ):
        result[key] = float(result[key])
        if not math.isfinite(result[key]) or result[key] < 0.0:
            raise ValueError("StreamingAppearanceLOD.{} must be finite and non-negative".format(key))
    if result["frequency_schedule_min_full_fraction"] > weights[0]:
        raise ValueError("minimum full fraction cannot exceed the base full fraction")
    for key in (
        "directional_step_budget_very_low_degrees",
        "directional_step_budget_low_degrees",
        "directional_step_budget_shuffle_very_low_fraction",
        "directional_step_budget_shuffle_low_fraction",
        "directional_step_budget_percentile_very_low_fraction",
        "directional_step_budget_percentile_low_fraction",
        "directional_view_budget_very_low_degrees",
        "directional_view_budget_low_degrees",
        "directional_view_budget_shuffle_very_low_fraction",
        "directional_view_budget_shuffle_low_fraction",
        "directional_view_budget_percentile_very_low_fraction",
        "directional_view_budget_percentile_low_fraction",
        "optimization_budget_max_relative_tail",
        "optimization_budget_max_high_band_ratio",
        "optimization_budget_decay_cap",
        "optimization_budget_tail_gain_factor",
    ):
        result[key] = float(result[key])
        if not math.isfinite(result[key]) or result[key] < 0.0:
            raise ValueError("StreamingAppearanceLOD.{} must be finite and non-negative".format(key))
    if not 0.0 < result["optimization_budget_decay_cap"] < 1.0:
        raise ValueError("optimization_budget_decay_cap must be in (0, 1)")
    if result["optimization_budget_tail_gain_factor"] > 2.0:
        raise ValueError("optimization_budget_tail_gain_factor must be in [0, 2]")
    if (
        result["directional_step_budget_low_degrees"]
        < result["directional_step_budget_very_low_degrees"]
    ):
        raise ValueError("directional novelty thresholds must be ordered")
    if (
        result["directional_step_budget_shuffle_very_low_fraction"]
        + result["directional_step_budget_shuffle_low_fraction"]
        > 1.0
    ):
        raise ValueError("directional shuffled budget fractions cannot exceed one")
    if (
        result["directional_step_budget_percentile_very_low_fraction"]
        + result["directional_step_budget_percentile_low_fraction"]
        > 1.0
    ):
        raise ValueError("directional step percentile fractions cannot exceed one")
    if (
        result["directional_view_budget_low_degrees"]
        < result["directional_view_budget_very_low_degrees"]
    ):
        raise ValueError("directional view novelty thresholds must be ordered")
    if (
        result["directional_view_budget_shuffle_very_low_fraction"]
        + result["directional_view_budget_shuffle_low_fraction"]
        > 1.0
    ):
        raise ValueError("directional shuffled view fractions cannot exceed one")
    if (
        result["directional_view_budget_percentile_very_low_fraction"]
        + result["directional_view_budget_percentile_low_fraction"]
        > 1.0
    ):
        raise ValueError("directional percentile fractions cannot exceed one")
    return result


def optimization_tail_certificate(
    current_view_losses,
    high_band_gradient_ratio,
    remaining_steps,
    config,
):
    """Certify that the predicted optimization tail has low marginal value.

    The last two non-negative loss improvements define a capped geometric
    decay model. Its finite tail bounds the expected current-view improvement;
    the coefficient-count-normalized SH3 gradient ratio independently prevents
    truncation while the routed high-frequency band still carries substantial
    descent energy.
    """

    losses = [float(value) for value in current_view_losses]
    remaining_steps = int(remaining_steps)
    high_band_gradient_ratio = float(high_band_gradient_ratio)
    result = {
        "stop": False,
        "reason": "insufficient_history",
        "relative_tail_bound": float("inf"),
        "high_band_gradient_ratio": high_band_gradient_ratio,
        "decay_ratio": None,
        "last_improvement": None,
    }
    if len(losses) < 3 or remaining_steps <= 0:
        return result
    if any(not math.isfinite(value) or value < 0.0 for value in losses[-3:]):
        result["reason"] = "invalid_loss"
        return result
    if not math.isfinite(high_band_gradient_ratio):
        result["reason"] = "invalid_gradient"
        return result

    previous_improvement = max(0.0, losses[-3] - losses[-2])
    last_improvement = max(0.0, losses[-2] - losses[-1])
    if previous_improvement <= 0.0:
        if last_improvement > 0.0:
            result.update(
                reason="nondecaying_tail",
                decay_ratio=float("inf"),
                last_improvement=float(last_improvement),
            )
            return result
        decay_ratio = 0.0
    else:
        decay_ratio = max(0.0, last_improvement / previous_improvement)
    if decay_ratio > float(config["optimization_budget_decay_cap"]):
        result.update(
            reason="nondecaying_tail",
            decay_ratio=float(decay_ratio),
            last_improvement=float(last_improvement),
        )
        return result
    if decay_ratio <= 0.0:
        tail_bound = 0.0
    else:
        tail_bound = (
            last_improvement
            * decay_ratio
            * (1.0 - decay_ratio**remaining_steps)
            / (1.0 - decay_ratio)
        )
    relative_tail_bound = tail_bound / max(losses[-1], 1.0e-8)
    low_tail = relative_tail_bound <= float(
        config["optimization_budget_max_relative_tail"]
    )
    low_band = high_band_gradient_ratio <= float(
        config["optimization_budget_max_high_band_ratio"]
    )
    result.update(
        stop=bool(low_tail and low_band),
        reason=(
            "certified"
            if low_tail and low_band
            else "tail_bound"
            if not low_tail
            else "high_band_gradient"
        ),
        relative_tail_bound=float(relative_tail_bound),
        decay_ratio=float(decay_ratio),
        last_improvement=float(last_improvement),
    )
    return result


def _largest_remainder_counts(total, weights):
    raw = [float(total) * float(weight) for weight in weights]
    counts = [int(math.floor(value)) for value in raw]
    remainder = int(total) - sum(counts)
    order = sorted(
        range(len(weights)), key=lambda index: (raw[index] - counts[index], -index), reverse=True
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def exact_replay_microbatch_ranges(view_count, microbatch_size):
    """Partition one logical replay batch without dropping any camera."""

    view_count = int(view_count)
    microbatch_size = int(microbatch_size)
    if view_count < 0 or microbatch_size < 0:
        raise ValueError("Replay view counts and microbatch sizes must be non-negative")
    if view_count == 0:
        return []
    if microbatch_size == 0 or microbatch_size >= view_count:
        return [(0, view_count)]
    return [
        (begin, min(view_count, begin + microbatch_size))
        for begin in range(0, view_count, microbatch_size)
    ]


def spectral_replay_microbatch_limit(target_fraction, config):
    """Bound simultaneous view gradients by TGBR's active SH basis load.

    TGBR has only birth-degree and target-degree rows, so the mean active basis
    count is the convex combination of their basis dimensions. The configured
    budget is expressed as basis terms per Gaussian across the whole view
    microbatch, making the limit independent of the current map size.
    """

    if set(DEFAULT_CONFIG) - set(config):
        config = validate_streaming_appearance_lod_config(config)
    if not config["spectral_replay_enabled"]:
        return 0, 0.0
    fraction = min(1.0, max(0.0, float(target_fraction)))
    base_terms = (int(config["birth_degree"]) + 1) ** 2
    target_terms = (int(config["target_degree"]) + 1) ** 2
    mean_terms = base_terms + fraction * (target_terms - base_terms)
    limit = int(float(config["spectral_replay_basis_budget"]) // mean_terms)
    limit = max(1, min(int(config["spectral_replay_max_views"]), limit))
    return limit, mean_terms


def spectral_residency_limit(target_fraction, config):
    """Keep the active-basis/view product within a fixed evidence budget."""

    if set(DEFAULT_CONFIG) - set(config):
        config = validate_streaming_appearance_lod_config(config)
    if not config["spectral_residency_enabled"]:
        return 0, 0.0
    fraction = min(1.0, max(0.0, float(target_fraction)))
    base_terms = (int(config["birth_degree"]) + 1) ** 2
    target_terms = (int(config["target_degree"]) + 1) ** 2
    mean_terms = base_terms + fraction * (target_terms - base_terms)
    limit = int(float(config["spectral_residency_basis_budget"]) // mean_terms)
    limit = max(1, min(int(config["spectral_residency_max_views"]), limit))
    return limit, mean_terms


def merge_replay_projection_info(records):
    """Merge packed projection records and restore logical camera indices."""

    records = list(records)
    if not records:
        return None
    merged = {}
    for key in ("gaussian_ids", "camera_ids", "radii", "depths"):
        values = []
        for camera_offset, projection_info in records:
            value = projection_info.get(key)
            if value is None:
                values = []
                break
            if key == "radii" and value.ndim > 1:
                value = value.amax(dim=-1)
            value = value.reshape(-1)
            if key == "camera_ids":
                value = value + int(camera_offset)
            values.append(value)
        if values:
            merged[key] = torch.cat(values, dim=0)
    return merged


def frequency_consistent_step_levels(
    optimization_steps,
    evidence_updates,
    target_fraction,
    config,
    base_level=0,
):
    """Allocate a deterministic low-band phase between full-resolution anchors."""

    config = validate_streaming_appearance_lod_config(config)
    steps = int(optimization_steps)
    if steps < 0:
        raise ValueError("optimization_steps must be non-negative")
    if steps == 0:
        return []
    if not config["frequency_schedule_enabled"] or int(base_level) != 0:
        return [int(base_level)] * steps

    weights = list(config["frequency_schedule_base_level_weights"])
    if int(evidence_updates) >= int(
        config["frequency_schedule_warmup_evidence_updates"]
    ):
        target_fraction = min(1.0, max(0.0, float(target_fraction)))
        inactive_fraction = 1.0 - target_fraction
        desired_full = max(
            float(config["frequency_schedule_min_full_fraction"]),
            weights[0]
            - float(config["frequency_schedule_inactive_gain"])
            * inactive_fraction,
        )
        reallocated = weights[0] - desired_full
        weights[0] = desired_full
        weights[int(config["frequency_schedule_reallocation_level"])] += reallocated

    counts = _largest_remainder_counts(steps, weights)
    anchor = min(int(config["frequency_schedule_full_anchor_steps"]), steps)
    tail = min(int(config["frequency_schedule_full_tail_steps"]), steps - anchor)
    required_full = anchor + tail
    while counts[0] < required_full:
        source = max(range(1, 4), key=lambda index: (counts[index], index))
        if counts[source] == 0:
            break
        counts[source] -= 1
        counts[0] += 1

    middle_full = counts[0] - required_full
    middle = []
    for level in range(3, 0, -1):
        middle.extend([level] * counts[level])
    middle.extend([0] * middle_full)
    schedule = [0] * anchor + middle + [0] * tail
    if len(schedule) != steps:
        raise RuntimeError("frequency schedule did not consume the optimization budget")
    return schedule


def view_direction_novelty_degrees(previous_pose, current_pose, median_depth):
    """Estimate an upper bound on per-point view-direction change."""

    previous = torch.as_tensor(previous_pose, dtype=torch.float64)
    current = torch.as_tensor(current_pose, dtype=torch.float64)
    if previous.shape != (4, 4) or current.shape != (4, 4):
        raise ValueError("Camera poses must have shape [4, 4]")
    depth = float(median_depth)
    if not math.isfinite(depth) or depth <= 0.0:
        return float("inf"), float("inf"), float("inf")

    relative_rotation = current[:3, :3] @ previous[:3, :3].T
    cosine = torch.clamp((torch.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    rotation_degrees = math.degrees(float(torch.acos(cosine).item()))
    previous_center = -previous[:3, :3].T @ previous[:3, 3]
    current_center = -current[:3, :3].T @ current[:3, 3]
    baseline = float(torch.linalg.norm(current_center - previous_center).item())
    parallax_degrees = math.degrees(math.atan2(baseline, depth))
    return (
        rotation_degrees + parallax_degrees,
        rotation_degrees,
        parallax_degrees,
    )


def directional_step_budget(
    novelty_degrees,
    full_steps,
    keyframe_index,
    config,
    novelty_percentile=None,
):
    """Select an optimization budget from causal directional novelty."""

    config = validate_streaming_appearance_lod_config(config)
    full_steps = int(full_steps)
    if full_steps < 1:
        raise ValueError("full_steps must be positive")
    if (
        not config["directional_step_budget_enabled"]
        or int(keyframe_index)
        < int(config["directional_step_budget_warmup_keyframes"])
    ):
        return full_steps, "full"

    very_low_steps = min(
        full_steps, int(config["directional_step_budget_very_low_steps"])
    )
    low_steps = min(full_steps, int(config["directional_step_budget_low_steps"]))
    if config["directional_step_budget_selection_mode"] == "percentile":
        if novelty_percentile is None or not math.isfinite(
            float(novelty_percentile)
        ):
            return full_steps, "full"
        percentile = min(1.0, max(0.0, float(novelty_percentile)))
        very_low_fraction = float(
            config["directional_step_budget_percentile_very_low_fraction"]
        )
        low_fraction = float(
            config["directional_step_budget_percentile_low_fraction"]
        )
        if percentile < very_low_fraction:
            return very_low_steps, "very_low"
        if percentile < very_low_fraction + low_fraction:
            return low_steps, "low"
        return full_steps, "full"
    if config["directional_step_budget_selection_mode"] == "shuffled":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config["shuffle_seed"]) + int(keyframe_index))
        sample = float(torch.rand((), generator=generator).item())
        very_low_fraction = float(
            config["directional_step_budget_shuffle_very_low_fraction"]
        )
        low_fraction = float(config["directional_step_budget_shuffle_low_fraction"])
        if sample < very_low_fraction:
            return very_low_steps, "very_low"
        if sample < very_low_fraction + low_fraction:
            return low_steps, "low"
        return full_steps, "full"

    novelty = float(novelty_degrees)
    if not math.isfinite(novelty):
        return full_steps, "full"
    if novelty < float(config["directional_step_budget_very_low_degrees"]):
        return very_low_steps, "very_low"
    if novelty < float(config["directional_step_budget_low_degrees"]):
        return low_steps, "low"
    return full_steps, "full"


def directional_compact_view_steps(
    novelty_degrees,
    optimization_steps,
    keyframe_index,
    config,
    novelty_percentile=None,
):
    """Allocate compact-view updates without changing the update count."""

    config = validate_streaming_appearance_lod_config(config)
    optimization_steps = int(optimization_steps)
    if optimization_steps < 1:
        raise ValueError("optimization_steps must be positive")
    if (
        not config["directional_view_budget_enabled"]
        or int(keyframe_index)
        < int(config["directional_view_budget_warmup_keyframes"])
    ):
        return 0, "full"

    available = max(
        0,
        optimization_steps
        - int(config["directional_view_budget_full_anchor_steps"])
        - int(config["directional_view_budget_full_tail_steps"]),
    )
    very_low_steps = min(
        available,
        int(config["directional_view_budget_very_low_compact_steps"]),
    )
    low_steps = min(
        available,
        int(config["directional_view_budget_low_compact_steps"]),
    )
    if config["directional_view_budget_selection_mode"] == "percentile":
        if novelty_percentile is None or not math.isfinite(
            float(novelty_percentile)
        ):
            return 0, "full"
        percentile = min(1.0, max(0.0, float(novelty_percentile)))
        very_low_fraction = float(
            config["directional_view_budget_percentile_very_low_fraction"]
        )
        low_fraction = float(
            config["directional_view_budget_percentile_low_fraction"]
        )
        if percentile < very_low_fraction:
            return very_low_steps, "very_low"
        if percentile < very_low_fraction + low_fraction:
            return low_steps, "low"
        return 0, "full"
    if config["directional_view_budget_selection_mode"] == "shuffled":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config["shuffle_seed"]) + int(keyframe_index))
        sample = float(torch.rand((), generator=generator).item())
        very_low_fraction = float(
            config["directional_view_budget_shuffle_very_low_fraction"]
        )
        low_fraction = float(config["directional_view_budget_shuffle_low_fraction"])
        if sample < very_low_fraction:
            return very_low_steps, "very_low"
        if sample < very_low_fraction + low_fraction:
            return low_steps, "low"
        return 0, "full"

    novelty = float(novelty_degrees)
    if not math.isfinite(novelty):
        return 0, "full"
    if novelty < float(config["directional_view_budget_very_low_degrees"]):
        return very_low_steps, "very_low"
    if novelty < float(config["directional_view_budget_low_degrees"]):
        return low_steps, "low"
    return 0, "full"


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
