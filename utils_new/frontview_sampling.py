"""Causal evidence-balanced candidate selection for forward-view mapping."""

from copy import deepcopy
import math

import torch
import torch.nn.functional as F


DEFAULT_FRONT_VIEW_SAMPLING_CONFIG = {
    "enabled": False,
    "selection_mode": "evidence_balanced",
    "pool_multiplier": 2,
    "evidence_fraction": 0.50,
    "reference_frames": 2,
    "photo_sigma": 0.08,
    "photo_mode": "consistency",
    "parallax_reference_deg": 2.0,
    "parallax_floor": 0.25,
    "confidence_power": 1.0,
    "shuffle_evidence": False,
    "shuffle_depth_bands": False,
    "projective_cell_px": 12,
    "shuffle_projective_coverage": False,
    "anchor_selection_mode": "random",
    "anchor_cell_px": 24,
    "shuffle_anchor_coverage": False,
    "shuffle_seed": 42,
    "depth_edges_m": [20.0, 50.0],
    "depth_fractions": [0.33, 0.34, 0.33],
}


def _stable_argsort(values: torch.Tensor, *, descending: bool = False):
    """Stable one-dimensional argsort compatible with older PyTorch builds."""

    try:
        return torch.argsort(values, descending=descending, stable=True)
    except TypeError:
        order = torch.argsort(values, descending=descending)
        if order.numel() < 2:
            return order
        sorted_values = values[order]
        starts_group = torch.ones_like(order, dtype=torch.bool)
        starts_group[1:] = sorted_values[1:] != sorted_values[:-1]
        group_ids = torch.cumsum(starts_group.long(), dim=0) - 1
        stable_keys = group_ids * (order.numel() + 1) + order
        return order[torch.argsort(stable_keys)]


def validate_front_view_sampling_config(config=None):
    merged = deepcopy(DEFAULT_FRONT_VIEW_SAMPLING_CONFIG)
    if config is not None:
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError(
                "Unknown FrontViewSampling options: {}".format(sorted(unknown))
            )
        merged.update(config)
    for key in (
        "enabled",
        "shuffle_evidence",
        "shuffle_depth_bands",
        "shuffle_projective_coverage",
        "shuffle_anchor_coverage",
    ):
        if not isinstance(merged[key], bool):
            raise TypeError("FrontViewSampling.{} must be boolean".format(key))
    for key in ("pool_multiplier", "reference_frames"):
        if not isinstance(merged[key], int) or merged[key] <= 0:
            raise ValueError("FrontViewSampling.{} must be positive".format(key))
    if not isinstance(merged["projective_cell_px"], int) or merged[
        "projective_cell_px"
    ] <= 0:
        raise ValueError("FrontViewSampling.projective_cell_px must be positive")
    if not isinstance(merged["anchor_cell_px"], int) or merged["anchor_cell_px"] <= 0:
        raise ValueError("FrontViewSampling.anchor_cell_px must be positive")
    if int(merged["pool_multiplier"]) < 2:
        raise ValueError("FrontViewSampling.pool_multiplier must be at least two")
    for key in ("evidence_fraction", "parallax_floor"):
        value = float(merged[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError("FrontViewSampling.{} must be in [0, 1]".format(key))
    for key in ("photo_sigma", "parallax_reference_deg", "confidence_power"):
        if float(merged[key]) <= 0.0:
            raise ValueError("FrontViewSampling.{} must be positive".format(key))
    if not isinstance(merged["shuffle_seed"], int):
        raise TypeError("FrontViewSampling.shuffle_seed must be an integer")
    if merged["photo_mode"] not in ("consistency", "disocclusion"):
        raise ValueError(
            "FrontViewSampling.photo_mode must be consistency or disocclusion"
        )
    if merged["selection_mode"] not in (
        "evidence_balanced",
        "uniform_survivor",
        "depth_stratified",
        "projective_coverage",
        "residual_importance",
        "adaptive_log_depth_random",
        "adaptive_log_depth_importance",
        "adaptive_log_depth_shuffled",
        "adaptive_log_depth_coverage",
        "adaptive_log_depth_residual_coverage",
        "adaptive_log_depth_coverage_shuffled",
        "adaptive_log_depth_rate_distortion",
        "adaptive_log_depth_rate_distortion_shuffled",
    ):
        raise ValueError("FrontViewSampling.selection_mode is invalid")
    if merged["anchor_selection_mode"] not in ("random", "projective_coverage"):
        raise ValueError("FrontViewSampling.anchor_selection_mode is invalid")
    edges = [float(value) for value in merged["depth_edges_m"]]
    fractions = [float(value) for value in merged["depth_fractions"]]
    if len(edges) != 2 or not 0.0 < edges[0] < edges[1]:
        raise ValueError("FrontViewSampling.depth_edges_m must be two increasing values")
    if len(fractions) != 3 or any(value < 0.0 for value in fractions):
        raise ValueError("FrontViewSampling.depth_fractions must have three nonnegative values")
    if abs(sum(fractions) - 1.0) > 1.0e-6:
        raise ValueError("FrontViewSampling.depth_fractions must sum to one")
    return merged


def _balanced_capacity_quotas(capacities, budget):
    """Solve max-min regime allocation with finite per-regime capacities."""

    capacities = [max(0, int(value)) for value in capacities]
    budget = min(max(0, int(budget)), sum(capacities))
    quotas = [0] * len(capacities)
    remaining = budget
    while remaining:
        active = [i for i, capacity in enumerate(capacities) if quotas[i] < capacity]
        if not active:
            break
        share = max(1, remaining // len(active))
        for regime in active:
            increment = min(share, capacities[regime] - quotas[regime], remaining)
            quotas[regime] += increment
            remaining -= increment
            if remaining == 0:
                break
    return quotas


def _lloyd_log_depth_regimes(depths: torch.Tensor, regime_count: int = 3):
    """Quantize relative depth error with causal one-dimensional Lloyd updates."""

    if depths.ndim != 1:
        raise ValueError("Candidate depths must be one-dimensional")
    if regime_count <= 0:
        raise ValueError("Depth regime count must be positive")
    if depths.numel() == 0:
        return (
            torch.empty(0, device=depths.device, dtype=torch.long),
            torch.empty(0, device=depths.device, dtype=depths.dtype),
            0,
            0.0,
        )
    if not bool(torch.all(torch.isfinite(depths) & (depths > 0.0))):
        raise ValueError("Candidate depths must be finite and positive")

    log_depths = torch.log(depths)
    quantiles = (
        torch.arange(regime_count, device=depths.device, dtype=depths.dtype) + 0.5
    ) / float(regime_count)
    centers = torch.quantile(log_depths, quantiles)
    previous_labels = None
    iterations = 0
    for iterations in range(1, 33):
        distances = torch.abs(log_depths[:, None] - centers[None, :])
        labels = torch.argmin(distances, dim=1)
        counts = torch.bincount(labels, minlength=regime_count)
        sums = torch.zeros_like(centers)
        sums.scatter_add_(0, labels, log_depths)
        updated = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
        updated = torch.sort(updated).values
        if previous_labels is not None and torch.equal(labels, previous_labels):
            centers = updated
            break
        previous_labels = labels
        centers = updated

    labels = torch.argmin(
        torch.abs(log_depths[:, None] - centers[None, :]), dim=1
    )
    objective = float(
        torch.mean((log_depths - centers[labels]).square()).item()
    )
    return labels, centers, iterations, objective


def adaptive_log_depth_indices(
    depths: torch.Tensor,
    confidences: torch.Tensor,
    residuals: torch.Tensor,
    budget: int,
    *,
    uv: torch.Tensor = None,
    image_size=None,
    pool_multiplier: int = 1,
    weighted: bool = False,
    shuffle_regimes: bool = False,
    coverage_priority: str = None,
    density_weights: torch.Tensor = None,
    shuffle_density: bool = False,
    seed: int = 42,
):
    """Select a fixed budget from online, scale-invariant depth regimes.

    Lloyd quantization minimizes squared relative-depth distortion in log space.
    A capped water-filling allocation then maximizes the minimum evidence count
    across the three regimes; a regime smaller than its allocation is retained
    in full and its unused budget is redistributed automatically.
    """

    if depths.ndim != 1 or confidences.ndim != 1 or residuals.ndim != 1:
        raise ValueError("Adaptive sampling inputs must be one-dimensional")
    if not (depths.shape == confidences.shape == residuals.shape):
        raise ValueError("Adaptive sampling inputs must align")
    if coverage_priority not in (None, "confidence", "residual_confidence"):
        raise ValueError("Adaptive coverage priority is invalid")
    if density_weights is not None:
        if coverage_priority is not None or weighted:
            raise ValueError(
                "Rate-distortion density is mutually exclusive with other priorities"
            )
        density_weights = torch.as_tensor(
            density_weights, device=depths.device, dtype=depths.dtype
        ).reshape(-1)
        if density_weights.shape != depths.shape:
            raise ValueError("Rate-distortion density must align with candidates")
        if not bool(torch.all(torch.isfinite(density_weights))) or bool(
            torch.any(density_weights < 0.0)
        ):
            raise ValueError("Rate-distortion density must be finite and nonnegative")
    elif shuffle_density:
        raise ValueError("Shuffled density requires rate-distortion density")
    if coverage_priority is not None:
        if uv is None or uv.ndim != 2 or uv.shape != (depths.numel(), 2):
            raise ValueError("Adaptive coverage pixels must have shape [N, 2]")
        if image_size is None or len(image_size) != 2:
            raise ValueError("Adaptive coverage requires image width and height")
        if int(image_size[0]) <= 0 or int(image_size[1]) <= 0:
            raise ValueError("Adaptive coverage image dimensions must be positive")
        if int(pool_multiplier) <= 0:
            raise ValueError("Adaptive coverage pool multiplier must be positive")
    count = int(depths.numel())
    budget = min(max(int(budget), 0), count)
    labels, centers, iterations, objective = _lloyd_log_depth_regimes(depths)
    learned_labels = labels

    generator = torch.Generator(device=depths.device)
    generator.manual_seed(int(seed))
    if shuffle_regimes and count > 1:
        labels = labels[
            torch.randperm(count, generator=generator, device=depths.device)
        ]

    capacities = torch.bincount(labels, minlength=3).tolist()
    quotas = _balanced_capacity_quotas(capacities, budget)
    selected = torch.zeros(count, device=depths.device, dtype=torch.bool)
    importance = torch.nan_to_num(
        torch.clamp(residuals, min=0.0) * torch.clamp(confidences, 0.0, 1.0),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    cell_ids = None
    cell_px = None
    coverage_representatives = [0, 0, 0]
    if coverage_priority is not None:
        width, height = (int(value) for value in image_size)
        cell_px = max(
            1.0,
            math.sqrt(
                float(width * height)
                / max(float(budget * int(pool_multiplier)), 1.0)
            ),
        )
        cells_per_row = max(1, int(math.ceil(float(width) / cell_px)))
        cell_x = torch.floor(uv[:, 0] / cell_px).long()
        cell_y = torch.floor(uv[:, 1] / cell_px).long()
        cell_ids = cell_y * cells_per_row + cell_x

    for regime, quota in enumerate(quotas):
        rows = torch.nonzero(labels == regime, as_tuple=False).flatten()
        if quota <= 0:
            continue
        if quota >= rows.numel():
            chosen = rows
        elif density_weights is not None:
            weights = density_weights[rows]
            if shuffle_density and rows.numel() > 1:
                weights = weights[
                    torch.randperm(
                        rows.numel(), generator=generator, device=depths.device
                    )
                ]
            if bool(torch.any(weights > 0.0)):
                chosen = rows[
                    torch.multinomial(
                        weights,
                        quota,
                        replacement=False,
                        generator=generator,
                    )
                ]
            else:
                chosen = rows[
                    torch.randperm(
                        rows.numel(), generator=generator, device=depths.device
                    )[:quota]
                ]
        elif coverage_priority is not None:
            priority = (
                importance[rows]
                if coverage_priority == "residual_confidence"
                else torch.clamp(confidences[rows], 0.0, 1.0)
            )
            ranked = rows[_stable_argsort(priority, descending=True)]
            by_cell = ranked[_stable_argsort(cell_ids[ranked])]
            grouped_cells = cell_ids[by_cell]
            first = torch.ones_like(grouped_cells, dtype=torch.bool)
            first[1:] = grouped_cells[1:] != grouped_cells[:-1]
            representatives = by_cell[first]
            coverage_representatives[regime] = int(representatives.numel())
            representative_order = torch.randperm(
                representatives.numel(),
                generator=generator,
                device=depths.device,
            )
            chosen = representatives[representative_order[:quota]]
            shortfall = quota - int(chosen.numel())
            if shortfall > 0:
                already_chosen = torch.zeros(
                    count, device=depths.device, dtype=torch.bool
                )
                already_chosen[chosen] = True
                remaining = ranked[~already_chosen[ranked]]
                chosen = torch.cat((chosen, remaining[:shortfall]))
        elif weighted and bool(torch.any(importance[rows] > 0.0)):
            positive = rows[importance[rows] > 0.0]
            if positive.numel() >= quota:
                chosen = positive[
                    torch.multinomial(
                        importance[positive],
                        quota,
                        replacement=False,
                        generator=generator,
                    )
                ]
            else:
                zero_weight = rows[importance[rows] <= 0.0]
                supplement = zero_weight[
                    torch.randperm(
                        zero_weight.numel(),
                        generator=generator,
                        device=depths.device,
                    )[: quota - positive.numel()]
                ]
                chosen = torch.cat((positive, supplement))
        else:
            chosen = rows[
                torch.randperm(
                    rows.numel(), generator=generator, device=depths.device
                )[:quota]
            ]
        selected[chosen] = True

    selected_indices = torch.nonzero(selected, as_tuple=False).flatten()
    learned_counts = torch.bincount(learned_labels, minlength=3)
    selected_counts = torch.bincount(
        learned_labels[selected_indices], minlength=3
    )
    boundaries = torch.exp(0.5 * (centers[:-1] + centers[1:]))
    metadata = {
        "boundaries_m": [float(value) for value in boundaries.tolist()],
        "centers_log_depth": [float(value) for value in centers.tolist()],
        "pool_counts": [int(value) for value in learned_counts.tolist()],
        "assigned_pool_counts": [int(value) for value in capacities],
        "quotas": [int(value) for value in quotas],
        "selected_counts": [int(value) for value in selected_counts.tolist()],
        "iterations": int(iterations),
        "objective": float(objective),
        "shuffled": bool(shuffle_regimes),
        "coverage_priority": coverage_priority,
        "coverage_cell_px": cell_px,
        "coverage_representatives": coverage_representatives,
        "density_weighted": density_weights is not None,
        "density_shuffled": bool(shuffle_density),
    }
    return selected_indices, metadata


def rate_distortion_density_weights(
    image: torch.Tensor,
    uv: torch.Tensor,
    confidences: torch.Tensor,
    *,
    image_size,
    budget: int,
    pool_multiplier: int,
):
    """Estimate the fixed-budget image sampling density at candidate pixels.

    With local first-order image energy ``E`` and sample density ``rho``, the
    piecewise-constant distortion is proportional to ``E / rho``. Minimizing
    its image integral under a fixed sample budget gives
    ``rho proportional to sqrt(E)``. Depth confidence is the probability that
    a sampled pixel creates usable geometry, yielding ``sqrt(confidence * E)``.
    The current-frame mean energy is an online textureless-region prior.
    """

    image = torch.as_tensor(image)
    uv = torch.as_tensor(uv, device=image.device, dtype=image.dtype)
    confidences = torch.as_tensor(
        confidences, device=image.device, dtype=image.dtype
    ).reshape(-1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Rate-distortion image must have shape [H, W, 3]")
    if uv.ndim != 2 or uv.shape != (confidences.numel(), 2):
        raise ValueError("Rate-distortion pixels must have shape [N, 2]")
    width, height = (int(value) for value in image_size)
    if image.shape[:2] != (height, width):
        raise ValueError("Rate-distortion image size does not match its contract")
    if int(budget) <= 0 or int(pool_multiplier) <= 0:
        raise ValueError("Rate-distortion budget and pool multiplier must be positive")
    if not bool(torch.all(torch.isfinite(confidences))) or bool(
        torch.any((confidences < 0.0) | (confidences > 1.0))
    ):
        raise ValueError("Rate-distortion confidence must be in [0, 1]")

    luminance = (
        image[..., 0] * 0.299
        + image[..., 1] * 0.587
        + image[..., 2] * 0.114
    )[None, None]
    kernels = luminance.new_tensor(
        [
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        ]
    )[:, None] / 8.0
    gradients = F.conv2d(luminance, kernels, padding=1)[0]
    energy = gradients.square().sum(dim=0)
    cell_px = math.sqrt(
        float(width * height) / float(int(budget) * int(pool_multiplier))
    )
    radius = max(1, int(math.ceil(0.5 * cell_px)))
    kernel_size = 2 * radius + 1
    local_energy = F.avg_pool2d(
        energy[None, None],
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )[0, 0]
    prior = torch.clamp(
        torch.mean(local_energy), min=torch.finfo(local_energy.dtype).eps
    )
    x = torch.clamp(torch.floor(uv[:, 0]).long(), 0, width - 1)
    y = torch.clamp(torch.floor(uv[:, 1]).long(), 0, height - 1)
    density = torch.sqrt(
        torch.clamp(confidences, 0.0, 1.0)
        * torch.clamp(local_energy[y, x] + prior, min=prior)
    )
    density /= torch.clamp(
        torch.mean(density), min=torch.finfo(density.dtype).eps
    )
    return density, {
        "cell_px": float(cell_px),
        "mean": float(torch.mean(density).item()),
        "min": float(torch.min(density).item()) if density.numel() else None,
        "max": float(torch.max(density).item()) if density.numel() else None,
    }


def residual_rate_distortion_radius_factors(
    residual: torch.Tensor,
    uv: torch.Tensor,
    eligible: torch.Tensor,
    *,
    image_size,
    budget: int,
    pool_multiplier: int,
    visibility: torch.Tensor = None,
    detail_protection: bool = False,
) -> torch.Tensor:
    """Allocate equal total footprint area by local residual rate-distortion.

    For piecewise-constant image reconstruction, optimal sample density obeys
    ``rho proportional to sqrt(E)``, where ``E`` is local gradient energy. A
    sample's responsibility area is inverse density, so its radius is
    proportional to ``E**(-1/4)``. RMS normalization preserves total footprint
    area over eligible candidates without a scene-specific scale parameter.
    """

    residual = torch.as_tensor(residual)
    uv = torch.as_tensor(uv, device=residual.device, dtype=residual.dtype)
    eligible = torch.as_tensor(
        eligible, device=residual.device, dtype=torch.bool
    ).reshape(-1)
    width, height = (int(value) for value in image_size)
    if residual.ndim == 3:
        residual = torch.mean(residual, dim=-1)
    if residual.shape != (height, width):
        raise ValueError("Residual map size does not match its contract")
    if uv.ndim != 2 or uv.shape != (eligible.numel(), 2):
        raise ValueError("Residual footprint pixels must have shape [N, 2]")
    if int(budget) <= 0 or int(pool_multiplier) <= 0:
        raise ValueError("Residual footprint budget must be positive")
    if not bool(torch.all(torch.isfinite(residual))):
        raise ValueError("Residual footprint map must be finite")

    factors = torch.ones(eligible.shape, device=residual.device, dtype=residual.dtype)
    if not bool(torch.any(eligible)):
        return factors
    kernels = residual.new_tensor(
        [
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        ]
    )[:, None] / 8.0
    gradients = F.conv2d(residual[None, None], kernels, padding=1)[0]
    energy = gradients.square().sum(dim=0)
    if visibility is not None:
        visibility = torch.as_tensor(
            visibility, device=residual.device, dtype=residual.dtype
        )
        if visibility.ndim == 3:
            visibility = torch.mean(visibility, dim=-1)
        if visibility.shape != (height, width):
            raise ValueError("Residual visibility size does not match its contract")
        if not bool(torch.all(torch.isfinite(visibility))) or bool(
            torch.any((visibility < 0.0) | (visibility > 1.0))
        ):
            raise ValueError("Residual visibility must lie in [0, 1]")
        energy = energy * visibility
    cell_px = math.sqrt(
        float(width * height) / float(int(budget) * int(pool_multiplier))
    )
    radius = max(1, int(math.ceil(0.5 * cell_px)))
    local_energy = F.avg_pool2d(
        energy[None, None],
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )[0, 0]
    prior = torch.clamp(
        torch.mean(local_energy), min=torch.finfo(local_energy.dtype).eps
    )
    x = torch.clamp(torch.floor(uv[:, 0]).long(), 0, width - 1)
    y = torch.clamp(torch.floor(uv[:, 1]).long(), 0, height - 1)
    raw = torch.pow(local_energy[y, x] + prior, -0.25)
    eligible_raw = raw[eligible]
    area_normalizer = torch.sqrt(torch.mean(eligible_raw.square()))
    factors[eligible] = eligible_raw / torch.clamp(
        area_normalizer, min=torch.finfo(raw.dtype).eps
    )
    if detail_protection:
        factors[eligible] = torch.clamp(factors[eligible], max=1.0)
    return factors


def residual_importance_indices(
    residuals: torch.Tensor,
    confidences: torch.Tensor,
    budget: int,
    *,
    seed: int = 42,
) -> torch.Tensor:
    """Sample a fixed birth budget by causal residual-gradient magnitude.

    For a photometric loss, sampling proportional to the magnitude of its color
    gradient minimizes the variance of an importance-sampled gradient estimate.
    Depth confidence discounts proposals whose 3D location is weakly supported.
    """

    if residuals.ndim != 1 or confidences.ndim != 1:
        raise ValueError("Residuals and confidences must be one-dimensional")
    if residuals.shape != confidences.shape:
        raise ValueError("Residuals and confidences must align")
    count = int(residuals.numel())
    budget = min(max(int(budget), 0), count)
    if budget == count:
        return torch.arange(count, device=residuals.device, dtype=torch.long)
    if budget == 0:
        return torch.empty(0, device=residuals.device, dtype=torch.long)

    weights = torch.nan_to_num(
        torch.clamp(residuals, min=0.0) * torch.clamp(confidences, 0.0, 1.0),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if not bool(torch.any(weights > 0.0)):
        weights = torch.ones_like(weights)
    generator = torch.Generator(device=residuals.device)
    generator.manual_seed(int(seed))
    selected = torch.multinomial(
        weights, budget, replacement=False, generator=generator
    )
    return torch.sort(selected).values


def projective_coverage_indices(
    uv: torch.Tensor,
    depths: torch.Tensor,
    confidences: torch.Tensor,
    budget: int,
    edges,
    fractions,
    *,
    image_width: int,
    cell_px: int,
    shuffle: bool = False,
    seed: int = 42,
) -> torch.Tensor:
    """Select depth-balanced, screen-covering candidates without random births."""

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("Candidate pixels must have shape [N, 2]")
    if depths.ndim != 1 or confidences.ndim != 1:
        raise ValueError("Candidate depths and confidences must be one-dimensional")
    if not (len(uv) == len(depths) == len(confidences)):
        raise ValueError("Projective coverage arrays must align")
    if image_width <= 0 or cell_px <= 0:
        raise ValueError("Projective image width and cell size must be positive")

    count = len(depths)
    budget = min(max(int(budget), 0), count)
    if budget == count:
        return torch.arange(count, device=depths.device, dtype=torch.long)
    if budget == 0:
        return torch.empty(0, device=depths.device, dtype=torch.long)

    edge0, edge1 = (float(value) for value in edges)
    quotas = [int(round(budget * float(value))) for value in fractions]
    quotas[-1] += budget - sum(quotas)
    band_ids = torch.bucketize(depths, depths.new_tensor([edge0, edge1]))
    cells_per_row = max(1, int(math.ceil(float(image_width) / float(cell_px))))
    cell_x = torch.floor(uv[:, 0] / float(cell_px)).long()
    cell_y = torch.floor(uv[:, 1] / float(cell_px)).long()
    cell_ids = cell_y * cells_per_row + cell_x
    selected = torch.zeros((count,), device=depths.device, dtype=torch.bool)
    generator = torch.Generator(device=depths.device)
    generator.manual_seed(int(seed))

    for band, quota in enumerate(quotas):
        rows = torch.nonzero(band_ids == band, as_tuple=False).flatten()
        if rows.numel() == 0 or quota <= 0:
            continue
        confidence = torch.nan_to_num(confidences[rows], nan=-torch.inf)
        confidence_order = _stable_argsort(confidence, descending=True)
        ranked = rows[confidence_order]
        cell_order = _stable_argsort(cell_ids[ranked])
        grouped = ranked[cell_order]
        grouped_cells = cell_ids[grouped]
        first = torch.ones_like(grouped_cells, dtype=torch.bool)
        first[1:] = grouped_cells[1:] != grouped_cells[:-1]
        representatives = grouped[first]
        if shuffle and representatives.numel() > 1:
            representatives = representatives[
                torch.randperm(
                    representatives.numel(),
                    generator=generator,
                    device=depths.device,
                )
            ]
        take = min(quota, representatives.numel())
        if take < representatives.numel():
            positions = torch.floor(
                (torch.arange(take, device=depths.device, dtype=torch.float64) + 0.5)
                * representatives.numel()
                / take
            ).long()
            chosen = representatives[positions]
        else:
            chosen = representatives
        selected[chosen] = True

        shortfall = quota - chosen.numel()
        if shortfall > 0:
            remaining = ranked[~selected[ranked]]
            selected[remaining[:shortfall]] = True

    shortfall = budget - int(selected.sum().item())
    if shortfall > 0:
        rows = torch.nonzero(~selected, as_tuple=False).flatten()
        confidence = torch.nan_to_num(confidences[rows], nan=-torch.inf)
        ranked = rows[_stable_argsort(confidence, descending=True)]
        selected[ranked[:shortfall]] = True

    return torch.nonzero(selected, as_tuple=False).flatten()


def evidence_balanced_indices(
    scores: torch.Tensor,
    budget: int,
    evidence_fraction: float,
    *,
    shuffle_evidence: bool = False,
    seed: int = 42,
) -> torch.Tensor:
    """Select top evidence rows plus a disjoint random coverage reserve."""

    if scores.ndim != 1:
        raise ValueError("Candidate scores must be one-dimensional")
    count = int(scores.numel())
    budget = min(max(int(budget), 0), count)
    if budget == count:
        return torch.arange(count, device=scores.device, dtype=torch.long)
    if budget == 0:
        return torch.empty(0, device=scores.device, dtype=torch.long)

    generator = torch.Generator(device=scores.device)
    generator.manual_seed(int(seed))
    ranked_scores = scores
    if shuffle_evidence and count > 1:
        permutation = torch.randperm(count, generator=generator, device=scores.device)
        ranked_scores = scores[permutation]

    evidence_count = min(budget, int(round(budget * float(evidence_fraction))))
    top = (
        torch.topk(ranked_scores, evidence_count, sorted=False).indices
        if evidence_count > 0
        else torch.empty(0, device=scores.device, dtype=torch.long)
    )
    selected = torch.zeros(count, device=scores.device, dtype=torch.bool)
    selected[top] = True
    remaining = torch.nonzero(~selected, as_tuple=False).flatten()
    coverage_count = budget - evidence_count
    if coverage_count > 0:
        order = torch.randperm(
            remaining.numel(), generator=generator, device=scores.device
        )
        coverage = remaining[order[:coverage_count]]
        top = torch.cat((top, coverage))
    return torch.sort(top).values


def depth_stratified_indices(
    depths: torch.Tensor,
    budget: int,
    edges,
    fractions,
    *,
    seed: int = 42,
    shuffle_depth_bands: bool = False,
) -> torch.Tensor:
    """Allocate survivor rows across front-view near, mid, and far depth bands."""

    if depths.ndim != 1:
        raise ValueError("Candidate depths must be one-dimensional")
    count = int(depths.numel())
    budget = min(max(int(budget), 0), count)
    if budget == count:
        return torch.arange(count, device=depths.device, dtype=torch.long)
    generator = torch.Generator(device=depths.device)
    generator.manual_seed(int(seed))
    band_depths = depths
    if shuffle_depth_bands and count > 1:
        shuffle_generator = torch.Generator(device=depths.device)
        shuffle_generator.manual_seed(int(seed) + 7919)
        permutation = torch.randperm(
            count, generator=shuffle_generator, device=depths.device
        )
        band_depths = depths[permutation]
    edge0, edge1 = (float(value) for value in edges)
    masks = (
        band_depths < edge0,
        (band_depths >= edge0) & (band_depths < edge1),
        band_depths >= edge1,
    )
    quotas = [int(round(budget * float(value))) for value in fractions]
    quotas[-1] += budget - sum(quotas)
    selected = torch.zeros(count, device=depths.device, dtype=torch.bool)
    for mask, quota in zip(masks, quotas):
        candidates = torch.nonzero(mask, as_tuple=False).flatten()
        if candidates.numel() == 0 or quota <= 0:
            continue
        order = torch.randperm(
            candidates.numel(), generator=generator, device=depths.device
        )
        selected[candidates[order[: min(quota, candidates.numel())]]] = True
    shortfall = budget - int(selected.sum().item())
    if shortfall > 0:
        remaining = torch.nonzero(~selected, as_tuple=False).flatten()
        order = torch.randperm(
            remaining.numel(), generator=generator, device=depths.device
        )
        selected[remaining[order[:shortfall]]] = True
    return torch.nonzero(selected, as_tuple=False).flatten()
