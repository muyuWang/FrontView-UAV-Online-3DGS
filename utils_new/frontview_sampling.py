"""Causal evidence-balanced candidate selection for forward-view mapping."""

from copy import deepcopy
import math

import torch


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
        confidence_order = torch.argsort(
            confidence, descending=True, stable=True
        )
        ranked = rows[confidence_order]
        cell_order = torch.argsort(cell_ids[ranked], stable=True)
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
        ranked = rows[
            torch.argsort(confidence, descending=True, stable=True)
        ]
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
