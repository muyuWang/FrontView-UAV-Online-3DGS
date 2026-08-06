"""Track-anchored, layered projective birth for forward-view UAV mapping."""

from copy import deepcopy
import math

import numpy as np
import torch


DEFAULT_FRONT_VIEW_BIRTH_CONFIG = {
    "enabled": False,
    "pool_multiplier": 2,
    "projective_cell_px": 12,
    "depth_bin_ratio": 1.10,
    "max_per_cell": 1,
    "overflow_max_per_cell": 0,
    "map_competition": True,
    "multi_layer_map_competition": False,
    "sparse_anchor_competition": False,
    "near_hash_competition": False,
    "near_hash_depth_m": 50.0,
    "atlas_min_opacity": 0.05,
    "explained_opacity": 0.65,
    "depth_consistency_ratio": 0.06,
    "residual_override": 0.16,
    "coverage_weight": 0.35,
    "confidence_weight": 0.20,
    "depth_novelty_weight": 0.20,
    "adaptive_layer_balance": False,
    "preserve_layer_budget": False,
    "shuffle_layer_assignments": False,
    "layer_quantiles": [0.55, 0.88],
    "layer_fractions": [0.33, 0.45, 0.22],
    "priority_fraction": 1.0,
    "strict_responsibility_budget": False,
    "track_refinement_ratio": 0.0,
    "responsibility_opacity": False,
    "depthcov_opacity_min": 0.30,
    "depthcov_opacity_max": 0.50,
    "opacity_residual_saturation": 0.25,
    "temporal_map_competition": False,
    "temporal_reference_frames": 2,
    "temporal_reject_views": 1,
    "temporal_opacity": 0.70,
    "temporal_free_space_ratio": 0.08,
    "temporal_reject_duplicates": False,
    "selection_seed": 42,
    "far_scale_control": False,
    "far_depth_quantile": 0.75,
    "far_max_scale_expansion": 2.0,
}


def validate_front_view_birth_config(config=None):
    merged = deepcopy(DEFAULT_FRONT_VIEW_BIRTH_CONFIG)
    if config is not None:
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError(
                "Unknown FrontViewBirth options: {}".format(sorted(unknown))
            )
        merged.update(config)
    for key in (
        "enabled",
        "map_competition",
        "multi_layer_map_competition",
        "sparse_anchor_competition",
        "near_hash_competition",
        "adaptive_layer_balance",
        "preserve_layer_budget",
        "shuffle_layer_assignments",
        "strict_responsibility_budget",
        "responsibility_opacity",
        "temporal_map_competition",
        "temporal_reject_duplicates",
        "far_scale_control",
    ):
        if not isinstance(merged[key], bool):
            raise TypeError("FrontViewBirth.{} must be boolean".format(key))
    if not isinstance(merged["pool_multiplier"], int) or merged["pool_multiplier"] < 1:
        raise ValueError("FrontViewBirth.pool_multiplier must be a positive integer")
    if int(merged["projective_cell_px"]) <= 0:
        raise ValueError("FrontViewBirth.projective_cell_px must be positive")
    if float(merged["depth_bin_ratio"]) <= 1.0:
        raise ValueError("FrontViewBirth.depth_bin_ratio must be greater than one")
    if not isinstance(merged["max_per_cell"], int) or merged["max_per_cell"] <= 0:
        raise ValueError("FrontViewBirth.max_per_cell must be a positive integer")
    overflow_maximum = merged["overflow_max_per_cell"]
    if not isinstance(overflow_maximum, int) or overflow_maximum < 0:
        raise ValueError(
            "FrontViewBirth.overflow_max_per_cell must be a nonnegative integer"
        )
    if overflow_maximum and overflow_maximum < merged["max_per_cell"]:
        raise ValueError(
            "FrontViewBirth.overflow_max_per_cell must be zero or at least max_per_cell"
        )
    for key in (
        "explained_opacity",
        "depth_consistency_ratio",
        "residual_override",
        "coverage_weight",
        "confidence_weight",
        "depth_novelty_weight",
    ):
        if float(merged[key]) < 0.0:
            raise ValueError("FrontViewBirth.{} must be nonnegative".format(key))
    if not 0.0 <= float(merged["priority_fraction"]) <= 1.0:
        raise ValueError("FrontViewBirth.priority_fraction must be in [0, 1]")
    refinement_ratio = float(merged["track_refinement_ratio"])
    if refinement_ratio != 0.0 and refinement_ratio <= 1.0:
        raise ValueError(
            "FrontViewBirth.track_refinement_ratio must be zero or greater than one"
        )
    opacity_min = float(merged["depthcov_opacity_min"])
    opacity_max = float(merged["depthcov_opacity_max"])
    if not 0.0 < opacity_min <= opacity_max < 1.0:
        raise ValueError(
            "FrontViewBirth DepthCov opacity bounds must satisfy 0 < min <= max < 1"
        )
    if float(merged["opacity_residual_saturation"]) <= 0.0:
        raise ValueError(
            "FrontViewBirth.opacity_residual_saturation must be positive"
        )
    reference_frames = int(merged["temporal_reference_frames"])
    reject_views = int(merged["temporal_reject_views"])
    if reference_frames < 1:
        raise ValueError("FrontViewBirth.temporal_reference_frames must be positive")
    if reject_views < 1 or reject_views > reference_frames:
        raise ValueError(
            "FrontViewBirth.temporal_reject_views must be within reference_frames"
        )
    if not 0.0 <= float(merged["temporal_opacity"]) <= 1.0:
        raise ValueError("FrontViewBirth.temporal_opacity must be in [0, 1]")
    if float(merged["temporal_free_space_ratio"]) <= 0.0:
        raise ValueError(
            "FrontViewBirth.temporal_free_space_ratio must be positive"
        )
    if not isinstance(merged["selection_seed"], int):
        raise TypeError("FrontViewBirth.selection_seed must be an integer")
    if not 0.0 <= float(merged["explained_opacity"]) <= 1.0:
        raise ValueError("FrontViewBirth.explained_opacity must be in [0, 1]")
    if not 0.0 <= float(merged["atlas_min_opacity"]) <= 1.0:
        raise ValueError("FrontViewBirth.atlas_min_opacity must be in [0, 1]")
    if float(merged["near_hash_depth_m"]) <= 0.0:
        raise ValueError("FrontViewBirth.near_hash_depth_m must be positive")
    if not 0.0 < float(merged["far_depth_quantile"]) < 1.0:
        raise ValueError("FrontViewBirth.far_depth_quantile must be in (0, 1)")
    if float(merged["far_max_scale_expansion"]) <= 0.0:
        raise ValueError("FrontViewBirth.far_max_scale_expansion must be positive")
    quantiles = [float(value) for value in merged["layer_quantiles"]]
    fractions = [float(value) for value in merged["layer_fractions"]]
    if len(quantiles) != 2 or not 0.0 < quantiles[0] < quantiles[1] < 1.0:
        raise ValueError(
            "FrontViewBirth.layer_quantiles must be two increasing values in (0, 1)"
        )
    if len(fractions) != 3 or any(value < 0.0 for value in fractions):
        raise ValueError(
            "FrontViewBirth.layer_fractions must have three nonnegative values"
        )
    if abs(sum(fractions) - 1.0) > 1.0e-6:
        raise ValueError("FrontViewBirth.layer_fractions must sum to one")
    return merged


def responsibility_initial_opacities(
    source_kinds,
    residual_scores,
    coverage_scores,
    depth_confidences,
    base_opacity,
    config,
):
    """Map untracked proposal responsibility to birth opacity."""

    source_kinds = np.asarray(source_kinds, dtype="U32").reshape(-1)
    count = len(source_kinds)

    def vector(values, name):
        result = np.asarray(values, dtype=np.float32).reshape(-1)
        if result.shape != (count,):
            raise ValueError("{} must match source_kinds".format(name))
        return result

    residual = vector(residual_scores, "residual_scores")
    coverage = vector(coverage_scores, "coverage_scores")
    confidence = vector(depth_confidences, "depth_confidences")
    opacities = np.full((count,), float(base_opacity), dtype=np.float32)
    depthcov = source_kinds == "depthcov"
    if not bool(config["responsibility_opacity"]) or not np.any(depthcov):
        return opacities

    residual_quality = np.clip(
        residual / float(config["opacity_residual_saturation"]), 0.0, 1.0
    )
    quality = np.clip(
        0.50 * residual_quality
        + 0.30 * np.clip(coverage, 0.0, 1.0)
        + 0.20 * np.clip(confidence, 0.0, 1.0),
        0.0,
        1.0,
    )
    lower = float(config["depthcov_opacity_min"])
    upper = float(config["depthcov_opacity_max"])
    opacities[depthcov] = lower + (upper - lower) * quality[depthcov]
    return opacities


def temporal_responsibility_rejections(
    candidate_depths_by_view,
    map_depths_by_view,
    map_opacities_by_view,
    valid_by_view,
    config,
):
    """Vote out births that contradict rendered free space in prior views."""

    candidate_depths = np.asarray(candidate_depths_by_view, dtype=np.float32)
    map_depths = np.asarray(map_depths_by_view, dtype=np.float32)
    map_opacities = np.asarray(map_opacities_by_view, dtype=np.float32)
    valid = np.asarray(valid_by_view, dtype=np.bool_)
    if not (
        candidate_depths.shape
        == map_depths.shape
        == map_opacities.shape
        == valid.shape
    ):
        raise ValueError("Temporal responsibility evidence shapes must match")
    if candidate_depths.ndim != 2:
        raise ValueError("Temporal responsibility evidence must be views by rows")

    observed = (
        valid
        & np.isfinite(candidate_depths)
        & (candidate_depths > 0.0)
        & np.isfinite(map_depths)
        & (map_depths > 0.0)
        & (map_opacities >= float(config["temporal_opacity"]))
    )
    ratio = float(config["temporal_free_space_ratio"])
    free_space = observed & (candidate_depths < map_depths * (1.0 - ratio))
    duplicates = np.zeros_like(free_space)
    if bool(config["temporal_reject_duplicates"]):
        log_gap = np.full(candidate_depths.shape, np.inf, dtype=np.float32)
        log_gap[observed] = np.abs(
            np.log(candidate_depths[observed]) - np.log(map_depths[observed])
        )
        duplicates = observed & (log_gap <= math.log1p(ratio))
    votes = free_space | duplicates
    reject = np.sum(votes, axis=0) >= int(config["temporal_reject_views"])
    return reject, {
        "tested_rows": int(np.sum(np.any(observed, axis=0))),
        "free_space_rows": int(np.sum(np.any(free_space, axis=0))),
        "duplicate_rows": int(np.sum(np.any(duplicates, axis=0))),
        "rejected_rows": int(np.sum(reject)),
    }


class TrackResponsibilityLedger:
    """Bind tracks once, with optional finer-depth octave refinements."""

    def __init__(self, refinement_ratio=0.0):
        self._committed = set()
        self._ever_committed = set()
        self._finest_depth = {}
        self.refinement_ratio = float(refinement_ratio)
        self.proposal_rejections = 0
        self.commit_rejections = 0
        self.refinement_births = 0
        self.release_events = 0
        self.rebirths = 0

    def new_indices(self, track_ids, depths=None, *, at_commit=False):
        ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        if depths is None:
            depth_values = np.full(ids.shape, np.nan, dtype=np.float32)
        else:
            depth_values = np.asarray(depths, dtype=np.float32).reshape(-1)
            if depth_values.shape != ids.shape:
                raise ValueError("Track depths must match track IDs")
        keep = []
        pending = set()
        rejected = 0
        for index, (track_id, depth) in enumerate(
            zip(ids.tolist(), depth_values.tolist())
        ):
            if track_id < 0:
                keep.append(index)
                continue
            committed_depth = self._finest_depth.get(track_id)
            refines = (
                committed_depth is not None
                and self.refinement_ratio > 1.0
                and np.isfinite(depth)
                and depth > 0.0
                and depth < committed_depth / self.refinement_ratio
            )
            if (track_id in self._committed and not refines) or track_id in pending:
                rejected += 1
                continue
            pending.add(track_id)
            keep.append(index)
        if at_commit:
            self.commit_rejections += rejected
        else:
            self.proposal_rejections += rejected
        return np.asarray(keep, dtype=np.int64)

    def mark_committed(self, track_ids, depths=None):
        ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        if depths is None:
            depth_values = np.full(ids.shape, np.nan, dtype=np.float32)
        else:
            depth_values = np.asarray(depths, dtype=np.float32).reshape(-1)
            if depth_values.shape != ids.shape:
                raise ValueError("Track depths must match track IDs")
        for track_id, depth in zip(ids.tolist(), depth_values.tolist()):
            if track_id < 0:
                continue
            if track_id in self._committed:
                self.refinement_births += 1
            elif track_id in self._ever_committed:
                self.rebirths += 1
            self._committed.add(int(track_id))
            self._ever_committed.add(int(track_id))
            if np.isfinite(depth) and depth > 0.0:
                previous = self._finest_depth.get(track_id, float("inf"))
                self._finest_depth[track_id] = min(float(depth), previous)

    def release(self, track_ids):
        ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        released = 0
        for track_id in np.unique(ids):
            if track_id < 0 or int(track_id) not in self._committed:
                continue
            self._committed.remove(int(track_id))
            self._finest_depth.pop(int(track_id), None)
            released += 1
        self.release_events += released
        return released

    def summary(self):
        return {
            "committed_tracks": len(self._ever_committed),
            "active_tracks": len(self._committed),
            "track_refinement_births": int(self.refinement_births),
            "track_release_events": int(self.release_events),
            "track_rebirths": int(self.rebirths),
            "proposal_track_rejections": int(self.proposal_rejections),
            "commit_track_rejections": int(self.commit_rejections),
        }


def _as_vector(value, count, device, default):
    if value is None:
        return torch.full((count,), float(default), device=device, dtype=torch.float32)
    tensor = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(-1)
    if tensor.shape != (count,):
        raise ValueError("Projective birth evidence has the wrong shape")
    return tensor


def multi_layer_projective_occupancy(
    candidate_uv, candidate_depths, map_uv, map_depths, config
):
    """Test candidate ray-depth cells against every projected map layer."""

    candidate_depths = torch.as_tensor(candidate_depths, dtype=torch.float32)
    device = candidate_depths.device
    candidate_depths = candidate_depths.reshape(-1)
    candidate_uv = torch.as_tensor(
        candidate_uv, device=device, dtype=torch.float32
    )
    map_depths = torch.as_tensor(map_depths, device=device, dtype=torch.float32).reshape(
        -1
    )
    map_uv = torch.as_tensor(map_uv, device=device, dtype=torch.float32)
    if candidate_uv.shape != (len(candidate_depths), 2):
        raise ValueError("Candidate atlas UV coordinates must have shape Nx2")
    if map_uv.shape != (len(map_depths), 2):
        raise ValueError("Map atlas UV coordinates must have shape Nx2")
    if len(candidate_depths) == 0 or len(map_depths) == 0:
        return torch.zeros(
            len(candidate_depths), device=device, dtype=torch.bool
        )

    uv = torch.cat((candidate_uv, map_uv), dim=0)
    depths = torch.cat((candidate_depths, map_depths), dim=0)
    cell_px = float(config["projective_cell_px"])
    xy = torch.floor(uv / cell_px).to(dtype=torch.int64)
    z = torch.floor(
        torch.log(torch.clamp(depths, min=1.0e-8))
        / math.log(float(config["depth_bin_ratio"]))
    ).to(dtype=torch.int64)
    x = xy[:, 0] - torch.min(xy[:, 0])
    y = xy[:, 1] - torch.min(xy[:, 1])
    z = z - torch.min(z)
    x_size = int(torch.max(x).item()) + 1
    y_size = int(torch.max(y).item()) + 1
    keys = (z * y_size + y) * x_size + x
    candidate_count = len(candidate_depths)
    return torch.isin(keys[:candidate_count], torch.unique(keys[candidate_count:]))


def layered_projective_birth_indices(
    uv,
    depths,
    depth_confidences,
    residual_scores,
    map_depths,
    map_opacities,
    budget,
    config,
    *,
    seed=None,
    map_occupied=None,
    anchor_occupied=None,
):
    """Select non-redundant candidates in screen-ray and log-depth cells."""

    depths = torch.as_tensor(depths)
    device = depths.device
    depths = depths.to(dtype=torch.float32).reshape(-1)
    count = int(depths.numel())
    uv = torch.as_tensor(uv, device=device, dtype=torch.float32)
    if uv.shape != (count, 2):
        raise ValueError("Projective birth UV coordinates must have shape Nx2")
    budget = min(max(int(budget), 0), count)
    if count == 0 or budget == 0:
        return torch.empty(0, device=device, dtype=torch.long), {
            "pool": count,
            "map_rejected": 0,
            "atlas_rejected": 0,
            "anchor_rejected": 0,
            "cell_rejected": 0,
            "selected": 0,
            "priority_selected": 0,
            "coverage_selected": 0,
            "fallback_selected": 0,
            "layer_edges": [],
            "pool_layer_counts": [0, 0, 0],
            "selected_layer_counts": [0, 0, 0],
        }

    confidence = _as_vector(depth_confidences, count, device, 1.0)
    residual = _as_vector(residual_scores, count, device, 1.0)
    map_depth = _as_vector(map_depths, count, device, -1.0)
    map_opacity = _as_vector(map_opacities, count, device, 0.0)
    atlas_occupied = torch.zeros(count, device=device, dtype=torch.bool)
    if map_occupied is not None:
        atlas_occupied = torch.as_tensor(
            map_occupied, device=device, dtype=torch.bool
        ).reshape(-1)
        if atlas_occupied.shape != (count,):
            raise ValueError("Projective atlas occupancy must match candidates")
    anchor_conflict = torch.zeros(count, device=device, dtype=torch.bool)
    if anchor_occupied is not None:
        anchor_conflict = torch.as_tensor(
            anchor_occupied, device=device, dtype=torch.bool
        ).reshape(-1)
        if anchor_conflict.shape != (count,):
            raise ValueError("Sparse-anchor occupancy must match candidates")
    valid_map = torch.isfinite(map_depth) & (map_depth > 0.0)
    log_gap = torch.full_like(depths, float("inf"))
    log_gap[valid_map] = torch.abs(
        torch.log(torch.clamp(depths[valid_map], min=1.0e-8))
        - torch.log(torch.clamp(map_depth[valid_map], min=1.0e-8))
    )
    log_tolerance = math.log1p(float(config["depth_consistency_ratio"]))
    explained = (
        bool(config["map_competition"])
        & valid_map
        & (map_opacity >= float(config["explained_opacity"]))
        & (log_gap <= log_tolerance)
    )
    residual_override = residual >= float(config["residual_override"])
    atlas_occupied &= bool(config["multi_layer_map_competition"])
    anchor_conflict &= bool(config["sparse_anchor_competition"])
    admissible = ~anchor_conflict & (
        (~explained & ~atlas_occupied) | residual_override
    )

    coverage = torch.clamp(1.0 - map_opacity, 0.0, 1.0)
    depth_novelty = torch.where(
        valid_map,
        torch.clamp(log_gap / max(log_tolerance, 1.0e-8), 0.0, 1.0),
        torch.ones_like(depths),
    )
    priority = (
        residual
        + float(config["coverage_weight"]) * coverage
        + float(config["confidence_weight"]) * torch.clamp(confidence, 0.0, 1.0)
        + float(config["depth_novelty_weight"]) * depth_novelty
    )
    candidates = torch.nonzero(admissible, as_tuple=False).flatten()
    if candidates.numel() == 0:
        return candidates, {
            "pool": count,
            "map_rejected": count,
            "atlas_rejected": int(atlas_occupied.sum().item()),
            "anchor_rejected": int(anchor_conflict.sum().item()),
            "cell_rejected": 0,
            "selected": 0,
            "priority_selected": 0,
            "coverage_selected": 0,
            "fallback_selected": 0,
            "layer_edges": [],
            "pool_layer_counts": [0, 0, 0],
            "selected_layer_counts": [0, 0, 0],
        }

    cell_px = float(config["projective_cell_px"])
    xy = torch.floor(uv / cell_px).to(dtype=torch.int64)
    depth_bin = torch.floor(
        torch.log(torch.clamp(depths, min=1.0e-8))
        / math.log(float(config["depth_bin_ratio"]))
    ).to(dtype=torch.int64)
    x = xy[:, 0] - torch.min(xy[:, 0])
    y = xy[:, 1] - torch.min(xy[:, 1])
    z = depth_bin - torch.min(depth_bin)
    x_size = int(torch.max(x).item()) + 1
    y_size = int(torch.max(y).item()) + 1
    keys = (z * y_size + y) * x_size + x

    ranked = candidates[
        torch.argsort(priority[candidates], descending=True, stable=True)
    ]
    ranked_cpu = ranked.detach().cpu().numpy()
    key_cpu = keys.detach().cpu().numpy()
    layer_cpu = None
    allocation_layer_cpu = None
    layer_edges = []
    pool_layer_counts = [0, 0, 0]
    if bool(config["adaptive_layer_balance"]):
        candidate_log_depth = torch.log(torch.clamp(depths[candidates], min=1.0e-8))
        boundaries = torch.quantile(
            candidate_log_depth,
            torch.as_tensor(
                config["layer_quantiles"], device=device, dtype=torch.float32
            ),
        )
        layer_ids = torch.bucketize(
            torch.log(torch.clamp(depths, min=1.0e-8)), boundaries
        )
        layer_cpu = layer_ids.detach().cpu().numpy()
        allocation_layer_cpu = layer_cpu.copy()
        if bool(config["shuffle_layer_assignments"]):
            candidate_cpu = candidates.detach().cpu().numpy()
            shuffled = allocation_layer_cpu[candidate_cpu].copy()
            allocation_rng = np.random.default_rng(
                (int(config["selection_seed"]) if seed is None else int(seed))
                + 1000003
            )
            allocation_rng.shuffle(shuffled)
            allocation_layer_cpu[candidate_cpu] = shuffled
        layer_edges = torch.exp(boundaries).detach().cpu().tolist()
        candidate_layer_ids = layer_ids[candidates]
        pool_layer_counts = [
            int((candidate_layer_ids == layer).sum().item()) for layer in range(3)
        ]

    per_cell = {}
    selected = []
    maximum = int(config["max_per_cell"])
    priority_selected = 0
    coverage_selected = 0

    def add_rows(rows, limit):
        if limit <= 0 or len(selected) >= budget:
            return 0
        added = 0
        selected_set = set(selected)
        for index in rows:
            if index in selected_set:
                continue
            key = int(key_cpu[index])
            used = per_cell.get(key, 0)
            if used >= maximum:
                continue
            per_cell[key] = used + 1
            selected.append(index)
            selected_set.add(index)
            added += 1
            if added >= limit or len(selected) >= budget:
                break
        return added

    def add_overflow_rows(rows, limit):
        if limit <= 0 or len(selected) >= budget:
            return 0
        added = 0
        selected_set = set(selected)
        overflow_maximum = int(config["overflow_max_per_cell"])
        for index in rows:
            if index in selected_set:
                continue
            key = int(key_cpu[index])
            used = per_cell.get(key, 0)
            if overflow_maximum and used >= overflow_maximum:
                continue
            per_cell[key] = used + 1
            selected.append(index)
            selected_set.add(index)
            added += 1
            if added >= limit or len(selected) >= budget:
                break
        return added

    ranked_list = ranked_cpu.tolist()
    if layer_cpu is None:
        priority_target = int(round(budget * float(config["priority_fraction"])))
        priority_selected += add_rows(ranked_list, priority_target)
        if len(selected) < budget:
            rng = np.random.default_rng(
                int(config["selection_seed"]) if seed is None else int(seed)
            )
            coverage_selected += add_rows(
                rng.permutation(ranked_list).tolist(), budget - len(selected)
            )
    else:
        quotas = [
            int(round(budget * float(value))) for value in config["layer_fractions"]
        ]
        quotas[-1] += budget - sum(quotas)
        rng = np.random.default_rng(
            int(config["selection_seed"]) if seed is None else int(seed)
        )
        for layer, quota in enumerate(quotas):
            layer_rows = [
                index
                for index in ranked_list
                if int(allocation_layer_cpu[index]) == layer
            ]
            priority_target = int(
                round(quota * float(config["priority_fraction"]))
            )
            priority_selected += add_rows(layer_rows, priority_target)
            already_in_layer = sum(
                int(allocation_layer_cpu[index]) == layer for index in selected
            )
            remaining = max(0, quota - already_in_layer)
            if remaining:
                coverage_selected += add_rows(
                    rng.permutation(layer_rows).tolist(), remaining
                )
        if not bool(config["preserve_layer_budget"]):
            priority_selected += add_rows(ranked_list, budget - len(selected))

    cell_selected = len(selected)
    if len(selected) < budget and not bool(config["strict_responsibility_budget"]):
        if layer_cpu is not None and bool(config["preserve_layer_budget"]):
            for layer, quota in enumerate(quotas):
                already_in_layer = sum(
                    int(allocation_layer_cpu[index]) == layer for index in selected
                )
                layer_rows = [
                    index
                    for index in ranked_list
                    if int(allocation_layer_cpu[index]) == layer
                ]
                add_overflow_rows(layer_rows, max(0, quota - already_in_layer))
        add_overflow_rows(ranked_list, budget - len(selected))
    selected = torch.as_tensor(sorted(selected), device=device, dtype=torch.long)
    selected_layer_counts = [0, 0, 0]
    if layer_cpu is not None:
        selected_layer_counts = [
            sum(int(layer_cpu[index]) == layer for index in selected.tolist())
            for layer in range(3)
        ]
    return selected, {
        "pool": count,
        "map_rejected": int((~admissible).sum().item()),
        "atlas_rejected": int((atlas_occupied & ~residual_override).sum().item()),
        "anchor_rejected": int(anchor_conflict.sum().item()),
        "cell_rejected": max(0, int(candidates.numel()) - cell_selected),
        "selected": int(selected.numel()),
        "priority_selected": int(priority_selected),
        "coverage_selected": int(coverage_selected),
        "fallback_selected": int(selected.numel()) - cell_selected,
        "layer_edges": layer_edges,
        "pool_layer_counts": pool_layer_counts,
        "selected_layer_counts": selected_layer_counts,
    }


def layered_scale_expansion_limits(depths, source_kinds, config):
    """Bound only the far quantile of DepthCov footprints within each birth batch."""

    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    kinds = np.asarray(source_kinds, dtype="U32").reshape(-1)
    if depths.shape != kinds.shape:
        raise ValueError("Depths and source kinds must align")
    limits = np.full(depths.shape, np.inf, dtype=np.float32)
    if not bool(config["far_scale_control"]):
        return limits
    depthcov = (kinds == "depthcov") & np.isfinite(depths) & (depths > 0.0)
    if not np.any(depthcov):
        return limits
    threshold = float(np.quantile(depths[depthcov], float(config["far_depth_quantile"])))
    limits[depthcov & (depths >= threshold)] = float(
        config["far_max_scale_expansion"]
    )
    return limits
