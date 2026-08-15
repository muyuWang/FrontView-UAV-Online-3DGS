"""Metric/projective responsibility for forward-view UAV Gaussian births."""

from copy import deepcopy
from collections import OrderedDict
import math

import numpy as np


DEFAULT_FRONT_VIEW_FAR_FIELD_CONFIG = {
    "enabled": False,
    "routing_mode": "fixed_depth",
    "depth_m": 50.0,
    "projective_cell_px": 12,
    "depth_bin_ratio": 1.10,
    "projective_nms_mode": "fixed_grid",
    "projective_covariance_mode": "isotropic",
    "fallback_support_mode": "legacy",
    "map_redundancy_gate": False,
    "map_redundancy_evidence": "geometry",
    "posterior_budget_refill": False,
    "shuffle_refill_evidence": False,
    "ray_atlas_enabled": False,
    "ray_atlas_shuffle_evidence": False,
    "ray_atlas_coordinate_mode": "camera_ray",
    "ray_atlas_competition_mode": "hard_cell",
    "responsibility_basis": "source",
    "shuffle_responsibility": False,
    "responsibility_shuffle_mode": "global",
    "footprint_trust_mode": "disabled",
    "footprint_trust_scope": "all_depthcov",
    "footprint_trust_dynamic_update": False,
    "footprint_trust_dynamic_shuffle": False,
    "footprint_trust_dynamic_shuffle_mode": "evidence",
    "unobservable_birth_policy": "projective",
    "shuffle_seed": 42,
}


def adaptive_log_depth_responsibility(depths, eligible, regime_count=3):
    """Return the farthest online log-depth Voronoi regime.

    The partition is fitted only to candidates available in the current frame.
    Log depth makes the assignment invariant to a global metric rescaling, and
    Lloyd quantization replaces a scene-specific metric near/far threshold with
    the minimum-distortion partition of the observed proposal distribution.
    """

    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    eligible = np.asarray(eligible, dtype=np.bool_).reshape(-1)
    if depths.shape != eligible.shape:
        raise ValueError("Adaptive responsibility arrays must align")
    if int(regime_count) < 2:
        raise ValueError("Adaptive responsibility requires at least two regimes")
    valid = eligible & np.isfinite(depths) & (depths > 0.0)
    rows = np.flatnonzero(valid)
    result = np.zeros(depths.shape, dtype=np.bool_)
    metadata = {
        "boundaries_m": [],
        "centers_log_depth": [],
        "regime_counts": [0] * int(regime_count),
        "far_rows": 0,
        "iterations": 0,
        "objective": 0.0,
    }
    if not len(rows):
        return result, metadata

    values = np.log(depths[rows].astype(np.float64))
    quantiles = (np.arange(int(regime_count), dtype=np.float64) + 0.5) / float(
        regime_count
    )
    centers = np.quantile(values, quantiles)
    previous = None
    iterations = 0
    for iterations in range(1, 33):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        updated = centers.copy()
        for regime in range(int(regime_count)):
            members = values[labels == regime]
            if len(members):
                updated[regime] = float(np.mean(members))
        updated.sort()
        if previous is not None and np.array_equal(labels, previous):
            centers = updated
            break
        previous = labels
        centers = updated

    labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
    occupied = np.flatnonzero(np.bincount(labels, minlength=regime_count) > 0)
    far_label = int(occupied[np.argmax(centers[occupied])])
    far_rows = rows[labels == far_label]
    result[far_rows] = True
    objective = float(np.mean((values - centers[labels]) ** 2))
    metadata.update(
        boundaries_m=[
            float(np.exp(0.5 * (centers[index] + centers[index + 1])))
            for index in range(int(regime_count) - 1)
        ],
        centers_log_depth=[float(value) for value in centers.tolist()],
        regime_counts=[
            int(value)
            for value in np.bincount(labels, minlength=regime_count).tolist()
        ],
        far_rows=int(len(far_rows)),
        iterations=int(iterations),
        objective=objective,
    )
    return result, metadata


def _adaptive_log_depth_regime_labels(depths, eligible, regime_count=3):
    """Fit causal Lloyd regimes and return one label per eligible row."""

    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    eligible = np.asarray(eligible, dtype=np.bool_).reshape(-1)
    if depths.shape != eligible.shape:
        raise ValueError("Adaptive responsibility arrays must align")
    if int(regime_count) < 2:
        raise ValueError("Adaptive responsibility requires at least two regimes")
    valid = eligible & np.isfinite(depths) & (depths > 0.0)
    rows = np.flatnonzero(valid)
    labels_full = np.full(depths.shape, -1, dtype=np.int16)
    if not len(rows):
        return labels_full

    values = np.log(depths[rows].astype(np.float64))
    quantiles = (np.arange(int(regime_count), dtype=np.float64) + 0.5) / float(
        regime_count
    )
    centers = np.quantile(values, quantiles)
    previous = None
    for _ in range(32):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        updated = centers.copy()
        for regime in range(int(regime_count)):
            members = values[labels == regime]
            if len(members):
                updated[regime] = float(np.mean(members))
        updated.sort()
        if previous is not None and np.array_equal(labels, previous):
            centers = updated
            break
        previous = labels
        centers = updated
    labels_full[rows] = np.argmin(
        np.abs(values[:, None] - centers[None, :]), axis=1
    ).astype(np.int16, copy=False)
    return labels_full


def matched_responsibility_shuffle(
    responsibility,
    eligible,
    depths,
    seed,
    *,
    mode="log_depth_regimes",
    regime_count=3,
):
    """Relocate responsibility with exactly matched causal depth-domain counts."""

    responsibility = np.asarray(responsibility, dtype=np.bool_).reshape(-1)
    eligible = np.asarray(eligible, dtype=np.bool_).reshape(-1)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    if not (responsibility.shape == eligible.shape == depths.shape):
        raise ValueError("Responsibility shuffle arrays must align")
    if np.any(responsibility & ~eligible):
        raise ValueError("Responsibility rows must be eligible for reassignment")
    if np.any(~np.isfinite(depths[eligible]) | (depths[eligible] <= 0.0)):
        raise ValueError("Eligible responsibility depths must be finite and positive")
    if mode not in ("global", "log_depth_regimes"):
        raise ValueError("Responsibility shuffle mode is invalid")

    result = np.zeros_like(responsibility)
    rng = np.random.default_rng(int(seed))
    if mode == "global":
        groups = [np.flatnonzero(eligible)]
    else:
        labels = _adaptive_log_depth_regime_labels(
            depths, eligible, regime_count=int(regime_count)
        )
        groups = [
            np.flatnonzero(eligible & (labels == regime))
            for regime in range(int(regime_count))
        ]

    for rows in groups:
        requested = int(np.count_nonzero(responsibility[rows]))
        if requested:
            result[rng.choice(rows, size=requested, replace=False)] = True
    return result


def observability_footprint_trust_limits(
    parallax_pixels,
    projected_radii,
    log_depth_stds,
    depths,
    eligible,
    image_size,
    birth_budget,
    pool_multiplier,
    *,
    mode="information",
    projective_owner=None,
    responsibility_radius_factors=None,
    seed=42,
):
    """Bound footprint growth by causal information and birth density.

    The dimensionless certificate combines parallax resolution and depth
    precision. Its information odds allocate log-radius headroom between the
    newborn support and half of the budget-derived projective cell spacing.
    """

    parallax = np.asarray(parallax_pixels, dtype=np.float32).reshape(-1)
    radii = np.asarray(projected_radii, dtype=np.float32).reshape(-1)
    log_stds = np.asarray(log_depth_stds, dtype=np.float32).reshape(-1)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    eligible = np.asarray(eligible, dtype=np.bool_).reshape(-1)
    if not (
        parallax.shape
        == radii.shape
        == log_stds.shape
        == depths.shape
        == eligible.shape
    ):
        raise ValueError("Footprint trust arrays must align")
    if mode not in (
        "information",
        "information_shuffled",
        "certificate_odds",
        "certificate_odds_shuffled",
        "certificate_equal_area",
        "certificate_equal_area_shuffled",
        "certificate_owner_area",
        "certificate_owner_area_shuffled",
        "certificate_residual_rd",
        "certificate_residual_rd_shuffled",
        "certificate_residual_rd_detail",
        "certificate_residual_rd_detail_shuffled",
        "certificate_residual_rd_visible_detail",
        "certificate_residual_rd_visible_detail_shuffled",
        "certificate_residual_rd_visible_detail_bounded_area",
        "certificate_residual_rd_visible_detail_bounded_area_shuffled",
    ):
        raise ValueError("Footprint trust mode is invalid")
    if (
        np.any(~np.isfinite(parallax))
        or np.any(~np.isfinite(radii))
        or np.any(~np.isfinite(log_stds))
        or np.any(~np.isfinite(depths))
        or np.any(parallax < 0.0)
        or np.any(radii <= 0.0)
        or np.any(log_stds < 0.0)
        or np.any(depths <= 0.0)
    ):
        raise ValueError("Footprint trust values must be finite and valid")

    limits = np.full(depths.shape, np.inf, dtype=np.float32)
    information = np.zeros(depths.shape, dtype=np.float32)
    rows = np.flatnonzero(eligible)
    if not len(rows):
        return limits, information, {
            "rows": 0,
            "cell_px": None,
            "mean_information": None,
            "mean_limit": None,
            "projective_owner_rows": 0,
            "shuffled": mode.endswith("_shuffled"),
        }

    p = parallax[rows].astype(np.float64)
    radius = radii[rows].astype(np.float64)
    sigma = log_stds[rows].astype(np.float64)
    parallax_margin = p / radius
    precision_margin = np.divide(
        radius,
        p * sigma,
        out=np.full_like(p, np.inf),
        where=(p * sigma) > 0.0,
    )
    bottleneck = np.maximum(0.0, np.minimum(parallax_margin, precision_margin))
    if mode.startswith("certificate_"):
        row_information = np.clip(bottleneck, 0.0, 1.0)
    else:
        row_information = bottleneck / (1.0 + bottleneck)

    owner_rows = None
    if mode.startswith("certificate_owner_area"):
        if projective_owner is None:
            raise ValueError("Owner-area footprint trust requires projective owners")
        projective_owner = np.asarray(projective_owner, dtype=np.bool_).reshape(-1)
        if projective_owner.shape != depths.shape:
            raise ValueError("Projective footprint owners must align")
        if np.any(projective_owner & ~eligible):
            raise ValueError("Projective footprint owners must be eligible")
        owner_rows = projective_owner[rows].copy()
    elif projective_owner is not None:
        raise ValueError("Projective owners require owner-area footprint trust")

    radius_factor_rows = None
    if mode.startswith("certificate_residual_rd"):
        if responsibility_radius_factors is None:
            raise ValueError("Residual rate-distortion trust requires radius factors")
        radius_factors = np.asarray(
            responsibility_radius_factors, dtype=np.float32
        ).reshape(-1)
        if radius_factors.shape != depths.shape:
            raise ValueError("Residual footprint radius factors must align")
        if np.any(~np.isfinite(radius_factors) | (radius_factors <= 0.0)):
            raise ValueError("Residual footprint radius factors must be positive")
        radius_factor_rows = radius_factors[rows].copy()
    elif responsibility_radius_factors is not None:
        raise ValueError("Radius factors require residual rate-distortion trust")

    if mode.endswith("_shuffled") and len(rows) > 1:
        labels = _adaptive_log_depth_regime_labels(depths, eligible)
        rng = np.random.default_rng(int(seed))
        shuffled = row_information.copy()
        shuffled_owners = None if owner_rows is None else owner_rows.copy()
        shuffled_radius_factors = (
            None if radius_factor_rows is None else radius_factor_rows.copy()
        )
        row_labels = labels[rows]
        for regime in np.unique(row_labels):
            positions = np.flatnonzero(row_labels == regime)
            if len(positions) > 1:
                permutation = positions[rng.permutation(len(positions))]
                if radius_factor_rows is None:
                    shuffled[positions] = row_information[permutation]
                if shuffled_owners is not None:
                    shuffled_owners[positions] = owner_rows[permutation]
                if shuffled_radius_factors is not None:
                    shuffled_radius_factors[positions] = radius_factor_rows[permutation]
        row_information = shuffled
        owner_rows = shuffled_owners
        radius_factor_rows = shuffled_radius_factors

    cell_px, _ = budget_cell_parameters(
        image_size,
        birth_budget,
        pool_multiplier,
        max(float(np.max(log_stds)), 1.0e-8),
    )
    responsibility_radius = np.full(
        row_information.shape, 0.5 * float(cell_px), dtype=np.float64
    )
    if mode.startswith("certificate_equal_area"):
        responsibility_radius[:] = float(cell_px) / math.sqrt(math.pi)
    elif mode.startswith("certificate_owner_area"):
        responsibility_radius[owner_rows] = float(cell_px) / math.sqrt(math.pi)
    elif mode.startswith("certificate_residual_rd"):
        responsibility_radius *= radius_factor_rows
    radius_ratio = np.maximum(responsibility_radius / radius, 1.0)
    if "bounded_area" in mode:
        row_limits = np.sqrt(
            (1.0 - row_information) + row_information * np.square(radius_ratio)
        )
    elif mode.startswith("certificate_"):
        unresolved = row_information < 1.0
        headroom = np.full(row_information.shape, np.inf, dtype=np.float64)
        headroom[unresolved] = row_information[unresolved] / np.maximum(
            1.0 - row_information[unresolved], np.finfo(np.float64).eps
        )
        log_limits = headroom * np.log(radius_ratio)
        row_limits = np.full(row_information.shape, np.inf, dtype=np.float64)
        finite = np.isfinite(log_limits) & (
            log_limits <= math.log(np.finfo(np.float32).max)
        )
        row_limits[finite] = np.exp(log_limits[finite])
    else:
        row_limits = np.exp(row_information * np.log(radius_ratio))
    limits[rows] = row_limits.astype(np.float32, copy=False)
    information[rows] = row_information.astype(np.float32, copy=False)
    return limits, information, {
        "rows": int(len(rows)),
        "cell_px": float(cell_px),
        "mean_information": float(np.mean(row_information)),
        "mean_limit": float(np.mean(row_limits)),
        "projective_owner_rows": (
            0 if owner_rows is None else int(np.count_nonzero(owner_rows))
        ),
        "mean_radius_factor": (
            None if radius_factor_rows is None else float(np.mean(radius_factor_rows))
        ),
        "shuffled": mode.endswith("_shuffled"),
    }


def validate_front_view_far_field_config(config=None):
    merged = deepcopy(DEFAULT_FRONT_VIEW_FAR_FIELD_CONFIG)
    if config is not None:
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError(
                "Unknown FrontViewFarField options: {}".format(sorted(unknown))
            )
        merged.update(config)
    if not isinstance(merged["enabled"], bool):
        raise TypeError("FrontViewFarField.enabled must be boolean")
    if not isinstance(merged["map_redundancy_gate"], bool):
        raise TypeError("FrontViewFarField.map_redundancy_gate must be boolean")
    if not isinstance(merged["posterior_budget_refill"], bool):
        raise TypeError("FrontViewFarField.posterior_budget_refill must be boolean")
    if not isinstance(merged["shuffle_refill_evidence"], bool):
        raise TypeError("FrontViewFarField.shuffle_refill_evidence must be boolean")
    for key in ("ray_atlas_enabled", "ray_atlas_shuffle_evidence"):
        if not isinstance(merged[key], bool):
            raise TypeError("FrontViewFarField.{} must be boolean".format(key))
    if merged["ray_atlas_shuffle_evidence"] and not merged["ray_atlas_enabled"]:
        raise ValueError("Shuffled ray-atlas evidence requires ray_atlas_enabled")
    if merged["ray_atlas_coordinate_mode"] not in (
        "camera_ray",
        "canonical_world",
    ):
        raise ValueError("FrontViewFarField ray-atlas coordinate mode is invalid")
    if merged["ray_atlas_competition_mode"] not in (
        "hard_cell",
        "continuous_kernel",
        "continuous_record",
        "continuous_dyadic",
    ):
        raise ValueError("FrontViewFarField ray-atlas competition mode is invalid")
    if (
        merged["ray_atlas_competition_mode"] in (
            "continuous_kernel",
            "continuous_record",
            "continuous_dyadic",
        )
        and merged["ray_atlas_coordinate_mode"] != "canonical_world"
    ):
        raise ValueError("Continuous ray responsibility requires canonical_world")
    if merged["unobservable_birth_policy"] not in ("projective", "reject"):
        raise ValueError(
            "FrontViewFarField.unobservable_birth_policy must be projective or reject"
        )
    if merged["map_redundancy_evidence"] not in (
        "geometry",
        "photometric",
        "photometric_shuffled",
    ):
        raise ValueError("FrontViewFarField.map_redundancy_evidence is invalid")
    if (
        merged["map_redundancy_evidence"] != "geometry"
        and not merged["map_redundancy_gate"]
    ):
        raise ValueError(
            "Photometric map evidence requires map_redundancy_gate=true"
        )
    if merged["posterior_budget_refill"] and not merged["map_redundancy_gate"]:
        raise ValueError("Posterior budget refill requires map_redundancy_gate=true")
    if merged["shuffle_refill_evidence"] and not merged["posterior_budget_refill"]:
        raise ValueError(
            "Shuffled refill evidence requires posterior_budget_refill=true"
        )
    if not isinstance(merged["shuffle_responsibility"], bool):
        raise TypeError("FrontViewFarField.shuffle_responsibility must be boolean")
    if merged["responsibility_shuffle_mode"] not in (
        "global",
        "log_depth_regimes",
    ):
        raise ValueError(
            "FrontViewFarField.responsibility_shuffle_mode must be global or "
            "log_depth_regimes"
        )
    if not isinstance(merged["shuffle_seed"], int):
        raise TypeError("FrontViewFarField.shuffle_seed must be an integer")
    if merged["responsibility_basis"] not in ("source", "persistent_identity"):
        raise ValueError(
            "FrontViewFarField.responsibility_basis must be source or "
            "persistent_identity"
        )
    if merged["footprint_trust_mode"] not in (
        "disabled",
        "information",
        "information_shuffled",
        "certificate_odds",
        "certificate_odds_shuffled",
        "certificate_equal_area",
        "certificate_equal_area_shuffled",
        "certificate_owner_area",
        "certificate_owner_area_shuffled",
        "certificate_residual_rd",
        "certificate_residual_rd_shuffled",
        "certificate_residual_rd_detail",
        "certificate_residual_rd_detail_shuffled",
        "certificate_residual_rd_visible_detail",
        "certificate_residual_rd_visible_detail_shuffled",
        "certificate_residual_rd_visible_detail_bounded_area",
        "certificate_residual_rd_visible_detail_bounded_area_shuffled",
    ):
        raise ValueError("FrontViewFarField.footprint_trust_mode is invalid")
    if merged["footprint_trust_scope"] not in (
        "all_depthcov",
        "projective_responsibility",
    ):
        raise ValueError("FrontViewFarField.footprint_trust_scope is invalid")
    for key in (
        "footprint_trust_dynamic_update",
        "footprint_trust_dynamic_shuffle",
    ):
        if not isinstance(merged[key], bool):
            raise TypeError("FrontViewFarField.{} must be boolean".format(key))
    if merged["footprint_trust_dynamic_update"] and merged[
        "footprint_trust_mode"
    ] == "disabled":
        raise ValueError("Dynamic footprint trust requires footprint trust")
    if merged["footprint_trust_dynamic_shuffle"] and not merged[
        "footprint_trust_dynamic_update"
    ]:
        raise ValueError("Shuffled dynamic trust requires dynamic trust")
    if merged["footprint_trust_dynamic_shuffle_mode"] not in (
        "evidence",
        "certificate",
    ):
        raise ValueError(
            "FrontViewFarField.footprint_trust_dynamic_shuffle_mode must be "
            "evidence or certificate"
        )
    if merged["routing_mode"] not in (
        "fixed_depth",
        "causal_observability",
        "adaptive_observability",
    ):
        raise ValueError(
            "FrontViewFarField.routing_mode must be fixed_depth, "
            "causal_observability, or adaptive_observability"
        )
    if merged["projective_nms_mode"] not in (
        "fixed_grid",
        "gaussian_support",
        "budget_cells",
    ):
        raise ValueError(
            "FrontViewFarField.projective_nms_mode must be fixed_grid, "
            "gaussian_support, or budget_cells"
        )
    if merged["projective_covariance_mode"] not in (
        "isotropic",
        "observability_rank",
        "observability_rank_shuffled",
        "surfel",
    ):
        raise ValueError("FrontViewFarField.projective_covariance_mode is invalid")
    if merged["fallback_support_mode"] not in (
        "legacy",
        "budget_isotropic",
        "budget_information",
        "budget_information_shuffled",
        "budget_structure",
        "budget_structure_shuffled",
        "budget_certificate_structure",
        "budget_certificate_structure_shuffled",
    ):
        raise ValueError("FrontViewFarField.fallback_support_mode is invalid")
    if (
        merged["projective_covariance_mode"] != "isotropic"
        and merged["routing_mode"] not in (
            "causal_observability",
            "adaptive_observability",
        )
    ):
        raise ValueError(
            "Projective covariance requires causal-observability routing"
        )
    if (
        merged["fallback_support_mode"] != "legacy"
        and merged["routing_mode"] not in (
            "causal_observability",
            "adaptive_observability",
        )
    ):
        raise ValueError(
            "Budgeted fallback support requires causal-observability routing"
        )
    if merged["ray_atlas_enabled"] and merged["routing_mode"] not in (
        "causal_observability",
        "adaptive_observability",
    ):
        raise ValueError(
            "Ray responsibility requires causal-observability or "
            "adaptive-observability routing"
        )
    if merged["footprint_trust_mode"] != "disabled" and merged[
        "routing_mode"
    ] not in ("causal_observability", "adaptive_observability"):
        raise ValueError("Footprint trust requires causal-observability routing")
    if (
        (
            "structure" in merged["fallback_support_mode"]
            or "information" in merged["fallback_support_mode"]
        )
        and merged["projective_covariance_mode"] != "isotropic"
    ):
        raise ValueError(
            "Fallback image support and radial covariance modes are mutually exclusive"
        )
    if float(merged["depth_m"]) <= 0.0:
        raise ValueError("FrontViewFarField.depth_m must be positive")
    if int(merged["projective_cell_px"]) <= 0:
        raise ValueError("FrontViewFarField.projective_cell_px must be positive")
    if float(merged["depth_bin_ratio"]) <= 1.0:
        raise ValueError("FrontViewFarField.depth_bin_ratio must be greater than one")
    return merged


class CausalRayResponsibilityAtlas:
    """Bounded online ownership of unresolved world-ray/depth cells.

    Angular resolution follows the candidate budget's image-plane sampling
    density. Depth resolution follows the paired DepthCov log-depth uncertainty.
    The atlas therefore introduces no metric near/far boundary. Its LRU capacity
    is one candidate-pool worth of cells, so memory remains independent of
    sequence length.
    """

    def __init__(
        self,
        enabled=False,
        shuffle_evidence=False,
        seed=42,
        coordinate_mode="camera_ray",
        competition_mode="hard_cell",
    ):
        self.enabled = bool(enabled)
        self.shuffle_evidence = bool(shuffle_evidence)
        self.seed = int(seed)
        self.coordinate_mode = str(coordinate_mode)
        self.competition_mode = str(competition_mode)
        if self.coordinate_mode not in ("camera_ray", "canonical_world"):
            raise ValueError("Ray-atlas coordinate mode is invalid")
        if self.competition_mode not in (
            "hard_cell",
            "continuous_kernel",
            "continuous_record",
            "continuous_dyadic",
        ):
            raise ValueError("Ray-atlas competition mode is invalid")
        if (
            self.competition_mode in (
                "continuous_kernel",
                "continuous_record",
                "continuous_dyadic",
            )
            and self.coordinate_mode != "canonical_world"
        ):
            raise ValueError("Continuous ray responsibility requires canonical_world")
        self._owners = OrderedDict()
        self._buckets = {}
        self._next_owner_id = 0
        self._max_angular_support = 0.0
        self._capacity = 0
        self._canonical_origin = None
        self.stats = {
            "calls": 0,
            "rows": 0,
            "rejected_rows": 0,
            "registered_rows": 0,
            "evicted_rows": 0,
            "shuffled_calls": 0,
            "last_angular_cell_rad": None,
            "last_log_depth_cell": None,
            "continuous_neighbor_checks": 0,
            "continuous_neighbor_rejections": 0,
            "record_admissions": 0,
            "record_rejections": 0,
            "record_replacements": 0,
            "dyadic_accumulations": 0,
            "dyadic_admissions": 0,
            "dyadic_rejections": 0,
            "dyadic_same_frame_rejections": 0,
            "dyadic_level_sum": 0,
        }

    @staticmethod
    def _parameters(image_size, birth_budget, pool_multiplier, focal_pixels, max_log_depth_std):
        cell_px, log_depth_cell = budget_cell_parameters(
            image_size,
            birth_budget,
            pool_multiplier,
            max_log_depth_std,
        )
        focal_pixels = float(focal_pixels)
        if not math.isfinite(focal_pixels) or focal_pixels <= 0.0:
            raise ValueError("Ray atlas focal length must be finite and positive")
        angular_cell = math.atan(cell_px / focal_pixels)
        capacity = max(1, int(birth_budget) * int(pool_multiplier))
        return angular_cell, log_depth_cell, capacity

    @staticmethod
    def _coordinates(directions, depths):
        directions = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        if len(directions) != len(depths):
            raise ValueError("Ray-atlas directions and depths must align")
        norms = np.linalg.norm(directions, axis=1)
        if (
            np.any(~np.isfinite(directions))
            or np.any(~np.isfinite(depths))
            or np.any(depths <= 0.0)
            or np.any(norms <= np.finfo(np.float32).eps)
        ):
            raise ValueError("Ray-atlas evidence must be finite and valid")
        unit = directions / norms[:, None]
        return unit, np.log(depths)

    @staticmethod
    def _keys_from_coordinates(unit, log_depths, angular_cell, log_depth_cell):
        angular = np.floor((unit + 1.0) / float(angular_cell)).astype(np.int64)
        depth_cells = np.floor(log_depths / float(log_depth_cell)).astype(np.int64)
        return [
            (int(ray[0]), int(ray[1]), int(ray[2]), int(depth_cell))
            for ray, depth_cell in zip(angular, depth_cells)
        ]

    @classmethod
    def _keys(cls, directions, depths, angular_cell, log_depth_cell):
        unit, log_depths = cls._coordinates(directions, depths)
        return cls._keys_from_coordinates(
            unit, log_depths, angular_cell, log_depth_cell
        )

    def _responsibility_coordinates(
        self, directions, depths, world_points, camera_center
    ):
        if self.coordinate_mode == "camera_ray":
            return self._coordinates(directions, depths)
        points = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
        center = np.asarray(camera_center, dtype=np.float32).reshape(3)
        if len(points) != len(depths):
            raise ValueError("Canonical ray-atlas world points must align")
        if np.any(~np.isfinite(points)) or np.any(~np.isfinite(center)):
            raise ValueError("Canonical ray-atlas coordinates must be finite")
        if self._canonical_origin is None:
            self._canonical_origin = center.copy()
        rays = points - self._canonical_origin[None, :]
        radii = np.linalg.norm(rays, axis=1)
        return self._coordinates(rays, radii)

    @staticmethod
    def _neighbor_keys(key, angular_radius=1):
        angular_radius = max(1, int(angular_radius))
        for dx in range(-angular_radius, angular_radius + 1):
            for dy in range(-angular_radius, angular_radius + 1):
                for dz in range(-angular_radius, angular_radius + 1):
                    for dd in (-1, 0, 1):
                        yield (
                            key[0] + dx,
                            key[1] + dy,
                            key[2] + dz,
                            key[3] + dd,
                        )

    @staticmethod
    def _within_kernel(
        lhs_unit,
        lhs_log_depth,
        lhs_angular_support,
        lhs_log_depth_std,
        rhs,
        angular_cell,
    ):
        angular_scale = max(
            float(angular_cell),
            float(lhs_angular_support),
            float(rhs["angular_support"]),
        )
        radial_scale = math.hypot(
            float(lhs_log_depth_std), float(rhs["log_depth_std"])
        )
        radial_scale = max(radial_scale, np.finfo(np.float32).eps)
        angular = np.linalg.norm(lhs_unit - rhs["unit"]) / angular_scale
        radial = abs(float(lhs_log_depth) - float(rhs["log_depth"])) / radial_scale
        return angular * angular + radial * radial <= 1.0

    def _continuous_owners(self, key, angular_radius):
        for neighbor in self._neighbor_keys(key, angular_radius):
            for owner_id in tuple(self._buckets.get(neighbor, ())):
                owner = self._owners.get(owner_id)
                if owner is not None:
                    yield owner_id, owner

    def _remove_continuous_owner(self, owner_id, *, evicted=False):
        owner = self._owners.pop(owner_id)
        bucket = self._buckets.get(owner["key"])
        if bucket is not None:
            bucket.discard(owner_id)
            if not bucket:
                del self._buckets[owner["key"]]
        if evicted:
            self.stats["evicted_rows"] += 1

    def _evict_continuous_owner(self):
        owner_id = next(iter(self._owners))
        self._remove_continuous_owner(owner_id, evicted=True)

    def admit(
        self,
        directions,
        depths,
        *,
        image_size,
        birth_budget,
        pool_multiplier,
        focal_pixels,
        max_log_depth_std,
        frame_id,
        world_points=None,
        camera_center=None,
        evidence_scores=None,
        projected_radii=None,
        log_depth_stds=None,
    ):
        count = len(depths)
        if not self.enabled or count == 0:
            return np.ones((count,), dtype=np.bool_), [None] * count
        angular_cell, log_depth_cell, capacity = self._parameters(
            image_size,
            birth_budget,
            pool_multiplier,
            focal_pixels,
            max_log_depth_std,
        )
        evidence_directions = np.asarray(directions, dtype=np.float32)
        evidence_depths = np.asarray(depths, dtype=np.float32)
        evidence_world_points = world_points
        shuffle_scores_only = (
            self.shuffle_evidence
            and self.competition_mode in ("continuous_record", "continuous_dyadic")
        )
        if self.shuffle_evidence and count > 1 and not shuffle_scores_only:
            rng = np.random.default_rng(self.seed + int(frame_id))
            permutation = rng.permutation(count)
            evidence_directions = evidence_directions[permutation]
            evidence_depths = evidence_depths[permutation]
            if world_points is not None:
                evidence_world_points = np.asarray(world_points)[permutation]
            self.stats["shuffled_calls"] += 1
        units, log_depths = self._responsibility_coordinates(
            evidence_directions,
            evidence_depths,
            evidence_world_points,
            camera_center,
        )
        keys = self._keys_from_coordinates(
            units,
            log_depths,
            angular_cell,
            log_depth_cell,
        )
        keep = np.ones((count,), dtype=np.bool_)
        claims = keys
        if self.competition_mode == "hard_cell":
            claimed = set()
            for index, key in enumerate(keys):
                if key in self._owners or key in claimed:
                    keep[index] = False
                    if key in self._owners:
                        self._owners.move_to_end(key)
                else:
                    claimed.add(key)
        else:
            scores = np.zeros((count,), dtype=np.float32)
            if evidence_scores is not None:
                scores = np.asarray(evidence_scores, dtype=np.float32).reshape(-1)
                if (
                    scores.shape != (count,)
                    or np.any(~np.isfinite(scores))
                    or np.any(scores < 0.0)
                ):
                    raise ValueError("Ray-atlas evidence scores must align and be finite")
            if shuffle_scores_only and count > 1:
                rng = np.random.default_rng(self.seed + int(frame_id))
                scores = scores[rng.permutation(count)]
                self.stats["shuffled_calls"] += 1
            radii = np.asarray(projected_radii, dtype=np.float32).reshape(-1)
            log_stds = np.asarray(log_depth_stds, dtype=np.float32).reshape(-1)
            if (
                radii.shape != (count,)
                or log_stds.shape != (count,)
                or np.any(~np.isfinite(radii))
                or np.any(~np.isfinite(log_stds))
                or np.any(radii <= 0.0)
                or np.any(log_stds < 0.0)
            ):
                raise ValueError("Continuous ray-atlas supports must align and be valid")
            angular_supports = np.arctan(radii / float(focal_pixels))
            order = np.lexsort(
                (log_depths, units[:, 2], units[:, 1], units[:, 0], -scores)
            )
            keep[:] = False
            local = {}
            for index in order.tolist():
                key = keys[index]
                overlapping_ids = []
                overlapping = []
                local_conflict = False
                angular_radius = int(
                    math.ceil(
                        max(
                            float(angular_supports[index]),
                            self._max_angular_support,
                        )
                        / float(angular_cell)
                    )
                )
                for owner_id, owner in self._continuous_owners(
                    key, angular_radius
                ):
                    self.stats["continuous_neighbor_checks"] += 1
                    if self._within_kernel(
                        units[index],
                        log_depths[index],
                        angular_supports[index],
                        log_stds[index],
                        owner,
                        angular_cell,
                    ):
                        overlapping_ids.append(owner_id)
                        overlapping.append(owner)
                for neighbor in self._neighbor_keys(key, angular_radius):
                    for owner in local.get(neighbor, ()):
                        if self._within_kernel(
                            units[index],
                            log_depths[index],
                            angular_supports[index],
                            log_stds[index],
                            owner,
                            angular_cell,
                        ):
                            overlapping.append(owner)
                            local_conflict = True
                for owner_id in overlapping_ids:
                    self._owners.move_to_end(owner_id)
                conflict = bool(overlapping)
                replace_owner_ids = []
                if conflict and self.competition_mode == "continuous_record":
                    record = max(float(owner.get("score", 0.0)) for owner in overlapping)
                    if float(scores[index]) > record:
                        conflict = False
                        replace_owner_ids = overlapping_ids
                        self.stats["record_admissions"] += 1
                    else:
                        self.stats["record_rejections"] += 1
                elif conflict and self.competition_mode == "continuous_dyadic":
                    if local_conflict or not overlapping_ids:
                        self.stats["dyadic_same_frame_rejections"] += 1
                    else:
                        owner_id = min(overlapping_ids)
                        owner = self._owners[owner_id]
                        if owner.get("last_evidence_frame") == int(frame_id):
                            self.stats["dyadic_same_frame_rejections"] += 1
                        else:
                            evidence = float(scores[index])
                            origin = float(owner.get("evidence_origin", 0.0))
                            cumulative = float(
                                owner.get("cumulative_evidence", origin)
                            )
                            if origin <= 0.0 and evidence > 0.0:
                                origin = evidence
                                cumulative = evidence
                                owner["evidence_origin"] = origin
                                owner["dyadic_level"] = 0
                            else:
                                cumulative += evidence
                            owner["cumulative_evidence"] = cumulative
                            owner["last_evidence_frame"] = int(frame_id)
                            self.stats["dyadic_accumulations"] += 1
                            level = int(owner.get("dyadic_level", 0))
                            target_level = level
                            if origin > 0.0 and cumulative >= origin:
                                target_level = int(
                                    math.floor(math.log2(cumulative / origin))
                                )
                            if target_level > level:
                                owner["dyadic_level"] = target_level
                                conflict = False
                                self.stats["dyadic_admissions"] += 1
                                self.stats["dyadic_level_sum"] += target_level - level
                                replace_owner_ids = [owner_id]
                            else:
                                self.stats["dyadic_rejections"] += 1
                if conflict:
                    self.stats["continuous_neighbor_rejections"] += 1
                    continue
                keep[index] = True
                local.setdefault(key, []).append({
                    "unit": units[index].copy(),
                    "log_depth": float(log_depths[index]),
                    "angular_support": float(angular_supports[index]),
                    "log_depth_std": float(log_stds[index]),
                    "score": float(scores[index]),
                })
                claims[index] = {
                    "key": key,
                    "unit": units[index].copy(),
                    "log_depth": float(log_depths[index]),
                    "angular_support": float(angular_supports[index]),
                    "log_depth_std": float(log_stds[index]),
                    "score": float(scores[index]),
                    "replace_owner_ids": tuple(replace_owner_ids),
                    "dyadic_update": self.competition_mode == "continuous_dyadic",
                    "last_evidence_frame": int(frame_id),
                    "evidence_origin": float(scores[index]),
                    "cumulative_evidence": float(scores[index]),
                    "dyadic_level": 0,
                }
        self._capacity = capacity
        self.stats["calls"] += 1
        self.stats["rows"] += count
        self.stats["rejected_rows"] += int(np.sum(~keep))
        self.stats["last_angular_cell_rad"] = float(angular_cell)
        self.stats["last_log_depth_cell"] = float(log_depth_cell)
        return keep, claims

    def register(self, keys, frame_id):
        if not self.enabled:
            return
        for claim in keys:
            if claim is None:
                continue
            key = tuple(claim[:4]) if not isinstance(claim, dict) else claim["key"]
            owner = int(frame_id)
            if self.competition_mode in (
                "continuous_kernel",
                "continuous_record",
                "continuous_dyadic",
            ):
                owner = dict(claim)
                owner.update({
                    "key": key,
                    "unit": np.asarray(claim["unit"], dtype=np.float32),
                    "frame_id": int(frame_id),
                })
                replace_owner_ids = owner.pop("replace_owner_ids", ())
                dyadic_update = bool(owner.pop("dyadic_update", False))
                if dyadic_update and replace_owner_ids:
                    owner_id = int(replace_owner_ids[0])
                    if owner_id in self._owners:
                        self._owners.move_to_end(owner_id)
                        self.stats["registered_rows"] += 1
                        continue
                for owner_id in replace_owner_ids:
                    if owner_id in self._owners:
                        self._remove_continuous_owner(owner_id)
                        self.stats["record_replacements"] += 1
                owner_id = self._next_owner_id
                self._next_owner_id += 1
                self._owners[owner_id] = owner
                self._buckets.setdefault(key, set()).add(owner_id)
                self._max_angular_support = max(
                    self._max_angular_support, owner["angular_support"]
                )
                self.stats["registered_rows"] += 1
                while len(self._owners) > self._capacity:
                    self._evict_continuous_owner()
                continue
            if key in self._owners:
                self._owners.move_to_end(key)
                self._owners[key] = owner
                continue
            self._owners[key] = owner
            self.stats["registered_rows"] += 1
            while len(self._owners) > self._capacity:
                self._owners.popitem(last=False)
                self.stats["evicted_rows"] += 1

    def summary(self):
        result = dict(self.stats)
        result.update(
            enabled=self.enabled,
            shuffle_evidence=self.shuffle_evidence,
            coordinate_mode=self.coordinate_mode,
            competition_mode=self.competition_mode,
            canonical_origin=(
                None
                if self._canonical_origin is None
                else [float(value) for value in self._canonical_origin.tolist()]
            ),
            active_cells=len(self._owners),
            capacity=self._capacity,
        )
        return result


def budgeted_fallback_radius(image_size, birth_budget):
    """Return half the square-cell spacing induced by the birth budget."""

    width, height = (int(value) for value in image_size)
    birth_budget = int(birth_budget)
    if width <= 0 or height <= 0 or birth_budget <= 0:
        raise ValueError("Fallback support requires positive image and budget")
    return 0.5 * math.sqrt(float(width * height) / float(birth_budget))


def projective_radial_scale_factors(
    parallax_pixels,
    projected_radii,
    log_depth_stds,
    projective_mask,
    *,
    mode="observability_rank",
    seed=42,
):
    """Convert the metric-observability certificate into covariance rank."""

    parallax = np.asarray(parallax_pixels, dtype=np.float32).reshape(-1)
    radii = np.asarray(projected_radii, dtype=np.float32).reshape(-1)
    log_stds = np.asarray(log_depth_stds, dtype=np.float32).reshape(-1)
    projective = np.asarray(projective_mask, dtype=np.bool_).reshape(-1)
    if not (parallax.shape == radii.shape == log_stds.shape == projective.shape):
        raise ValueError("Projective covariance arrays must align")
    if (
        np.any(~np.isfinite(parallax))
        or np.any(~np.isfinite(radii))
        or np.any(~np.isfinite(log_stds))
        or np.any(parallax < 0.0)
        or np.any(radii <= 0.0)
        or np.any(log_stds < 0.0)
    ):
        raise ValueError("Projective covariance values must be finite and valid")
    if mode not in (
        "isotropic",
        "observability_rank",
        "observability_rank_shuffled",
        "surfel",
    ):
        raise ValueError("Projective covariance mode is invalid")

    factors = np.ones(parallax.shape, dtype=np.float32)
    rows = np.flatnonzero(projective)
    if not len(rows) or mode == "isotropic":
        return factors
    variance_floor = np.finfo(np.float32).eps
    if mode == "surfel":
        factors[rows] = math.sqrt(variance_floor)
        return factors

    p = parallax[rows].astype(np.float64)
    rho = radii[rows].astype(np.float64)
    sigma = log_stds[rows].astype(np.float64)
    parallax_margin = p / rho
    precision_margin = np.divide(
        rho,
        p * sigma,
        out=np.full_like(p, np.inf),
        where=(p * sigma) > 0.0,
    )
    information = np.minimum(1.0, np.minimum(parallax_margin, precision_margin))
    factors[rows] = np.sqrt(
        np.maximum(information, variance_floor)
    ).astype(np.float32)
    if mode == "observability_rank_shuffled" and len(rows) > 1:
        rng = np.random.default_rng(int(seed))
        factors[rows] = factors[rows[rng.permutation(len(rows))]]
    return factors


def ray_aligned_quaternions(view_directions):
    """Return wxyz rotations whose local +Z axes follow world-space rays."""

    directions = np.asarray(view_directions, dtype=np.float32).reshape(-1, 3)
    if len(directions) == 0:
        return np.empty((0, 4), dtype=np.float32)
    norms = np.linalg.norm(directions, axis=1)
    if np.any(~np.isfinite(directions)) or np.any(norms <= 1.0e-8):
        raise ValueError("Projective covariance requires finite nonzero rays")
    directions = directions / norms[:, None]
    quaternions = np.zeros((len(directions), 4), dtype=np.float32)
    quaternions[:, 0] = 1.0 + directions[:, 2]
    quaternions[:, 1] = -directions[:, 1]
    quaternions[:, 2] = directions[:, 0]
    opposite = quaternions[:, 0] <= np.finfo(np.float32).eps
    quaternions[opposite] = np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float32)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions.astype(np.float32, copy=False)


def projected_gaussian_radii(log_scales, depths, focal_pixels):
    """Return the one-sigma image support of isotropic newborn Gaussians."""

    log_scales = np.asarray(log_scales, dtype=np.float32)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    if log_scales.ndim == 2:
        log_scales = log_scales[:, 0]
    log_scales = log_scales.reshape(-1)
    if log_scales.shape != depths.shape:
        raise ValueError("Gaussian scales and depths must align")
    focal_pixels = float(focal_pixels)
    if focal_pixels <= 0.0:
        raise ValueError("Focal length must be positive")
    if np.any(~np.isfinite(log_scales)) or np.any(~np.isfinite(depths)):
        raise ValueError("Gaussian scales and depths must be finite")
    if np.any(depths <= 0.0):
        raise ValueError("Gaussian depths must be positive")
    return (
        focal_pixels * np.exp(log_scales) / np.maximum(depths, 1.0e-8)
    ).astype(np.float32, copy=False)


def visible_parallax_pixels(
    world_points,
    current_world_to_camera,
    reference_world_to_camera,
    reference_intrinsics,
    reference_image_sizes,
    focal_pixels,
):
    """Maximum causally visible ray parallax measured in current-view pixels."""

    points = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
    current_pose = np.asarray(current_world_to_camera, dtype=np.float32).reshape(4, 4)
    reference_poses = [
        np.asarray(pose, dtype=np.float32).reshape(4, 4)
        for pose in reference_world_to_camera
    ]
    intrinsics = [
        np.asarray(matrix, dtype=np.float32).reshape(3, 3)
        for matrix in reference_intrinsics
    ]
    image_sizes = [tuple(int(value) for value in size) for size in reference_image_sizes]
    if not (len(reference_poses) == len(intrinsics) == len(image_sizes)):
        raise ValueError("Reference camera arrays must align")
    focal_pixels = float(focal_pixels)
    if focal_pixels <= 0.0:
        raise ValueError("Focal length must be positive")
    if len(points) == 0 or len(reference_poses) == 0:
        return (
            np.zeros((len(points),), dtype=np.float32),
            np.zeros((len(points),), dtype=np.int32),
        )

    current_center = -current_pose[:3, :3].T @ current_pose[:3, 3]
    current_rays = points - current_center[None, :]
    current_rays /= np.maximum(
        np.linalg.norm(current_rays, axis=1, keepdims=True), 1.0e-8
    )
    maximum = np.zeros((len(points),), dtype=np.float32)
    support = np.zeros((len(points),), dtype=np.int32)
    for pose, intrinsic, (width, height) in zip(
        reference_poses, intrinsics, image_sizes
    ):
        camera_points = points @ pose[:3, :3].T + pose[:3, 3]
        z = camera_points[:, 2]
        projected = camera_points @ intrinsic.T
        uv = projected[:, :2] / np.maximum(z[:, None], 1.0e-8)
        valid = (
            (z > 0.0)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < float(width))
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < float(height))
        )
        reference_center = -pose[:3, :3].T @ pose[:3, 3]
        reference_rays = points - reference_center[None, :]
        reference_rays /= np.maximum(
            np.linalg.norm(reference_rays, axis=1, keepdims=True), 1.0e-8
        )
        cosine = np.sum(current_rays * reference_rays, axis=1)
        sine = np.sqrt(np.clip(1.0 - cosine * cosine, 0.0, 1.0))
        parallax = focal_pixels * sine
        maximum = np.maximum(maximum, np.where(valid, parallax, 0.0))
        support += valid.astype(np.int32)
    return maximum.astype(np.float32, copy=False), support


def far_field_responsibility_mask(
    depths,
    sparse_valid,
    track_ids,
    config,
    *,
    parallax_pixels=None,
    projected_radii=None,
    log_depth_stds=None,
    return_metadata=False,
):
    """Select far candidates whose configured owner is projective space."""

    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    sparse_valid = np.asarray(sparse_valid, dtype=np.bool_).reshape(-1)
    track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
    if not (depths.shape == sparse_valid.shape == track_ids.shape):
        raise ValueError("Far-field responsibility arrays must align")
    if not config["enabled"]:
        result = np.zeros(depths.shape, dtype=np.bool_)
        return (result, None) if return_metadata else result
    if config["responsibility_basis"] == "persistent_identity":
        lacks_metric_owner = track_ids < 0
    else:
        lacks_metric_owner = ~sparse_valid
    if config["routing_mode"] == "fixed_depth":
        result = lacks_metric_owner & (depths >= float(config["depth_m"]))
        return (result, None) if return_metadata else result

    parallax_pixels = np.asarray(parallax_pixels, dtype=np.float32).reshape(-1)
    projected_radii = np.asarray(projected_radii, dtype=np.float32).reshape(-1)
    log_depth_stds = np.asarray(log_depth_stds, dtype=np.float32).reshape(-1)
    if not (
        parallax_pixels.shape
        == projected_radii.shape
        == log_depth_stds.shape
        == depths.shape
    ):
        raise ValueError("Causal observability arrays must align")
    if (
        np.any(~np.isfinite(parallax_pixels))
        or np.any(~np.isfinite(projected_radii))
        or np.any(~np.isfinite(log_depth_stds))
        or np.any(projected_radii <= 0.0)
        or np.any(log_depth_stds < 0.0)
    ):
        raise ValueError("Causal observability values must be finite and valid")

    observable = parallax_pixels >= projected_radii
    depth_precise = parallax_pixels * log_depth_stds <= projected_radii
    has_metric_certificate = observable & depth_precise
    unresolved = lacks_metric_owner & ~has_metric_certificate
    metadata = None
    if config["routing_mode"] == "adaptive_observability":
        adaptive_far, metadata = adaptive_log_depth_responsibility(
            depths, lacks_metric_owner
        )
        unresolved &= adaptive_far
    return (unresolved, metadata) if return_metadata else unresolved


def projective_map_posterior_log_odds(
    depths,
    map_depths,
    map_opacities,
    parallax_pixels,
    projected_radii,
    residuals=None,
    residual_scale=None,
):
    """Return log posterior odds that the rendered map explains each birth.

    Existing opacity supplies prior odds.  The log-depth discrepancy is converted
    to a causal disparity and evaluated under the newborn Gaussian's projected
    one-sigma support.  Non-negative posterior log odds mean the map already owns
    the observation.
    """

    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    map_depths = np.asarray(map_depths, dtype=np.float32).reshape(-1)
    map_opacities = np.asarray(map_opacities, dtype=np.float32).reshape(-1)
    parallax_pixels = np.asarray(parallax_pixels, dtype=np.float32).reshape(-1)
    projected_radii = np.asarray(projected_radii, dtype=np.float32).reshape(-1)
    if not (
        depths.shape
        == map_depths.shape
        == map_opacities.shape
        == parallax_pixels.shape
        == projected_radii.shape
    ):
        raise ValueError("Projective map-gate arrays must align")
    valid = (
        np.isfinite(depths)
        & np.isfinite(map_depths)
        & np.isfinite(map_opacities)
        & np.isfinite(parallax_pixels)
        & np.isfinite(projected_radii)
        & (depths > 0.0)
        & (map_depths > 0.0)
        & (projected_radii > 0.0)
        & (map_opacities > 0.0)
        & (map_opacities <= 1.0)
    )
    log_odds = np.full(depths.shape, -np.inf, dtype=np.float64)
    if not np.any(valid):
        return log_odds
    alpha = np.clip(map_opacities[valid], 1.0e-6, 1.0 - 1.0e-6)
    disparity = parallax_pixels[valid] * np.abs(
        np.log(depths[valid]) - np.log(map_depths[valid])
    )
    normalized = disparity / projected_radii[valid]
    posterior_log_odds = np.log(alpha) - np.log1p(-alpha) - 0.5 * normalized**2
    if residuals is not None or residual_scale is not None:
        if residuals is None or residual_scale is None:
            raise ValueError("Photometric map evidence requires residuals and scale")
        residuals = np.asarray(residuals, dtype=np.float32).reshape(-1)
        residual_scale = float(residual_scale)
        if residuals.shape != depths.shape or np.any(~np.isfinite(residuals)):
            raise ValueError("Map residuals must align and be finite")
        if not math.isfinite(residual_scale) or residual_scale <= 0.0:
            raise ValueError("Map residual scale must be finite and positive")
        posterior_log_odds -= 0.5 * (
            residuals[valid].astype(np.float64) / residual_scale
        ) ** 2
    log_odds[valid] = posterior_log_odds
    return log_odds


def projective_map_redundancy_mask(
    depths,
    map_depths,
    map_opacities,
    parallax_pixels,
    projected_radii,
    residuals=None,
    residual_scale=None,
):
    """Reject births with non-negative map-explanation posterior log odds."""

    return projective_map_posterior_log_odds(
        depths,
        map_depths,
        map_opacities,
        parallax_pixels,
        projected_radii,
        residuals=residuals,
        residual_scale=residual_scale,
    ) >= 0.0


def posterior_budget_refill_mask(
    budget_primary,
    sparse_valid,
    map_log_odds,
    residual_scores,
    requested_refill_count,
    *,
    reserve_eligible=None,
    shuffle_evidence=False,
    seed=42,
):
    """Project surviving proposals back onto the original DepthCov budget.

    Primary and sparse rows are mandatory. Reserve rows have already passed the
    same admission tests; the least map-explained rows refill missing primary
    DepthCov slots without increasing the configured per-frame birth budget.
    """

    primary = np.asarray(budget_primary, dtype=np.bool_).reshape(-1)
    sparse = np.asarray(sparse_valid, dtype=np.bool_).reshape(-1)
    log_odds = np.asarray(map_log_odds, dtype=np.float64).reshape(-1)
    residuals = np.asarray(residual_scores, dtype=np.float32).reshape(-1)
    if not (primary.shape == sparse.shape == log_odds.shape == residuals.shape):
        raise ValueError("Posterior refill arrays must align")
    eligible = (
        np.ones(primary.shape, dtype=np.bool_)
        if reserve_eligible is None
        else np.asarray(reserve_eligible, dtype=np.bool_).reshape(-1)
    )
    if eligible.shape != primary.shape:
        raise ValueError("Posterior refill eligibility must align")
    if np.any(np.isnan(log_odds)) or np.any(~np.isfinite(residuals)):
        raise ValueError("Posterior refill evidence must not contain NaNs")
    requested = int(requested_refill_count)
    if requested < 0:
        raise ValueError("Posterior refill request must be non-negative")

    mandatory = primary | sparse
    reserves = np.flatnonzero(~mandatory & ~sparse & eligible)
    keep = mandatory.copy()
    if requested <= 0 or not len(reserves):
        return keep, {
            "requested": requested,
            "reserves": int(len(reserves)),
            "selected": 0,
        }

    evidence = log_odds[reserves].copy()
    if shuffle_evidence and len(reserves) > 1:
        rng = np.random.default_rng(int(seed))
        evidence = evidence[rng.permutation(len(evidence))]
    ranked = reserves[np.lexsort((reserves, -residuals[reserves], evidence))]
    selected = min(requested, len(ranked))
    keep[ranked[:selected]] = True
    return keep, {
        "requested": requested,
        "reserves": int(len(reserves)),
        "selected": int(selected),
    }


def _priority_order(scores, primary_mask):
    indices = np.arange(len(scores), dtype=np.int64)
    if primary_mask is None:
        return np.argsort(-scores, kind="stable")
    primary = np.asarray(primary_mask, dtype=np.bool_).reshape(-1)
    if primary.shape != scores.shape:
        raise ValueError("Projective primary mask must align")
    return np.lexsort((indices, -scores, ~primary))


def _support_aware_survivor_mask(
    uv,
    depths,
    scores,
    projected_radii,
    log_depth_stds,
    primary_mask=None,
):
    projected_radii = np.asarray(projected_radii, dtype=np.float32).reshape(-1)
    log_depth_stds = np.asarray(log_depth_stds, dtype=np.float32).reshape(-1)
    if projected_radii.shape != depths.shape or log_depth_stds.shape != depths.shape:
        raise ValueError("Projective support arrays must align")
    if (
        np.any(~np.isfinite(projected_radii))
        or np.any(projected_radii <= 0.0)
        or np.any(~np.isfinite(log_depth_stds))
        or np.any(log_depth_stds < 0.0)
    ):
        raise ValueError("Projective support values must be finite and valid")

    keep = np.zeros((len(depths),), dtype=np.bool_)
    if len(depths) == 0:
        return keep
    base_cell = float(np.median(projected_radii))
    base_cell = max(base_cell, np.finfo(np.float32).eps)
    log_depths = np.log(np.maximum(depths, 1.0e-8))
    order = _priority_order(scores, primary_mask)
    buckets = {}
    maximum_radius = 0.0
    for index in order.tolist():
        radius = float(projected_radii[index])
        search = math.sqrt(radius * radius + maximum_radius * maximum_radius)
        x0 = int(math.floor((float(uv[index, 0]) - search) / base_cell))
        x1 = int(math.floor((float(uv[index, 0]) + search) / base_cell))
        y0 = int(math.floor((float(uv[index, 1]) - search) / base_cell))
        y1 = int(math.floor((float(uv[index, 1]) + search) / base_cell))
        conflict = False
        for cell_y in range(y0, y1 + 1):
            if conflict:
                break
            for cell_x in range(x0, x1 + 1):
                for other in buckets.get((cell_x, cell_y), ()):
                    delta_uv = uv[index] - uv[other]
                    spatial_variance = radius * radius + float(
                        projected_radii[other]
                    ) ** 2
                    if float(np.dot(delta_uv, delta_uv)) > spatial_variance:
                        continue
                    depth_sigma = math.sqrt(
                        float(log_depth_stds[index]) ** 2
                        + float(log_depth_stds[other]) ** 2
                    )
                    if abs(float(log_depths[index] - log_depths[other])) <= max(
                        depth_sigma, np.finfo(np.float32).eps
                    ):
                        conflict = True
                        break
                if conflict:
                    break
        if conflict:
            continue
        keep[index] = True
        maximum_radius = max(maximum_radius, radius)
        cell = (
            int(math.floor(float(uv[index, 0]) / base_cell)),
            int(math.floor(float(uv[index, 1]) / base_cell)),
        )
        buckets.setdefault(cell, []).append(index)
    return keep


def budget_cell_parameters(
    image_size,
    birth_budget,
    pool_multiplier,
    max_log_depth_std,
):
    """Derive projective responsibility cells from compute and uncertainty."""

    width, height = (int(value) for value in image_size)
    birth_budget = int(birth_budget)
    pool_multiplier = int(pool_multiplier)
    max_log_depth_std = float(max_log_depth_std)
    if width <= 0 or height <= 0:
        raise ValueError("Projective image size must be positive")
    if birth_budget <= 0 or pool_multiplier <= 0:
        raise ValueError("Birth budget and pool multiplier must be positive")
    if not math.isfinite(max_log_depth_std) or max_log_depth_std <= 0.0:
        raise ValueError("Maximum log-depth standard deviation must be positive")
    cell_px = math.sqrt(
        float(width * height) / float(birth_budget * pool_multiplier)
    )
    paired_log_depth_std = math.sqrt(2.0) * max_log_depth_std
    return cell_px, paired_log_depth_std


def _budget_cell_survivor_mask(
    uv,
    depths,
    scores,
    image_size,
    birth_budget,
    pool_multiplier,
    max_log_depth_std,
    primary_mask=None,
):
    cell_px, log_depth_width = budget_cell_parameters(
        image_size,
        birth_budget,
        pool_multiplier,
        max_log_depth_std,
    )
    xy = np.floor(uv / cell_px).astype(np.int64)
    depth_bin = np.floor(
        np.log(np.maximum(depths, 1.0e-8)) / log_depth_width
    ).astype(np.int64)
    order = _priority_order(scores, primary_mask)
    keep = np.zeros((len(depths),), dtype=np.bool_)
    occupied = set()
    for index in order.tolist():
        key = (int(xy[index, 0]), int(xy[index, 1]), int(depth_bin[index]))
        if key in occupied:
            continue
        occupied.add(key)
        keep[index] = True
    return keep


def projective_survivor_mask(
    uv,
    depths,
    scores,
    config,
    *,
    projected_radii=None,
    log_depth_stds=None,
    image_size=None,
    birth_budget=None,
    pool_multiplier=None,
    max_log_depth_std=None,
    primary_mask=None,
):
    """Keep the strongest row in each image/log-depth responsibility cell."""

    uv = np.asarray(uv, dtype=np.float32)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if uv.shape != (len(depths), 2) or scores.shape != depths.shape:
        raise ValueError("Projective far-field arrays must align")
    keep = np.zeros((len(depths),), dtype=np.bool_)
    if len(depths) == 0:
        return keep
    if config["projective_nms_mode"] == "gaussian_support":
        return _support_aware_survivor_mask(
            uv,
            depths,
            scores,
            projected_radii,
            log_depth_stds,
            primary_mask=primary_mask,
        )
    if config["projective_nms_mode"] == "budget_cells":
        return _budget_cell_survivor_mask(
            uv,
            depths,
            scores,
            image_size,
            birth_budget,
            pool_multiplier,
            max_log_depth_std,
            primary_mask=primary_mask,
        )
    xy = np.floor(uv / float(config["projective_cell_px"])).astype(np.int64)
    depth_bin = np.floor(
        np.log(np.maximum(depths, 1.0e-8))
        / math.log(float(config["depth_bin_ratio"]))
    ).astype(np.int64)
    order = _priority_order(scores, primary_mask)
    occupied = set()
    for index in order.tolist():
        key = (int(xy[index, 0]), int(xy[index, 1]), int(depth_bin[index]))
        if key in occupied:
            continue
        occupied.add(key)
        keep[index] = True
    return keep
