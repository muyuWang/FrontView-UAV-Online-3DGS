"""HashBlock-free residual-cover births for front-view UAV mapping."""

from __future__ import annotations

from copy import deepcopy
import heapq
import math

import numpy as np
import torch


DEFAULT_FRONT_VIEW_RESIDUAL_COVER_CONFIG = {
    "enabled": False,
    "selection_mode": "utility_coverage",
    "utility_fraction": 1.0,
    "pool_multiplier": 2,
    "budget_scale": 0.78,
    "depth_edges_m": [20.0, 50.0],
    "depth_fractions": [0.25, 0.45, 0.30],
    "initial_opacity": 0.5,
    "support_sigma_multiplier": 3.0,
    "selection_sigma_floor_px": 0.35,
    "min_support_radius_px": 1,
    "max_support_radius_px": 3,
    "use_depth_visibility": False,
    "depth_tolerance_ratio": 0.05,
    "behind_visibility_floor": 0.25,
    "confidence_power": 1.0,
    "confidence_floor": 0.25,
    "area_normalization_power": 0.0,
    "edge_gain_weight": 0.0,
    "edge_residual_scale": 0.10,
    "use_covariance_lod": False,
    "covariance_competition_enabled": False,
    "covariance_neighbors": 8,
    "covariance_overlap_sigma": 3.0,
    "covariance_refinement_ratio": 2.0,
    "covariance_refinement_min_radius_px": 1.5,
    "covariance_duplicate_floor": 0.10,
    "covariance_opacity_min": 0.01,
    "shuffle_covariance": False,
    "shuffle_covariance_competition": False,
    "retirement_enabled": False,
    "retirement_capacity_base": 35000,
    "retirement_capacity_per_frame": 255.0,
    "retirement_start_frame": 0,
    "retirement_min_age_frames": 100,
    "retirement_expansion_weight": 1.0,
    "retirement_score_mode": "opacity",
    "shuffle_retirement": False,
    "shuffle_evidence": False,
    "shuffle_seed": 42,
    "refill_depthcov_pool": True,
}


def validate_front_view_residual_cover_config(config=None):
    merged = deepcopy(DEFAULT_FRONT_VIEW_RESIDUAL_COVER_CONFIG)
    if config is not None:
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError(
                "Unknown FrontViewResidualCover options: {}".format(sorted(unknown))
            )
        merged.update(config)
    for key in (
        "enabled",
        "shuffle_evidence",
        "refill_depthcov_pool",
        "use_depth_visibility",
        "use_covariance_lod",
        "covariance_competition_enabled",
        "shuffle_covariance",
        "shuffle_covariance_competition",
        "retirement_enabled",
        "shuffle_retirement",
    ):
        if not isinstance(merged[key], bool):
            raise TypeError("FrontViewResidualCover.{} must be boolean".format(key))
    if merged["selection_mode"] not in (
        "utility",
        "utility_coverage",
        "pbsd_order",
        "utility_mix",
    ):
        raise ValueError(
            "FrontViewResidualCover.selection_mode is invalid"
        )
    if not 0.0 <= float(merged["utility_fraction"]) <= 1.0:
        raise ValueError("FrontViewResidualCover.utility_fraction must be in [0, 1]")
    if not isinstance(merged["pool_multiplier"], int) or merged["pool_multiplier"] < 1:
        raise ValueError("FrontViewResidualCover.pool_multiplier must be positive")
    if not 0.0 < float(merged["budget_scale"]) <= 1.0:
        raise ValueError("FrontViewResidualCover.budget_scale must be in (0, 1]")
    edges = [float(value) for value in merged["depth_edges_m"]]
    if len(edges) != 2 or not 0.0 < edges[0] < edges[1]:
        raise ValueError("FrontViewResidualCover.depth_edges_m must be increasing")
    fractions = np.asarray(merged["depth_fractions"], dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions < 0.0) or fractions.sum() <= 0.0:
        raise ValueError(
            "FrontViewResidualCover.depth_fractions must contain three nonnegative values"
        )
    merged["depth_fractions"] = (fractions / fractions.sum()).tolist()
    if not 0.0 < float(merged["initial_opacity"]) < 1.0:
        raise ValueError("FrontViewResidualCover.initial_opacity must be in (0, 1)")
    for key in (
        "support_sigma_multiplier",
        "selection_sigma_floor_px",
        "depth_tolerance_ratio",
        "confidence_power",
        "confidence_floor",
        "edge_gain_weight",
        "covariance_overlap_sigma",
        "covariance_opacity_min",
        "covariance_refinement_min_radius_px",
    ):
        if float(merged[key]) < 0.0:
            raise ValueError("FrontViewResidualCover.{} cannot be negative".format(key))
    if not 0.0 <= float(merged["confidence_floor"]) <= 1.0:
        raise ValueError("FrontViewResidualCover.confidence_floor must be in [0, 1]")
    if not 0.0 <= float(merged["behind_visibility_floor"]) <= 1.0:
        raise ValueError(
            "FrontViewResidualCover.behind_visibility_floor must be in [0, 1]"
        )
    if not 0.0 <= float(merged["area_normalization_power"]) <= 1.0:
        raise ValueError(
            "FrontViewResidualCover.area_normalization_power must be in [0, 1]"
        )
    if float(merged["edge_residual_scale"]) <= 0.0:
        raise ValueError("FrontViewResidualCover.edge_residual_scale must be positive")
    if (
        not isinstance(merged["covariance_neighbors"], int)
        or merged["covariance_neighbors"] < 1
    ):
        raise ValueError(
            "FrontViewResidualCover.covariance_neighbors must be positive"
        )
    if float(merged["covariance_refinement_ratio"]) <= 1.0:
        raise ValueError(
            "FrontViewResidualCover.covariance_refinement_ratio must exceed one"
        )
    if not 0.0 <= float(merged["covariance_duplicate_floor"]) <= 1.0:
        raise ValueError(
            "FrontViewResidualCover.covariance_duplicate_floor must be in [0, 1]"
        )
    if not 0.0 <= float(merged["covariance_opacity_min"]) <= 1.0:
        raise ValueError(
            "FrontViewResidualCover.covariance_opacity_min must be in [0, 1]"
        )
    for key in ("min_support_radius_px", "max_support_radius_px"):
        if not isinstance(merged[key], int) or merged[key] < 1:
            raise ValueError("FrontViewResidualCover.{} must be positive".format(key))
    if merged["max_support_radius_px"] < merged["min_support_radius_px"]:
        raise ValueError("FrontViewResidualCover support radii are reversed")
    if not isinstance(merged["shuffle_seed"], int):
        raise TypeError("FrontViewResidualCover.shuffle_seed must be an integer")
    if (
        not isinstance(merged["retirement_capacity_base"], int)
        or merged["retirement_capacity_base"] < 0
    ):
        raise ValueError(
            "FrontViewResidualCover.retirement_capacity_base cannot be negative"
        )
    if float(merged["retirement_capacity_per_frame"]) < 0.0:
        raise ValueError(
            "FrontViewResidualCover.retirement_capacity_per_frame cannot be negative"
        )
    if (
        not isinstance(merged["retirement_min_age_frames"], int)
        or merged["retirement_min_age_frames"] < 0
    ):
        raise ValueError(
            "FrontViewResidualCover.retirement_min_age_frames cannot be negative"
        )
    if (
        not isinstance(merged["retirement_start_frame"], int)
        or merged["retirement_start_frame"] < 0
    ):
        raise ValueError(
            "FrontViewResidualCover.retirement_start_frame cannot be negative"
        )
    if float(merged["retirement_expansion_weight"]) < 0.0:
        raise ValueError(
            "FrontViewResidualCover.retirement_expansion_weight cannot be negative"
        )
    if merged["retirement_score_mode"] not in (
        "opacity",
        "gradient",
        "gradient_opacity",
    ):
        raise ValueError("FrontViewResidualCover.retirement_score_mode is invalid")
    return merged


def _image_array(image):
    if torch.is_tensor(image):
        image = image.detach().float().cpu().numpy()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Residual-cover images must have shape HxWx3")
    return image


def _depth_bands(depths, edges):
    return np.digitize(np.asarray(depths), np.asarray(edges), right=False).astype(
        np.int8
    )


def _partition_quotas(bands, budget, fractions):
    availability = np.bincount(bands, minlength=3).astype(np.int64)
    budget = min(int(budget), int(availability.sum()))
    raw = np.asarray(fractions, dtype=np.float64) * budget
    quotas = np.floor(raw).astype(np.int64)
    remainder_order = np.argsort(-(raw - quotas), kind="stable")
    for band in remainder_order[: budget - int(quotas.sum())]:
        quotas[band] += 1
    quotas = np.minimum(quotas, availability)
    while int(quotas.sum()) < budget:
        remaining = availability - quotas
        band = int(np.argmax(remaining))
        if remaining[band] <= 0:
            break
        quotas[band] += 1
    return quotas


class FrontViewResidualCover:
    """Select fixed-budget births by their marginal rendered-residual coverage."""

    def __init__(self, config=None):
        self.config = validate_front_view_residual_cover_config(config)
        self.active_sparse_tracks = set()
        self.stats = {
            "calls": 0,
            "candidate_rows": 0,
            "sparse_rows": 0,
            "sparse_track_rejected": 0,
            "depthcov_pool_rows": 0,
            "depthcov_budget_rows": 0,
            "depthcov_selected_rows": 0,
            "positive_gain_rows": 0,
            "selected_gain_sum": 0.0,
            "selected_band_counts": [0, 0, 0],
            "covariance_overlap_rows": 0,
            "covariance_refinement_rows": 0,
            "covariance_novel_rows": 0,
            "covariance_weight_sum": 0.0,
            "covariance_competition_admissible_rows": 0,
            "covariance_competition_rejected_rows": 0,
            "covariance_competition_selected_novel_rows": 0,
            "covariance_competition_selected_refinement_rows": 0,
            "covariance_competition_selected_duplicate_rows": 0,
            "committed_rows": 0,
            "pruned_sparse_tracks": 0,
            "retirement_calls": 0,
            "retirement_rows": 0,
            "retirement_eligible_rows": 0,
            "retirement_last_capacity": 0,
            "hash_query_rows": 0,
            "hash_set_rows": 0,
        }

    @property
    def enabled(self):
        return bool(self.config["enabled"])

    def _counterfactual_support(
        self,
        uv,
        depths,
        log_scales,
        colors,
        depth_confidences,
        rendered,
        target,
        focal_px,
        map_depth=None,
        map_opacity=None,
    ):
        rendered = _image_array(rendered)
        target = _image_array(target)
        if rendered.shape != target.shape:
            raise ValueError("Rendered and target images must align")
        height, width = target.shape[:2]
        count = len(depths)
        max_radius = int(self.config["max_support_radius_px"])
        offsets = np.asarray(
            [
                (dy, dx)
                for dy in range(-max_radius, max_radius + 1)
                for dx in range(-max_radius, max_radius + 1)
            ],
            dtype=np.int32,
        )
        dy = offsets[:, 0][None, :]
        dx = offsets[:, 1][None, :]
        center_x = np.floor(uv[:, 0]).astype(np.int32)[:, None]
        center_y = np.floor(uv[:, 1]).astype(np.int32)[:, None]
        x = center_x + dx
        y = center_y + dy
        valid_pixel = (x >= 0) & (x < width) & (y >= 0) & (y < height)

        scale = np.exp(np.mean(np.asarray(log_scales), axis=1))
        sigma = float(focal_px) * scale / np.maximum(depths, 1.0e-6)
        sigma = np.maximum(
            sigma, float(self.config["selection_sigma_floor_px"])
        )
        radius = np.ceil(
            sigma * float(self.config["support_sigma_multiplier"])
        ).astype(np.int32)
        radius = np.clip(
            radius,
            int(self.config["min_support_radius_px"]),
            max_radius,
        )
        radial = dx.astype(np.float32) ** 2 + dy.astype(np.float32) ** 2
        valid_pixel &= radial <= radius[:, None].astype(np.float32) ** 2
        safe_x = np.clip(x, 0, width - 1)
        safe_y = np.clip(y, 0, height - 1)
        current = rendered[safe_y, safe_x]
        desired = target[safe_y, safe_x]

        sigma = sigma[:, None]
        alpha = float(self.config["initial_opacity"]) * np.exp(
            -0.5 * radial / (sigma**2)
        )
        if self.config["use_depth_visibility"]:
            if map_depth is None or map_opacity is None:
                raise ValueError(
                    "Depth visibility requires rendered depth and opacity maps"
                )
            if torch.is_tensor(map_depth):
                map_depth = map_depth.detach().float().cpu().numpy()
            if torch.is_tensor(map_opacity):
                map_opacity = map_opacity.detach().float().cpu().numpy()
            map_depth = np.asarray(map_depth, dtype=np.float32).squeeze()
            map_opacity = np.asarray(map_opacity, dtype=np.float32).squeeze()
            if map_depth.shape != (height, width) or map_opacity.shape != (
                height,
                width,
            ):
                raise ValueError("Depth visibility maps must align with the image")
            local_depth = map_depth[safe_y, safe_x]
            local_opacity = np.clip(map_opacity[safe_y, safe_x], 0.0, 1.0)
            behind = (
                valid_pixel
                & np.isfinite(local_depth)
                & (local_depth > 0.0)
                & (
                    depths[:, None]
                    > local_depth
                    * (1.0 + float(self.config["depth_tolerance_ratio"]))
                )
            )
            floor = float(self.config["behind_visibility_floor"])
            visibility = floor + (1.0 - floor) * (1.0 - local_opacity)
            alpha *= np.where(behind, visibility, 1.0)
        alpha *= valid_pixel
        predicted = current + alpha[..., None] * (colors[:, None, :] - current)
        before = np.mean((desired - current) ** 2, axis=-1)
        after = np.mean((desired - predicted) ** 2, axis=-1)
        gain = np.maximum(before - after, 0.0).astype(np.float32)
        edge_weight = float(self.config["edge_gain_weight"])
        if edge_weight > 0.0:
            target_gray = np.mean(target, axis=-1)
            rendered_gray = np.mean(rendered, axis=-1)
            target_dx = np.zeros_like(target_gray)
            target_dy = np.zeros_like(target_gray)
            rendered_dx = np.zeros_like(rendered_gray)
            rendered_dy = np.zeros_like(rendered_gray)
            target_dx[:, 1:] = target_gray[:, 1:] - target_gray[:, :-1]
            target_dy[1:, :] = target_gray[1:, :] - target_gray[:-1, :]
            rendered_dx[:, 1:] = rendered_gray[:, 1:] - rendered_gray[:, :-1]
            rendered_dy[1:, :] = rendered_gray[1:, :] - rendered_gray[:-1, :]
            edge_residual = np.sqrt(
                (target_dx - rendered_dx) ** 2 + (target_dy - rendered_dy) ** 2
            )
            edge_residual = np.clip(
                edge_residual / float(self.config["edge_residual_scale"]),
                0.0,
                1.0,
            )
            gain *= 1.0 + edge_weight * edge_residual[safe_y, safe_x]
        confidence = np.clip(depth_confidences, 0.0, 1.0)
        confidence = float(self.config["confidence_floor"]) + (
            1.0 - float(self.config["confidence_floor"])
        ) * confidence ** float(self.config["confidence_power"])
        gain *= confidence[:, None]
        area_power = float(self.config["area_normalization_power"])
        if area_power > 0.0:
            area = np.maximum(valid_pixel.sum(axis=1), 1).astype(np.float32)
            gain /= area[:, None] ** area_power
        gain *= valid_pixel
        pixel_ids = (safe_y * width + safe_x).astype(np.int64)
        pixel_ids[~valid_pixel] = -1
        residual_energy = np.mean((target - rendered) ** 2, axis=-1).reshape(-1)
        return pixel_ids, gain, residual_energy.astype(np.float32, copy=False)

    def _shuffle_support(self, pixel_ids, gain, bands, frame_id):
        if not self.config["shuffle_evidence"]:
            return pixel_ids, gain
        rng = np.random.default_rng(int(self.config["shuffle_seed"]) + int(frame_id))
        shuffled_ids = pixel_ids.copy()
        shuffled_gain = gain.copy()
        for band in range(3):
            rows = np.flatnonzero(bands == band)
            if len(rows) <= 1:
                continue
            source = rows[rng.permutation(len(rows))]
            shuffled_ids[rows] = pixel_ids[source]
            shuffled_gain[rows] = gain[source]
        return shuffled_ids, shuffled_gain

    @staticmethod
    def _quaternion_matrices(quaternions):
        quaternions = np.asarray(quaternions, dtype=np.float32)
        norm = np.linalg.norm(quaternions, axis=-1, keepdims=True)
        quaternions = quaternions / np.maximum(norm, 1.0e-8)
        w, x, y, z = np.moveaxis(quaternions, -1, 0)
        matrices = np.empty(quaternions.shape[:-1] + (3, 3), dtype=np.float32)
        matrices[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        matrices[..., 0, 1] = 2.0 * (x * y - z * w)
        matrices[..., 0, 2] = 2.0 * (x * z + y * w)
        matrices[..., 1, 0] = 2.0 * (x * y + z * w)
        matrices[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        matrices[..., 1, 2] = 2.0 * (y * z - x * w)
        matrices[..., 2, 0] = 2.0 * (x * z - y * w)
        matrices[..., 2, 1] = 2.0 * (y * z + x * w)
        matrices[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
        return matrices

    def _covariance_lod_weights(
        self,
        candidate_world,
        candidate_log_scales,
        global_means,
        global_scales,
        global_quaternions,
        global_opacities,
        *,
        return_equivalent_scale=False,
    ):
        from scipy.spatial import cKDTree

        candidate_world = np.asarray(candidate_world, dtype=np.float32).reshape(-1, 3)
        candidate_scale = np.exp(
            np.asarray(candidate_log_scales, dtype=np.float32)
            .reshape(len(candidate_world), -1)
            .mean(axis=1)
        )
        global_means = np.asarray(global_means, dtype=np.float32).reshape(-1, 3)
        global_scales = np.asarray(global_scales, dtype=np.float32).reshape(-1, 3)
        global_quaternions = np.asarray(global_quaternions, dtype=np.float32).reshape(-1, 4)
        global_opacities = np.asarray(global_opacities, dtype=np.float32).reshape(-1)
        count = len(candidate_world)
        if not (
            len(global_means)
            == len(global_scales)
            == len(global_quaternions)
            == len(global_opacities)
        ):
            raise ValueError("Covariance LOD Gaussian arrays must align")
        active = (
            np.isfinite(global_means).all(axis=1)
            & np.isfinite(global_scales).all(axis=1)
            & (global_scales > 0.0).all(axis=1)
            & (global_opacities >= float(self.config["covariance_opacity_min"]))
        )
        active_rows = np.flatnonzero(active)
        if count == 0 or len(active_rows) == 0:
            result = (
                np.ones((count,), dtype=np.float32),
                np.zeros((count,), dtype=np.bool_),
                np.zeros((count,), dtype=np.bool_),
            )
            if return_equivalent_scale:
                return result + (np.zeros((count,), dtype=np.float32),)
            return result

        neighbor_count = min(
            int(self.config["covariance_neighbors"]), len(active_rows)
        )
        tree = cKDTree(global_means[active_rows])
        _, local_neighbors = tree.query(candidate_world, k=neighbor_count, workers=1)
        if neighbor_count == 1:
            local_neighbors = local_neighbors[:, None]
        neighbors = active_rows[np.asarray(local_neighbors, dtype=np.int64)]
        neighbor_means = global_means[neighbors]
        neighbor_scales = global_scales[neighbors]
        rotations = self._quaternion_matrices(global_quaternions[neighbors])
        scale_covariance = np.zeros(rotations.shape, dtype=np.float32)
        scale_covariance[..., 0, 0] = neighbor_scales[..., 0] ** 2
        scale_covariance[..., 1, 1] = neighbor_scales[..., 1] ** 2
        scale_covariance[..., 2, 2] = neighbor_scales[..., 2] ** 2
        neighbor_covariance = rotations @ scale_covariance @ np.swapaxes(
            rotations, -1, -2
        )
        identity = np.eye(3, dtype=np.float32)
        combined = neighbor_covariance + (
            candidate_scale[:, None, None, None] ** 2
        ) * identity
        delta = candidate_world[:, None, :] - neighbor_means
        try:
            solved = np.linalg.solve(combined, delta[..., None])[..., 0]
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(combined) @ delta[..., None]
            solved = solved[..., 0]
        mahalanobis2 = np.sum(delta * solved, axis=-1)
        overlap = (
            np.isfinite(mahalanobis2)
            & (
                mahalanobis2
                <= float(self.config["covariance_overlap_sigma"]) ** 2
            )
        )
        best_distance = np.where(overlap, mahalanobis2, np.inf)
        best_neighbor = np.argmin(best_distance, axis=1)
        has_overlap = np.isfinite(
            best_distance[np.arange(count), best_neighbor]
        )
        best_scales = neighbor_scales[np.arange(count), best_neighbor]
        equivalent_scale = np.cbrt(np.prod(best_scales, axis=1))
        ratio = equivalent_scale / np.maximum(candidate_scale, 1.0e-8)
        refinement = has_overlap & (
            ratio >= float(self.config["covariance_refinement_ratio"])
        )
        denominator = math.log(float(self.config["covariance_refinement_ratio"]))
        refinement_strength = np.clip(
            np.log(np.maximum(ratio, 1.0)) / max(denominator, 1.0e-8),
            0.0,
            1.0,
        )
        floor = float(self.config["covariance_duplicate_floor"])
        weights = np.where(
            has_overlap,
            floor + (1.0 - floor) * refinement_strength,
            1.0,
        ).astype(np.float32)
        result = (weights, has_overlap, refinement)
        if return_equivalent_scale:
            return result + (equivalent_scale.astype(np.float32, copy=False),)
        return result

    def _covariance_competition_mask(
        self,
        *,
        depths,
        focal_px,
        overlap,
        refinement,
        equivalent_scale,
        bands,
        frame_id,
    ):
        """Admit novel or newly resolvable finer LOD births in the current view."""

        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        overlap = np.asarray(overlap, dtype=np.bool_).reshape(-1)
        refinement = np.asarray(refinement, dtype=np.bool_).reshape(-1)
        equivalent_scale = np.asarray(equivalent_scale, dtype=np.float32).reshape(-1)
        if not (
            depths.shape == overlap.shape == refinement.shape == equivalent_scale.shape
        ):
            raise ValueError("Covariance competition arrays must align")
        projected_radius = (
            float(focal_px)
            * equivalent_scale
            / np.maximum(depths, 1.0e-6)
        )
        resolvable_refinement = refinement & (
            projected_radius
            >= float(self.config["covariance_refinement_min_radius_px"])
        )
        admissible = (~overlap) | resolvable_refinement
        if self.config["shuffle_covariance_competition"] and len(admissible) > 1:
            rng = np.random.default_rng(
                int(self.config["shuffle_seed"]) + int(frame_id)
            )
            shuffled = admissible.copy()
            for band in range(3):
                rows = np.flatnonzero(np.asarray(bands) == band)
                if len(rows) > 1:
                    shuffled[rows] = admissible[rows[rng.permutation(len(rows))]]
            admissible = shuffled
        return admissible, resolvable_refinement

    @staticmethod
    def _utility_indices(scores, bands, quotas):
        selected = []
        for band in range(3):
            rows = np.flatnonzero(bands == band)
            order = rows[np.argsort(-scores[rows], kind="stable")]
            selected.extend(order[: int(quotas[band])].tolist())
        return np.asarray(selected, dtype=np.int64)

    @classmethod
    def _mixed_indices(cls, scores, bands, quotas, utility_fraction):
        priority_quotas = np.floor(
            quotas.astype(np.float64) * float(utility_fraction)
        ).astype(np.int64)
        selected = cls._utility_indices(scores, bands, priority_quotas).tolist()
        selected_mask = np.zeros((len(scores),), dtype=np.bool_)
        selected_mask[selected] = True
        counts = np.bincount(
            bands[np.asarray(selected, dtype=np.int64)], minlength=3
        )
        for index, band in enumerate(bands.tolist()):
            if selected_mask[index] or counts[band] >= quotas[band]:
                continue
            selected.append(index)
            selected_mask[index] = True
            counts[band] += 1
            if np.all(counts >= quotas):
                break
        return np.asarray(selected, dtype=np.int64)

    @staticmethod
    def _coverage_indices(pixel_ids, gains, residual_energy, bands, quotas):
        scores = gains.sum(axis=1)
        heap = [(-float(score), int(index), -1) for index, score in enumerate(scores)]
        heapq.heapify(heap)
        selected = []
        band_counts = np.zeros((3,), dtype=np.int64)
        covered = np.zeros_like(residual_energy, dtype=np.float32)
        iteration = 0
        while heap and len(selected) < int(quotas.sum()):
            negative_score, index, stamp = heapq.heappop(heap)
            band = int(bands[index])
            if band_counts[band] >= quotas[band]:
                continue
            if stamp != iteration:
                ids = pixel_ids[index]
                valid = ids >= 0
                ids = ids[valid]
                contribution = gains[index, valid]
                remaining = np.maximum(residual_energy[ids] - covered[ids], 0.0)
                marginal = float(np.minimum(contribution, remaining).sum())
                heapq.heappush(heap, (-marginal, index, iteration))
                continue
            selected.append(index)
            band_counts[band] += 1
            ids = pixel_ids[index]
            valid = ids >= 0
            ids = ids[valid]
            if len(ids):
                covered[ids] = np.minimum(
                    residual_energy[ids], covered[ids] + gains[index, valid]
                )
            iteration += 1
        return np.asarray(selected, dtype=np.int64)

    def filter_candidates(
        self,
        *,
        frame_id,
        uv,
        depths,
        world_points,
        log_scales,
        colors,
        residual_scores,
        depth_confidences,
        sparse_valid,
        track_ids,
        rendered,
        target,
        focal_px,
        depthcov_budget,
        map_depth=None,
        map_opacity=None,
        global_means=None,
        global_scales=None,
        global_quaternions=None,
        global_opacities=None,
    ):
        uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        world_points = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
        log_scales = np.asarray(log_scales, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        residual_scores = np.asarray(residual_scores, dtype=np.float32).reshape(-1)
        depth_confidences = np.asarray(depth_confidences, dtype=np.float32).reshape(-1)
        sparse_valid = np.asarray(sparse_valid, dtype=np.bool_).reshape(-1)
        track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        count = len(depths)
        aligned = (
            uv,
            world_points,
            log_scales,
            colors,
            residual_scores,
            depth_confidences,
            sparse_valid,
            track_ids,
        )
        if any(len(value) != count for value in aligned):
            raise ValueError("FrontViewResidualCover candidate arrays must align")

        accepted_sparse = []
        pending_tracks = set()
        for index in np.flatnonzero(sparse_valid):
            track_id = int(track_ids[index])
            if track_id >= 0 and (
                track_id in self.active_sparse_tracks or track_id in pending_tracks
            ):
                self.stats["sparse_track_rejected"] += 1
                continue
            accepted_sparse.append(int(index))
            if track_id >= 0:
                pending_tracks.add(track_id)

        dense_rows = np.flatnonzero(~sparse_valid)
        fixed_budget = int(
            round(float(depthcov_budget) * float(self.config["budget_scale"]))
        )
        fixed_budget = min(fixed_budget, len(dense_rows))
        selected_dense = np.empty((0,), dtype=np.int64)
        selected_scores = np.empty((0,), dtype=np.float32)
        bands = np.empty((0,), dtype=np.int8)
        covariance_weights = np.empty((0,), dtype=np.float32)
        overlap = np.empty((0,), dtype=np.bool_)
        refinement = np.empty((0,), dtype=np.bool_)
        competition_admissible = np.empty((0,), dtype=np.bool_)
        resolvable_refinement = np.empty((0,), dtype=np.bool_)
        if fixed_budget > 0:
            pixel_ids, gains, residual_energy = self._counterfactual_support(
                uv[dense_rows],
                depths[dense_rows],
                log_scales[dense_rows],
                colors[dense_rows],
                depth_confidences[dense_rows],
                rendered,
                target,
                focal_px,
                map_depth,
                map_opacity,
            )
            bands = _depth_bands(
                depths[dense_rows], self.config["depth_edges_m"]
            )
            needs_covariance = (
                self.config["use_covariance_lod"]
                or self.config["covariance_competition_enabled"]
            )
            if needs_covariance:
                if any(
                    value is None
                    for value in (
                        global_means,
                        global_scales,
                        global_quaternions,
                        global_opacities,
                    )
                ):
                    raise ValueError(
                        "Covariance LOD requires current Gaussian geometry"
                    )
                covariance_weights, overlap, refinement, equivalent_scale = (
                    self._covariance_lod_weights(
                        world_points[dense_rows],
                        log_scales[dense_rows],
                        global_means,
                        global_scales,
                        global_quaternions,
                        global_opacities,
                        return_equivalent_scale=True,
                    )
                )
            else:
                covariance_weights = np.ones((len(dense_rows),), dtype=np.float32)
                overlap = np.zeros((len(dense_rows),), dtype=np.bool_)
                refinement = np.zeros((len(dense_rows),), dtype=np.bool_)
                equivalent_scale = np.zeros((len(dense_rows),), dtype=np.float32)
            if self.config["covariance_competition_enabled"]:
                competition_admissible, resolvable_refinement = (
                    self._covariance_competition_mask(
                        depths=depths[dense_rows],
                        focal_px=focal_px,
                        overlap=overlap,
                        refinement=refinement,
                        equivalent_scale=equivalent_scale,
                        bands=bands,
                        frame_id=frame_id,
                    )
                )
            else:
                competition_admissible = np.ones(
                    (len(dense_rows),), dtype=np.bool_
                )
                resolvable_refinement = refinement.copy()
            if self.config["shuffle_covariance"] and len(dense_rows) > 1:
                rng = np.random.default_rng(
                    int(self.config["shuffle_seed"]) + int(frame_id)
                )
                shuffled_weights = covariance_weights.copy()
                for band in range(3):
                    rows = np.flatnonzero(bands == band)
                    if len(rows) > 1:
                        shuffled_weights[rows] = covariance_weights[
                            rows[rng.permutation(len(rows))]
                        ]
                covariance_weights = shuffled_weights
            if self.config["use_covariance_lod"]:
                gains *= covariance_weights[:, None]
            pixel_ids, gains = self._shuffle_support(
                pixel_ids, gains, bands, frame_id
            )
            scores = gains.sum(axis=1)
            fallback = np.maximum(residual_scores[dense_rows], 0.0) * (
                float(self.config["confidence_floor"])
                + (1.0 - float(self.config["confidence_floor"]))
                * np.clip(depth_confidences[dense_rows], 0.0, 1.0)
            )
            scores = scores + fallback * 1.0e-8
            eligible_rows = np.flatnonzero(competition_admissible)
            eligible_budget = min(fixed_budget, len(eligible_rows))
            eligible_bands = bands[eligible_rows]
            quotas = _partition_quotas(
                eligible_bands, eligible_budget, self.config["depth_fractions"]
            )
            if self.config["selection_mode"] == "pbsd_order":
                eligible_selection = np.arange(eligible_budget, dtype=np.int64)
            elif self.config["selection_mode"] == "utility_mix":
                eligible_selection = self._mixed_indices(
                    scores[eligible_rows],
                    eligible_bands,
                    quotas,
                    self.config["utility_fraction"],
                )
            elif self.config["selection_mode"] == "utility":
                eligible_selection = self._utility_indices(
                    scores[eligible_rows], eligible_bands, quotas
                )
            else:
                eligible_selection = self._coverage_indices(
                    pixel_ids[eligible_rows],
                    gains[eligible_rows],
                    residual_energy,
                    eligible_bands,
                    quotas,
                )
            if len(eligible_selection) < eligible_budget:
                used = np.zeros((len(eligible_rows),), dtype=np.bool_)
                used[eligible_selection] = True
                reserve = np.flatnonzero(~used)
                reserve = reserve[
                    np.argsort(-scores[eligible_rows[reserve]], kind="stable")
                ]
                eligible_selection = np.concatenate(
                    (
                        eligible_selection,
                        reserve[: eligible_budget - len(eligible_selection)],
                    )
                )
            local_selection = eligible_rows[eligible_selection]
            selected_dense = dense_rows[local_selection]
            selected_scores = scores[local_selection]
            selected_band_counts = np.bincount(
                bands[local_selection], minlength=3
            ).tolist()
        else:
            selected_band_counts = [0, 0, 0]

        self.stats["calls"] += 1
        self.stats["candidate_rows"] += count
        self.stats["sparse_rows"] += len(accepted_sparse)
        self.stats["depthcov_pool_rows"] += len(dense_rows)
        self.stats["depthcov_budget_rows"] += fixed_budget
        self.stats["depthcov_selected_rows"] += len(selected_dense)
        self.stats["positive_gain_rows"] += int(np.sum(selected_scores > 1.0e-8))
        self.stats["selected_gain_sum"] += float(selected_scores.sum())
        self.stats["covariance_overlap_rows"] += int(np.sum(overlap))
        self.stats["covariance_refinement_rows"] += int(np.sum(refinement))
        self.stats["covariance_novel_rows"] += int(np.sum(~overlap))
        self.stats["covariance_weight_sum"] += float(covariance_weights.sum())
        self.stats["covariance_competition_admissible_rows"] += int(
            np.sum(competition_admissible)
        )
        self.stats["covariance_competition_rejected_rows"] += int(
            len(competition_admissible) - np.sum(competition_admissible)
        )
        if len(selected_dense):
            selected_local = local_selection
            self.stats["covariance_competition_selected_novel_rows"] += int(
                np.sum(~overlap[selected_local])
            )
            self.stats[
                "covariance_competition_selected_refinement_rows"
            ] += int(np.sum(resolvable_refinement[selected_local]))
            self.stats["covariance_competition_selected_duplicate_rows"] += int(
                np.sum(
                    overlap[selected_local]
                    & ~resolvable_refinement[selected_local]
                )
            )
        self.stats["selected_band_counts"] = [
            int(old + new)
            for old, new in zip(
                self.stats["selected_band_counts"], selected_band_counts
            )
        ]
        return np.asarray(accepted_sparse + selected_dense.tolist(), dtype=np.int64)

    def prepare_commit(self, proposals):
        keep = []
        pending_tracks = set()
        for index, track_id in enumerate(np.asarray(proposals.track_ids, dtype=np.int64)):
            track_id = int(track_id)
            if track_id >= 0 and (
                track_id in self.active_sparse_tracks or track_id in pending_tracks
            ):
                continue
            keep.append(index)
            if track_id >= 0:
                pending_tracks.add(track_id)
        return np.asarray(keep, dtype=np.int64)

    def mark_committed(self, proposals):
        tracks = np.asarray(proposals.track_ids, dtype=np.int64)
        self.active_sparse_tracks.update(int(value) for value in tracks if value >= 0)
        self.stats["committed_rows"] += len(proposals)

    def release(self, track_ids):
        released = 0
        for value in np.asarray(track_ids, dtype=np.int64).reshape(-1):
            if int(value) in self.active_sparse_tracks:
                self.active_sparse_tracks.remove(int(value))
                released += 1
        self.stats["pruned_sparse_tracks"] += released

    def record_retirement(self, *, capacity, eligible, retired):
        self.stats["retirement_calls"] += 1
        self.stats["retirement_rows"] += int(retired)
        self.stats["retirement_eligible_rows"] += int(eligible)
        self.stats["retirement_last_capacity"] = int(capacity)

    def summary(self):
        result = deepcopy(self.stats)
        result["enabled"] = self.enabled
        result["selection_mode"] = self.config["selection_mode"]
        result["shuffle_evidence"] = bool(self.config["shuffle_evidence"])
        result["use_covariance_lod"] = bool(self.config["use_covariance_lod"])
        result["covariance_competition_enabled"] = bool(
            self.config["covariance_competition_enabled"]
        )
        result["shuffle_covariance"] = bool(self.config["shuffle_covariance"])
        result["shuffle_covariance_competition"] = bool(
            self.config["shuffle_covariance_competition"]
        )
        result["active_sparse_tracks"] = len(self.active_sparse_tracks)
        result["hash_calls_zero"] = (
            result["hash_query_rows"] == 0 and result["hash_set_rows"] == 0
        )
        return result
