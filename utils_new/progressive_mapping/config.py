"""Configuration defaults and validation for the progressive mapping MVP."""

from copy import deepcopy
from typing import Any, Dict, Mapping, Tuple


DEFAULT_PROGRESSIVE_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "bootstrap_frames": 5,
    "replace_original_densification_after_bootstrap": True,
    "process_keyframes_only": False,
    "process_frame_interval": 1,
    "admission_only": False,
    "use_keyframes_for_spawn": True,
    "spawn_interval": 3,
    "patch_size": 16,
    "patch_stride": 16,
    "descriptor_resize": 4,
    "appearance_grid_size": 8,
    "max_observations_per_frame": 1024,
    "max_new_anchors_per_keyframe": 256,
    "near_observation_depth_m": 30.0,
    "near_observation_fraction": 0.60,
    "near_spawn_fraction": 0.75,
    "spawn_requires_valid_depth": False,
    "candidate_opacity_threshold": 0.15,
    "candidate_residual_threshold": 0.15,
    "candidate_min_gradient": 0.01,
    "num_inverse_depth_modes": 4,
    "inverse_depth_log_offsets": [-0.7, -0.25, 0.25, 0.7],
    "association_radius_px": 12.0,
    "association_feature_threshold": 0.35,
    "association_photo_weight": 0.25,
    "association_pixel_weight": 0.10,
    "association_temperature": 0.10,
    "minimum_probability_floor": 1.0e-4,
    "descriptor_ema": 0.10,
    "color_ema": 0.10,
    "residual_ema": 0.10,
    "promotion_min_observations": 3,
    "promotion_min_best_weight": 0.70,
    "promotion_max_normalized_entropy": 0.60,
    "promotion_max_relative_std": 0.20,
    "promotion_min_parallax_deg": 1.0,
    "promotion_max_match_error": 0.30,
    "near_promotion_max_depth_m": 30.0,
    "near_promotion_min_observations": 2,
    "near_promotion_min_best_weight": 0.50,
    "near_promotion_max_normalized_entropy": 0.95,
    "near_promotion_max_relative_std": 0.60,
    "near_promotion_min_parallax_deg": 0.20,
    "near_promotion_max_match_error": 0.45,
    # This is an empirical admission score, not an identifiability certificate.
    "commitment_score_enabled": False,
    "commitment_min_valid_updates": 2,
    "commitment_parallax_reference_deg": 1.0,
    "commitment_relative_std_reference": 0.25,
    "commitment_match_error_reference": 0.25,
    "commitment_score_threshold": 0.0,
    "near_inverse_depth_log_offsets": [-0.20, -0.05, 0.05, 0.20],
    "enable_sparse_plane_initialization": False,
    "sparse_plane_min_points": 5,
    "sparse_plane_depth_mad_scale": 3.0,
    "sparse_plane_min_relative_depth_band": 0.10,
    "sparse_plane_max_relative_rmse": 0.01,
    "sparse_plane_min_confidence": 0.70,
    "sparse_plane_min_uv_variance": 4.0,
    "sparse_plane_max_tilt_deg": 80.0,
    "sparse_plane_max_scale_multiplier": 1.25,
    "metric_merge_radius_factor": 0.75,
    "metric_merge_feature_threshold": 0.80,
    "metric_initial_opacity": 0.10,
    "metric_scale_factor": 1.0,
    "enable_metric_depth_correction": False,
    "metric_correction_min_depth_support": 2,
    "metric_correction_max_pixel_error_px": 12.0,
    "metric_correction_max_feature_error": 0.30,
    "metric_correction_max_relative_depth_error": 0.45,
    "metric_correction_max_relative_uncertainty": 0.15,
    "metric_correction_ema": 0.50,
    "metric_correction_min_updates_for_refine": 1,
    "metric_correction_max_wait_frames": 2,
    "surface_initial_opacity_floor": 0.0,
    "surface_opacity_min": 0.0,
    "refine_min_observations": 4,
    "refine_min_projected_radius_px": 12.0,
    "refine_min_residual": 0.08,
    "refine_min_confidence": 0.50,
    "children_per_root": 4,
    "child_scale_ratio": 0.55,
    "near_surface_depth_m": 50.0,
    "very_near_surface_depth_m": 25.0,
    "near_surface_children": 9,
    "very_near_surface_children": 16,
    "near_child_scale_ratio": 0.38,
    "very_near_child_scale_ratio": 0.30,
    "surface_scale_min_factor": 0.50,
    "surface_scale_max_factor": 1.50,
    "surface_max_projected_sigma_px": 6.0,
    "near_surface_projected_radius_px": 24.0,
    "very_near_surface_projected_radius_px": 48.0,
    "very_near_surface_min_residual": 0.12,
    "depth_histogram_edges_m": [20.0, 50.0],
    "enable_center_regularization": False,
    "center_regularization_weight": 0.01,
    "center_regularization_allowed_factor": 1.5,
    "optimize_visible_roots_only": False,
    "optimization_visibility_margin_px": 32.0,
    "current_view_optimization_fraction": 0.0,
    "newborn_optimization_iters": 0,
    "max_promotions_per_frame": 0,
    "max_refinements_per_frame": 0,
    "max_optimized_roots_per_step": 0,
    "means_lr_multiplier": 1.0,
    "scales_lr_multiplier": 1.0,
    "quats_lr_multiplier": 1.0,
    "appearance_lr_multiplier": 1.0,
    "post_refinement_optimize_progressive": True,
    "post_refinement_merge_into_baseline": False,
    "archive_after_unseen_frames": 30,
    "archive_dtype": "float16",
    "archive_to_cpu": True,
    "enable_reactivation": True,
    "reactivate_min_projected_radius_px": 10.0,
    "max_projective_anchors": 5000,
    "projective_prune_grace_frames": 20,
    "projective_prune_stale_frames": 40,
    "max_metric_roots": 50000,
    "max_surface_gaussians": 300000,
    "max_active_gaussians": 400000,
    "projective_top_k_render_modes": 2,
    "projective_base_opacity": 0.08,
    "debug": True,
    "debug_save_interval": 10,
}


_EXPECTED_TYPES: Dict[str, Tuple[type, ...]] = {
    key: (bool,) if isinstance(value, bool) else (int,) if isinstance(value, int) else (float, int)
    if isinstance(value, float)
    else (list, tuple)
    if isinstance(value, list)
    else (str,)
    for key, value in DEFAULT_PROGRESSIVE_CONFIG.items()
}


def validate_progressive_config(config: Mapping[str, Any] = None) -> Dict[str, Any]:
    """Merge user values with defaults and reject invalid MVP configurations."""
    merged = deepcopy(DEFAULT_PROGRESSIVE_CONFIG)
    if config is not None:
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError("Unknown ProgressiveMapping options: {}".format(sorted(unknown)))
        merged.update(dict(config))

    for key, expected in _EXPECTED_TYPES.items():
        value = merged[key]
        if not isinstance(value, expected) or (bool not in expected and isinstance(value, bool)):
            raise TypeError(
                "ProgressiveMapping.{} must have type {}, got {}".format(
                    key, "/".join(t.__name__ for t in expected), type(value).__name__
                )
            )

    positive_ints = (
        "spawn_interval",
        "process_frame_interval",
        "patch_size",
        "patch_stride",
        "descriptor_resize",
        "appearance_grid_size",
        "max_observations_per_frame",
        "near_promotion_min_observations",
        "commitment_min_valid_updates",
        "sparse_plane_min_points",
        "metric_correction_min_depth_support",
        "num_inverse_depth_modes",
        "children_per_root",
        "near_surface_children",
        "very_near_surface_children",
        "debug_save_interval",
        "newborn_optimization_iters",
        "max_promotions_per_frame",
        "max_refinements_per_frame",
        "max_optimized_roots_per_step",
        "metric_correction_min_updates_for_refine",
        "metric_correction_max_wait_frames",
        "projective_prune_grace_frames",
        "projective_prune_stale_frames",
    )
    for key in positive_ints:
        if key in (
            "newborn_optimization_iters",
            "max_promotions_per_frame",
            "max_refinements_per_frame",
            "max_optimized_roots_per_step",
            "metric_correction_min_updates_for_refine",
            "metric_correction_max_wait_frames",
        ):
            if merged[key] < 0:
                raise ValueError("ProgressiveMapping.{} must be non-negative".format(key))
            continue
        if merged[key] <= 0:
            raise ValueError("ProgressiveMapping.{} must be positive".format(key))
    if merged["num_inverse_depth_modes"] != 4:
        raise ValueError("The MVP requires exactly four inverse-depth modes")
    if len(merged["inverse_depth_log_offsets"]) != merged["num_inverse_depth_modes"]:
        raise ValueError("inverse_depth_log_offsets must match num_inverse_depth_modes")
    if len(merged["near_inverse_depth_log_offsets"]) != merged["num_inverse_depth_modes"]:
        raise ValueError("near_inverse_depth_log_offsets must match num_inverse_depth_modes")
    if merged["minimum_probability_floor"] * merged["num_inverse_depth_modes"] >= 1.0:
        raise ValueError("minimum_probability_floor must be below 1 / num_inverse_depth_modes")
    for key in (
        "children_per_root",
        "near_surface_children",
        "very_near_surface_children",
    ):
        grid_size = int(round(merged[key] ** 0.5))
        if grid_size * grid_size != merged[key]:
            raise ValueError("ProgressiveMapping.{} must be a square grid size".format(key))
    if merged["archive_dtype"] != "float16":
        raise ValueError("The MVP archive_dtype must be float16")
    if not merged["archive_to_cpu"]:
        raise ValueError("The MVP requires archive_to_cpu=true")
    if merged["projective_prune_stale_frames"] < merged["projective_prune_grace_frames"]:
        raise ValueError(
            "projective_prune_stale_frames must be greater than or equal to "
            "projective_prune_grace_frames"
        )
    for key in (
        "candidate_opacity_threshold",
        "candidate_residual_threshold",
        "association_feature_threshold",
        "minimum_probability_floor",
        "metric_initial_opacity",
        "projective_base_opacity",
        "surface_initial_opacity_floor",
        "surface_opacity_min",
        "current_view_optimization_fraction",
        "near_observation_fraction",
        "near_spawn_fraction",
        "near_promotion_min_best_weight",
        "near_promotion_max_normalized_entropy",
        "sparse_plane_min_confidence",
        "metric_correction_max_feature_error",
        "metric_correction_ema",
    ):
        if not 0.0 <= float(merged[key]) <= 1.0:
            raise ValueError("ProgressiveMapping.{} must be in [0, 1]".format(key))
    positive_values = (
        "near_observation_depth_m",
        "near_promotion_max_depth_m",
        "near_surface_depth_m",
        "very_near_surface_depth_m",
        "near_child_scale_ratio",
        "very_near_child_scale_ratio",
        "surface_scale_min_factor",
        "surface_scale_max_factor",
        "surface_max_projected_sigma_px",
        "near_surface_projected_radius_px",
        "very_near_surface_projected_radius_px",
        "sparse_plane_depth_mad_scale",
        "sparse_plane_min_relative_depth_band",
        "sparse_plane_max_relative_rmse",
        "sparse_plane_min_uv_variance",
        "sparse_plane_max_scale_multiplier",
        "metric_correction_max_pixel_error_px",
        "commitment_parallax_reference_deg",
        "commitment_relative_std_reference",
        "commitment_match_error_reference",
    )
    for key in positive_values:
        if float(merged[key]) <= 0.0:
            raise ValueError("ProgressiveMapping.{} must be positive".format(key))
    for key in (
        "near_promotion_max_relative_std",
        "near_promotion_min_parallax_deg",
        "near_promotion_max_match_error",
        "very_near_surface_min_residual",
        "optimization_visibility_margin_px",
        "means_lr_multiplier",
        "scales_lr_multiplier",
        "quats_lr_multiplier",
        "appearance_lr_multiplier",
        "sparse_plane_max_tilt_deg",
        "metric_correction_max_relative_depth_error",
        "metric_correction_max_relative_uncertainty",
        "commitment_score_threshold",
    ):
        if float(merged[key]) < 0.0:
            raise ValueError("ProgressiveMapping.{} must be non-negative".format(key))
    if merged["very_near_surface_depth_m"] > merged["near_surface_depth_m"]:
        raise ValueError("very_near_surface_depth_m must not exceed near_surface_depth_m")
    if merged["surface_scale_min_factor"] > merged["surface_scale_max_factor"]:
        raise ValueError(
            "surface_scale_min_factor must not exceed surface_scale_max_factor"
        )
    if merged["very_near_surface_children"] < merged["near_surface_children"]:
        raise ValueError("very_near_surface_children must be >= near_surface_children")
    if merged["near_surface_children"] < merged["children_per_root"]:
        raise ValueError("near_surface_children must be >= children_per_root")
    if float(merged["sparse_plane_max_tilt_deg"]) > 90.0:
        raise ValueError("sparse_plane_max_tilt_deg must not exceed 90 degrees")
    depth_edges = [float(value) for value in merged["depth_histogram_edges_m"]]
    if len(depth_edges) != 2 or depth_edges[0] <= 0.0 or depth_edges[1] <= depth_edges[0]:
        raise ValueError("depth_histogram_edges_m must contain two increasing positive values")
    return merged
