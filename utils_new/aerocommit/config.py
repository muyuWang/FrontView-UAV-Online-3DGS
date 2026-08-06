"""Configuration defaults and validation for AeroCommit-MVP."""

from copy import deepcopy
from typing import Any, Dict, Mapping


DEFAULT_AEROCOMMIT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "mode": "baseline",
    "bootstrap_frames": 5,
    "diagnostic_allow_unsafe_hybrid": False,
    "admission": {
        "policy": "npo_lite",
        "trusted_sparse_fast_path": False,
        "trusted_depthcov_fast_path": False,
        "allow_depthcov_candidates": True,
        "depthcov_candidate_stable_depth_ratio": 0.0,
        "fast_path_max_gaussians_per_frame": 0,
        "fast_path_frequency_fraction": 0.5,
        "fast_path_group_frames": 0,
        "fast_path_initial_opacity": 0.5,
        "trusted_depth_confidence_threshold": 0.70,
        "frequency_gate_enabled": False,
        "frequency_gate_sparse": True,
        "frequency_gate_score_threshold": 1.0,
        "trusted_frequency_depth_confidence_threshold": 0.70,
        "frequency_probation_enabled": False,
        "frequency_probation_initial_opacity": 0.15,
        "frequency_probation_max_opacity": 0.45,
        "frequency_candidate_enabled": False,
        "frequency_candidate_score_threshold": 0.65,
        "frequency_candidate_min_support": 2,
        "frequency_candidate_min_parallax_deg": 0.10,
        "frequency_candidate_risk_threshold": 0.20,
        "gate_interval": 3,
        "max_risk_candidates_per_keyframe": 32,
        "min_support": 3,
        "max_support_edges": 4,
        "patch_size": 8,
        "descriptor_resize": 4,
        "max_patch_samples": 16,
        "candidate_group_cell_px": 48,
        "candidate_group_log_depth_bin": 0.25,
        "max_proposals_per_candidate": 64,
        "fuse_support_proposals": False,
        "commit_snapshot_policy": "reference",
        "max_fused_proposals_per_candidate": 128,
        "fusion_relative_depth_threshold": 0.35,
        "fusion_voxel_scale_ratio": 0.35,
        "side_fusion_capacity_multiplier": 1.5,
        "max_candidate_groups_per_frame": 700,
        "max_candidate_bank_size": 3000,
        "candidate_max_age": 60,
        "association_radius_px": 16.0,
        "association_descriptor_threshold": 0.40,
        "association_residual_threshold": 0.35,
        "association_relative_depth_threshold": 0.50,
        "descriptor_ema": 0.15,
        "candidate_ema": 0.20,
        "fixed_delay_frames": 2,
        "posterior_variance_threshold": 0.02,
        "residual_threshold": 0.25,
        "pose_rotation_sigma_deg": 0.5,
        "pose_translation_sigma_scene_units": 0.01,
        "pose_covariance_mode": "fixed_diagonal",
        "finite_difference_rho_relative_step": 0.01,
        "huber_delta": 0.05,
        "lambda_num": 1.0e-6,
        "curvature_margin": 0.0,
        "min_information": 1.0e-8,
        "noise_weight": 1.0,
        "association_weight": 0.5,
        "pose_weight": 0.5,
        "risk_threshold": 0.20,
        "minimum_sanity_parallax_deg": 0.20,
        "side_band_start": 0.65,
        "side_quota_fraction": 0.35,
        "side_priority_weight": 0.75,
    },
    "commit_refinement": {
        "enabled": True,
        "max_commits_per_keyframe": 64,
        "iterations": 10,
        "learning_rate": 0.05,
        "max_depth_correction_ratio": 0.20,
        "color_fusion_strength": 0.35,
        "initial_opacity": 0.20,
        "force_new_group": True,
        "group_chunk_keyframes": 4,
        "newborn_optimization_iters": 0,
        "newborn_multiview": False,
    },
    "detail_refinement": {
        "enabled": True,
        "refine_min_views": 3,
        "refine_min_projected_radius_px": 4.0,
        "refine_min_stable_residual": 0.06,
        "child_count": 4,
        "child_scale_ratio": 0.55,
        "max_splits_per_keyframe": 64,
        "side_score_boost": 2.0,
    },
    "track_detail": {
        "enabled": False,
        "mode": "carrier",
        "track_quantization": 1.0e-4,
        "min_support_views": 3,
        "max_observations_per_track": 12,
        "gradient_threshold": 0.035,
        "side_start": 0.35,
        "near_depth_m": 60.0,
        "color_mad_threshold": 0.12,
        "projected_scale_px": 0.60,
        "initial_opacity": 0.35,
        "max_scale_expansion": 1.25,
        "max_commits_per_frame": 192,
        "max_total_gaussians": 12000,
        "side_score_boost": 2.0,
        "near_score_boost": 1.0,
        "freeze_geometry": True,
        "reassign_max_distance": 0.50,
        "reassign_scale_multiplier": 1.25,
        "reassign_color_blend": 0.50,
        "reassign_mean_blend": 0.25,
        "reassign_opacity_floor": 0.25,
    },
    "surface_detail": {
        "enabled": False,
        "depth_source": "stable",
        "precomputed_depth_directory": None,
        "depth_confidence_threshold": 0.55,
        "stable_depth_consistency_ratio": 0.50,
        "gradient_threshold": 0.035,
        "residual_threshold": 0.05,
        "opacity_threshold": 0.65,
        "side_start": 0.35,
        "near_depth_m": 60.0,
        "voxel_size": 0.05,
        "projected_scale_px": 0.60,
        "initial_opacity": 0.35,
        "max_scale_expansion": 1.25,
        "max_commits_per_keyframe": 384,
        "max_total_gaussians": 10000,
        "side_score_boost": 2.0,
        "freeze_geometry": True,
    },
    "flow_detail": {
        "enabled": False,
        "start_frame": 20,
        "track_views": 3,
        "side_start": 0.35,
        "vertical_start": 0.20,
        "near_depth_m": 60.0,
        "max_corners": 2500,
        "quality_level": 0.005,
        "min_corner_distance_px": 3.0,
        "corner_block_size": 5,
        "lk_window_size": 31,
        "lk_max_level": 4,
        "lk_iterations": 30,
        "lk_epsilon": 0.01,
        "forward_backward_threshold_px": 1.0,
        "min_parallax_deg": 0.20,
        "max_parallax_deg": 10.0,
        "reprojection_threshold_px": 2.0,
        "color_consistency_threshold": 0.20,
        "voxel_size": 0.05,
        "projected_scale_px": 0.60,
        "initial_opacity": 0.30,
        "max_scale_expansion": 1.25,
        "max_commits_per_frame": 384,
        "max_total_gaussians": 10000,
        "freeze_geometry": True,
    },
    "budget": {
        "enabled": True,
        "max_candidate_bank_size": 3000,
        "max_active_trainable_gaussians": 350000,
        "max_active_trainable_bytes": None,
        "archive_after_unseen_frames": 40,
    },
    "archive": {
        "enabled": True,
        "cpu_dtype": "float16",
        "enable_reactivation": True,
        "archive_directory": None,
    },
    "debug": {
        "enabled": True,
        "save_interval": 10,
        "log_candidate_stats": True,
        "log_memory_breakdown": True,
    },
}


def _merge_known(default: Mapping[str, Any], user: Mapping[str, Any], path: str):
    unknown = set(user) - set(default)
    if unknown:
        raise ValueError("Unknown {} options: {}".format(path, sorted(unknown)))
    merged = deepcopy(dict(default))
    for key, value in user.items():
        if isinstance(default[key], Mapping):
            if not isinstance(value, Mapping):
                raise TypeError("{}.{} must be a mapping".format(path, key))
            merged[key] = _merge_known(default[key], value, "{}.{}".format(path, key))
        else:
            merged[key] = value
    return merged


def validate_aerocommit_config(config: Mapping[str, Any] = None) -> Dict[str, Any]:
    merged = _merge_known(
        DEFAULT_AEROCOMMIT_CONFIG, config or {}, "AeroCommit"
    )
    if merged["mode"] not in ("baseline", "npo_gate", "aerocommit_mvp"):
        raise ValueError("AeroCommit.mode must be baseline, npo_gate, or aerocommit_mvp")
    policies = {
        "immediate",
        "fixed_delay",
        "parallax",
        "posterior_variance",
        "residual",
        "depth_confidence",
        "cheap_hessian_no_pose",
        "npo_lite",
    }
    if merged["admission"]["policy"] not in policies:
        raise ValueError("Unsupported AeroCommit admission policy")
    for key in (
        "gate_interval",
        "max_risk_candidates_per_keyframe",
        "min_support",
        "max_support_edges",
        "patch_size",
        "descriptor_resize",
        "max_patch_samples",
        "candidate_group_cell_px",
        "max_proposals_per_candidate",
        "max_fused_proposals_per_candidate",
        "max_candidate_groups_per_frame",
        "max_candidate_bank_size",
        "candidate_max_age",
        "frequency_candidate_min_support",
    ):
        if int(merged["admission"][key]) <= 0:
            raise ValueError("AeroCommit.admission.{} must be positive".format(key))
    for key in (
        "fast_path_max_gaussians_per_frame",
        "fast_path_group_frames",
    ):
        if int(merged["admission"][key]) < 0:
            raise ValueError(
                "AeroCommit.admission.{} cannot be negative".format(key)
            )
    fast_path_frequency_fraction = float(
        merged["admission"]["fast_path_frequency_fraction"]
    )
    if not 0.0 <= fast_path_frequency_fraction <= 1.0:
        raise ValueError(
            "AeroCommit fast path frequency fraction must be in [0, 1]"
        )
    for key in (
        "fusion_relative_depth_threshold",
        "fusion_voxel_scale_ratio",
        "side_fusion_capacity_multiplier",
    ):
        if float(merged["admission"][key]) <= 0.0:
            raise ValueError("AeroCommit.admission.{} must be positive".format(key))
    stable_depth_ratio = float(
        merged["admission"]["depthcov_candidate_stable_depth_ratio"]
    )
    if not 0.0 <= stable_depth_ratio <= 1.0:
        raise ValueError(
            "AeroCommit stable-depth candidate ratio must be in [0, 1]"
        )
    if int(merged["commit_refinement"]["group_chunk_keyframes"]) <= 0:
        raise ValueError("AeroCommit commit group chunk must be positive")
    if int(merged["commit_refinement"]["newborn_optimization_iters"]) < 0:
        raise ValueError("AeroCommit newborn optimization iterations cannot be negative")
    if not 0.0 <= float(merged["commit_refinement"]["max_depth_correction_ratio"]) < 1.0:
        raise ValueError("AeroCommit max depth correction ratio must be in [0, 1)")
    if not 0.0 <= float(merged["commit_refinement"]["color_fusion_strength"]) <= 1.0:
        raise ValueError("AeroCommit color fusion strength must be in [0, 1]")
    if merged["admission"]["max_support_edges"] > 4:
        raise ValueError("AeroCommit-MVP retains at most four support edges")
    if merged["admission"]["commit_snapshot_policy"] not in (
        "reference",
        "latest_consistent",
    ):
        raise ValueError("Unsupported AeroCommit commit snapshot policy")
    if merged["detail_refinement"]["child_count"] != 4:
        raise ValueError("AeroCommit-MVP detail split requires four children")
    for key in (
        "min_support_views",
        "max_observations_per_track",
        "max_commits_per_frame",
        "max_total_gaussians",
    ):
        if int(merged["track_detail"][key]) <= 0:
            raise ValueError("AeroCommit.track_detail.{} must be positive".format(key))
    for key in (
        "track_quantization",
        "gradient_threshold",
        "near_depth_m",
        "projected_scale_px",
        "max_scale_expansion",
    ):
        if float(merged["track_detail"][key]) <= 0.0:
            raise ValueError("AeroCommit.track_detail.{} must be positive".format(key))
    for key in ("side_start", "initial_opacity"):
        value = float(merged["track_detail"][key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "AeroCommit.track_detail.{} must be in [0, 1]".format(key)
            )
    if merged["track_detail"]["mode"] not in ("carrier", "reassign"):
        raise ValueError("AeroCommit.track_detail.mode must be carrier or reassign")
    for key in (
        "reassign_color_blend",
        "reassign_mean_blend",
        "reassign_opacity_floor",
    ):
        value = float(merged["track_detail"][key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "AeroCommit.track_detail.{} must be in [0, 1]".format(key)
            )
    for key in (
        "gradient_threshold",
        "residual_threshold",
        "opacity_threshold",
        "near_depth_m",
        "voxel_size",
        "projected_scale_px",
        "max_scale_expansion",
    ):
        if float(merged["surface_detail"][key]) <= 0.0:
            raise ValueError("AeroCommit.surface_detail.{} must be positive".format(key))
    for key in ("side_start", "initial_opacity", "opacity_threshold"):
        value = float(merged["surface_detail"][key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "AeroCommit.surface_detail.{} must be in [0, 1]".format(key)
            )
    for key in ("max_commits_per_keyframe", "max_total_gaussians"):
        if int(merged["surface_detail"][key]) <= 0:
            raise ValueError("AeroCommit.surface_detail.{} must be positive".format(key))
    if merged["surface_detail"]["depth_source"] not in (
        "stable",
        "depthcov",
        "precomputed",
    ):
        raise ValueError("AeroCommit.surface_detail.depth_source is invalid")
    if (
        merged["surface_detail"]["depth_source"] == "precomputed"
        and not merged["surface_detail"]["precomputed_depth_directory"]
    ):
        raise ValueError("AeroCommit precomputed depth directory is required")
    if not 0.0 <= float(
        merged["surface_detail"]["depth_confidence_threshold"]
    ) <= 1.0:
        raise ValueError("AeroCommit surface detail confidence must be in [0, 1]")
    for key in (
        "max_corners",
        "corner_block_size",
        "lk_window_size",
        "lk_max_level",
        "lk_iterations",
        "max_commits_per_frame",
        "max_total_gaussians",
        "track_views",
    ):
        if int(merged["flow_detail"][key]) <= 0:
            raise ValueError("AeroCommit.flow_detail.{} must be positive".format(key))
    for key in (
        "near_depth_m",
        "quality_level",
        "min_corner_distance_px",
        "lk_epsilon",
        "forward_backward_threshold_px",
        "min_parallax_deg",
        "max_parallax_deg",
        "reprojection_threshold_px",
        "color_consistency_threshold",
        "voxel_size",
        "projected_scale_px",
        "max_scale_expansion",
    ):
        if float(merged["flow_detail"][key]) <= 0.0:
            raise ValueError("AeroCommit.flow_detail.{} must be positive".format(key))
    if float(merged["flow_detail"]["max_parallax_deg"]) <= float(
        merged["flow_detail"]["min_parallax_deg"]
    ):
        raise ValueError("AeroCommit.flow_detail parallax bounds are invalid")
    if int(merged["flow_detail"]["track_views"]) < 3:
        raise ValueError("AeroCommit.flow_detail.track_views must be at least three")
    if merged["archive"]["cpu_dtype"] != "float16":
        raise ValueError("AeroCommit-MVP archive dtype must be float16")
    for key in ("side_band_start", "side_quota_fraction"):
        value = float(merged["admission"][key])
        if not 0.0 <= value <= 1.0:
            raise ValueError("AeroCommit.admission.{} must be in [0, 1]".format(key))
    confidence_threshold = float(
        merged["admission"]["trusted_depth_confidence_threshold"]
    )
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("AeroCommit trusted depth confidence must be in [0, 1]")
    for key in (
        "fast_path_initial_opacity",
        "frequency_gate_score_threshold",
        "trusted_frequency_depth_confidence_threshold",
        "frequency_probation_initial_opacity",
        "frequency_probation_max_opacity",
        "frequency_candidate_score_threshold",
        "frequency_candidate_risk_threshold",
    ):
        value = float(merged["admission"][key])
        if not 0.0 <= value <= 1.0:
            raise ValueError("AeroCommit {} must be in [0, 1]".format(key))
    if (
        merged["admission"]["frequency_gate_enabled"]
        or merged["admission"]["frequency_probation_enabled"]
    ) and float(
        merged["admission"]["trusted_frequency_depth_confidence_threshold"]
    ) < confidence_threshold:
        raise ValueError(
            "AeroCommit trusted frequency depth confidence cannot be below the base threshold"
        )
    if float(
        merged["admission"]["frequency_probation_max_opacity"]
    ) < float(merged["admission"]["frequency_probation_initial_opacity"]):
        raise ValueError(
            "AeroCommit frequency probation max opacity cannot be below its initial opacity"
        )
    if float(merged["admission"]["frequency_candidate_min_parallax_deg"]) < 0.0:
        raise ValueError("AeroCommit frequency candidate parallax cannot be negative")
    return merged
