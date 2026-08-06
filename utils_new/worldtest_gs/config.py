"""Configuration and validation for Canonical WorldTest-GS."""

from __future__ import annotations

import math
from copy import deepcopy


DEFAULT_WORLDTEST_CONFIG = {
    "enabled": False,
    "admission_mode": "true_qg",
    "allow_invalid_world_stress": False,
    "calibration_version": "panoair_v1",
    "qg_threshold": math.log(19.0),
    "min_views": 3,
    "max_views": 8,
    "max_groups": 12000,
    "max_evaluations_per_frame": 128,
    "max_commits_per_frame": 128,
    "freeze_committed_means": False,
    "committed_mean_trust_radius_m": None,
    "offline_sparse_track_cache": True,
    "offline_cache_frames": 200,
    "candidate_max_age": 60,
    "flow_cycle_threshold_px": 1.5,
    "minimum_parallax_deg": 0.2,
    "rank_ratio_threshold": 1.0e-6,
    "pixel_sigma": 1.0,
    "inverse_depth_sigma_floor": 1.0e-3,
    "pose_rotation_sigma_deg": 0.5,
    "pose_translation_sigma_m": 0.01,
    "nuisance_window_frames": 16,
    "nuisance_update_interval": 4,
    "nuisance_knot_stride": 4,
    "nuisance_huber_delta": 3.0,
    "nuisance_damping": 1.0e-6,
    "prior_scales": [0.5, 1.0, 2.0],
    "scene_prior_near": 0.05,
    "scene_prior_far": 120.0,
    "shadow_alpha_cap": 0.1,
    "schedule_path": None,
    "random_seed": 42,
    "debug": True,
}


def validate_worldtest_config(config=None):
    user = config or {}
    unknown = set(user) - set(DEFAULT_WORLDTEST_CONFIG)
    if unknown:
        raise ValueError("Unknown WorldTestGS options: {}".format(sorted(unknown)))
    merged = deepcopy(DEFAULT_WORLDTEST_CONFIG)
    merged.update(user)
    modes = {
        "true_qg",
        "matched_delay",
        "equal_count_random",
        "shuffled_qg",
        "npo_lite",
    }
    if merged["admission_mode"] not in modes:
        raise ValueError("Unsupported WorldTestGS admission mode")
    if int(merged["min_views"]) < 3:
        raise ValueError("WorldTestGS requires at least three support views")
    if int(merged["max_views"]) < int(merged["min_views"]):
        raise ValueError("WorldTestGS max_views must be >= min_views")
    if float(merged["qg_threshold"]) <= 0.0:
        raise ValueError("WorldTestGS q_g threshold must be positive")
    if sorted(float(value) for value in merged["prior_scales"]) != [0.5, 1.0, 2.0]:
        raise ValueError("WorldTestGS prior scales must be exactly 0.5, 1, 2")
    if not 0.0 < float(merged["shadow_alpha_cap"]) <= 0.1:
        raise ValueError("WorldTestGS shadow alpha cap must be in (0, 0.1]")
    trust_radius = merged["committed_mean_trust_radius_m"]
    if trust_radius is not None and (
        not math.isfinite(float(trust_radius)) or float(trust_radius) <= 0.0
    ):
        raise ValueError("WorldTestGS mean trust radius must be finite and positive")
    return merged
