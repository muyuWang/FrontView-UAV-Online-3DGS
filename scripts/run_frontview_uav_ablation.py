#!/usr/bin/env python3
"""Run matched PanoAir front-view method arms on independent GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CONFIGS = {
    "modp": REPO / "configs/frontview_uav/panoair_orbvi_modp_200.yaml",
    "equal_count": REPO
    / "configs/frontview_uav/panoair_orbvi_equal_count_200.yaml",
    "matched_76k": REPO
    / "configs/frontview_uav/panoair_orbvi_matched_76k_200.yaml",
    "matched_full": REPO
    / "configs/frontview_uav/panoair_orbvi_matched_76k_full.yaml",
    "pm_only": REPO / "configs/frontview_uav/panoair_orbvi_pm_only_200.yaml",
    "pcro": REPO / "configs/frontview_uav/panoair_orbvi_pcro_200.yaml",
    "pcro_gentle": REPO
    / "configs/frontview_uav/panoair_orbvi_pcro_gentle_200.yaml",
    "pcro_strong": REPO
    / "configs/frontview_uav/panoair_orbvi_pcro_strong_200.yaml",
    "pcro_online_only": REPO
    / "configs/frontview_uav/panoair_orbvi_pcro_online_only_200.yaml",
    "pcro_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_pcro_shuffled_200.yaml",
    "crwd": REPO / "configs/frontview_uav/panoair_orbvi_crwd_200.yaml",
    "crwd_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_crwd_shuffled_200.yaml",
    "cdwd": REPO / "configs/frontview_uav/panoair_orbvi_cdwd_200.yaml",
    "pool_uniform": REPO
    / "configs/frontview_uav/panoair_orbvi_pool_uniform_200.yaml",
    "pbsd_equal": REPO
    / "configs/frontview_uav/panoair_orbvi_pbsd_equal_200.yaml",
    "pbsd_uav": REPO / "configs/frontview_uav/panoair_orbvi_pbsd_uav_200.yaml",
    "pbsd_uav_full": REPO
    / "configs/frontview_uav/panoair_orbvi_pbsd_uav_full.yaml",
    "pbsd_uav_4000": REPO
    / "configs/frontview_uav/panoair_orbvi_pbsd_uav_4000_200.yaml",
    "pbsd_uav_4400": REPO
    / "configs/frontview_uav/panoair_orbvi_pbsd_uav_4400_200.yaml",
    "pbsd_uav_4800": REPO
    / "configs/frontview_uav/panoair_orbvi_pbsd_uav_4800_200.yaml",
    "pbsd_uav_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_pbsd_uav_shuffled_200.yaml",
    "tlpb_track": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_track_200.yaml",
    "tlpb_layered": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_layered_200.yaml",
    "tlpb_layered_scale": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_layered_scale_200.yaml",
    "tlpb_adaptive": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_adaptive_200.yaml",
    "tlpb_adaptive_nomap": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_adaptive_nomap_200.yaml",
    "tlpb_mix0": REPO / "configs/frontview_uav/panoair_orbvi_tlpb_mix0_200.yaml",
    "tlpb_mix25": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_mix25_200.yaml",
    "tlpb_mix50": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_mix50_200.yaml",
    "tlpb_strict12": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_strict12_200.yaml",
    "tlpb_strict16": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_strict16_200.yaml",
    "tlpb_strict16_mix25": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_strict16_mix25_200.yaml",
    "tlpb_strict20": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_strict20_200.yaml",
    "tlpb_opacity20": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_opacity20_200.yaml",
    "tlpb_opacity30": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_opacity30_200.yaml",
    "tlpb_opacity40": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_opacity40_200.yaml",
    "tlpb_opacity_const40": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_opacity_const40_200.yaml",
    "tlpb_trackscale125": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_trackscale125_200.yaml",
    "tlpb_trackscale150": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_trackscale150_200.yaml",
    "tlpb_trackscale150_mix25": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_trackscale150_mix25_200.yaml",
    "tlpb_trackscale200": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_trackscale200_200.yaml",
    "tlpb_temporal1": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_temporal1_200.yaml",
    "tlpb_temporal2_any": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_temporal2_any_200.yaml",
    "tlpb_temporal2_vote": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_temporal2_vote_200.yaml",
    "tlpb_temporal2_dup": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_temporal2_dup_200.yaml",
    "tlpb_replay_range": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_replay_range_200.yaml",
    "tlpb_replay_age": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_replay_age_200.yaml",
    "tlpb_replay_uniform": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_replay_uniform_200.yaml",
    "tlpb_replay_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_replay_shuffled_200.yaml",
    "tlpb_replay_uniform_k2": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_replay_uniform_k2_200.yaml",
    "tlpb_replay_uniform_k3": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_replay_uniform_k3_200.yaml",
    "tlpb_replay_uniform_k3_s8": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_replay_uniform_k3_s8_200.yaml",
    "tlpb_replay_uniform_k4_s6": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_replay_uniform_k4_s6_200.yaml",
    "tlpb_replay_cyclic_k1_s3": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_replay_cyclic_k1_s3_200.yaml",
    "tlpb_replay_cyclic_k1_s4": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_replay_cyclic_k1_s4_200.yaml",
    "tlpb_replay_cyclic_k2_s4": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_replay_cyclic_k2_s4_200.yaml",
    "tlpb_replay_cyclic_k2_s6": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_replay_cyclic_k2_s6_200.yaml",
    "tlpb_post_range_rr": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_range_rr_200.yaml",
    "tlpb_post_range_rr_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_range_rr_shuffled_200.yaml",
    "pbsd_post_range_rr": REPO
    / "configs/frontview_uav/panoair_orbvi_pbsd_post_range_rr_200.yaml",
    "tlpb_post_rr_freeze60": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_rr_freeze60_200.yaml",
    "tlpb_post_rr_freeze80": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_rr_freeze80_200.yaml",
    "tlpb_post_rr_freeze90": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_rr_freeze90_200.yaml",
    "tlpb_post_rr_mse25": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_rr_mse25_200.yaml",
    "tlpb_post_rr_mse50": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_rr_mse50_200.yaml",
    "tlpb_post_rr_mse100": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_rr_mse100_200.yaml",
    "tlpb_post_rr_mse100_s4": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_rr_mse100_s4_200.yaml",
    "tlpb_post_rr_layer_budget": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_rr_layer_budget_200.yaml",
    "tlpb_post_rr_layer_budget_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_rr_layer_budget_full.yaml",
    "tlpb_post_rr_layer_budget_far30": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_rr_layer_budget_far30_200.yaml",
    "tlpb_post_rr_layer_budget_mid50": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_rr_layer_budget_mid50_200.yaml",
    "tlpb_post_rr_layer_budget_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_rr_layer_budget_shuffled_200.yaml",
    "tlpb_post_rr_layer_budget_76k": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_post_rr_layer_budget_76k_200.yaml",
    "tlpb_bounded2_4000": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_bounded2_4000_200.yaml",
    "tlpb_bounded3_4000": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_bounded3_4000_200.yaml",
    "tlpb_bounded2_4000_far30": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_bounded2_4000_far30_200.yaml",
    "tlpb_layer_budget_pool3": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_layer_budget_pool3_200.yaml",
    "tlpb_layer_budget_pool4": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_layer_budget_pool4_200.yaml",
    "tlpb_layer_budget_pool3_bounded2": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_layer_budget_pool3_bounded2_200.yaml",
    "tlpb_atlas_3200": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_atlas_3200_200.yaml",
    "tlpb_atlas_4000": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_atlas_4000_200.yaml",
    "tlpb_atlas_4800": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_atlas_4800_200.yaml",
    "tlpb_anchor_pool3_3200": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_anchor_pool3_3200_200.yaml",
    "tlpb_anchor_pool3_4000": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_anchor_pool3_4000_200.yaml",
    "tlpb_anchor_pool3_4000_p25": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_anchor_pool3_4000_p25_200.yaml",
    "tlpb_hybrid20": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_hybrid20_200.yaml",
    "tlpb_hybrid50": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_hybrid50_200.yaml",
    "tlpb_hybrid80": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_hybrid80_200.yaml",
    "tlpb_hybrid20_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_hybrid20_full.yaml",
    "tlpb_hybrid80_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tlpb_hybrid80_full.yaml",
    "pbsd_far_projective20": REPO
    / "configs/frontview_uav/panoair_orbvi_pbsd_far_projective20_200.yaml",
    "pbsd_far_projective50": REPO
    / "configs/frontview_uav/panoair_orbvi_pbsd_far_projective50_200.yaml",
    "pbsd_far_projective80": REPO
    / "configs/frontview_uav/panoair_orbvi_pbsd_far_projective80_200.yaml",
    "pbsd_far_projective80_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_pbsd_far_projective80_shuffled_200.yaml",
    "pbsd_far_projective80_full": REPO
    / "configs/frontview_uav/panoair_orbvi_pbsd_far_projective80_full.yaml",
    "rift_identity": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_identity_200.yaml",
    "rift_lod": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_lod_200.yaml",
    "rift_lod_relaxed": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_lod_relaxed_200.yaml",
    "rift_lod_slots2": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_lod_slots2_200.yaml",
    "rift_lod_slots2_tight": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_lod_slots2_tight_200.yaml",
    "rift_lod_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_lod_shuffled_200.yaml",
    "rift_footprint": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_footprint_200.yaml",
    "rift_footprint_dense": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_footprint_dense_200.yaml",
    "rift_footprint_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_footprint_shuffled_200.yaml",
    "rift_footprint_l2": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_footprint_l2_200.yaml",
    "rift_footprint_dense_l2": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_footprint_dense_l2_200.yaml",
    "rift_footprint_dense_l2_allres": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_footprint_dense_l2_allres_200.yaml",
    "rift_footprint_dense_l2_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_footprint_dense_l2_shuffled_200.yaml",
    "tfl": REPO / "configs/frontview_uav/panoair_orbvi_tfl_200.yaml",
    "tfl_dense": REPO / "configs/frontview_uav/panoair_orbvi_tfl_dense_200.yaml",
    "tfl_allres": REPO / "configs/frontview_uav/panoair_orbvi_tfl_allres_200.yaml",
    "cgtfl_w10": REPO / "configs/frontview_uav/panoair_orbvi_cgtfl_w10_200.yaml",
    "cgtfl_w15": REPO / "configs/frontview_uav/panoair_orbvi_cgtfl_w15_200.yaml",
    "cgtfl_w20": REPO / "configs/frontview_uav/panoair_orbvi_cgtfl_w20_200.yaml",
    "residual_utility": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_utility_200.yaml",
    "residual_cover": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_cover_200.yaml",
    "residual_cover_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_cover_shuffled_200.yaml",
    "residual_cover_visibility0": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_cover_visibility0_200.yaml",
    "residual_cover_visibility25": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_cover_visibility25_200.yaml",
    "residual_cover_wide": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_cover_wide_200.yaml",
    "residual_cover_visibility25_wide": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_cover_visibility25_wide_200.yaml",
    "residual_cover_dual25": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_cover_dual25_200.yaml",
    "residual_cover_dual50": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_cover_dual50_200.yaml",
    "residual_cover_dual100": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_cover_dual100_200.yaml",
    "residual_cover_dual50_wide15": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_cover_dual50_wide15_200.yaml",
    "covlod_strict": REPO
    / "configs/frontview_uav/panoair_orbvi_covlod_strict_200.yaml",
    "covlod_floor10": REPO
    / "configs/frontview_uav/panoair_orbvi_covlod_floor10_200.yaml",
    "covlod_floor25": REPO
    / "configs/frontview_uav/panoair_orbvi_covlod_floor25_200.yaml",
    "covlod_floor10_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_covlod_floor10_shuffled_200.yaml",
    "rift_handoff_floor0": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_handoff_floor0_200.yaml",
    "rift_handoff_floor25": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_handoff_floor25_200.yaml",
    "rift_handoff_floor50": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_handoff_floor50_200.yaml",
    "rift_handoff_floor25_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_rift_handoff_floor25_shuffled_200.yaml",
    "pbsd_far_projective80_4000": REPO
    / "configs/frontview_uav/panoair_orbvi_pbsd_far_projective80_4000_200.yaml",
    "residual_utility_budget1": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_utility_budget1_200.yaml",
    "residual_cover_wide_budget1": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_cover_wide_budget1_200.yaml",
    "residual_cover_wide_budget1_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_cover_wide_budget1_shuffled_200.yaml",
    "residual_utility_budget1_sigma75": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_utility_budget1_sigma75_200.yaml",
    "residual_utility_budget1_confidence": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_utility_budget1_confidence_200.yaml",
    "residual_utility_budget1_edge25": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_utility_budget1_edge25_200.yaml",
    "residual_utility_budget1_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_utility_budget1_shuffled_200.yaml",
    "residual_pbsd_order_budget1": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_pbsd_order_budget1_200.yaml",
    "residual_mix25_budget1": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_mix25_budget1_200.yaml",
    "residual_mix50_budget1": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_mix50_budget1_200.yaml",
    "residual_mix75_budget1": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_mix75_budget1_200.yaml",
    "residual_utility_budget1_full": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_utility_budget1_full.yaml",
    "residual_retire_opacity": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_retire_opacity_200.yaml",
    "residual_retire_expansion1": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_retire_expansion1_200.yaml",
    "residual_retire_expansion2": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_retire_expansion2_200.yaml",
    "residual_retire_expansion1_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_retire_expansion1_shuffled_200.yaml",
    "residual_retire_gradient": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_retire_gradient_200.yaml",
    "residual_retire_gradient_opacity": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_retire_gradient-opacity_200.yaml",
    "residual_retire_gradient_expansion": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_retire_gradient-expansion_200.yaml",
    "residual_retire_gradient_opacity_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_retire_gradient-opacity-shuffled_200.yaml",
    "residual_retire_gradient_opacity_full": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_retire_gradient-opacity_full.yaml",
    "residual_retire_late170_full": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_retire_late170_full.yaml",
    "residual_retire_late190_full": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_retire_late190_full.yaml",
    "residual_retire_late170_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_retire_late170-shuffled_full.yaml",
    "residual_retire_late170_opacity_full": REPO
    / "configs/frontview_uav/panoair_orbvi_residual_retire_late170-opacity_full.yaml",
    "pcra_s3_r1p5": REPO
    / "configs/frontview_uav/panoair_orbvi_pcra_s3_r1p5_200.yaml",
    "pcra_s3_r2p5": REPO
    / "configs/frontview_uav/panoair_orbvi_pcra_s3_r2p5_200.yaml",
    "pcra_s2_r1p5": REPO
    / "configs/frontview_uav/panoair_orbvi_pcra_s2_r1p5_200.yaml",
    "pcra_s3_r1p5_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_pcra_s3_r1p5-shuffled_200.yaml",
    "pcra_smoke10": REPO
    / "configs/frontview_uav/panoair_orbvi_pcra_smoke10.yaml",
    "tsc_r1": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r1_200.yaml",
    "tsc_r0p5": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p5_200.yaml",
    "tsc_r1p5": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r1p5_200.yaml",
    "tsc_r1_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r1-shuffled_200.yaml",
    "tsc_smoke10": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_smoke10.yaml",
    "tsc_r0p2": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p2_200.yaml",
    "tsc_r0p3": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_200.yaml",
    "tsc_tracks": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_tracks_200.yaml",
    "tsc_trackfusion_e002": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_trackfusion_e002_200.yaml",
    "tsc_trackfusion_e005": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_trackfusion_e005_200.yaml",
    "tsc_trackfusion_e010": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_trackfusion_e010_200.yaml",
    "tsc_trackfusion_e005_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_trackfusion_e005_shuffled_200.yaml",
    "tsc_pfc_floor25": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor25_200.yaml",
    "tsc_pfc_floor25_gpu": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor25_gpu_200.yaml",
    "tsc_upfc_floor25_gpu": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_upfc_floor25_gpu_200.yaml",
    "tsc_upfc_floor25_max4_gpu": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_upfc_floor25_max4_gpu_200.yaml",
    "tsc_pfc_floor25_gpu_smoke10": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor25_gpu_smoke10.yaml",
    "tsc_pfc_pfh_r0p5_gpu": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_pfh_r0p5_gpu_200.yaml",
    "tsc_pfc_pfh_r0p5_gpu_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_pfh_r0p5_gpu_shuffled_200.yaml",
    "tsc_pfc_isdo_gpu": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_isdo_gpu_200.yaml",
    "tsc_pfc_isdo_gpu_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_isdo_gpu_shuffled_200.yaml",
    "tsc_pfc_isdo_gpu_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_isdo_gpu_full.yaml",
    "tsc_pfc_isdo_gpu_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_isdo_gpu_shuffled_full.yaml",
    "tsc_pfc_revisit1700_gpu": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_revisit1700_gpu.yaml",
    "tsc_pfc_dcsc45_revisit1700_gpu": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_dcsc45_revisit1700_gpu.yaml",
    "tsc_pfc_dcsc45_shuffled_revisit1700_gpu": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_dcsc45_shuffled_revisit1700_gpu.yaml",
    "tsc_pfc_dcsc45_gpu_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_dcsc45_gpu_full.yaml",
    "tsc_pfc_dcsc45_shuffled_gpu_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_dcsc45_shuffled_gpu_full.yaml",
    "tsc_color15_revisit1700": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_color15_revisit1700.yaml",
    "tsc_fcbo64_revisit1700": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_fcbo64_revisit1700.yaml",
    "tsc_fcbo64_shuffled_revisit1700": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_fcbo64_shuffled_revisit1700.yaml",
    "tsc_pfc_floor25_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor25_shuffled_200.yaml",
    "tsc_pfc_floor25_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor25_full.yaml",
    "tsc_pfc_floor25_gpu_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor25_gpu_full.yaml",
    "tsc_pfc_floor25_color10_gpu_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor25_color10_gpu_full.yaml",
    "tsc_upfc_floor25_gpu_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_upfc_floor25_gpu_full.yaml",
    "tsc_pfc_floor25_color10": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor25_color10_200.yaml",
    "tsc_pfc_floor25_color15": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor25_color15_200.yaml",
    "tsc_pfc_floor25_color20": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor25_color20_200.yaml",
    "tsc_pfc_floor25_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor25_shuffled_full.yaml",
    "tsc_pfc_floor35": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor35_200.yaml",
    "tsc_pfc_floor35_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor35_full.yaml",
    "tsc_pfc_floor50": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor50_200.yaml",
    "tsc_pfc_floor50_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pfc_floor50_shuffled_200.yaml",
    "tsc_r0p4": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p4_200.yaml",
    "tsc_r0p266": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p266_200.yaml",
    "tsc_r0p266_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p266_full.yaml",
    "tsc_r0p3_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3-shuffled_200.yaml",
    "tsc_r0p3_color15": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color15_200.yaml",
    "tsc_r0p3_color15_sh1": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color15_sh1_200.yaml",
    "tsc_r0p3_color15_sh2": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color15_sh2_200.yaml",
    "tsc_r0p3_color15_sh3": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color15_sh3_200.yaml",
    "tsc_cgbr75_sh3": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_cgbr75_sh3_200.yaml",
    "tsc_cgbr75_sh3_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_cgbr75_sh3_shuffled_200.yaml",
    "tsc_tgbr75_sh3": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_tgbr75_sh3_200.yaml",
    "tsc_tgbr75_sh3_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_tgbr75_sh3_shuffled_200.yaml",
    "tsc_r0p3_color15_sh3_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color15_sh3_full.yaml",
    "tsc_tgbr75_sh3_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_tgbr75_sh3_full.yaml",
    "tsc_tgbr75_sh3_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_tgbr75_sh3_shuffled_full.yaml",
    "tsc_r0p3_color15_sh2_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color15_sh2_shuffled_200.yaml",
    "tsc_r0p2_color15_sh2_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p2_color15_sh2_shuffled_200.yaml",
    "tsc_r0p2_color15_sh2_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p2_color15_sh2_shuffled_full.yaml",
    "tsc_eqr_sh2": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_eqr_sh2_200.yaml",
    "tsc_eqr_sh2_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_eqr_sh2_shuffled_200.yaml",
    "tsc_eqr_sh2_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_eqr_sh2_full.yaml",
    "tsc_eqr_sh2_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_eqr_sh2_shuffled_full.yaml",
    "tsc_eqr_far20_sh2": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_eqr_far20_sh2_200.yaml",
    "tsc_eqr_far20_sh2_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_eqr_far20_sh2_shuffled_200.yaml",
    "tsc_eqr_far50_sh2": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_eqr_far50_sh2_200.yaml",
    "tsc_eqr_far50_sh2_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_eqr_far50_sh2_shuffled_200.yaml",
    "tsc_pqr_sh2": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pqr_sh2_200.yaml",
    "tsc_pqr_sh2_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pqr_sh2_shuffled_200.yaml",
    "tsc_pqr_sh2_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pqr_sh2_full.yaml",
    "tsc_pqr_sh2_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pqr_sh2_shuffled_full.yaml",
    "tsc_dpqr_sh2": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_dpqr_sh2_200.yaml",
    "tsc_dpqr_sh2_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_dpqr_sh2_shuffled_200.yaml",
    "tsc_pcpr_sh2": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pcpr_sh2_200.yaml",
    "tsc_pcpr_sh2_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pcpr_sh2_full.yaml",
    "tsc_pcpr_turn_ablated_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pcpr_turn_ablated_full.yaml",
    "tsc_pcpr_fixed1900_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_pcpr_fixed1900_full.yaml",
    "tsc_mvsqr_w1_sh2": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_mvsqr_w1_sh2_200.yaml",
    "tsc_mvsqr_w1_sh2_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_mvsqr_w1_sh2_shuffled_200.yaml",
    "tsc_mvsqr_w1_projective_sh2": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_mvsqr_w1_projective_sh2_200.yaml",
    "tsc_r0p3_color15_sh1_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color15_sh1_full.yaml",
    "tsc_r0p3_color15_sh2_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color15_sh2_full.yaml",
    "tsc_r0p3_color15_sh2_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color15_sh2_shuffled_full.yaml",
    "tsc_cpal50": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_cpal50_200.yaml",
    "tsc_cpal75": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_cpal75_200.yaml",
    "tsc_cpal75_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_cpal75_shuffled_200.yaml",
    "tsc_cpal75_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_cpal75_full.yaml",
    "tsc_cpal75_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_cpal75_shuffled_full.yaml",
    "tsc_ogau_v2_i10": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_ogau_v2_i10_200.yaml",
    "tsc_ogau_v2_i5": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_ogau_v2_i5_200.yaml",
    "tsc_ogau_v1_i10": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_ogau_v1_i10_200.yaml",
    "tsc_ogau_v2_i10_shuffled_global": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_ogau_v2_i10_shuffled_global_200.yaml",
    "tsc_ogau_v2_i10_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_ogau_v2_i10_full.yaml",
    "tsc_ogau_v2_i10_shuffled_global_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_ogau_v2_i10_shuffled_global_full.yaml",
    "tsc_sh2_r0p2": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_sh2_r0p2_200.yaml",
    "tsc_sh2_r0p266": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_sh2_r0p266_200.yaml",
    "tsc_sh2_r0p4": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_sh2_r0p4_200.yaml",
    "tsc_sh2_r0p2_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_sh2_r0p2_full.yaml",
    "tsc_rcas_r008": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_rcas_r008_200.yaml",
    "tsc_rcas_r015": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_rcas_r015_200.yaml",
    "tsc_rcas_r015_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_rcas_r015_shuffled_200.yaml",
    "tsc_rcas_track_r008": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_rcas_track_r008_200.yaml",
    "tsc_rcas_track_r015": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_rcas_track_r015_200.yaml",
    "tsc_rcas_track_r008_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_rcas_track_r008_shuffled_200.yaml",
    "tsc_rcas_track_r015_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_rcas_track_r015_full.yaml",
    "tsc_rcas_track_r015_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_rcas_track_r015_shuffled_full.yaml",
    "tsc_rcas_track_r008_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_rcas_track_r008_shuffled_full.yaml",
    "tsc_r0p3_color30": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color30_200.yaml",
    "tsc_r0p3_color50": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color50_200.yaml",
    "tsc_r0p3_color30_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color30-shuffled_200.yaml",
    "tsc_dpcs_c8": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_dpcs_c8_200.yaml",
    "tsc_dpcs_c12": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_dpcs_c12_200.yaml",
    "tsc_dpcs_c16": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_dpcs_c16_200.yaml",
    "tsc_dpcs_c12_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_dpcs_c12_shuffled_200.yaml",
    "tsc_vial_c16": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_vial_c16_200.yaml",
    "tsc_vial_c24": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_vial_c24_200.yaml",
    "tsc_vial_c32": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_vial_c32_200.yaml",
    "tsc_vial_c24_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_vial_c24_shuffled_200.yaml",
    "tsc_r0p3_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_full.yaml",
    "tsc_r0p2_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p2_full.yaml",
    "tsc_r0p3_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3-shuffled_full.yaml",
    "tsc_r0p3_color15_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color15_full.yaml",
    "tsc_r0p3_color15_gpu_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color15_gpu_full.yaml",
    "tsc_r0p3_color15_gpu_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_color15_gpu_shuffled_full.yaml",
    "tsc_r0p3_ratio1p25": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_ratio1p25_200.yaml",
    "tsc_r0p3_ratio1p5": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_ratio1p5_200.yaml",
    "tsc_r0p3_ratio2": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_ratio2_200.yaml",
    "tsc_r0p3_ratio1p5_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_r0p3_ratio1p5-shuffled_200.yaml",
    "tsc_handoff_floor75": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_handoff_floor75_200.yaml",
    "tsc_handoff_floor50": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_handoff_floor50_200.yaml",
    "tsc_handoff_floor25": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_handoff_floor25_200.yaml",
    "tsc_handoff_floor50_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_handoff_floor50_shuffled_200.yaml",
    "tsc_handoff_smoke10": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_handoff_smoke10.yaml",
    "tsc_active_r0p3": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_active_r0p3_200.yaml",
    "tsc_active_color15": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_active_color15_200.yaml",
    "tsc_active_r0p3_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_active_r0p3_full.yaml",
    "tsc_active_color15_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_active_color15_full.yaml",
    "tsc_active_r0p3_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_active_r0p3_shuffled_200.yaml",
    "tsc_active_r0p3_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_tsc_active_r0p3_shuffled_full.yaml",
    "eosc_r0p3": REPO
    / "configs/frontview_uav/panoair_orbvi_eosc_r0p3_200.yaml",
    "eosc_r0p3_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_eosc_r0p3_shuffled_200.yaml",
    "eosc_color15": REPO
    / "configs/frontview_uav/panoair_orbvi_eosc_color15_200.yaml",
    "eosc_r0p3_full": REPO
    / "configs/frontview_uav/panoair_orbvi_eosc_r0p3_full.yaml",
    "eosc_r0p3_shuffled_full": REPO
    / "configs/frontview_uav/panoair_orbvi_eosc_r0p3_shuffled_full.yaml",
    "eosc_color15_full": REPO
    / "configs/frontview_uav/panoair_orbvi_eosc_color15_full.yaml",
    "clsm_l7": REPO
    / "configs/frontview_uav/panoair_orbvi_clsm_l7_200.yaml",
    "clsm_l8": REPO
    / "configs/frontview_uav/panoair_orbvi_clsm_l8_200.yaml",
    "clsm_l9": REPO
    / "configs/frontview_uav/panoair_orbvi_clsm_l9_200.yaml",
    "clsm_smoke10": REPO
    / "configs/frontview_uav/panoair_orbvi_clsm_smoke10.yaml",
    "clsm_l8_full": REPO
    / "configs/frontview_uav/panoair_orbvi_clsm_l8_full.yaml",
    "acdt_split": REPO
    / "configs/frontview_uav/panoair_orbvi_acdt_split_200.yaml",
    "acdt": REPO / "configs/frontview_uav/panoair_orbvi_acdt_200.yaml",
    "acdt_shuffled": REPO
    / "configs/frontview_uav/panoair_orbvi_acdt_shuffled_200.yaml",
    "qg": REPO / "configs/frontview_uav/panoair_orbvi_qg_200.yaml",
    "pmsa": REPO / "configs/frontview_uav/panoair_orbvi_pmsa_200.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="modp,qg,pmsa")
    parser.add_argument("--gpu-ids", default="1,3,4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def run_one(name: str, config: Path, gpu: int, seed: int, tag: str, log: Path):
    from utils_new.tool_utils import load_config

    resolved = load_config(str(config))
    experiment = f"frontview_{name}_{tag}_seed{seed}_gpu{gpu}"
    command = [
        sys.executable,
        "slam_new.py",
        "--config",
        str(config),
        "--exp_name",
        experiment,
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("CUDA_HOME", "/usr/local/cuda-11.8")
    env["PATH"] = f"{env['CUDA_HOME']}/bin:{env.get('PATH', '')}"
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env["TORCH_EXTENSIONS_DIR"] = str(
        Path.home() / ".cache" / "torch_extensions" / f"online3dgs_gpu{gpu}"
    )
    wall_start = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    run_root = Path(resolved["Results"]["save_dir"]) / resolved["Dataset"]["name"]
    candidates = sorted(
        run_root.glob(f"*_{experiment}"), key=lambda path: path.stat().st_mtime
    )
    run_dir = candidates[-1] if candidates else None
    metrics = None
    if run_dir is not None and (run_dir / "results.json").is_file():
        payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        metrics = {
            "online_recon_time": payload.get("online_recon_time"),
            "num_processed_frames": payload.get("num_processed_frames"),
            "num_gaussians": payload.get("num_gaussians"),
            "num_keyframes": payload.get("num_keyframes"),
            "eval_res": payload.get("eval_res"),
            "frontview_scale_cover": payload.get("frontview_scale_cover"),
            "streaming_appearance_lod": payload.get("streaming_appearance_lod"),
            "frontview_track_fusion": payload.get("frontview_track_fusion"),
        }
    return {
        "name": name,
        "gpu": gpu,
        "config": str(config),
        "command": command,
        "returncode": process.returncode,
        "wall_time_s": time.perf_counter() - wall_start,
        "console_log": str(log),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "metrics": metrics,
    }


def main() -> int:
    args = parse_args()
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    gpu_ids = [int(item) for item in args.gpu_ids.split(",") if item.strip()]
    unknown = set(arms) - set(CONFIGS)
    if unknown:
        raise ValueError(f"Unknown arms: {sorted(unknown)}")
    if len(gpu_ids) < len(arms):
        raise ValueError("One independent GPU ID is required per arm")
    tag = args.tag or datetime.now().strftime("ablation200_%Y-%m-%d-%H-%M-%S")
    output = REPO / "Logs_frontview_uav" / "benchmarks" / tag
    output.mkdir(parents=True, exist_ok=False)

    results = []
    with ThreadPoolExecutor(max_workers=len(arms)) as executor:
        futures = {
            executor.submit(
                run_one,
                arm,
                CONFIGS[arm],
                gpu,
                args.seed,
                tag,
                output / f"{arm}.console.log",
            ): arm
            for arm, gpu in zip(arms, gpu_ids)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)

    manifest = {
        "tag": tag,
        "seed": args.seed,
        "baseline_psnr": 28.112104682922364,
        "results": sorted(results, key=lambda item: item["name"]),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 1 if any(item["returncode"] != 0 for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
