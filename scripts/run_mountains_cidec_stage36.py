#!/usr/bin/env python3
"""Run matched causal inverse-depth certificate ablations on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "Logs_mountains_adaptive_goal_8_12_8_13"
    / "final/stage35_full_765/batch_20260813_095909"
    / "runtime_configs/A_visible_residual_detail_real.yaml"
)
runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_far_depth_goal_8_13/stage36"


def cidec(**overrides):
    config = {
        "enabled": True,
        "reference_frames": 3,
        "history_frames": 24,
        "hypotheses": 9,
        "prior_span_stds": 6.0,
        "minimum_log_depth_span": 1.0,
        "patch_radius_px": 2,
        "photometric_temperature": 0.04,
        "information_gain_min": 0.20,
        "posterior_std_ratio_max": 0.85,
        "minimum_valid_views": 1,
        "conflict_nll_margin": 1.0,
        "shuffle_evidence": False,
        "shuffle_seed": 43,
        "uncertified_policy": "projective_reject_conflict",
    }
    config.update(overrides)
    return {"inverse_depth_certificate": config}


runner.VARIANTS = {
    "A_stage35_baseline": {},
    "B_cidec_conflict": cidec(),
    "C_cidec_fail_closed": cidec(uncertified_policy="reject"),
    "D_cidec_matched_shuffled": cidec(shuffle_evidence=True),
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
