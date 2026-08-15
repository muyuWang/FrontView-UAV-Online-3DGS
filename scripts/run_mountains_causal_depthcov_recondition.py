#!/usr/bin/env python3
"""Test tracking-reconditioned DepthCov on Mountains GPUs 4-7."""

import math
import sys

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "Logs_mountains_adaptive_goal_8_12_8_13"
    / "final/stage35_full_765/batch_20260813_095909"
    / "runtime_configs/A_visible_residual_detail_real.yaml"
)
runner.DEFAULT_OUTPUT = (
    runner.ROOT / "Logs_mountains_far_depth_goal_8_13/causal_depthcov_620"
)


def recondition(shuffle=False):
    return {
        "enabled": True,
        "history_frames": 4,
        "max_features": 2048,
        "minimum_match_score": 0.15,
        "maximum_sampson_error_px": 1.5,
        "pixel_sigma_px": 0.5,
        "consistency_chi2": 3.841458820694124,
        "minimum_references": 2,
        "minimum_information_gain": math.log(math.sqrt(2.0)),
        "maximum_reprojection_error_px": 1.5,
        "shuffle_evidence": False,
        "shuffle_seed": 43,
        "cache_frames": 8,
        "support_mode": "point",
        "birth_mode": "depthcov_recondition",
        "field_neighbors": 6,
        "field_confidence_probability": 0.95,
        "shuffle_field_binding": bool(shuffle),
    }


runner.VARIANTS = {
    "A_stage35_fallback": {},
    "B_no_fallback": {
        "coverage_recovery": {"depth_fallback_enabled": False},
    },
    "C_depthcov_recondition": {
        "causal_metric_birth": recondition(False),
    },
    "D_depthcov_recondition_shuffled": {
        "causal_metric_birth": recondition(True),
    },
}


if __name__ == "__main__":
    if "--frames" not in sys.argv:
        sys.argv.extend(("--frames", "620"))
    if "--fixed-far-mask-dir" not in sys.argv:
        sys.argv.extend(
            (
                "--fixed-far-mask-dir",
                str(
                    runner.ROOT
                    / "Logs_mountains_adaptive_goal_8_12_8_13/evaluation"
                    / "fixed_far_masks_q80_545_619_v2"
                ),
                "--fixed-far-begin",
                "545",
                "--fixed-far-end",
                "619",
            )
        )
    raise SystemExit(runner.main())
