#!/usr/bin/env python3
"""Test budget-derived support for certified Mountains metric births."""

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
    runner.ROOT / "Logs_mountains_far_depth_goal_8_13/causal_metric_support_620"
)


def causal(support_mode, shuffle=False):
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
        "shuffle_evidence": bool(shuffle),
        "shuffle_seed": 43,
        "cache_frames": 8,
        "support_mode": support_mode,
    }


runner.VARIANTS = {
    "A_stage35_fallback": {},
    "B_metric_budget_isotropic": {
        "causal_metric_birth": causal("budget_isotropic"),
    },
    "C_metric_budget_structure": {
        "causal_metric_birth": causal("budget_structure"),
    },
    "D_metric_budget_structure_shuffled": {
        "causal_metric_birth": causal("budget_structure", shuffle=True),
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
