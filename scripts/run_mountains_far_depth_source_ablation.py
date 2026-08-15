#!/usr/bin/env python3
"""Separate Mountains projective-birth and sparse-dropout depth errors."""

import sys

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "Logs_mountains_adaptive_goal_8_12_8_13"
    / "final/stage35_full_765/batch_20260813_095909"
    / "runtime_configs/A_visible_residual_detail_real.yaml"
)
runner.DEFAULT_OUTPUT = (
    runner.ROOT / "Logs_mountains_far_depth_goal_8_13/source_ablation_620"
)
runner.VARIANTS = {
    "A_stage35": {},
    "B_reject_unobservable": {
        "far_field": {"unobservable_birth_policy": "reject"},
    },
    "C_no_dropout_depth_fallback": {
        "coverage_recovery": {"depth_fallback_enabled": False},
    },
    "D_reject_both": {
        "far_field": {"unobservable_birth_policy": "reject"},
        "coverage_recovery": {"depth_fallback_enabled": False},
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
