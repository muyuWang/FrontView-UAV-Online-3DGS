#!/usr/bin/env python3
"""Isolate threshold-free causal responsibility from footprint trust."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "configs/360dvo_coverage_recovery/mountains_adaptive_responsibility_best.yaml"
)
runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage31_8_13"

COMMON = {
    "routing_mode": "causal_observability",
    "ray_atlas_enabled": False,
    "ray_atlas_shuffle_evidence": False,
    "footprint_trust_mode": "disabled",
}

runner.VARIANTS = {
    "A_causal_responsibility_real": {
        "far_field": {
            **COMMON,
            "shuffle_responsibility": False,
        },
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
    "B_causal_responsibility_matched_shuffled": {
        "far_field": {
            **COMMON,
            "shuffle_responsibility": True,
            "responsibility_shuffle_mode": "log_depth_regimes",
        },
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
