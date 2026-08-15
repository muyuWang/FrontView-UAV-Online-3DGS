#!/usr/bin/env python3
"""Test threshold-free certificate responsibility on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "configs/360dvo_coverage_recovery/mountains_adaptive_responsibility_best.yaml"
)
runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage30_8_13"

COMMON = {
    "routing_mode": "causal_observability",
    "ray_atlas_enabled": False,
    "ray_atlas_shuffle_evidence": False,
    "shuffle_responsibility": False,
    "footprint_trust_scope": "all_depthcov",
}

runner.VARIANTS = {
    "A_certificate_responsibility_real": {
        "far_field": {
            **COMMON,
            "footprint_trust_mode": "certificate_odds",
        },
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
    "B_certificate_responsibility_shuffled": {
        "far_field": {
            **COMMON,
            "footprint_trust_mode": "certificate_odds_shuffled",
        },
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
