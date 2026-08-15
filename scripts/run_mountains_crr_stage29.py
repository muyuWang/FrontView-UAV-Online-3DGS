#!/usr/bin/env python3
"""Test dynamic trust against a count-matched certificate shuffle."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "configs/360dvo_coverage_recovery/mountains_adaptive_responsibility_best.yaml"
)
runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage29_8_13"

COMMON_ROUTE = {
    "routing_mode": "adaptive_observability",
    "ray_atlas_enabled": False,
    "ray_atlas_shuffle_evidence": False,
    "shuffle_responsibility": False,
    "footprint_trust_mode": "certificate_odds",
    "footprint_trust_scope": "all_depthcov",
    "footprint_trust_dynamic_update": True,
    "footprint_trust_dynamic_shuffle_mode": "certificate",
}

runner.VARIANTS = {
    "A_dynamic_real": {
        "far_field": {
            **COMMON_ROUTE,
            "footprint_trust_dynamic_shuffle": False,
        },
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
    "B_certificate_matched_shuffled": {
        "far_field": {
            **COMMON_ROUTE,
            "footprint_trust_dynamic_shuffle": True,
        },
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
