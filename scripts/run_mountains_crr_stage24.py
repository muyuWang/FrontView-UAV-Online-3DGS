#!/usr/bin/env python3
"""Test causal observability-conditioned footprint trust on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "configs/360dvo_coverage_recovery/mountains_adaptive_responsibility_best.yaml"
)
runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage24_8_13"

COMMON_ROUTE = {
    "routing_mode": "adaptive_observability",
    "ray_atlas_enabled": False,
    "ray_atlas_shuffle_evidence": False,
    "shuffle_responsibility": False,
}

runner.VARIANTS = {
    "A_footprint_trust_real": {
        "far_field": {**COMMON_ROUTE, "footprint_trust_mode": "information"},
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
    "B_footprint_trust_shuffled": {
        "far_field": {
            **COMMON_ROUTE,
            "footprint_trust_mode": "information_shuffled",
        },
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
