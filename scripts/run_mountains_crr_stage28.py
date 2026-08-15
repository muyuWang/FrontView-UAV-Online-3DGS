#!/usr/bin/env python3
"""Test causal one-way release of Stage 25 footprint trust on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "configs/360dvo_coverage_recovery/mountains_adaptive_responsibility_best.yaml"
)
runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage28_8_13"

COMMON_ROUTE = {
    "routing_mode": "adaptive_observability",
    "ray_atlas_enabled": False,
    "ray_atlas_shuffle_evidence": False,
    "shuffle_responsibility": False,
    "footprint_trust_mode": "certificate_odds",
    "footprint_trust_scope": "all_depthcov",
    "footprint_trust_dynamic_update": True,
}

runner.VARIANTS = {
    "A_dynamic_real": {
        "far_field": {
            **COMMON_ROUTE,
            "footprint_trust_dynamic_shuffle": False,
        },
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
    "B_dynamic_shuffled": {
        "far_field": {
            **COMMON_ROUTE,
            "footprint_trust_dynamic_shuffle": True,
        },
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
