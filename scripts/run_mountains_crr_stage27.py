#!/usr/bin/env python3
"""Test equal-area observability footprint trust on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "configs/360dvo_coverage_recovery/mountains_adaptive_responsibility_best.yaml"
)
runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage27_8_13"

COMMON_ROUTE = {
    "routing_mode": "adaptive_observability",
    "ray_atlas_enabled": False,
    "ray_atlas_shuffle_evidence": False,
    "shuffle_responsibility": False,
    "footprint_trust_scope": "all_depthcov",
}

runner.VARIANTS = {
    "A_equal_area_real": {
        "far_field": {
            **COMMON_ROUTE,
            "footprint_trust_mode": "certificate_equal_area",
        },
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
    "B_equal_area_shuffled": {
        "far_field": {
            **COMMON_ROUTE,
            "footprint_trust_mode": "certificate_equal_area_shuffled",
        },
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
