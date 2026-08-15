#!/usr/bin/env python3
"""Test posterior rate-distortion birth density on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "configs/360dvo_coverage_recovery/mountains_adaptive_responsibility_best.yaml"
)
runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage23_8_13"

ADAPTIVE_ROUTE = {
    "routing_mode": "adaptive_observability",
    "ray_atlas_enabled": False,
    "ray_atlas_shuffle_evidence": False,
    "shuffle_responsibility": False,
}

runner.VARIANTS = {
    "A_rate_distortion_real": {
        "far_field": ADAPTIVE_ROUTE,
        "sampling": {
            "selection_mode": "adaptive_log_depth_rate_distortion",
        },
    },
    "B_rate_distortion_shuffled": {
        "far_field": ADAPTIVE_ROUTE,
        "sampling": {
            "selection_mode": "adaptive_log_depth_rate_distortion_shuffled",
        },
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
