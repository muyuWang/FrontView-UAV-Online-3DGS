#!/usr/bin/env python3
"""Test visibility-conditioned residual detail footprints on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "configs/360dvo_coverage_recovery/mountains_adaptive_responsibility_best.yaml"
)
runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage35_8_13"

COMMON = {
    "routing_mode": "adaptive_observability",
    "ray_atlas_enabled": False,
    "ray_atlas_shuffle_evidence": False,
    "shuffle_responsibility": False,
    "footprint_trust_scope": "all_depthcov",
}

runner.VARIANTS = {
    "A_visible_residual_detail_real": {
        "far_field": {
            **COMMON,
            "footprint_trust_mode": "certificate_residual_rd_visible_detail",
        },
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
    "B_visible_residual_detail_matched_shuffled": {
        "far_field": {
            **COMMON,
            "footprint_trust_mode": (
                "certificate_residual_rd_visible_detail_shuffled"
            ),
        },
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
