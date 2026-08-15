#!/usr/bin/env python3
"""Isolate adaptive far-regime routing from cross-frame atlas suppression."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "configs/360dvo_coverage_recovery/mountains_adaptive_responsibility_best.yaml"
)
runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage22_8_13"

ADAPTIVE_FRAME_RESPONSIBILITY = {
    "routing_mode": "adaptive_observability",
    "ray_atlas_enabled": False,
    "ray_atlas_shuffle_evidence": False,
}

runner.VARIANTS = {
    "A_adaptive_route_real": {
        "far_field": {
            **ADAPTIVE_FRAME_RESPONSIBILITY,
            "shuffle_responsibility": False,
        },
    },
    "B_adaptive_route_shuffled": {
        "far_field": {
            **ADAPTIVE_FRAME_RESPONSIBILITY,
            "shuffle_responsibility": True,
        },
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
