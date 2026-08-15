#!/usr/bin/env python3
"""Screen adaptive far-regime projective responsibility on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "configs/360dvo_coverage_recovery/mountains_adaptive_responsibility_best.yaml"
)
runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage21_8_13"

ADAPTIVE_ROUTE = {
    "routing_mode": "adaptive_observability",
    "ray_atlas_enabled": True,
    "ray_atlas_coordinate_mode": "camera_ray",
    "ray_atlas_competition_mode": "hard_cell",
}

runner.VARIANTS = {
    "A_adaptive_far_real": {
        "far_field": {
            **ADAPTIVE_ROUTE,
            "ray_atlas_shuffle_evidence": False,
        },
    },
    "B_adaptive_far_shuffled": {
        "far_field": {
            **ADAPTIVE_ROUTE,
            "ray_atlas_shuffle_evidence": True,
        },
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
