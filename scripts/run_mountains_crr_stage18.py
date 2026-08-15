#!/usr/bin/env python3
"""Ablate canonical and continuous causal ray responsibility on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "configs/360dvo_coverage_recovery/mountains_adaptive_responsibility_best.yaml"
)
runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage18_8_13"

runner.VARIANTS = {
    "A_camera_hard": {
        "far_field": {
            "ray_atlas_enabled": True,
            "ray_atlas_shuffle_evidence": False,
            "ray_atlas_coordinate_mode": "camera_ray",
            "ray_atlas_competition_mode": "hard_cell",
        },
    },
    "B_canonical_hard": {
        "far_field": {
            "ray_atlas_enabled": True,
            "ray_atlas_shuffle_evidence": False,
            "ray_atlas_coordinate_mode": "canonical_world",
            "ray_atlas_competition_mode": "hard_cell",
        },
    },
    "C_canonical_continuous": {
        "far_field": {
            "ray_atlas_enabled": True,
            "ray_atlas_shuffle_evidence": False,
            "ray_atlas_coordinate_mode": "canonical_world",
            "ray_atlas_competition_mode": "continuous_kernel",
        },
    },
    "D_continuous_shuffled": {
        "far_field": {
            "ray_atlas_enabled": True,
            "ray_atlas_shuffle_evidence": True,
            "ray_atlas_coordinate_mode": "canonical_world",
            "ray_atlas_competition_mode": "continuous_kernel",
        },
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
