#!/usr/bin/env python3
"""Run matched causal ray-responsibility atlas ablations on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_CONFIG = (
    runner.ROOT
    / "configs/360dvo_coverage_recovery/mountains_adaptive_responsibility_best.yaml"
)
runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage17_8_12"

runner.VARIANTS = {
    "A_atlas_off": {
        "far_field": {
            "ray_atlas_enabled": False,
            "ray_atlas_shuffle_evidence": False,
        },
    },
    "B_ray_atlas": {
        "far_field": {
            "ray_atlas_enabled": True,
            "ray_atlas_shuffle_evidence": False,
        },
    },
    "C_ray_atlas_shuffled": {
        "far_field": {
            "ray_atlas_enabled": True,
            "ray_atlas_shuffle_evidence": True,
        },
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
