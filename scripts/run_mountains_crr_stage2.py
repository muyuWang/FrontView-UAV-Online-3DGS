#!/usr/bin/env python3
"""Run second-stage Mountains causal-resolution ablations on GPUs 4-7."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage2_8_12"
runner.VARIANTS = {
    "A_causal_fixed_nms": {
        "far_field": {
            "routing_mode": "causal_observability",
            "map_redundancy_gate": False,
            "projective_nms_mode": "fixed_grid",
        },
    },
    "B_causal_support_nms": {
        "far_field": {
            "routing_mode": "causal_observability",
            "map_redundancy_gate": False,
            "projective_nms_mode": "gaussian_support",
        },
    },
    "C_causal_support_covtsc": {
        "far_field": {
            "routing_mode": "causal_observability",
            "map_redundancy_gate": False,
            "projective_nms_mode": "gaussian_support",
        },
        "scale_cover": {
            "target_size_mode": "gaussian_support",
            "distance_mode": "gaussian_overlap",
            "color_distance_threshold": -1.0,
        },
    },
    "D_causal_support_rip": {
        "far_field": {
            "routing_mode": "causal_observability",
            "map_redundancy_gate": False,
            "projective_nms_mode": "gaussian_support",
        },
        "observability": {
            "enabled": True,
            "learning_scale_mode": "resolution_information",
            "apply_post_refinement": True,
        },
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
