#!/usr/bin/env python3
"""Run threshold-free responsibility-cell and covariance-TSC ablations."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage3_8_12"

CAUSAL_FIXED = {
    "routing_mode": "causal_observability",
    "map_redundancy_gate": False,
    "projective_nms_mode": "fixed_grid",
}
CAUSAL_BUDGET = {
    "routing_mode": "causal_observability",
    "map_redundancy_gate": False,
    "projective_nms_mode": "budget_cells",
}
COVARIANCE_TSC = {
    "target_size_mode": "gaussian_support",
    "distance_mode": "gaussian_overlap",
    "color_distance_threshold": -1.0,
}

runner.VARIANTS = {
    "A_causal_fixed": {"far_field": CAUSAL_FIXED},
    "B_causal_budget": {"far_field": CAUSAL_BUDGET},
    "C_causal_fixed_covtsc": {
        "far_field": CAUSAL_FIXED,
        "scale_cover": COVARIANCE_TSC,
    },
    "D_causal_budget_covtsc": {
        "far_field": CAUSAL_BUDGET,
        "scale_cover": COVARIANCE_TSC,
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
