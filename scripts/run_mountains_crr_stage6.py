#!/usr/bin/env python3
"""Run observability-rank projective covariance ablations on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage6_8_12"

COMMON = {
    "routing_mode": "causal_observability",
    "map_redundancy_gate": False,
    "projective_nms_mode": "budget_cells",
}

runner.VARIANTS = {
    "A_isotropic": {
        "far_field": {**COMMON, "projective_covariance_mode": "isotropic"},
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "B_observability_rank": {
        "far_field": {
            **COMMON,
            "projective_covariance_mode": "observability_rank",
        },
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "C_rank_shuffled": {
        "far_field": {
            **COMMON,
            "projective_covariance_mode": "observability_rank_shuffled",
        },
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "D_surfel": {
        "far_field": {**COMMON, "projective_covariance_mode": "surfel"},
        "sampling": {"selection_mode": "depth_stratified"},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
