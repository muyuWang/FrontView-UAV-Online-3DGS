#!/usr/bin/env python3
"""Screen posterior-budget refill after causal projective map ownership."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage10_8_12"

COMMON = {
    "routing_mode": "causal_observability",
    "projective_nms_mode": "budget_cells",
    "projective_covariance_mode": "isotropic",
    "fallback_support_mode": "budget_isotropic",
}

runner.VARIANTS = {
    "A_no_map_gate": {
        "far_field": {
            **COMMON,
            "map_redundancy_gate": False,
            "map_redundancy_evidence": "geometry",
            "posterior_budget_refill": False,
            "shuffle_refill_evidence": False,
        },
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "B_photometric_gate": {
        "far_field": {
            **COMMON,
            "map_redundancy_gate": True,
            "map_redundancy_evidence": "photometric",
            "posterior_budget_refill": False,
            "shuffle_refill_evidence": False,
        },
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "C_posterior_refill": {
        "far_field": {
            **COMMON,
            "map_redundancy_gate": True,
            "map_redundancy_evidence": "photometric",
            "posterior_budget_refill": True,
            "shuffle_refill_evidence": False,
        },
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "D_refill_shuffled": {
        "far_field": {
            **COMMON,
            "map_redundancy_gate": True,
            "map_redundancy_evidence": "photometric",
            "posterior_budget_refill": True,
            "shuffle_refill_evidence": True,
        },
        "sampling": {"selection_mode": "depth_stratified"},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
