#!/usr/bin/env python3
"""Screen residual-conditioned projective map ownership on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage9_8_12"

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
        },
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "B_geometry_gate": {
        "far_field": {
            **COMMON,
            "map_redundancy_gate": True,
            "map_redundancy_evidence": "geometry",
        },
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "C_photometric_gate": {
        "far_field": {
            **COMMON,
            "map_redundancy_gate": True,
            "map_redundancy_evidence": "photometric",
        },
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "D_photometric_shuffled": {
        "far_field": {
            **COMMON,
            "map_redundancy_gate": True,
            "map_redundancy_evidence": "photometric_shuffled",
        },
        "sampling": {"selection_mode": "depth_stratified"},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
