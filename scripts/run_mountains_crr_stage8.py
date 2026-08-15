#!/usr/bin/env python3
"""Screen certificate-modulated projective support on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage8_8_12"

COMMON = {
    "routing_mode": "causal_observability",
    "map_redundancy_gate": False,
    "projective_nms_mode": "budget_cells",
    "projective_covariance_mode": "isotropic",
}

runner.VARIANTS = {
    "A_budget_isotropic": {
        "far_field": {**COMMON, "fallback_support_mode": "budget_isotropic"},
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "B_certificate_structure": {
        "far_field": {
            **COMMON,
            "fallback_support_mode": "budget_certificate_structure",
        },
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "C_certificate_shuffled": {
        "far_field": {
            **COMMON,
            "fallback_support_mode": "budget_certificate_structure_shuffled",
        },
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "D_point_structure": {
        "far_field": {**COMMON, "fallback_support_mode": "budget_structure"},
        "sampling": {"selection_mode": "depth_stratified"},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
