#!/usr/bin/env python3
"""Screen budget-normalized information support on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage11_8_12"

COMMON = {
    "routing_mode": "causal_observability",
    "map_redundancy_gate": False,
    "projective_nms_mode": "budget_cells",
    "projective_covariance_mode": "isotropic",
}

runner.VARIANTS = {
    "A_legacy_support": {
        "far_field": {**COMMON, "fallback_support_mode": "legacy"},
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "B_budget_isotropic": {
        "far_field": {**COMMON, "fallback_support_mode": "budget_isotropic"},
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "C_budget_information": {
        "far_field": {**COMMON, "fallback_support_mode": "budget_information"},
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "D_information_shuffled": {
        "far_field": {
            **COMMON,
            "fallback_support_mode": "budget_information_shuffled",
        },
        "sampling": {"selection_mode": "depth_stratified"},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
