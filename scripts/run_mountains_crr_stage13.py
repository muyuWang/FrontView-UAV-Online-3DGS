#!/usr/bin/env python3
"""Run coverage-debt multiview responsibility controls on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage13_8_12"

FAR_FIELD = {
    "routing_mode": "causal_observability",
    "map_redundancy_gate": False,
    "projective_nms_mode": "budget_cells",
    "projective_covariance_mode": "isotropic",
    "fallback_support_mode": "budget_isotropic",
}

DEBT = {
    "trigger_mode": "projective_debt",
    "min_translation_m": 0.0,
    "min_rotation_deg": 0.0,
}

MULTIVIEW = {
    **DEBT,
    "multiview_depth_enabled": True,
    "multiview_depth_hypotheses": 8,
    "multiview_depth_seed": 43,
}

runner.VARIANTS = {
    "A_fixed_gap": {
        "far_field": FAR_FIELD,
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "B_coverage_debt": {
        "far_field": FAR_FIELD,
        "sampling": {"selection_mode": "depth_stratified"},
        "coverage_recovery": DEBT,
    },
    "C_multiview_depth": {
        "far_field": FAR_FIELD,
        "sampling": {"selection_mode": "depth_stratified"},
        "coverage_recovery": {
            **MULTIVIEW,
            "shuffle_multiview_depth": False,
        },
    },
    "D_multiview_shuffled": {
        "far_field": FAR_FIELD,
        "sampling": {"selection_mode": "depth_stratified"},
        "coverage_recovery": {
            **MULTIVIEW,
            "shuffle_multiview_depth": True,
        },
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
