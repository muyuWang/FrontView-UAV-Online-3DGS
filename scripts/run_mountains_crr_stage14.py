#!/usr/bin/env python3
"""Run entropy-routed inverse-depth responsibility controls on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage14_8_12"

FAR_FIELD = {
    "routing_mode": "causal_observability",
    "map_redundancy_gate": False,
    "projective_nms_mode": "budget_cells",
    "projective_covariance_mode": "isotropic",
    "fallback_support_mode": "budget_isotropic",
}

POSTERIOR = {
    "trigger_mode": "projective_debt",
    "min_translation_m": 0.0,
    "min_rotation_deg": 0.0,
    "multiview_depth_enabled": True,
    "multiview_depth_hypotheses": 8,
    "multiview_depth_mode": "posterior_inverse_depth",
    "multiview_depth_seed": 43,
}

runner.VARIANTS = {
    "C_posterior_depth": {
        "far_field": FAR_FIELD,
        "sampling": {"selection_mode": "depth_stratified"},
        "coverage_recovery": {
            **POSTERIOR,
            "shuffle_multiview_depth": False,
        },
    },
    "D_posterior_shuffled": {
        "far_field": FAR_FIELD,
        "sampling": {"selection_mode": "depth_stratified"},
        "coverage_recovery": {
            **POSTERIOR,
            "shuffle_multiview_depth": True,
        },
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
