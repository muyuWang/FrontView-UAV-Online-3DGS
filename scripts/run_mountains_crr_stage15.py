#!/usr/bin/env python3
"""Run causal posterior-responsibility controls on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage15_8_12"

FAR_FIELD = {
    "routing_mode": "causal_observability",
    "map_redundancy_gate": False,
    "projective_nms_mode": "budget_cells",
    "projective_covariance_mode": "isotropic",
    "fallback_support_mode": "budget_isotropic",
}

POSTERIOR = {
    "enabled": True,
    "learning_scale_mode": "posterior_information",
    "optimization_mode": "post_step_projection",
    "responsibility_scope": "projective_only",
    "evidence_update_interval": 10,
    "apply_post_refinement": True,
}

runner.VARIANTS = {
    "A_budget_support": {
        "far_field": FAR_FIELD,
        "sampling": {"selection_mode": "depth_stratified"},
        "observability": {"enabled": False},
    },
    "B_posterior_responsibility": {
        "far_field": FAR_FIELD,
        "sampling": {"selection_mode": "depth_stratified"},
        "observability": {**POSTERIOR, "shuffle_evidence": False},
    },
    "C_posterior_shuffled": {
        "far_field": FAR_FIELD,
        "sampling": {"selection_mode": "depth_stratified"},
        "observability": {**POSTERIOR, "shuffle_evidence": True},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
