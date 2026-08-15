#!/usr/bin/env python3
"""Run matched adaptive-budget and isotropic-support ablations on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage16_8_12"

FAR_FIELD = {
    "routing_mode": "causal_observability",
    "map_redundancy_gate": False,
    "projective_nms_mode": "budget_cells",
    "projective_covariance_mode": "isotropic",
    "fallback_support_mode": "budget_isotropic",
}

runner.VARIANTS = {
    "A_adaptive_isotropic": {
        "far_field": FAR_FIELD,
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
    "B_adaptive_shuffled_isotropic": {
        "far_field": FAR_FIELD,
        "sampling": {"selection_mode": "adaptive_log_depth_shuffled"},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
