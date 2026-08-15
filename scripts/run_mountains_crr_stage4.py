#!/usr/bin/env python3
"""Run matched adaptive log-depth birth-budget ablations on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage4_8_12"

CAUSAL_BUDGET = {
    "routing_mode": "causal_observability",
    "map_redundancy_gate": False,
    "projective_nms_mode": "budget_cells",
}

runner.VARIANTS = {
    "A_pbsd_control": {
        "far_field": CAUSAL_BUDGET,
        "sampling": {"selection_mode": "depth_stratified"},
    },
    "B_adaptive_random": {
        "far_field": CAUSAL_BUDGET,
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
    "C_adaptive_importance": {
        "far_field": CAUSAL_BUDGET,
        "sampling": {"selection_mode": "adaptive_log_depth_importance"},
    },
    "D_adaptive_shuffled": {
        "far_field": CAUSAL_BUDGET,
        "sampling": {"selection_mode": "adaptive_log_depth_shuffled"},
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
