#!/usr/bin/env python3
"""Run matched adaptive regime-coverage ablations on Mountains."""

import run_mountains_crr_ablation as runner


runner.DEFAULT_OUTPUT = runner.ROOT / "Logs_mountains_crr_stage5_8_12"

CAUSAL_BUDGET = {
    "routing_mode": "causal_observability",
    "map_redundancy_gate": False,
    "projective_nms_mode": "budget_cells",
}

runner.VARIANTS = {
    "A_adaptive_random": {
        "far_field": CAUSAL_BUDGET,
        "sampling": {"selection_mode": "adaptive_log_depth_random"},
    },
    "B_adaptive_coverage": {
        "far_field": CAUSAL_BUDGET,
        "sampling": {"selection_mode": "adaptive_log_depth_coverage"},
    },
    "C_adaptive_residual_coverage": {
        "far_field": CAUSAL_BUDGET,
        "sampling": {
            "selection_mode": "adaptive_log_depth_residual_coverage"
        },
    },
    "D_adaptive_coverage_shuffled": {
        "far_field": CAUSAL_BUDGET,
        "sampling": {
            "selection_mode": "adaptive_log_depth_coverage_shuffled"
        },
    },
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
