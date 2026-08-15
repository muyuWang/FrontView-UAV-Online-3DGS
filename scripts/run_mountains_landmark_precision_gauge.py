#!/usr/bin/env python3
"""Test posterior-precision landmark transport on Mountains GPUs 4-7."""

from __future__ import annotations

import run_mountains_landmark_admitted_mean as experiment


def memory(*, shuffle=False, propagate_uncertainty=False):
    return {
        "enabled": True,
        "conditioning_mode": "admitted_mean",
        "minimum_observations": 1,
        "maximum_conditioning_points": 500,
        "transport_rule": "full",
        "propagate_conditioned_uncertainty": bool(propagate_uncertainty),
        "shuffle_depths": bool(shuffle),
        "shuffle_seed": 43,
    }


POSTERIOR_GAUGE = {
    "enabled": True,
    "learning_scale_mode": "posterior_information",
    "optimization_mode": "post_step_projection",
    "responsibility_scope": "all_depthcov",
    "evidence_update_interval": 10,
    "apply_post_refinement": True,
}


experiment.OUTPUT_ROOT = (
    experiment.runner.ROOT
    / "Logs_mountains_far_depth_goal_8_13/landmark_precision_gauge_620"
)
experiment.METHODS = {
    "A_stage35": {},
    "B_landmark_mean": {
        "causal_landmark_memory": memory(),
    },
    "C_landmark_precision_gauge": {
        "causal_landmark_memory": memory(propagate_uncertainty=True),
        "observability": POSTERIOR_GAUGE,
    },
    "D_precision_gauge_shuffled": {
        "causal_landmark_memory": memory(
            shuffle=True, propagate_uncertainty=True
        ),
        "observability": POSTERIOR_GAUGE,
    },
}
experiment.BASELINE_VARIANT = "A_stage35"
experiment.CAUSAL_REAL_VARIANT = "C_landmark_precision_gauge"
experiment.CAUSAL_CONTROL_VARIANT = "D_precision_gauge_shuffled"
experiment.REFERENCE_VARIANT = "B_landmark_mean"


if __name__ == "__main__":
    raise SystemExit(experiment.main())
