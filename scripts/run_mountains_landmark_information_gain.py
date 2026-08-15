#!/usr/bin/env python3
"""Compare full and information-gain causal landmark transport on Mountains."""

from __future__ import annotations

from pathlib import Path

import run_mountains_landmark_admitted_mean as experiment


def memory(rule, *, shuffle=False):
    return {
        "enabled": True,
        "conditioning_mode": "admitted_mean",
        "minimum_observations": 1,
        "maximum_conditioning_points": 500,
        "transport_rule": rule,
        "shuffle_depths": bool(shuffle),
        "shuffle_seed": 43,
    }


experiment.OUTPUT_ROOT = (
    experiment.runner.ROOT
    / "Logs_mountains_far_depth_goal_8_13/landmark_information_gain_620"
)
experiment.METHODS = {
    "A_stage35": {},
    "B_admitted_full_real": {
        "causal_landmark_memory": memory("full"),
    },
    "C_information_gain_real": {
        "causal_landmark_memory": memory("variance_gain"),
    },
    "D_information_gain_shuffled": {
        "causal_landmark_memory": memory("variance_gain", shuffle=True),
    },
}
experiment.BASELINE_VARIANT = "A_stage35"
experiment.CAUSAL_REAL_VARIANT = "C_information_gain_real"
experiment.CAUSAL_CONTROL_VARIANT = "D_information_gain_shuffled"
experiment.REFERENCE_VARIANT = "B_admitted_full_real"


def main():
    return experiment.main()


if __name__ == "__main__":
    raise SystemExit(main())
