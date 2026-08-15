#!/usr/bin/env python3
"""Run same-GPU cross-fitted gauge pairs on free GPUs 6 and 7."""

import sys

import run_mountains_causal_depth_posterior_paired as paired
import run_mountains_cross_fitted_depth_gauge as experiment


paired.experiment = experiment
paired.PAIR_SCHEDULE = {
    "6": ("A_real_observe", "B_real_gauge"),
    "7": ("C_shuffled_observe", "D_shuffled_gauge"),
}
experiment.runner.VARIANTS = experiment.runner.VARIANTS


if __name__ == "__main__":
    if "--save-dir" not in sys.argv:
        sys.argv.extend(
            (
                "--save-dir",
                str(
                    paired.runner.ROOT
                    / "Logs_mountains_far_depth_goal_8_13"
                    / "cross_fitted_depth_gauge_paired_620"
                ),
            )
        )
    raise SystemExit(paired.main())
