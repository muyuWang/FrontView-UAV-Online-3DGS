#!/usr/bin/env python3
"""Run same-GPU Fisher metric-birth pairs on GPUs 6 and 7."""

import sys

import run_mountains_causal_depth_posterior_paired as paired
import run_mountains_fisher_metric_birth as experiment


paired.experiment = experiment
paired.PAIR_SCHEDULE = {
    "6": ("A_real_observe", "B_real_fisher"),
    "7": ("C_shuffled_observe", "D_shuffled_fisher"),
}


if __name__ == "__main__":
    if "--save-dir" not in sys.argv:
        sys.argv.extend(
            (
                "--save-dir",
                str(
                    paired.runner.ROOT
                    / "Logs_mountains_far_depth_goal_8_13"
                    / "fisher_metric_birth_paired_620"
                ),
            )
        )
    raise SystemExit(paired.main())
