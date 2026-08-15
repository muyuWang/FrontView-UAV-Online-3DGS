#!/usr/bin/env python3
"""Run same-GPU observe/abstain/exact metric birth triples on GPUs 6 and 7."""

import sys

import run_mountains_causal_depth_posterior_paired as paired
import run_mountains_support_certified_metric_birth as experiment


paired.experiment = experiment
paired.PAIR_SCHEDULE = {
    "6": ("A_real_observe", "B_real_abstain", "C_real_exact"),
    "7": ("D_shuffled_observe", "E_shuffled_abstain", "F_shuffled_exact"),
}


if __name__ == "__main__":
    if "--save-dir" not in sys.argv:
        sys.argv.extend(
            (
                "--save-dir",
                str(
                    paired.runner.ROOT
                    / "Logs_mountains_far_depth_goal_8_13"
                    / "support_certified_metric_birth_paired_620"
                ),
            )
        )
    raise SystemExit(paired.main())
