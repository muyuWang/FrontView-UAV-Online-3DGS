#!/usr/bin/env python3
"""Cross-validate track-calibrated relative depth from cached Mountains samples."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_INPUT = (
    ROOT
    / "Logs_mountains_far_depth_goal_8_13/diagnostics"
    / "parallax_tracking_consensus_v2/parallax_tracking_samples.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "Logs_mountains_far_depth_goal_8_13/diagnostics"
    / "track_depth_gauge_cross_validation.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def metrics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": None if not len(values) else float(np.mean(values)),
        "median": None if not len(values) else float(np.median(values)),
        "p90": None if not len(values) else float(np.quantile(values, 0.9)),
    }


def main():
    args = parse_args()
    from utils_new.frontview_track_depth_gauge import cross_fitted_track_depth_gauge

    rows = json.loads(args.input.expanduser().read_text(encoding="utf-8"))
    certified = []
    threshold = math.log(math.sqrt(2.0))
    for row in rows:
        if (
            row.get("matched_multi_depth_m") is not None
            and float(row.get("matched_multi_maximum_reprojection_error_px", math.inf))
            <= 1.5
            and float(row.get("matched_multi_information_gain", -math.inf)) > threshold
        ):
            certified.append(row)
    by_frame = {}
    for row in certified:
        by_frame.setdefault(int(row["frame"]), []).append(row)

    frame_results = []
    baseline_errors = []
    corrected_errors = []
    for frame, local in sorted(by_frame.items()):
        if len(local) < 4:
            continue
        pixels = np.asarray([[row["u"], row["v"]] for row in local])
        predicted = np.asarray([row["pred_depth_m"] for row in local])
        predicted_std = np.asarray([row["pred_log_std"] for row in local])
        tracked = np.asarray([row["matched_multi_depth_m"] for row in local])
        tracked_std = np.asarray([row["matched_multi_log_depth_std"] for row in local])
        truth = np.asarray([row["true_depth_m"] for row in local])
        sparse_prior = float(np.median(tracked))
        gauge = cross_fitted_track_depth_gauge(
            pixels,
            predicted,
            predicted_std,
            tracked,
            tracked_std,
            [sparse_prior, 300.0],
            fallback_log_std=0.05,
            shuffle_binding=args.shuffle,
            seed=args.seed + frame,
        )
        baseline = np.abs(np.log(predicted) - np.log(truth))
        corrected = np.abs(
            np.log(predicted) + (gauge.log_scale if gauge.accepted_field else 0.0)
            - np.log(truth)
        )
        baseline_errors.extend(baseline.tolist())
        corrected_errors.extend(corrected.tolist())
        frame_results.append(
            {
                "frame": frame,
                "rows": len(local),
                "accepted_field": gauge.accepted_field,
                "log_scale": gauge.log_scale,
                "fold_nll_gains": list(gauge.fold_nll_gains),
                "baseline_log_mae": float(np.mean(baseline)),
                "corrected_log_mae": float(np.mean(corrected)),
            }
        )
    payload = {
        "status": "success",
        "protocol": "entropy-certified multi-view tracks; frame-wise two-fold predictive-risk selection",
        "shuffle_binding": bool(args.shuffle),
        "certified_rows": len(certified),
        "evaluated_frames": len(frame_results),
        "accepted_frames": sum(row["accepted_field"] for row in frame_results),
        "baseline_absolute_log_error": metrics(baseline_errors),
        "selected_absolute_log_error": metrics(corrected_errors),
        "frames": frame_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({key: value for key, value in payload.items() if key != "frames"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
