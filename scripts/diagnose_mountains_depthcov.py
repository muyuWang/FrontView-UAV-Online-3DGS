#!/usr/bin/env python3
"""Cross-validate DepthCov against held-out persistent world tracks."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = (
    ROOT
    / "Logs_mountains_adaptive_goal_8_12_8_13"
    / "final/stage35_full_765/batch_20260813_095909"
    / "runtime_configs/A_visible_residual_detail_real.yaml"
)
DEFAULT_OUTPUT = ROOT / "Logs_mountains_far_depth_goal_8_13/diagnostics/depthcov"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--min-training", type=int, default=8)
    parser.add_argument("--min-holdout", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cidec", action="store_true")
    parser.add_argument("--cidec-track-support", action="store_true")
    parser.add_argument("--cidec-causal-frustum", action="store_true")
    parser.add_argument("--cidec-shuffle-support", action="store_true")
    parser.add_argument("--cidec-shuffle", action="store_true")
    parser.add_argument("--cidec-reference-frames", type=int, default=3)
    parser.add_argument("--cidec-history-frames", type=int, default=24)
    parser.add_argument("--cidec-hypotheses", type=int, default=17)
    parser.add_argument("--cidec-support-neighbors", type=int, default=8)
    parser.add_argument(
        "--cidec-support-confidence-chi2",
        type=float,
        default=3.841458820694124,
    )
    parser.add_argument("--cidec-mode-nll-margin-min", type=float, default=0.0)
    parser.add_argument("--cidec-leave-one-out", action="store_true")
    parser.add_argument(
        "--cidec-photometric-cost",
        choices=("centered_l1", "zncc"),
        default="centered_l1",
    )
    parser.add_argument(
        "--cidec-view-aggregation",
        choices=("mean", "median", "consensus"),
        default="mean",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def finite_quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: None for key in ("mean", "median", "p75", "p90", "p95")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    true = np.asarray([row["true_depth_m"] for row in rows])
    pred = np.asarray([row["pred_depth_m"] for row in rows])
    std = np.asarray([row["pred_log_std"] for row in rows])
    error = np.abs(np.log(pred) - np.log(true))
    valid = np.asarray([row["passes_mapper_std"] for row in rows], dtype=bool)
    ratio = pred / true
    result = {
        "count": int(len(rows)),
        "true_depth_m": finite_quantiles(true),
        "pred_depth_m": finite_quantiles(pred),
        "pred_over_true": finite_quantiles(ratio),
        "absolute_log_error": finite_quantiles(error),
        "pred_log_std": finite_quantiles(std),
        "mapper_std_pass_fraction": float(np.mean(valid)),
    }
    if np.any(valid):
        accepted_error = error[valid]
        result["accepted"] = {
            "count": int(np.sum(valid)),
            "absolute_log_error": finite_quantiles(accepted_error),
            "error_gt_0p06_fraction": float(np.mean(accepted_error > 0.06)),
            "error_gt_0p10_fraction": float(np.mean(accepted_error > 0.10)),
            "error_gt_0p20_fraction": float(np.mean(accepted_error > 0.20)),
            "under_depth_20pct_fraction": float(np.mean(ratio[valid] < 0.8)),
            "over_depth_20pct_fraction": float(np.mean(ratio[valid] > 1.2)),
        }
    if "cidec_depth_m" in rows[0]:
        cidec_depth = np.asarray([row["cidec_depth_m"] for row in rows])
        certified = np.asarray(
            [row["cidec_certified"] for row in rows], dtype=np.bool_
        )
        conflicted = np.asarray(
            [row["cidec_conflicted"] for row in rows], dtype=np.bool_
        )
        cidec_error = np.abs(np.log(cidec_depth) - np.log(true))
        cidec_result = {
            "certified_count": int(certified.sum()),
            "certified_fraction": float(certified.mean()),
            "conflicted_count": int(conflicted.sum()),
            "conflicted_fraction": float(conflicted.mean()),
        }
        if np.any(certified):
            cidec_result.update(
                certified_absolute_log_error=finite_quantiles(cidec_error[certified]),
                certified_pred_over_true=finite_quantiles(
                    cidec_depth[certified] / true[certified]
                ),
                depthcov_same_rows_absolute_log_error=finite_quantiles(
                    error[certified]
                ),
                error_improvement=finite_quantiles(
                    error[certified] - cidec_error[certified]
                ),
            )
        result["cidec"] = cidec_result
    if "support_contains_true" in rows[0]:
        contains = np.asarray(
            [row["support_contains_true"] for row in rows], dtype=np.bool_
        )
        oracle_error = np.asarray(
            [row["support_oracle_log_error"] for row in rows], dtype=np.float64
        )
        support_result = {
            "coverage_count": int(contains.sum()),
            "coverage_fraction": float(contains.mean()),
            "oracle_absolute_log_error": finite_quantiles(oracle_error),
        }
        if np.any(contains):
            support_result["covered_oracle_absolute_log_error"] = finite_quantiles(
                oracle_error[contains]
            )
        if "cidec_depth_m" in rows[0]:
            cidec_depth = np.asarray([row["cidec_depth_m"] for row in rows])
            certified = np.asarray(
                [row["cidec_certified"] for row in rows], dtype=np.bool_
            )
            selected_error = np.abs(np.log(cidec_depth) - np.log(true))
            covered_certified = contains & certified
            missed_certified = ~contains & certified
            support_result.update(
                certified_covered_count=int(covered_certified.sum()),
                certified_domain_miss_count=int(missed_certified.sum()),
                certified_coverage_precision=(
                    float(covered_certified.sum() / certified.sum())
                    if np.any(certified)
                    else None
                ),
                certification_fraction_when_covered=(
                    float(certified[contains].mean()) if np.any(contains) else None
                ),
                certification_fraction_when_missed=(
                    float(certified[~contains].mean()) if np.any(~contains) else None
                ),
            )
            if np.any(covered_certified):
                support_result[
                    "covered_certified_selected_absolute_log_error"
                ] = finite_quantiles(selected_error[covered_certified])
                support_result[
                    "covered_certified_excess_over_oracle"
                ] = finite_quantiles(
                    selected_error[covered_certified]
                    - oracle_error[covered_certified]
                )
        result["track_support_domain"] = support_result
    return result


def main() -> int:
    args = parse_args()
    if args.stride <= 0 or args.min_training <= 0 or args.min_holdout <= 0:
        raise ValueError("Stride and split sizes must be positive")
    if not 0.0 < args.holdout_fraction < 0.5:
        raise ValueError("--holdout-fraction must lie in (0, 0.5)")

    from utils_new.dataset import ArialDataset
    from depth_cov.depth_cov_estimator import DepthCovEstimator
    from utils_new.tool_utils import load_config
    from utils_new.frontview_inverse_depth_certificate import (
        causal_frustum_inverse_depth_hypotheses,
        causal_inverse_depth_posterior,
        locally_track_supported_inverse_depth_hypotheses,
    )

    if args.cidec_track_support and args.cidec_causal_frustum:
        raise ValueError("Choose either local track support or causal frustum")

    config = load_config(str(args.config.expanduser().resolve()))
    dataset_config = dict(config["Dataset"])
    dataset_config["num_threads"] = 0
    dataset_config["scene_exposure_gain"] = float(
        config.get("Mapper", {}).get("scene_exposure_gain", 20.0)
    )
    dataset = ArialDataset(dataset_config)
    estimator_config = dict(config["Model"]["DepthCovEstimator"])
    estimator_config["device"] = args.device
    estimator = DepthCovEstimator(estimator_config)

    rows: list[dict] = []
    frame_rows: list[dict] = []
    skipped = {"too_few_tracks": 0, "no_finite_prediction": 0}
    reference_cameras = []
    for frame_index in range(0, len(dataset), args.stride):
        camera = dataset[frame_index]
        sparse_depth = camera.get_sparse_depth(0)
        coords_yx = torch.nonzero(sparse_depth > 0.0, as_tuple=False)
        count = int(len(coords_yx))
        minimum = args.min_training + args.min_holdout
        if count < minimum:
            skipped["too_few_tracks"] += 1
            reference_cameras.append(camera)
            reference_cameras = reference_cameras[-args.cidec_history_frames :]
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + frame_index)
        order = torch.randperm(count, generator=generator)
        holdout_count = max(args.min_holdout, int(round(count * args.holdout_fraction)))
        holdout_count = min(holdout_count, count - args.min_training)
        holdout = order[:holdout_count]
        training = order[holdout_count:]
        coords_xy = torch.stack(
            (coords_yx[:, 1].float() + 0.5, coords_yx[:, 0].float() + 0.5), dim=1
        )
        depths = sparse_depth[coords_yx[:, 0], coords_yx[:, 1]]
        pred, passes, pred_std = estimator.query_tensor(
            camera.get_gt_image(0),
            depths[training].to(args.device),
            coords_xy[training].to(args.device),
            coords_xy[holdout].to(args.device),
            return_std=True,
        )
        pred = pred.detach().cpu().numpy()
        passes = passes.detach().cpu().numpy()
        pred_std = pred_std.detach().cpu().numpy()
        true = depths[holdout].cpu().numpy()
        uv = coords_xy[holdout].cpu().numpy()
        finite = (
            np.isfinite(pred)
            & np.isfinite(pred_std)
            & np.isfinite(true)
            & (pred > 0.0)
            & (true > 0.0)
        )
        if not np.any(finite):
            skipped["no_finite_prediction"] += 1
            reference_cameras.append(camera)
            reference_cameras = reference_cameras[-args.cidec_history_frames :]
            continue
        cidec = None
        support_diagnostics = None
        if args.cidec:
            if args.cidec_track_support or args.cidec_causal_frustum:
                if args.cidec_causal_frustum:
                    shared_inverse = causal_frustum_inverse_depth_hypotheses(
                        depths[training].to(args.device),
                        torch.full_like(depths[training].to(args.device), 0.02),
                        args.cidec_hypotheses,
                        far_depth=float(camera.far),
                        confidence_chi2=args.cidec_support_confidence_chi2,
                    )
                    support_inverse = shared_inverse[:, None].expand(
                        -1, len(holdout)
                    )
                else:
                    support_inverse = locally_track_supported_inverse_depth_hypotheses(
                        coords_xy[holdout].to(args.device),
                        coords_xy[training].to(args.device),
                        depths[training].to(args.device),
                        torch.full_like(depths[training].to(args.device), 0.02),
                        args.cidec_hypotheses,
                        neighbors=args.cidec_support_neighbors,
                        confidence_chi2=args.cidec_support_confidence_chi2,
                        shuffle_binding=args.cidec_shuffle_support,
                        seed=args.seed + int(camera.cam_idx),
                    )
                support_depth_hypotheses = torch.reciprocal(
                    torch.clamp(support_inverse, min=1.0e-8)
                )
                true_depth = depths[holdout].to(args.device)
                true_inverse = torch.reciprocal(torch.clamp(true_depth, min=1.0e-8))
                support_lower = support_inverse.amin(dim=0)
                support_upper = support_inverse.amax(dim=0)
                support_diagnostics = {
                    "contains": (
                        (true_inverse >= support_lower)
                        & (true_inverse <= support_upper)
                    ).detach().cpu().numpy(),
                    "oracle_error": torch.amin(
                        torch.abs(
                            torch.log(support_depth_hypotheses)
                            - torch.log(true_depth)[None]
                        ),
                        dim=0,
                    ).detach().cpu().numpy(),
                    "minimum_depth": support_depth_hypotheses.amin(dim=0)
                    .detach()
                    .cpu()
                    .numpy(),
                    "maximum_depth": support_depth_hypotheses.amax(dim=0)
                    .detach()
                    .cpu()
                    .numpy(),
                }
            cidec = causal_inverse_depth_posterior(
                camera,
                reference_cameras,
                coords_xy[holdout].to(args.device),
                torch.as_tensor(pred, device=args.device),
                torch.as_tensor(pred_std, device=args.device),
                0,
                {
                    "enabled": True,
                    "hypothesis_source": (
                        "causal_frustum"
                        if args.cidec_causal_frustum
                        else (
                            "track_support"
                            if args.cidec_track_support
                            else "local_prior"
                        )
                    ),
                    "shuffle_support_binding": args.cidec_shuffle_support,
                    "reference_frames": args.cidec_reference_frames,
                    "history_frames": args.cidec_history_frames,
                    "hypotheses": args.cidec_hypotheses,
                    "support_neighbors": args.cidec_support_neighbors,
                    "support_confidence_chi2": args.cidec_support_confidence_chi2,
                    "mode_nll_margin_min": args.cidec_mode_nll_margin_min,
                    "leave_one_out_consistency": args.cidec_leave_one_out,
                    "minimum_log_depth_span": 1.0,
                    "photometric_cost": args.cidec_photometric_cost,
                    "view_aggregation": args.cidec_view_aggregation,
                    "shuffle_evidence": args.cidec_shuffle,
                    "shuffle_seed": args.seed,
                },
                support_depths=depths[training].to(args.device),
                support_log_depth_stds=torch.full_like(
                    depths[training].to(args.device), 0.02
                ),
                support_uv=coords_xy[training].to(args.device),
            )
            cidec = {
                key: value.detach().cpu().numpy()
                for key, value in cidec.items()
            }
        local_rows = []
        for sample_index in np.flatnonzero(finite):
            row = {
                "frame": int(frame_index),
                "u": float(uv[sample_index, 0]),
                "v": float(uv[sample_index, 1]),
                "true_depth_m": float(true[sample_index]),
                "pred_depth_m": float(pred[sample_index]),
                "pred_log_std": float(pred_std[sample_index]),
                "passes_mapper_std": bool(passes[sample_index]),
            }
            if cidec is not None:
                row.update(
                    cidec_depth_m=float(cidec["depths"][sample_index]),
                    cidec_posterior_log_std=float(
                        cidec["posterior_log_stds"][sample_index]
                    ),
                    cidec_certified=bool(cidec["certified"][sample_index]),
                    cidec_conflicted=bool(cidec["conflicted"][sample_index]),
                    cidec_information_gain=float(
                        cidec["information_gain"][sample_index]
                    ),
                    cidec_valid_views=int(cidec["valid_views"][sample_index]),
                    cidec_baseline_information=float(
                        cidec["baseline_information"][sample_index]
                    ),
                    cidec_mode_nll_margin=float(
                        cidec["mode_nll_margin"][sample_index]
                    ),
                    cidec_leave_one_out_views=int(
                        cidec["leave_one_out_views"][sample_index]
                    ),
                    cidec_leave_one_out_chi2=float(
                        cidec["leave_one_out_chi2"][sample_index]
                    ),
                )
            if support_diagnostics is not None:
                row.update(
                    support_contains_true=bool(
                        support_diagnostics["contains"][sample_index]
                    ),
                    support_oracle_log_error=float(
                        support_diagnostics["oracle_error"][sample_index]
                    ),
                    support_minimum_depth_m=float(
                        support_diagnostics["minimum_depth"][sample_index]
                    ),
                    support_maximum_depth_m=float(
                        support_diagnostics["maximum_depth"][sample_index]
                    ),
                )
            rows.append(row)
            local_rows.append(row)
        frame_rows.append(
            {
                "frame": int(frame_index),
                "sparse_tracks": count,
                "training_tracks": int(len(training)),
                "held_out_tracks": int(len(local_rows)),
                "summary": summarize(local_rows),
            }
        )
        reference_cameras.append(camera)
        reference_cameras = reference_cameras[-args.cidec_history_frames :]

    depth_bands = {
        "near_lt_20m": [row for row in rows if row["true_depth_m"] < 20.0],
        "mid_20_50m": [row for row in rows if 20.0 <= row["true_depth_m"] < 50.0],
        "far_ge_50m": [row for row in rows if row["true_depth_m"] >= 50.0],
        "far_ge_80m": [row for row in rows if row["true_depth_m"] >= 80.0],
    }
    temporal_bands = {}
    bin_count = 10
    for index in range(bin_count):
        start = math.floor(index * len(dataset) / bin_count)
        end = math.floor((index + 1) * len(dataset) / bin_count)
        temporal_bands[f"{start:04d}_{end - 1:04d}"] = [
            row for row in rows if start <= row["frame"] < end
        ]
    payload = {
        "status": "success",
        "protocol": "held-out persistent-track DepthCov cross-validation",
        "config": str(args.config.expanduser().resolve()),
        "dataset": dataset_config["dataset_path"],
        "device": args.device,
        "stride": args.stride,
        "holdout_fraction": args.holdout_fraction,
        "cidec": args.cidec,
        "cidec_track_support": args.cidec_track_support,
        "cidec_causal_frustum": args.cidec_causal_frustum,
        "cidec_shuffle_support": args.cidec_shuffle_support,
        "cidec_shuffle": args.cidec_shuffle,
        "cidec_photometric_cost": args.cidec_photometric_cost,
        "cidec_view_aggregation": args.cidec_view_aggregation,
        "cidec_support_neighbors": args.cidec_support_neighbors,
        "cidec_support_confidence_chi2": args.cidec_support_confidence_chi2,
        "cidec_mode_nll_margin_min": args.cidec_mode_nll_margin_min,
        "cidec_leave_one_out": args.cidec_leave_one_out,
        "mapper_log_std_threshold": float(estimator.std_valid_threshold),
        "evaluated_frames": len(frame_rows),
        "skipped_frames": skipped,
        "overall": summarize(rows),
        "by_true_depth": {name: summarize(band) for name, band in depth_bands.items()},
        "by_sequence_bin": {
            name: summarize(band) for name, band in temporal_bands.items()
        },
        "frames": frame_rows,
    }
    output = args.output.expanduser().resolve()
    atomic_json(output / "depthcov_cross_validation.json", payload)
    atomic_json(output / "depthcov_cross_validation_samples.json", rows)
    print(json.dumps({key: payload[key] for key in ("status", "evaluated_frames", "overall", "by_true_depth")}, indent=2))
    print(output / "depthcov_cross_validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
