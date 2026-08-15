#!/usr/bin/env python3
"""Cross-validate causal landmark-memory conditioning on Mountains."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = (
    ROOT
    / "Logs_mountains_adaptive_goal_8_12_8_13"
    / "final/stage35_full_765/batch_20260813_095909"
    / "runtime_configs/A_visible_residual_detail_real.yaml"
)
DEFAULT_OUTPUT = (
    ROOT / "Logs_mountains_far_depth_goal_8_13/diagnostics/landmark_memory"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=("baseline", "real_novel", "shuffled_novel", "recurrent_allowed"),
        required=True,
    )
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--min-training", type=int, default=1)
    parser.add_argument("--min-holdout", type=int, default=1)
    parser.add_argument("--conditioning-budget", type=int, default=500)
    parser.add_argument("--minimum-observations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def finite_summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def summarize(rows):
    if not rows:
        return {"count": 0}
    true = np.asarray([row["true_depth_m"] for row in rows], dtype=np.float64)
    pred = np.asarray([row["pred_depth_m"] for row in rows], dtype=np.float64)
    std = np.asarray([row["pred_log_std"] for row in rows], dtype=np.float64)
    passed = np.asarray([row["passes_mapper_std"] for row in rows], dtype=np.bool_)
    error = np.abs(np.log(pred) - np.log(true))
    ratio = pred / true
    result = {
        "count": int(len(rows)),
        "true_depth_m": finite_summary(true),
        "pred_depth_m": finite_summary(pred),
        "pred_over_true": finite_summary(ratio),
        "absolute_log_error": finite_summary(error),
        "pred_log_std": finite_summary(std),
        "mapper_std_pass_fraction": float(np.mean(passed)),
        "conditioning_points": finite_summary(
            [row["conditioning_points"] for row in rows]
        ),
        "memory_points": finite_summary([row["memory_points"] for row in rows]),
    }
    if np.any(passed):
        accepted_error = error[passed]
        result["accepted"] = {
            "count": int(np.sum(passed)),
            "absolute_log_error": finite_summary(accepted_error),
            "error_gt_0p06_fraction": float(np.mean(accepted_error > 0.06)),
            "error_gt_0p10_fraction": float(np.mean(accepted_error > 0.10)),
            "error_gt_0p20_fraction": float(np.mean(accepted_error > 0.20)),
            "under_depth_20pct_fraction": float(np.mean(ratio[passed] < 0.8)),
            "over_depth_20pct_fraction": float(np.mean(ratio[passed] > 1.2)),
        }
    return result


def main():
    args = parse_args()
    if args.stride < 1 or args.min_training < 1 or args.min_holdout < 1:
        raise ValueError("Stride and split sizes must be positive")
    if not 0.0 < args.holdout_fraction < 0.5:
        raise ValueError("Holdout fraction must lie in (0, 0.5)")
    if args.conditioning_budget < args.min_training:
        raise ValueError("Conditioning budget is smaller than minimum training set")

    from depth_cov.depth_cov_estimator import DepthCovEstimator
    from utils_new.dataset import ArialDataset
    from utils_new.frontview_causal_landmark_memory import (
        CausalPersistentLandmarkMemory,
    )
    from utils_new.tool_utils import load_config

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
    memory = CausalPersistentLandmarkMemory(
        {
            "enabled": args.mode != "baseline",
            "minimum_observations": args.minimum_observations,
            "maximum_conditioning_points": args.conditioning_budget,
            "shuffle_depths": args.mode == "shuffled_novel",
            "shuffle_seed": args.seed,
        }
    )

    rows = []
    frame_rows = []
    skipped = {"too_few_tracks": 0, "no_finite_prediction": 0}
    for frame_index in range(len(dataset)):
        camera = dataset[frame_index]
        sparse_depth = camera.get_sparse_depth(0)
        coords_yx = torch.nonzero(sparse_depth > 0.0, as_tuple=False)
        count = int(len(coords_yx))
        evaluate = frame_index % args.stride == 0
        minimum = args.min_training + args.min_holdout
        if evaluate and count >= minimum:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(args.seed + frame_index)
            order = torch.randperm(count, generator=generator)
            holdout_count = max(
                args.min_holdout, int(round(count * args.holdout_fraction))
            )
            holdout_count = min(holdout_count, count - args.min_training)
            holdout = order[:holdout_count]
            training = order[holdout_count:]
            if len(training) > args.conditioning_budget:
                training = training[: args.conditioning_budget]
            coords_xy = torch.stack(
                (coords_yx[:, 1].float() + 0.5, coords_yx[:, 0].float() + 0.5),
                dim=1,
            )
            depths = sparse_depth[coords_yx[:, 0], coords_yx[:, 1]]
            train_uv = coords_xy[training]
            train_depth = depths[training]
            memory_count = 0
            if args.mode != "baseline":
                sparse_ids = camera.get_sparse_point_ids(0).detach().cpu().numpy()
                if len(sparse_ids) != count:
                    raise RuntimeError("Sparse point identities do not align with depth")
                if args.mode in ("real_novel", "shuffled_novel"):
                    exclude_ids = camera.get_point_ids()
                else:
                    exclude_ids = sparse_ids[training.numpy()]
                occupied = (
                    coords_yx[training, 0] * camera.get_width(0)
                    + coords_yx[training, 1]
                ).numpy()
                landmark_batch = memory.project(
                    camera,
                    0,
                    exclude_ids=exclude_ids,
                    occupied_pixel_indices=occupied,
                    maximum_points=max(0, args.conditioning_budget - len(training)),
                )
                memory_count = len(landmark_batch)
                if memory_count:
                    train_uv = torch.cat(
                        (train_uv, torch.from_numpy(landmark_batch.uv)), dim=0
                    )
                    train_depth = torch.cat(
                        (train_depth, torch.from_numpy(landmark_batch.depths)), dim=0
                    )
            pred, passed, pred_std = estimator.query_tensor(
                camera.get_gt_image(0),
                train_depth.to(args.device),
                train_uv.to(args.device),
                coords_xy[holdout].to(args.device),
                return_std=True,
            )
            pred = pred.detach().cpu().numpy()
            passed = passed.detach().cpu().numpy()
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
            local_rows = []
            for sample_index in np.flatnonzero(finite):
                row = {
                    "frame": int(frame_index),
                    "u": float(uv[sample_index, 0]),
                    "v": float(uv[sample_index, 1]),
                    "true_depth_m": float(true[sample_index]),
                    "pred_depth_m": float(pred[sample_index]),
                    "pred_log_std": float(pred_std[sample_index]),
                    "passes_mapper_std": bool(passed[sample_index]),
                    "current_training_points": int(len(training)),
                    "memory_points": int(memory_count),
                    "conditioning_points": int(len(train_depth)),
                }
                rows.append(row)
                local_rows.append(row)
            if not local_rows:
                skipped["no_finite_prediction"] += 1
            frame_rows.append(
                {
                    "frame": int(frame_index),
                    "sparse_tracks": count,
                    "training_tracks": int(len(training)),
                    "memory_tracks": int(memory_count),
                    "held_out_tracks": int(len(local_rows)),
                    "summary": summarize(local_rows),
                }
            )
        elif evaluate:
            skipped["too_few_tracks"] += 1
        memory.observe(camera)

    depth_bands = {
        "near_lt_20m": [row for row in rows if row["true_depth_m"] < 20.0],
        "mid_20_50m": [
            row for row in rows if 20.0 <= row["true_depth_m"] < 50.0
        ],
        "far_ge_50m": [row for row in rows if row["true_depth_m"] >= 50.0],
        "far_ge_80m": [row for row in rows if row["true_depth_m"] >= 80.0],
    }
    temporal_bands = {}
    for index in range(10):
        start = math.floor(index * len(dataset) / 10)
        end = math.floor((index + 1) * len(dataset) / 10)
        temporal_bands[f"{start:04d}_{end - 1:04d}"] = [
            row for row in rows if start <= row["frame"] < end
        ]
    payload = {
        "status": "success",
        "protocol": "strictly causal held-out persistent-track DepthCov conditioning",
        "mode": args.mode,
        "config": str(args.config.expanduser().resolve()),
        "dataset": dataset_config["dataset_path"],
        "device": args.device,
        "stride": args.stride,
        "holdout_fraction": args.holdout_fraction,
        "conditioning_budget": args.conditioning_budget,
        "minimum_observations": args.minimum_observations,
        "mapper_log_std_threshold": float(estimator.std_valid_threshold),
        "evaluated_frames": len(frame_rows),
        "skipped_frames": skipped,
        "overall": summarize(rows),
        "by_true_depth": {
            name: summarize(band) for name, band in depth_bands.items()
        },
        "frame_588": summarize([row for row in rows if row["frame"] == 588]),
        "by_sequence_bin": {
            name: summarize(band) for name, band in temporal_bands.items()
        },
        "memory": memory.summary(),
        "frames": frame_rows,
    }
    output = args.output.expanduser().resolve() / args.mode
    atomic_json(output / "landmark_memory_cross_validation.json", payload)
    atomic_json(output / "landmark_memory_cross_validation_samples.json", rows)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "mode": args.mode,
                "evaluated_frames": payload["evaluated_frames"],
                "overall": payload["overall"],
                "far_ge_50m": payload["by_true_depth"]["far_ge_50m"],
                "frame_588": payload["frame_588"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
