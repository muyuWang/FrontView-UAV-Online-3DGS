#!/usr/bin/env python3
"""Audit whether causal feature tracking can repair Mountains far depth."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_DATASET = (
    ROOT
    / "data/Online3DGS_360DVO_pose_contract_v4"
    / "mountains_auto_pose_contract_tracks"
)
DEFAULT_CACHE = (
    ROOT
    / "data/Online3DGS_360DVO_orbmono_fixed_dense_v3_cache"
    / "mountains_disklg2048"
)
DEFAULT_DEPTHCOV = (
    ROOT
    / "Logs_mountains_far_depth_goal_8_13/diagnostics/depthcov_full"
    / "depthcov_cross_validation_samples.json"
)
DEFAULT_OUTPUT = (
    ROOT / "Logs_mountains_far_depth_goal_8_13/diagnostics/parallax_tracking"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--depthcov-samples", type=Path, default=DEFAULT_DEPTHCOV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pair-gaps", default="1,2,4,8,16,32,64")
    parser.add_argument("--observation-radius-px", type=float, default=1.5)
    parser.add_argument("--feature-radius-px", type=float, default=1.0)
    parser.add_argument("--minimum-match-score", type=float, default=0.15)
    parser.add_argument("--maximum-epipolar-error-px", type=float, default=1.5)
    parser.add_argument("--pixel-sigma-px", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def finite_summary(values) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
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


def depth_error(predicted: float, truth: float) -> float:
    if predicted <= 0.0 or truth <= 0.0:
        return math.inf
    return abs(math.log(predicted) - math.log(truth))


def camera_centers(poses: np.ndarray) -> np.ndarray:
    return np.asarray(
        [-pose[:3, :3].T @ pose[:3, 3] for pose in poses], dtype=np.float64
    )


def fundamental_from_poses(
    pose0: np.ndarray, pose1: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    relative = pose1 @ np.linalg.inv(pose0)
    rotation, translation = relative[:3, :3], relative[:3, 3]
    skew = np.asarray(
        [
            [0.0, -translation[2], translation[1]],
            [translation[2], 0.0, -translation[0]],
            [-translation[1], translation[0], 0.0],
        ]
    )
    inverse = np.linalg.inv(intrinsics)
    return inverse.T @ skew @ rotation @ inverse


def sampson_error(fundamental: np.ndarray, pixel0, pixel1) -> float:
    first = np.asarray([pixel0[0], pixel0[1], 1.0], dtype=np.float64)
    second = np.asarray([pixel1[0], pixel1[1], 1.0], dtype=np.float64)
    line1 = fundamental @ first
    line0 = fundamental.T @ second
    numerator = float(second @ fundamental @ first) ** 2
    denominator = (
        line1[0] ** 2 + line1[1] ** 2 + line0[0] ** 2 + line0[1] ** 2
    )
    return math.sqrt(numerator / max(float(denominator), 1.0e-12))


def load_feature(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def nearest_index(points: np.ndarray, target: np.ndarray) -> tuple[int, float]:
    if not len(points):
        return -1, math.inf
    distances = np.linalg.norm(points - np.asarray(target).reshape(1, 2), axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def estimate_record(poses, intrinsics, current_frame, current_uv, references, sigma):
    from utils_new.frontview_parallax_depth_certificate import (
        certificate_information_gain,
        triangulate_parallax_depth,
    )

    if not references:
        return None
    estimates = []
    for reference in references:
        estimate = triangulate_parallax_depth(
            [poses[reference["frame"]], poses[current_frame]],
            intrinsics,
            [reference["uv"], current_uv],
            current_view_index=1,
            pixel_sigma_px=sigma,
        )
        if estimate.finite and estimate.minimum_positive_depth > 0.0:
            estimates.append((estimate, reference))
    if not estimates:
        return None
    estimate, reference = min(
        estimates,
        key=lambda item: (
            item[0].log_depth_std,
            float(np.max(item[0].reprojection_errors_px)),
        ),
    )
    prior_std = float(reference["prior_log_std"])
    return {
        "depth_m": float(estimate.current_depth),
        "log_depth_std": float(estimate.log_depth_std),
        "position_std_m": float(estimate.position_std),
        "maximum_parallax_deg": float(estimate.maximum_parallax_deg),
        "maximum_reprojection_error_px": float(
            np.max(estimate.reprojection_errors_px)
        ),
        "reference_frame": int(reference["frame"]),
        "match_score": reference.get("score"),
        "epipolar_error_px": reference.get("epipolar_error_px"),
        "information_gain": certificate_information_gain(
            prior_std, estimate.log_depth_std
        ),
    }


def estimate_consensus_record(
    poses, intrinsics, current_frame, current_uv, references, sigma
):
    from utils_new.frontview_parallax_depth_certificate import (
        certificate_information_gain,
        consensus_triangulate_parallax_depth,
    )

    if len(references) < 2:
        return None
    consensus = consensus_triangulate_parallax_depth(
        poses[current_frame],
        current_uv,
        [poses[reference["frame"]] for reference in references],
        [reference["uv"] for reference in references],
        intrinsics,
        pixel_sigma_px=sigma,
    )
    if consensus is None:
        return None
    estimate = consensus.estimate
    prior_std = float(references[0]["prior_log_std"])
    selected = [references[index] for index in consensus.reference_indices]
    return {
        "depth_m": float(estimate.current_depth),
        "log_depth_std": float(estimate.log_depth_std),
        "position_std_m": float(estimate.position_std),
        "maximum_parallax_deg": float(estimate.maximum_parallax_deg),
        "maximum_reprojection_error_px": float(
            np.max(estimate.reprojection_errors_px)
        ),
        "reference_frame": int(min(item["frame"] for item in selected)),
        "match_score": (
            None
            if selected[0].get("score") is None
            else float(min(item["score"] for item in selected))
        ),
        "epipolar_error_px": (
            None
            if selected[0].get("epipolar_error_px") is None
            else float(max(item["epipolar_error_px"] for item in selected))
        ),
        "information_gain": certificate_information_gain(
            prior_std, estimate.log_depth_std
        ),
        "support": int(consensus.support),
        "maximum_pairwise_chi2": float(consensus.maximum_pairwise_chi2),
    }


def summarize_method(rows: list[dict], prefix: str) -> dict:
    available = [row for row in rows if row.get(f"{prefix}_depth_m") is not None]
    if not available:
        return {"available": 0, "coverage": 0.0}
    errors = np.asarray(
        [depth_error(row[f"{prefix}_depth_m"], row["true_depth_m"]) for row in available]
    )
    depthcov = np.asarray(
        [depth_error(row["pred_depth_m"], row["true_depth_m"]) for row in available]
    )
    result = {
        "available": int(len(available)),
        "coverage": float(len(available) / max(len(rows), 1)),
        "absolute_log_error": finite_summary(errors),
        "depthcov_same_rows_absolute_log_error": finite_summary(depthcov),
        "error_improvement": finite_summary(depthcov - errors),
        "better_than_depthcov_fraction": float(np.mean(errors < depthcov)),
        "pred_over_true": finite_summary(
            row[f"{prefix}_depth_m"] / row["true_depth_m"] for row in available
        ),
        "log_depth_std": finite_summary(
            row[f"{prefix}_log_depth_std"] for row in available
        ),
        "maximum_parallax_deg": finite_summary(
            row[f"{prefix}_maximum_parallax_deg"] for row in available
        ),
        "maximum_reprojection_error_px": finite_summary(
            row[f"{prefix}_maximum_reprojection_error_px"] for row in available
        ),
    }
    certified = [
        row
        for row in available
        if row[f"{prefix}_maximum_reprojection_error_px"] <= 1.5
        and row[f"{prefix}_information_gain"] > math.log(math.sqrt(2.0))
    ]
    result["entropy_certified"] = {
        "count": int(len(certified)),
        "coverage": float(len(certified) / max(len(rows), 1)),
        "absolute_log_error": finite_summary(
            depth_error(row[f"{prefix}_depth_m"], row["true_depth_m"])
            for row in certified
        ),
        "depthcov_same_rows_absolute_log_error": finite_summary(
            depth_error(row["pred_depth_m"], row["true_depth_m"])
            for row in certified
        ),
    }
    return result


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    cache = args.feature_cache.expanduser().resolve()
    samples = json.loads(args.depthcov_samples.expanduser().read_text(encoding="utf-8"))
    trajectory = json.loads((dataset / "trajectory.json").read_text(encoding="utf-8"))
    cameras = trajectory["cameras"]
    poses = np.asarray([camera["T_camera_world"] for camera in cameras], dtype=np.float64)
    first_intrinsic = cameras[0]["intrinsic"]
    intrinsics = np.asarray(
        [
            [first_intrinsic["fx"], 0.0, first_intrinsic["cx"]],
            [0.0, first_intrinsic["fy"], first_intrinsic["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    world_points = np.load(dataset / "preprocess/global_sparse_points.npy")
    with np.load(dataset / "preprocess/track_observations.npz") as data:
        observation_ids = data["point_ids"]
        observation_frames = data["frame_ids"]
        observation_uv = data["uv"]
    observations_by_frame = defaultdict(list)
    for point_id, frame, uv in zip(
        observation_ids.tolist(), observation_frames.tolist(), observation_uv
    ):
        observations_by_frame[int(frame)].append((int(point_id), uv.astype(np.float64)))
    observations_by_point = defaultdict(list)
    for frame, observations in observations_by_frame.items():
        for point_id, uv in observations:
            observations_by_point[point_id].append((frame, uv))

    gaps = sorted({int(value) for value in args.pair_gaps.split(",") if value})
    feature_memory = {}
    match_memory = {}

    def feature(frame):
        if frame not in feature_memory:
            feature_memory[frame] = load_feature(cache / f"features/{frame:05d}.npz")
        return feature_memory[frame]

    def matches(reference, current):
        key = (reference, current)
        if key not in match_memory:
            path = cache / f"matches/{reference:05d}_{current:05d}.npz"
            if not path.is_file():
                match_memory[key] = None
            else:
                with np.load(path) as data:
                    match_memory[key] = {
                        "indices": data["indices"],
                        "scores": data["scores"].astype(np.float32),
                    }
        return match_memory[key]

    rng = np.random.default_rng(args.seed)
    evaluated = []
    mapping_failures = defaultdict(int)
    for sample in samples:
        frame = int(sample["frame"])
        frame_observations = observations_by_frame.get(frame, [])
        if not frame_observations:
            mapping_failures["frame_without_observations"] += 1
            continue
        target = np.asarray([sample["u"] - 0.5, sample["v"] - 0.5])
        frame_uv = np.asarray([item[1] for item in frame_observations])
        observation_index, observation_distance = nearest_index(frame_uv, target)
        if observation_distance > args.observation_radius_px:
            mapping_failures["no_track_near_depthcov_sample"] += 1
            continue
        point_id, current_uv = frame_observations[observation_index]
        current_features = feature(frame)
        current_feature_index, feature_distance = nearest_index(
            current_features["keypoints"], current_uv
        )
        if feature_distance > args.feature_radius_px:
            mapping_failures["no_feature_near_track_observation"] += 1
            continue

        prior_std = float(sample["pred_log_std"])
        oracle_references = [
            {
                "frame": int(reference_frame),
                "uv": np.asarray(reference_uv, dtype=np.float64),
                "prior_log_std": prior_std,
            }
            for reference_frame, reference_uv in observations_by_point[point_id]
            if reference_frame < frame and frame - reference_frame in gaps
        ]
        matched_references = []
        shuffled_references = []
        true_point = world_points[point_id]
        for gap in gaps:
            reference = frame - gap
            if reference < 0:
                continue
            match = matches(reference, frame)
            if match is None:
                continue
            rows = np.flatnonzero(match["indices"][:, 1] == current_feature_index)
            if not len(rows):
                continue
            best_row = int(rows[np.argmax(match["scores"][rows])])
            score = float(match["scores"][best_row])
            reference_feature_index = int(match["indices"][best_row, 0])
            reference_features = feature(reference)
            matched_uv = reference_features["keypoints"][reference_feature_index].astype(
                np.float64
            )
            fundamental = fundamental_from_poses(
                poses[reference], poses[frame], intrinsics
            )
            epipolar = sampson_error(fundamental, matched_uv, current_uv)
            projected_true, true_reference_depth = (
                __import__(
                    "utils_new.frontview_parallax_depth_certificate",
                    fromlist=["project_world_point"],
                ).project_world_point(poses[reference], intrinsics, true_point)
            )
            if score >= args.minimum_match_score and epipolar <= args.maximum_epipolar_error_px:
                matched_references.append(
                    {
                        "frame": reference,
                        "uv": matched_uv,
                        "score": score,
                        "epipolar_error_px": epipolar,
                        "prior_log_std": prior_std,
                        "true_match_error_px": float(
                            np.linalg.norm(matched_uv - projected_true)
                        ),
                    }
                )
                eligible = np.flatnonzero(match["scores"] >= args.minimum_match_score)
                if len(eligible) > 1:
                    shuffled_row = int(rng.choice(eligible))
                    if shuffled_row == best_row:
                        shuffled_row = int(eligible[(np.flatnonzero(eligible == shuffled_row)[0] + 1) % len(eligible)])
                    shuffled_index = int(match["indices"][shuffled_row, 0])
                    shuffled_uv = reference_features["keypoints"][shuffled_index].astype(
                        np.float64
                    )
                    shuffled_epipolar = sampson_error(
                        fundamental, shuffled_uv, current_uv
                    )
                    if shuffled_epipolar <= args.maximum_epipolar_error_px:
                        shuffled_references.append(
                            {
                                "frame": reference,
                                "uv": shuffled_uv,
                                "score": float(match["scores"][shuffled_row]),
                                "epipolar_error_px": shuffled_epipolar,
                                "prior_log_std": prior_std,
                            }
                        )

        oracle = estimate_record(
            poses, intrinsics, frame, current_uv, oracle_references, args.pixel_sigma_px
        )
        matched = estimate_record(
            poses, intrinsics, frame, current_uv, matched_references, args.pixel_sigma_px
        )
        shuffled = estimate_record(
            poses, intrinsics, frame, current_uv, shuffled_references, args.pixel_sigma_px
        )
        oracle_multi = estimate_consensus_record(
            poses, intrinsics, frame, current_uv, oracle_references, args.pixel_sigma_px
        )
        matched_multi = estimate_consensus_record(
            poses, intrinsics, frame, current_uv, matched_references, args.pixel_sigma_px
        )
        shuffled_multi = estimate_consensus_record(
            poses, intrinsics, frame, current_uv, shuffled_references, args.pixel_sigma_px
        )
        record = dict(sample)
        record.update(
            point_id=int(point_id),
            track_mapping_distance_px=observation_distance,
            current_feature_distance_px=feature_distance,
            causal_pair_count=len(matched_references),
            correct_pair_count=int(
                sum(ref["true_match_error_px"] <= 2.0 for ref in matched_references)
            ),
        )
        for prefix, estimate in (
            ("oracle", oracle),
            ("matched", matched),
            ("shuffled", shuffled),
            ("oracle_multi", oracle_multi),
            ("matched_multi", matched_multi),
            ("shuffled_multi", shuffled_multi),
        ):
            if estimate is not None:
                for name, value in estimate.items():
                    record[f"{prefix}_{name}"] = value
            else:
                record[f"{prefix}_depth_m"] = None
        evaluated.append(record)

    true_far = [row for row in evaluated if row["true_depth_m"] >= 50.0]
    accepted = [row for row in evaluated if row["passes_mapper_std"]]
    payload = {
        "status": "success",
        "protocol": "held-out persistent points; causal cached feature matching",
        "dataset": str(dataset),
        "feature_cache": str(cache),
        "depthcov_samples": str(args.depthcov_samples.expanduser().resolve()),
        "pair_gaps": gaps,
        "mapping_failures": dict(sorted(mapping_failures.items())),
        "evaluated_rows": len(evaluated),
        "direct_match_pair_rows": int(sum(row["causal_pair_count"] for row in evaluated)),
        "correct_direct_match_pair_rows": int(
            sum(row["correct_pair_count"] for row in evaluated)
        ),
        "direct_match_correct_fraction": (
            float(
                sum(row["correct_pair_count"] for row in evaluated)
                / max(sum(row["causal_pair_count"] for row in evaluated), 1)
            )
        ),
        "overall": {
            name: summarize_method(evaluated, name)
            for name in (
                "oracle", "matched", "shuffled",
                "oracle_multi", "matched_multi", "shuffled_multi",
            )
        },
        "mapper_accepted": {
            name: summarize_method(accepted, name)
            for name in (
                "oracle", "matched", "shuffled",
                "oracle_multi", "matched_multi", "shuffled_multi",
            )
        },
        "true_far_ge_50m": {
            name: summarize_method(true_far, name)
            for name in (
                "oracle", "matched", "shuffled",
                "oracle_multi", "matched_multi", "shuffled_multi",
            )
        },
    }
    output = args.output.expanduser().resolve()
    atomic_json(output / "parallax_tracking_audit.json", payload)
    atomic_json(output / "parallax_tracking_samples.json", evaluated)
    print(json.dumps(payload, indent=2))
    print(output / "parallax_tracking_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
