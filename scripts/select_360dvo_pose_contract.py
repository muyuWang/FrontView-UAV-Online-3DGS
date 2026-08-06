#!/usr/bin/env python3
"""Select and certify a 360DVO pose contract from image correspondences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scripts.audit_360dvo_pose_contract import (
        pose_conventions,
        score_candidate,
        signed_axis_rotations,
        temporal_scores,
    )
    from scripts.refine_360dvo_rotation_prior import (
        load_samples,
        read_json,
        select_pair_paths,
    )
except ModuleNotFoundError:
    from audit_360dvo_pose_contract import (
        pose_conventions,
        score_candidate,
        signed_axis_rotations,
        temporal_scores,
    )
    from refine_360dvo_rotation_prior import (
        load_samples,
        read_json,
        select_pair_paths,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--max-matches-per-pair", type=int, default=256)
    parser.add_argument("--min-match-score", type=float, default=0.15)
    parser.add_argument("--epipolar-threshold-px", type=float, default=1.5)
    parser.add_argument("--temporal-bins", type=int, default=10)
    parser.add_argument("--max-frame-shift", type=int, default=8)
    parser.add_argument("--top-alignment-hypotheses", type=int, default=4)
    parser.add_argument("--min-validation-inlier-fraction", type=float, default=0.65)
    parser.add_argument("--min-bin-inlier-fraction", type=float, default=0.50)
    parser.add_argument("--min-passing-bin-fraction", type=float, default=0.80)
    parser.add_argument("--min-fit-winner-margin", type=float, default=0.05)
    return parser.parse_args()


def camera_matrix(cameras: list[dict]) -> np.ndarray:
    intrinsic = cameras[0]["intrinsic"]
    return np.asarray(
        [
            [intrinsic["fx"], 0.0, intrinsic["cx"]],
            [0.0, intrinsic["fy"], intrinsic["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def select_contract(
    rows: np.ndarray,
    fit: list[tuple[int, int, np.ndarray, np.ndarray]],
    validation: list[tuple[int, int, np.ndarray, np.ndarray]],
    k_inverse: np.ndarray,
    args: argparse.Namespace,
) -> tuple[object, np.ndarray, bool, dict]:
    if args.max_frame_shift < 0:
        raise ValueError("max-frame-shift must be non-negative")
    if args.top_alignment_hypotheses <= 0:
        raise ValueError("top-alignment-hypotheses must be positive")
    coarse_candidates = []
    conventions = pose_conventions(rows)
    axes = signed_axis_rotations()
    for convention in conventions:
        for reverse in (False, True):
            for axis_label, axis in axes:
                score = score_candidate(
                    convention,
                    axis,
                    fit,
                    k_inverse,
                    args.epipolar_threshold_px,
                    reverse=reverse,
                )
                coarse_candidates.append(
                    {
                        "convention": convention,
                        "axis_label": axis_label,
                        "axis": axis,
                        "reverse": reverse,
                        "fit": score,
                    }
                )
    coarse_candidates.sort(
        key=lambda item: (
            -item["fit"]["inlier_fraction"],
            item["fit"]["median_error_px"],
        )
    )
    candidates = []
    for coarse in coarse_candidates[: args.top_alignment_hypotheses]:
        shifted = []
        for shift in range(-args.max_frame_shift, args.max_frame_shift + 1):
            shifted.append(
                {
                    **coarse,
                    "shift": shift,
                    "fit": score_candidate(
                        coarse["convention"],
                        coarse["axis"],
                        fit,
                        k_inverse,
                        args.epipolar_threshold_px,
                        shift=shift,
                        reverse=coarse["reverse"],
                    ),
                }
            )
        shifted.sort(
            key=lambda item: (
                -item["fit"]["inlier_fraction"],
                item["fit"]["median_error_px"],
                abs(item["shift"]),
            )
        )
        candidates.append(shifted[0])
    candidates.sort(
        key=lambda item: (
            -item["fit"]["inlier_fraction"],
            item["fit"]["median_error_px"],
            abs(item["shift"]),
        )
    )
    winner = candidates[0]
    runner_up = candidates[1]
    winner_margin = (
        winner["fit"]["inlier_fraction"] - runner_up["fit"]["inlier_fraction"]
    )
    validation_score = score_candidate(
        winner["convention"],
        winner["axis"],
        validation,
        k_inverse,
        args.epipolar_threshold_px,
        shift=winner["shift"],
        reverse=winner["reverse"],
    )
    bins = temporal_scores(
        winner["convention"],
        winner["axis"],
        validation,
        k_inverse,
        args.epipolar_threshold_px,
        args.temporal_bins,
        shift=winner["shift"],
        reverse=winner["reverse"],
    )
    populated_bins = [item for item in bins if item["match_count"] > 0]
    passing_bins = sum(
        item["inlier_fraction"] >= args.min_bin_inlier_fraction
        for item in populated_bins
    )
    required_bins = int(np.ceil(args.temporal_bins * args.min_passing_bin_fraction))
    checks = {
        "fit_winner_margin": winner_margin >= args.min_fit_winner_margin,
        "validation_inlier_fraction": validation_score["inlier_fraction"]
        >= args.min_validation_inlier_fraction,
        "all_temporal_bins_populated": len(populated_bins) == args.temporal_bins,
        "passing_temporal_bins": passing_bins >= required_bins,
    }
    certificate = {
        "selection_scope": (
            "four quaternion/translation interpretations, 24 proper signed-axis "
            "rotations, forward/reversed sequence ordering, and bounded frame shift"
        ),
        "max_frame_shift": args.max_frame_shift,
        "top_alignment_hypotheses": args.top_alignment_hypotheses,
        "fit_pair_count": len(fit),
        "validation_pair_count": len(validation),
        "epipolar_threshold_px": args.epipolar_threshold_px,
        "selected_convention": winner["convention"].label,
        "selected_axis_label": winner["axis_label"],
        "selected_axis_matrix": winner["axis"].tolist(),
        "selected_reverse_order": winner["reverse"],
        "selected_frame_shift": winner["shift"],
        "fit": winner["fit"],
        "runner_up": {
            "convention": runner_up["convention"].label,
            "axis_label": runner_up["axis_label"],
            "reverse": runner_up["reverse"],
            "frame_shift": runner_up["shift"],
            **runner_up["fit"],
        },
        "fit_winner_margin": winner_margin,
        "required_fit_winner_margin": args.min_fit_winner_margin,
        "validation": validation_score,
        "required_validation_inlier_fraction": (args.min_validation_inlier_fraction),
        "temporal_bins": bins,
        "min_bin_inlier_fraction": args.min_bin_inlier_fraction,
        "passing_temporal_bin_count": passing_bins,
        "required_passing_temporal_bin_count": required_bins,
        "checks": checks,
        "accepted": all(checks.values()),
    }
    if not certificate["accepted"]:
        raise RuntimeError(f"Pose contract certificate failed: {certificate}")
    return (
        winner["convention"],
        winner["axis"],
        bool(winner["reverse"]),
        certificate,
    )


def alignment_indices(
    frame_count: int, shift: int, reverse: bool
) -> tuple[np.ndarray, np.ndarray]:
    image_start = max(0, -shift)
    image_end = min(frame_count, frame_count - shift)
    if image_start >= image_end:
        raise ValueError(
            f"Frame shift {shift} leaves no overlap for {frame_count} frames"
        )
    image_indices = np.arange(image_start, image_end, dtype=np.int64)
    pose_indices = image_indices + shift
    if reverse:
        pose_indices = frame_count - 1 - pose_indices
    return image_indices, pose_indices


def canonical_poses(
    convention: object, axis: np.ndarray, reverse: bool, shift: int = 0
) -> tuple[np.ndarray, np.ndarray, dict]:
    rotations = np.asarray(convention.rotations)
    centers = np.asarray(convention.centers)
    image_indices, pose_indices = alignment_indices(len(rotations), shift, reverse)
    rotations = rotations[pose_indices]
    centers = centers[pose_indices]
    camera_to_world = np.repeat(np.eye(4)[None], len(rotations), axis=0)
    camera_to_world[:, :3, :3] = rotations @ axis
    camera_to_world[:, :3, 3] = centers
    canonical_from_world = np.linalg.inv(camera_to_world[0])
    canonical_c2w = canonical_from_world[None] @ camera_to_world
    poses = np.linalg.inv(canonical_c2w)
    center_steps = np.linalg.norm(np.diff(canonical_c2w[:, :3, 3], axis=0), axis=1)
    coordinate = {
        "normalized_first_camera": True,
        "source_frame_count": len(convention.rotations),
        "selected_frame_count": len(image_indices),
        "source_image_start": int(image_indices[0]),
        "source_image_end_inclusive": int(image_indices[-1]),
        "source_pose_start": int(pose_indices[0]),
        "source_pose_end_inclusive": int(pose_indices[-1]),
        "frame_shift": int(shift),
        "reverse_order": bool(reverse),
        "first_pose_identity_max_abs": float(np.abs(poses[0] - np.eye(4)).max()),
        "trajectory_length_m": float(center_steps.sum()),
        "trajectory_step_m": {
            "median": float(np.median(center_steps)),
            "p95": float(np.percentile(center_steps, 95)),
            "max": float(center_steps.max()),
        },
    }
    return poses, image_indices, coordinate


def write_dataset(
    source: Path,
    output: Path,
    cameras: list[dict],
    poses: np.ndarray,
    image_indices: np.ndarray,
    certificate: dict,
    coordinate: dict,
    cache: Path,
) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {output}")
    staging = output.with_name(output.name + ".staging")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Stale staging directory exists: {staging}")
    rectified = staging / "rectified"
    rectified.mkdir(parents=True)
    selected_cameras = []
    for index, (source_index, pose) in enumerate(zip(image_indices, poses)):
        camera = cameras[int(source_index)]
        source_image = source / "rectified" / camera["image"]
        image_link = rectified / f"aria_{index:05d}.png"
        image_link.symlink_to(source_image.resolve())
        selected = dict(camera)
        selected["T_camera_world"] = pose.tolist()
        selected["image"] = image_link.name
        selected["frame_index"] = index
        selected["source_frame_index"] = int(source_index)
        selected_cameras.append(selected)
    trajectory = json.dumps({"cameras": selected_cameras}, indent=2) + "\n"
    (staging / "trajectory.json").write_text(trajectory, encoding="utf-8")
    (staging / "trajectory_orb.json").write_text(trajectory, encoding="utf-8")
    statistics = {
        "schema_version": 1,
        "source": str(source),
        "frame_count": len(selected_cameras),
        "source_frame_count": len(cameras),
        "pose_source": "360dvo_gt_auto_image_certified",
        "pose_mode": "automatic_pose_contract_selection",
        "sparse_world_geometry": "none_pose_only",
        "feature_cache": str(cache),
        "pose_contract_certificate": certificate,
        "coordinate_certificate": coordinate,
    }
    (staging / "conversion_stats.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
    )
    (staging / "README.md").write_text(
        "# Image-certified 360DVO pose contract\n\n"
        "The trajectory convention and virtual-camera axes are selected from a "
        "finite hypothesis set on image correspondences, then certified on "
        "disjoint held-out pairs and temporal bins. No scene-name rule is used. "
        "This pose-only dataset is intended for subsequent persistent-track "
        "triangulation.\n",
        encoding="utf-8",
    )
    staging.replace(output)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    cache = args.feature_cache.resolve()
    output = args.output.resolve()
    cameras = read_json(source / "trajectory_orb.json")["cameras"]
    source_stats = read_json(source / "conversion_stats.json")
    input_dir = Path(source_stats["input_dir"])
    rows = np.atleast_2d(np.loadtxt(input_dir / "trajectory.txt", dtype=np.float64))
    if len(rows) != len(cameras):
        raise RuntimeError(
            f"Trajectory rows={len(rows)} do not match cameras={len(cameras)}"
        )
    pair_paths = select_pair_paths(cache / "matches", args.max_pairs)
    fit, validation = load_samples(
        cache,
        pair_paths,
        args.min_match_score,
        args.max_matches_per_pair,
    )
    convention, axis, reverse, certificate = select_contract(
        rows,
        fit,
        validation,
        np.linalg.inv(camera_matrix(cameras)),
        args,
    )
    shift = int(certificate["selected_frame_shift"])
    poses, image_indices, coordinate = canonical_poses(convention, axis, reverse, shift)
    write_dataset(
        source,
        output,
        cameras,
        poses,
        image_indices,
        certificate,
        coordinate,
        cache,
    )
    print(
        f"Selected {certificate['selected_convention']} "
        f"{certificate['selected_axis_label']} reverse={reverse}; "
        f"shift={shift}; "
        f"held-out inliers={certificate['validation']['inlier_fraction']:.4f}"
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
