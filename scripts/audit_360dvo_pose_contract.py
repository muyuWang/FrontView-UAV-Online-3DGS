#!/usr/bin/env python3
"""Audit 360DVO pose conventions against cached image correspondences."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from scripts.refine_360dvo_rotation_prior import (
        load_samples,
        read_json,
        sampson_errors,
        select_pair_paths,
    )
except ModuleNotFoundError:
    from refine_360dvo_rotation_prior import (
        load_samples,
        read_json,
        sampson_errors,
        select_pair_paths,
    )


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", required=True)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data" / "Online3DGS_360DVO"
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "data" / "Online3DGS_360DVO_orbmono_fixed_dense_v3_cache",
    )
    parser.add_argument("--cache-suffix", default="disklg2048")
    parser.add_argument("--max-pairs", type=int, default=192)
    parser.add_argument("--max-matches-per-pair", type=int, default=96)
    parser.add_argument("--min-match-score", type=float, default=0.15)
    parser.add_argument("--epipolar-threshold-px", type=float, default=1.5)
    parser.add_argument("--max-frame-shift", type=int, default=8)
    parser.add_argument("--temporal-bins", type=int, default=10)
    parser.add_argument("--top-conventions", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def quaternion_rotations(rows: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(rows[:, 3:7]).as_matrix()


def signed_axis_rotations() -> list[tuple[str, np.ndarray]]:
    candidates = []
    basis = np.eye(3, dtype=np.float64)
    for permutation in (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ):
        for signs in np.ndindex(2, 2, 2):
            sign_values = np.asarray(
                [1.0 if value else -1.0 for value in signs], dtype=np.float64
            )
            matrix = basis[:, permutation] * sign_values[None]
            if np.linalg.det(matrix) < 0.0:
                continue
            label = "p{}-s{}".format(
                "".join(str(value) for value in permutation),
                "".join("p" if value > 0 else "m" for value in sign_values),
            )
            candidates.append((label, matrix))
    return candidates


@dataclass(frozen=True)
class PoseConvention:
    label: str
    rotations: np.ndarray
    centers: np.ndarray


def pose_conventions(rows: np.ndarray) -> list[PoseConvention]:
    rotations = quaternion_rotations(rows)
    values = rows[:, :3]
    return [
        PoseConvention("quat_c2w_xyz_center", rotations, values),
        PoseConvention("quat_w2c_xyz_center", rotations.transpose(0, 2, 1), values),
        PoseConvention(
            "rows_are_w2c_matrix",
            rotations.transpose(0, 2, 1),
            -np.einsum("nij,nj->ni", rotations.transpose(0, 2, 1), values),
        ),
        PoseConvention(
            "inverse_rotation_xyz_w2c_translation",
            rotations,
            -np.einsum("nij,nj->ni", rotations, values),
        ),
    ]


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def fundamental(
    rotation_i: np.ndarray,
    center_i: np.ndarray,
    rotation_j: np.ndarray,
    center_j: np.ndarray,
    k_inverse: np.ndarray,
) -> np.ndarray:
    relative_rotation = rotation_j.T @ rotation_i
    relative_translation = rotation_j.T @ (center_i - center_j)
    essential = skew(relative_translation) @ relative_rotation
    return k_inverse.T @ essential @ k_inverse


def pose_index(frame: int, count: int, shift: int, reverse: bool) -> int:
    shifted = frame + shift
    return count - 1 - shifted if reverse else shifted


def score_candidate(
    convention: PoseConvention,
    axis: np.ndarray,
    samples: list[tuple[int, int, np.ndarray, np.ndarray]],
    k_inverse: np.ndarray,
    threshold: float,
    *,
    shift: int = 0,
    reverse: bool = False,
) -> dict:
    count = len(convention.rotations)
    errors = []
    used_pairs = 0
    for frame_i, frame_j, points_i, points_j in samples:
        pose_i = pose_index(frame_i, count, shift, reverse)
        pose_j = pose_index(frame_j, count, shift, reverse)
        if not (0 <= pose_i < count and 0 <= pose_j < count):
            continue
        errors.append(
            sampson_errors(
                fundamental(
                    convention.rotations[pose_i] @ axis,
                    convention.centers[pose_i],
                    convention.rotations[pose_j] @ axis,
                    convention.centers[pose_j],
                    k_inverse,
                ),
                points_i,
                points_j,
            )
        )
        used_pairs += 1
    if not errors:
        return {
            "pair_count": 0,
            "match_count": 0,
            "inlier_fraction": 0.0,
            "median_error_px": None,
            "p95_error_px": None,
        }
    values = np.concatenate(errors)
    return {
        "pair_count": used_pairs,
        "match_count": len(values),
        "inlier_fraction": float(np.mean(values <= threshold)),
        "median_error_px": float(np.median(values)),
        "p95_error_px": float(np.percentile(values, 95)),
    }


def temporal_scores(
    convention: PoseConvention,
    axis: np.ndarray,
    samples: list[tuple[int, int, np.ndarray, np.ndarray]],
    k_inverse: np.ndarray,
    threshold: float,
    bins: int,
    *,
    shift: int,
    reverse: bool,
) -> list[dict]:
    frame_count = len(convention.rotations)
    results = []
    for bin_index in range(bins):
        start = int(np.floor(bin_index * frame_count / bins))
        end = int(np.floor((bin_index + 1) * frame_count / bins))
        selected = [
            sample for sample in samples if start <= (sample[0] + sample[1]) // 2 < end
        ]
        score = score_candidate(
            convention,
            axis,
            selected,
            k_inverse,
            threshold,
            shift=shift,
            reverse=reverse,
        )
        results.append({"bin": bin_index, "start": start, "end": end, **score})
    return results


def audit_scene(scene: str, args: argparse.Namespace) -> dict:
    source = args.data_root.resolve() / scene
    statistics = read_json(source / "conversion_stats.json")
    input_dir = Path(statistics["input_dir"])
    rows = np.atleast_2d(np.loadtxt(input_dir / "trajectory.txt", dtype=np.float64))
    cameras = read_json(source / "trajectory_orb.json")["cameras"]
    if len(rows) != len(cameras):
        raise RuntimeError(
            f"{scene}: trajectory rows={len(rows)} cameras={len(cameras)}"
        )
    intrinsic = cameras[0]["intrinsic"]
    matrix = np.asarray(
        [
            [intrinsic["fx"], 0.0, intrinsic["cx"]],
            [0.0, intrinsic["fy"], intrinsic["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    k_inverse = np.linalg.inv(matrix)
    cache = args.cache_root.resolve() / f"{scene}_{args.cache_suffix}"
    pair_paths = select_pair_paths(cache / "matches", args.max_pairs)
    fit, validation = load_samples(
        cache,
        pair_paths,
        args.min_match_score,
        args.max_matches_per_pair,
    )
    samples = fit + validation
    axes = signed_axis_rotations()
    conventions = pose_conventions(rows)

    coarse = []
    for convention in conventions:
        for reverse in (False, True):
            for axis_label, axis in axes:
                score = score_candidate(
                    convention,
                    axis,
                    samples,
                    k_inverse,
                    args.epipolar_threshold_px,
                    reverse=reverse,
                )
                coarse.append(
                    {
                        "convention": convention.label,
                        "reverse": reverse,
                        "axis": axis_label,
                        "axis_matrix": axis.tolist(),
                        **score,
                    }
                )
    coarse.sort(key=lambda item: (-item["inlier_fraction"], item["median_error_px"]))

    scanned = []
    unique_top = []
    for candidate in coarse:
        key = (candidate["convention"], candidate["reverse"], candidate["axis"])
        if key in unique_top:
            continue
        unique_top.append(key)
        if len(unique_top) >= args.top_conventions:
            break
    convention_by_label = {item.label: item for item in conventions}
    axis_by_label = dict(axes)
    for convention_label, reverse, axis_label in unique_top:
        convention = convention_by_label[convention_label]
        axis = axis_by_label[axis_label]
        for shift in range(-args.max_frame_shift, args.max_frame_shift + 1):
            score = score_candidate(
                convention,
                axis,
                samples,
                k_inverse,
                args.epipolar_threshold_px,
                shift=shift,
                reverse=reverse,
            )
            scanned.append(
                {
                    "convention": convention_label,
                    "reverse": reverse,
                    "axis": axis_label,
                    "axis_matrix": axis.tolist(),
                    "shift": shift,
                    **score,
                }
            )
    scanned.sort(key=lambda item: (-item["inlier_fraction"], item["median_error_px"]))
    best = scanned[0]
    best_convention = convention_by_label[best["convention"]]
    best_axis = np.asarray(best["axis_matrix"], dtype=np.float64)
    return {
        "scene": scene,
        "frame_count": len(rows),
        "sample_pair_count": len(samples),
        "threshold_px": args.epipolar_threshold_px,
        "best": best,
        "temporal_bins": temporal_scores(
            best_convention,
            best_axis,
            samples,
            k_inverse,
            args.epipolar_threshold_px,
            args.temporal_bins,
            shift=int(best["shift"]),
            reverse=bool(best["reverse"]),
        ),
        "top_shift_scan": scanned[:16],
        "top_zero_shift": coarse[:16],
    }


def main() -> int:
    args = parse_args()
    scenes = [value.strip() for value in args.scenes.split(",") if value.strip()]
    if not scenes:
        raise ValueError("--scenes must contain at least one scene")
    payload = {
        "schema_version": 1,
        "data_root": str(args.data_root.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "scenes": [audit_scene(scene, args) for scene in scenes],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
