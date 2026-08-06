#!/usr/bin/env python3
"""Merge multiple overlapping ORB-SLAM3 pose windows over 360DVO GT centers."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--segment", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blend-ramp-frames", type=int, default=50)
    parser.add_argument("--max-overlap-disagreement-deg", type=float, default=15.0)
    parser.add_argument("--max-gap-fill-frames", type=int, default=15)
    parser.add_argument("--max-gap-rotation-deg", type=float, default=30.0)
    parser.add_argument("--max-edge-fill-frames", type=int, default=10)
    parser.add_argument(
        "--select-covering-segments",
        action="store_true",
        help="Select a consistent overlapping path and ignore redundant failed windows.",
    )
    parser.add_argument(
        "--allow-source-rotation-prior-fallback",
        action="store_true",
        help=(
            "If visual windows cannot cover the sequence, calibrate one constant "
            "camera-axis rotation on the source trajectory under strict consensus."
        ),
    )
    parser.add_argument("--source-prior-max-residual-deg", type=float, default=20.0)
    parser.add_argument(
        "--source-prior-min-inlier-fraction", type=float, default=0.55
    )
    parser.add_argument("--source-prior-min-frame-span-fraction", type=float, default=0.75)
    parser.add_argument("--source-prior-min-observations", type=int, default=100)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--keep-gt-world", action="store_true")
    return parser.parse_args()


def load_cameras(dataset: Path) -> list[dict]:
    path = dataset / "trajectory_orb.json"
    return json.loads(path.read_text(encoding="utf-8"))["cameras"]


def source_index(camera: dict, fallback: int) -> int:
    return int(camera.get("source_frame_index", camera.get("frame_index", fallback)))


def c2w(camera: dict) -> np.ndarray:
    pose = np.asarray(camera["T_camera_world"], dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 pose, got {pose.shape}")
    return np.linalg.inv(pose)


def rotation_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left).T @ np.asarray(right)
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def summarize(values: np.ndarray) -> dict:
    if not len(values):
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def index_segment(cameras: list[dict]) -> dict[int, np.ndarray]:
    indexed: dict[int, np.ndarray] = {}
    for fallback, camera in enumerate(cameras):
        index = source_index(camera, fallback)
        if index in indexed:
            raise ValueError(f"Duplicate source frame {index} in one pose window")
        indexed[index] = c2w(camera)[:3, :3]
    if not indexed:
        raise ValueError("Pose window is empty")
    return indexed


def mean_rotation(
    rotations: list[np.ndarray], weights: list[float] | None = None
) -> np.ndarray:
    values = Rotation.from_matrix(np.asarray(rotations))
    return values.mean(
        weights=None if weights is None else np.asarray(weights)
    ).as_matrix()


def calibrate_source_rotation_prior(
    source_rotations: np.ndarray,
    segments: list[dict[int, np.ndarray]],
    max_residual_deg: float,
    min_inlier_fraction: float,
    min_observations: int,
    min_frame_span_fraction: float,
) -> tuple[np.ndarray, dict]:
    """Calibrate one camera-axis correction only when visual consensus is dominant."""
    if max_residual_deg <= 0.0:
        raise ValueError("source-prior residual threshold must be positive")
    if not 0.0 < min_inlier_fraction <= 1.0:
        raise ValueError("source-prior inlier fraction must be in (0, 1]")
    if min_observations <= 0:
        raise ValueError("source-prior minimum observations must be positive")
    if not 0.0 < min_frame_span_fraction <= 1.0:
        raise ValueError("source-prior frame span fraction must be in (0, 1]")

    corrections = []
    observation_frames = []
    for segment in segments:
        for index, visual_rotation in segment.items():
            corrections.append(source_rotations[index].T @ visual_rotation)
            observation_frames.append(index)
    if len(corrections) < min_observations:
        raise RuntimeError(
            f"Source rotation prior has only {len(corrections)} visual observations; "
            f"minimum is {min_observations}"
        )

    correction_array = np.stack(corrections)
    correction = mean_rotation(list(correction_array))
    inliers = np.ones(len(correction_array), dtype=bool)
    for _ in range(4):
        residuals = np.asarray(
            [rotation_angle_deg(correction, value) for value in correction_array]
        )
        inliers = residuals <= max_residual_deg
        if int(inliers.sum()) < min_observations:
            break
        correction = mean_rotation(list(correction_array[inliers]))

    residuals = np.asarray(
        [rotation_angle_deg(correction, value) for value in correction_array]
    )
    inliers = residuals <= max_residual_deg
    inlier_count = int(inliers.sum())
    inlier_fraction = float(inliers.mean())
    inlier_frames = np.asarray(observation_frames, dtype=np.int64)[inliers]
    frame_span_fraction = (
        float(inlier_frames.max() - inlier_frames.min() + 1) / len(source_rotations)
        if len(inlier_frames)
        else 0.0
    )
    if (
        inlier_count < min_observations
        or inlier_fraction < min_inlier_fraction
        or frame_span_fraction < min_frame_span_fraction
    ):
        raise RuntimeError(
            "Source rotation prior lacks visual consensus: "
            f"inliers={inlier_count}/{len(correction_array)} "
            f"({inlier_fraction:.3f}), required fraction={min_inlier_fraction:.3f}; "
            f"frame span={frame_span_fraction:.3f}, "
            f"required span={min_frame_span_fraction:.3f}"
        )

    calibrated = source_rotations @ correction
    adjacent = np.asarray(
        [
            rotation_angle_deg(left, right)
            for left, right in zip(calibrated[:-1], calibrated[1:])
        ]
    )
    certificate = {
        "mode": "source_rotation_prior_calibrated_by_orb",
        "visual_observation_count": len(correction_array),
        "inlier_count": inlier_count,
        "inlier_fraction": inlier_fraction,
        "max_residual_deg": float(max_residual_deg),
        "min_inlier_fraction": float(min_inlier_fraction),
        "inlier_frame_start": int(inlier_frames.min()),
        "inlier_frame_end_inclusive": int(inlier_frames.max()),
        "inlier_frame_span_fraction": frame_span_fraction,
        "min_frame_span_fraction": float(min_frame_span_fraction),
        "camera_axis_correction": correction.tolist(),
        "all_visual_residual_deg": summarize(residuals),
        "inlier_visual_residual_deg": summarize(residuals[inliers]),
        "adjacent_rotation_deg": summarize(adjacent),
    }
    return calibrated, certificate


def slerp(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((left, right))))(
        [fraction]
    ).as_matrix()[0]


def edge_weight(frame: int, start: int, end: int, ramp: int) -> float:
    effective_ramp = max(1, min(ramp, (end - start + 2) // 2))
    distance = min(frame - start + 1, end - frame + 1, effective_ramp)
    return float(distance) / float(effective_ramp)


def select_covering_segments(
    segments: list[dict[int, np.ndarray]],
    frame_count: int,
    max_overlap_disagreement_deg: float,
    max_gap_fill_frames: int,
    max_gap_rotation_deg: float,
    max_edge_fill_frames: int,
    min_overlap_frames: int = 3,
) -> tuple[list[dict[int, np.ndarray]], dict]:
    """Choose an overlap-consistent segment path that covers the source trajectory."""
    ordered = sorted(segments, key=lambda item: (min(item), -max(item)))
    initial = [item for item in ordered if min(item) <= max_edge_fill_frames]
    if not initial:
        raise RuntimeError(
            "No pose window starts close enough to the source trajectory edge"
        )
    first = max(initial, key=lambda item: (max(item), len(item)))
    selected = [first]
    selected_ids = {id(first)}
    reference: dict[int, list[np.ndarray]] = {
        index: [rotation] for index, rotation in first.items()
    }
    decisions = [
        {
            "source_frame_start": min(first),
            "source_frame_end_inclusive": max(first),
            "overlap_frame_count": 0,
            "overlap_disagreement_deg": summarize(np.asarray([], dtype=np.float64)),
        }
    ]

    target = frame_count - 1 - max_edge_fill_frames
    while max(reference) < target:
        candidates = []
        for segment in ordered:
            if id(segment) in selected_ids or max(segment) <= max(reference):
                continue
            overlap = sorted(set(segment).intersection(reference))
            if len(overlap) >= min_overlap_frames:
                disagreement = np.asarray(
                    [
                        rotation_angle_deg(
                            mean_rotation(reference[index]), segment[index]
                        )
                        for index in overlap
                    ],
                    dtype=np.float64,
                )
                p95 = float(np.percentile(disagreement, 95))
                if p95 > max_overlap_disagreement_deg:
                    continue
                connection = "overlap"
                gap = 0
            else:
                gap = max(0, min(segment) - max(reference) - 1)
                if gap > max_gap_fill_frames:
                    continue
                if overlap:
                    disagreement = np.asarray(
                        [
                            rotation_angle_deg(
                                mean_rotation(reference[index]), segment[index]
                            )
                            for index in overlap
                        ],
                        dtype=np.float64,
                    )
                else:
                    disagreement = np.asarray(
                        [
                            rotation_angle_deg(
                                mean_rotation(reference[max(reference)]),
                                segment[min(segment)],
                            )
                        ],
                        dtype=np.float64,
                    )
                p95 = float(np.percentile(disagreement, 95))
                if p95 > max_gap_rotation_deg:
                    continue
                connection = "short_overlap" if overlap else "certified_gap"
            candidates.append(
                (
                    int(connection == "overlap"),
                    -gap,
                    -p95,
                    max(segment),
                    len(overlap),
                    segment,
                    disagreement,
                    connection,
                    gap,
                )
            )
        if not candidates:
            raise RuntimeError(
                f"No consistent pose window extends coverage beyond frame {max(reference)}"
            )
        _, _, _, _, overlap_count, chosen, disagreement, connection, gap = max(
            candidates, key=lambda item: item[:5]
        )
        selected.append(chosen)
        selected_ids.add(id(chosen))
        for index, rotation in chosen.items():
            reference.setdefault(index, []).append(rotation)
        decisions.append(
            {
                "source_frame_start": min(chosen),
                "source_frame_end_inclusive": max(chosen),
                "overlap_frame_count": overlap_count,
                "connection": connection,
                "filled_gap_frames": gap,
                "overlap_disagreement_deg": summarize(disagreement),
            }
        )

    certificate = {
        "input_segment_count": len(segments),
        "selected_segment_count": len(selected),
        "ignored_segment_count": len(segments) - len(selected),
        "minimum_overlap_frames": min_overlap_frames,
        "max_gap_fill_frames": max_gap_fill_frames,
        "max_gap_rotation_deg": max_gap_rotation_deg,
        "selected_path": decisions,
        "covered_source_frame_start": min(reference),
        "covered_source_frame_end_inclusive": max(reference),
    }
    return selected, certificate


def merge_segment_rotations(
    segments: list[dict[int, np.ndarray]],
    frame_count: int,
    blend_ramp_frames: int,
    max_overlap_disagreement_deg: float,
    max_gap_fill_frames: int,
    max_edge_fill_frames: int,
    allow_certified_gaps: bool = False,
) -> tuple[np.ndarray, dict]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if blend_ramp_frames <= 0:
        raise ValueError("blend ramp must be positive")
    ordered = sorted(segments, key=lambda item: (min(item), max(item)))
    supports: dict[int, list[tuple[np.ndarray, float]]] = {}
    reference: dict[int, list[np.ndarray]] = {}
    segment_stats = []

    for number, segment in enumerate(ordered):
        start, end = min(segment), max(segment)
        if start < 0 or end >= frame_count:
            raise ValueError(f"Pose window [{start}, {end}] exceeds source trajectory")
        overlap = sorted(set(segment).intersection(reference))
        raw_disagreement = np.asarray(
            [
                rotation_angle_deg(mean_rotation(reference[index]), segment[index])
                for index in overlap
            ],
            dtype=np.float64,
        )
        if number > 0:
            gap = max(0, start - max(reference) - 1)
            if len(overlap) < 3 and not (
                allow_certified_gaps and gap <= max_gap_fill_frames
            ):
                raise RuntimeError(
                    f"Pose window [{start}, {end}] has only {len(overlap)} shared frames"
                )
            if len(overlap) >= 3 and (
                float(np.percentile(raw_disagreement, 95))
                > max_overlap_disagreement_deg
            ):
                raise RuntimeError(
                    f"Pose window [{start}, {end}] overlap rotation p95 "
                    f"{np.percentile(raw_disagreement, 95):.3f} deg exceeds "
                    f"{max_overlap_disagreement_deg:.3f} deg"
                )
        for index, value in segment.items():
            weight = edge_weight(index, start, end, blend_ramp_frames)
            supports.setdefault(index, []).append((value, weight))
            reference.setdefault(index, []).append(value)
        segment_stats.append(
            {
                "segment_index": number,
                "source_frame_start": start,
                "source_frame_end_inclusive": end,
                "frame_count": len(segment),
                "overlap_frame_count": len(overlap),
                "overlap_disagreement_deg": summarize(raw_disagreement),
            }
        )

    rotations: list[np.ndarray | None] = [None] * frame_count
    for index, values in supports.items():
        rotations[index] = mean_rotation(
            [value for value, _ in values], [weight for _, weight in values]
        )
    covered = [index for index, value in enumerate(rotations) if value is not None]
    if not covered:
        raise RuntimeError("No source frame is covered by a pose window")

    filled_leading = 0
    filled_trailing = 0
    filled_internal = 0
    for index, value in enumerate(rotations):
        if value is not None:
            continue
        position = bisect.bisect_left(covered, index)
        left = covered[position - 1] if position > 0 else None
        right = covered[position] if position < len(covered) else None
        if left is None:
            gap = int(right) - index
            if gap > max_edge_fill_frames:
                raise RuntimeError(f"Leading pose gap of {gap} frames exceeds limit")
            rotations[index] = rotations[int(right)].copy()
            filled_leading += 1
        elif right is None:
            gap = index - int(left)
            if gap > max_edge_fill_frames:
                raise RuntimeError(f"Trailing pose gap of {gap} frames exceeds limit")
            rotations[index] = rotations[int(left)].copy()
            filled_trailing += 1
        else:
            gap = int(right) - int(left) - 1
            if gap > max_gap_fill_frames:
                raise RuntimeError(
                    f"Internal pose gap {left + 1}..{right - 1} exceeds limit"
                )
            fraction = float(index - left) / float(right - left)
            rotations[index] = slerp(rotations[left], rotations[right], fraction)
            filled_internal += 1

    merged = np.asarray(rotations, dtype=np.float64)
    adjacent = np.asarray(
        [
            rotation_angle_deg(left, right)
            for left, right in zip(merged[:-1], merged[1:])
        ]
    )
    certificate = {
        "segment_count": len(ordered),
        "max_overlap_disagreement_deg": float(max_overlap_disagreement_deg),
        "segments": segment_stats,
        "directly_covered_frame_count": len(covered),
        "filled_leading_frames": filled_leading,
        "filled_internal_frames": filled_internal,
        "filled_trailing_frames": filled_trailing,
        "adjacent_rotation_deg": summarize(adjacent),
    }
    return merged, certificate


def build_poses(
    source_cameras: list[dict],
    rotations: np.ndarray,
    normalize_first: bool,
) -> tuple[np.ndarray, dict]:
    gt_c2w = np.asarray([c2w(camera) for camera in source_cameras])
    merged_c2w = np.repeat(np.eye(4)[None], len(source_cameras), axis=0)
    merged_c2w[:, :3, :3] = rotations
    merged_c2w[:, :3, 3] = gt_c2w[:, :3, 3]
    canonical_from_gt = np.linalg.inv(merged_c2w[0]) if normalize_first else np.eye(4)
    canonical_c2w = canonical_from_gt[None] @ merged_c2w
    poses = np.linalg.inv(canonical_c2w)
    expected_centers = (
        canonical_from_gt[:3, :3] @ gt_c2w[:, :3, 3].T
    ).T + canonical_from_gt[:3, 3]
    center_error = np.linalg.norm(canonical_c2w[:, :3, 3] - expected_centers, axis=1)
    certificate = {
        "normalized_first_camera": bool(normalize_first),
        "first_pose_identity_max_abs": float(np.abs(poses[0] - np.eye(4)).max()),
        "gt_center_preservation_max_error_m": float(center_error.max()),
        "trajectory_length_m": float(
            np.linalg.norm(np.diff(canonical_c2w[:, :3, 3], axis=0), axis=1).sum()
        ),
    }
    return poses, certificate


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    segments = [path.resolve() for path in args.segment]
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists: {staging}")

    source_cameras = load_cameras(source)
    all_indexed_segments = [index_segment(load_cameras(path)) for path in segments]
    indexed_segments = all_indexed_segments
    selection = None
    pose_mode = "overlap_certified_orb_windows"
    merge_error = None
    try:
        if args.select_covering_segments:
            indexed_segments, selection = select_covering_segments(
                indexed_segments,
                len(source_cameras),
                args.max_overlap_disagreement_deg,
                args.max_gap_fill_frames,
                args.max_gap_rotation_deg,
                args.max_edge_fill_frames,
            )
        rotations, continuity = merge_segment_rotations(
            indexed_segments,
            len(source_cameras),
            args.blend_ramp_frames,
            args.max_overlap_disagreement_deg,
            args.max_gap_fill_frames,
            args.max_edge_fill_frames,
            allow_certified_gaps=args.select_covering_segments,
        )
        if selection is not None:
            continuity["selection_certificate"] = selection
    except RuntimeError as error:
        if not args.allow_source_rotation_prior_fallback:
            raise
        merge_error = str(error)
        source_rotations = np.asarray(
            [c2w(camera)[:3, :3] for camera in source_cameras]
        )
        rotations, continuity = calibrate_source_rotation_prior(
            source_rotations,
            all_indexed_segments,
            args.source_prior_max_residual_deg,
            args.source_prior_min_inlier_fraction,
            args.source_prior_min_observations,
            args.source_prior_min_frame_span_fraction,
        )
        continuity["orb_merge_failure"] = merge_error
        pose_mode = "source_rotation_prior_calibrated_by_orb"
    poses, coordinate = build_poses(
        source_cameras, rotations, normalize_first=not args.keep_gt_world
    )

    rectified = staging / "rectified"
    rectified.mkdir(parents=True)
    cameras = []
    for index, (source_camera, pose) in enumerate(zip(source_cameras, poses)):
        source_image = source / "rectified" / source_camera["image"]
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        link = rectified / f"aria_{index:05d}.png"
        link.symlink_to(source_image.resolve())
        camera = dict(source_camera)
        camera["T_camera_world"] = pose.tolist()
        camera["image"] = link.name
        camera["frame_index"] = index
        camera["source_frame_index"] = index
        camera["timestamp"] = float(index) / args.fps
        cameras.append(camera)

    trajectory = json.dumps({"cameras": cameras}, indent=2) + "\n"
    (staging / "trajectory.json").write_text(trajectory, encoding="utf-8")
    (staging / "trajectory_orb.json").write_text(trajectory, encoding="utf-8")
    stats = {
        "schema_version": 1,
        "method": (
            "overlap-certified windowed ORB-SLAM3 rotations with 360DVO GT centers"
            if pose_mode == "overlap_certified_orb_windows"
            else "ORB-certified source rotation prior with 360DVO GT centers"
        ),
        "pose_source": "orbslam3_mono_windowed",
        "sparse_world_geometry": "not_exported_pose_stage",
        "source": str(source),
        "pose_segments": [str(path) for path in segments],
        "frame_count": len(cameras),
        "pose_mode": pose_mode,
        "pose_fusion": (
            "edge-weighted averaging of independently Sim(3)-aligned ORB-SLAM3 "
            "rotations with overlap rejection and per-frame 360DVO GT camera centers"
            if pose_mode == "overlap_certified_orb_windows"
            else "one consensus-certified camera-axis correction transfers source "
            "trajectory rotations into the ORB visual frame; camera centers remain GT"
        ),
        "continuity_certificate": continuity,
        "coordinate_certificate": coordinate,
    }
    (staging / "conversion_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    (staging / "README.md").write_text(
        "# Windowed 360DVO ORB-SLAM3 front-view poses\n\n"
        "Overlapping monocular ORB-SLAM3 windows provide image-consistent rotations. "
        "When those windows cannot cover the sequence, a source rotation prior is "
        "used only after one constant camera-axis correction has dominant ORB visual "
        "consensus. Every camera center comes from the matching 360DVO GT frame.\n",
        encoding="utf-8",
    )
    staging.replace(output)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
