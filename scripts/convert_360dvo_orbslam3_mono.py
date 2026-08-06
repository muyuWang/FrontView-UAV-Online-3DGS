#!/usr/bin/env python3
"""Align ORB-SLAM3 mono poses to 360DVO GT and export canonical camera poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from convert_panoair_orbslam3_vi import (
    apply_similarity,
    interpolate_orb_poses,
    robust_similarity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--orb-input", type=Path, required=True)
    parser.add_argument("--orb-trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment-threshold-m", type=float, default=0.75)
    parser.add_argument("--max-interpolation-gap-s", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--keep-gt-world",
        action="store_true",
        help="Keep the Sim(3)-aligned GT world instead of normalizing camera zero.",
    )
    parser.add_argument(
        "--use-gt-centers",
        action="store_true",
        help="Fuse visual rotations with the source GT camera centers.",
    )
    return parser.parse_args()


def load_orb_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.loadtxt(path, dtype=np.float64)
    rows = np.atleast_2d(rows)
    if rows.shape[1] != 8:
        raise RuntimeError(f"Expected timestamp xyz qx qy qz qw, got {rows.shape}")
    order = np.argsort(rows[:, 0])
    rows = rows[order]
    unique = np.concatenate(([True], np.diff(rows[:, 0]) > 0))
    rows = rows[unique]
    return rows[:, 0].astype(np.int64), rows[:, 1:4], rows[:, 4:8]


def camera_centers(cameras: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            np.linalg.inv(np.asarray(camera["T_camera_world"], dtype=np.float64))[
                :3, 3
            ]
            for camera in cameras
        ]
    )


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    orb_input = args.orb_input.resolve()
    orb_trajectory = args.orb_trajectory.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")

    source_cameras = json.loads(
        (source / "trajectory_orb.json").read_text(encoding="utf-8")
    )["cameras"]
    manifest = json.loads((orb_input / "manifest.json").read_text(encoding="utf-8"))
    frame_ns = np.asarray(manifest["timestamps_ns"], dtype=np.int64)
    source_indices = np.asarray(manifest["source_frame_indices"], dtype=np.int32)
    if len(frame_ns) != len(source_indices):
        raise RuntimeError("ORB input timestamps and source indices have different lengths")
    selected_source = [source_cameras[int(index)] for index in source_indices]
    gt_centers = camera_centers(selected_source)

    pose_ns, orb_positions, orb_quaternions = load_orb_trajectory(orb_trajectory)
    timestamp_to_frame = {int(value): index for index, value in enumerate(frame_ns)}
    matched_pose_rows = []
    matched_frame_rows = []
    for pose_index, timestamp in enumerate(pose_ns):
        frame_index = timestamp_to_frame.get(int(timestamp))
        if frame_index is not None:
            matched_pose_rows.append(pose_index)
            matched_frame_rows.append(frame_index)
    if len(matched_pose_rows) < 8:
        raise RuntimeError("Fewer than eight ORB poses match the prepared frame timestamps")
    matched_pose_rows = np.asarray(matched_pose_rows, dtype=np.int32)
    matched_frame_rows = np.asarray(matched_frame_rows, dtype=np.int32)
    similarity, inliers, residuals = robust_similarity(
        orb_positions[matched_pose_rows],
        gt_centers[matched_frame_rows],
        args.alignment_threshold_m,
        args.seed,
    )

    covered = (frame_ns >= pose_ns[0]) & (frame_ns <= pose_ns[-1])
    if not covered.any():
        raise RuntimeError("ORB trajectory does not cover any prepared frame")
    first = int(np.flatnonzero(covered)[0])
    last = int(np.flatnonzero(covered)[-1]) + 1
    frame_ns = frame_ns[first:last]
    source_indices = source_indices[first:last]
    selected_source = selected_source[first:last]
    retained_gt_centers = gt_centers[first:last]
    frame_positions, frame_rotations, tracking = interpolate_orb_poses(
        frame_ns,
        pose_ns,
        orb_positions,
        orb_quaternions,
        args.max_interpolation_gap_s,
    )
    scale, alignment_rotation, alignment_translation = similarity
    aligned_positions = apply_similarity(frame_positions, similarity)
    aligned_rotations = alignment_rotation[None] @ frame_rotations
    if args.use_gt_centers:
        aligned_positions = retained_gt_centers.copy()

    aligned_c2w = np.repeat(np.eye(4, dtype=np.float64)[None], len(frame_ns), axis=0)
    aligned_c2w[:, :3, :3] = aligned_rotations
    aligned_c2w[:, :3, 3] = aligned_positions
    canonical_from_aligned = (
        np.eye(4, dtype=np.float64)
        if args.keep_gt_world
        else np.linalg.inv(aligned_c2w[0])
    )
    canonical_c2w = canonical_from_aligned[None] @ aligned_c2w
    canonical_w2c = np.linalg.inv(canonical_c2w)

    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists: {staging}")
    rectified = staging / "rectified"
    rectified.mkdir(parents=True)
    cameras = []
    for output_index, (source_camera, source_index, pose) in enumerate(
        zip(selected_source, source_indices, canonical_w2c)
    ):
        source_image = source / "rectified" / source_camera["image"]
        link = rectified / f"aria_{output_index:05d}.png"
        link.symlink_to(source_image.resolve())
        camera = dict(source_camera)
        camera["T_camera_world"] = pose.tolist()
        camera["image"] = link.name
        camera["frame_index"] = output_index
        camera["source_frame_index"] = int(source_index)
        camera["timestamp"] = float(frame_ns[output_index]) / 1.0e9
        cameras.append(camera)
    trajectory = json.dumps({"cameras": cameras}, indent=2) + "\n"
    (staging / "trajectory.json").write_text(trajectory, encoding="utf-8")
    (staging / "trajectory_orb.json").write_text(trajectory, encoding="utf-8")

    matched_errors = residuals
    canonical_centers = np.linalg.inv(canonical_w2c)[:, :3, 3]
    stats = {
        "schema_version": 1,
        "method": "ORB-SLAM3 monocular BA with one global Sim(3) to 360DVO GT",
        "pose_source": "orbslam3_mono",
        "sparse_world_geometry": "not_exported_pose_stage",
        "source": str(source),
        "orb_input": str(orb_input),
        "orb_trajectory": str(orb_trajectory),
        "requested_frame_count": int(len(manifest["timestamps_ns"])),
        "frame_count": len(cameras),
        "trimmed_leading_frames": first,
        "trimmed_trailing_frames": int(len(manifest["timestamps_ns"]) - last),
        "tracked_pose_count": len(pose_ns),
        "tracking": tracking,
        "gt_alignment": {
            "scale": float(scale),
            "rotation": alignment_rotation.tolist(),
            "translation": alignment_translation.tolist(),
            "matched_pose_count": int(len(matched_pose_rows)),
            "inlier_count": int(inliers.sum()),
            "threshold_m": float(args.alignment_threshold_m),
            "rmse_m": float(np.sqrt(np.mean(matched_errors**2))),
            "median_m": float(np.median(matched_errors)),
            "p90_m": float(np.percentile(matched_errors, 90)),
            "max_m": float(matched_errors.max()),
        },
        "coordinate_contract": (
            "one Sim(3) aligns the complete ORB-SLAM3 trajectory to GT positions; "
            + (
                "the aligned GT world is retained"
                if args.keep_gt_world
                else "one global rigid transform then makes the first retained camera identity"
            )
        ),
        "kept_gt_world": bool(args.keep_gt_world),
        "used_gt_centers": bool(args.use_gt_centers),
        "pose_fusion": (
            "ORB-SLAM3 monocular rotations with 360DVO GT camera centers"
            if args.use_gt_centers
            else "globally Sim(3)-aligned ORB-SLAM3 monocular poses"
        ),
        "source_frame_start": int(source_indices[0]),
        "source_frame_end_inclusive": int(source_indices[-1]),
        "trajectory_span_m": np.ptp(canonical_centers, axis=0).tolist(),
        "trajectory_length_m": float(
            np.linalg.norm(np.diff(canonical_centers, axis=0), axis=1).sum()
        ),
    }
    (staging / "conversion_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    (staging / "README.md").write_text(
        "# 360DVO ORB-SLAM3 monocular canonical poses\n\n"
        "The virtual front-view orientation is estimated from RGB by ORB-SLAM3. "
        "A single global Sim(3) aligns the visual trajectory to 360DVO GT camera "
        "centers. This pose-stage dataset intentionally contains no sparse points.\n",
        encoding="utf-8",
    )
    staging.replace(output)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
