#!/usr/bin/env python3
"""Merge reset ORB-SLAM3 rotation segments over fixed 360DVO GT centers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--suffix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-source-frame", type=int, required=True)
    parser.add_argument("--end-source-frame", type=int, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--keep-gt-world",
        action="store_true",
        help="Do not apply the one rigid transform that makes the first camera identity.",
    )
    return parser.parse_args()


def load_cameras(dataset: Path) -> list[dict]:
    trajectory = dataset / "trajectory_orb.json"
    return json.loads(trajectory.read_text(encoding="utf-8"))["cameras"]


def camera_source_index(camera: dict, fallback: int) -> int:
    return int(camera.get("source_frame_index", camera.get("frame_index", fallback)))


def index_cameras(cameras: list[dict]) -> dict[int, dict]:
    indexed: dict[int, dict] = {}
    for fallback, camera in enumerate(cameras):
        source_index = camera_source_index(camera, fallback)
        if source_index in indexed:
            raise ValueError(f"Duplicate source frame {source_index}")
        indexed[source_index] = camera
    return indexed


def c2w_from_camera(camera: dict) -> np.ndarray:
    pose = np.asarray(camera["T_camera_world"], dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera pose, got {pose.shape}")
    return np.linalg.inv(pose)


def rotation_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left, dtype=np.float64).T @ np.asarray(
        right, dtype=np.float64
    )
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def slerp_rotation(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Slerp fraction must be in [0, 1]")
    rotations = Rotation.from_matrix(np.stack((left, right)))
    return Slerp([0.0, 1.0], rotations)([fraction]).as_matrix()[0]


def percentile_summary(values: np.ndarray) -> dict:
    if not len(values):
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def merge_rotations(
    source_indices: list[int],
    prefix: dict[int, dict],
    suffix: dict[int, dict],
) -> tuple[np.ndarray, dict]:
    overlap = sorted(set(prefix).intersection(suffix).intersection(source_indices))
    if len(overlap) < 2:
        raise RuntimeError("Pose segments need at least two overlapping source frames")
    overlap_start, overlap_end = overlap[0], overlap[-1]
    if overlap != list(range(overlap_start, overlap_end + 1)):
        raise RuntimeError("Pose-segment overlap is not contiguous")

    rotations = []
    assignment = {"prefix": 0, "blended_overlap": 0, "suffix": 0}
    for source_index in source_indices:
        prefix_camera = prefix.get(source_index)
        suffix_camera = suffix.get(source_index)
        if prefix_camera is not None and suffix_camera is not None:
            fraction = (source_index - overlap_start) / (overlap_end - overlap_start)
            rotation = slerp_rotation(
                c2w_from_camera(prefix_camera)[:3, :3],
                c2w_from_camera(suffix_camera)[:3, :3],
                fraction,
            )
            assignment["blended_overlap"] += 1
        elif prefix_camera is not None:
            rotation = c2w_from_camera(prefix_camera)[:3, :3]
            assignment["prefix"] += 1
        elif suffix_camera is not None:
            rotation = c2w_from_camera(suffix_camera)[:3, :3]
            assignment["suffix"] += 1
        else:
            raise RuntimeError(f"Neither pose segment covers source frame {source_index}")
        rotations.append(rotation)

    disagreement = np.asarray(
        [
            rotation_angle_deg(
                c2w_from_camera(prefix[index])[:3, :3],
                c2w_from_camera(suffix[index])[:3, :3],
            )
            for index in overlap
        ]
    )
    certificate = {
        "overlap_start": overlap_start,
        "overlap_end_inclusive": overlap_end,
        "overlap_frame_count": len(overlap),
        "assignment": assignment,
        "segment_rotation_disagreement_deg": percentile_summary(disagreement),
    }
    return np.asarray(rotations), certificate


def build_merged_poses(
    source_cameras: list[dict],
    prefix_cameras: list[dict],
    suffix_cameras: list[dict],
    start_source_frame: int,
    end_source_frame: int,
    normalize_first: bool,
) -> tuple[list[int], np.ndarray, dict]:
    if start_source_frame > end_source_frame:
        raise ValueError("start source frame must not exceed end source frame")
    if start_source_frame < 0 or end_source_frame >= len(source_cameras):
        raise ValueError("Requested source-frame range is outside the source trajectory")

    source_indices = list(range(start_source_frame, end_source_frame + 1))
    prefix = index_cameras(prefix_cameras)
    suffix = index_cameras(suffix_cameras)
    rotations, certificate = merge_rotations(source_indices, prefix, suffix)
    gt_c2w = np.asarray([c2w_from_camera(source_cameras[i]) for i in source_indices])

    merged_c2w = np.repeat(np.eye(4, dtype=np.float64)[None], len(source_indices), axis=0)
    merged_c2w[:, :3, :3] = rotations
    merged_c2w[:, :3, 3] = gt_c2w[:, :3, 3]
    canonical_from_gt = np.linalg.inv(merged_c2w[0]) if normalize_first else np.eye(4)
    canonical_c2w = canonical_from_gt[None] @ merged_c2w
    merged_w2c = np.linalg.inv(canonical_c2w)

    adjacent_angles = np.asarray(
        [
            rotation_angle_deg(left, right)
            for left, right in zip(
                canonical_c2w[:-1, :3, :3], canonical_c2w[1:, :3, :3]
            )
        ]
    )
    canonical_gt_centers = (
        canonical_from_gt[:3, :3] @ gt_c2w[:, :3, 3].T
    ).T + canonical_from_gt[:3, 3]
    center_errors = np.linalg.norm(
        canonical_c2w[:, :3, 3] - canonical_gt_centers, axis=1
    )
    orthogonality_errors = np.linalg.norm(
        np.swapaxes(canonical_c2w[:, :3, :3], 1, 2)
        @ canonical_c2w[:, :3, :3]
        - np.eye(3)[None],
        axis=(1, 2),
    )
    centers = canonical_c2w[:, :3, 3]
    certificate.update(
        {
            "source_frame_start": start_source_frame,
            "source_frame_end_inclusive": end_source_frame,
            "frame_count": len(source_indices),
            "missing_source_frames": [],
            "normalized_first_camera": bool(normalize_first),
            "first_pose_identity_max_abs": float(
                np.abs(merged_w2c[0] - np.eye(4)).max()
            ),
            "gt_center_preservation_max_error_m": float(center_errors.max()),
            "rotation_orthogonality_max_error": float(orthogonality_errors.max()),
            "rotation_determinant_min": float(
                np.linalg.det(canonical_c2w[:, :3, :3]).min()
            ),
            "adjacent_rotation_deg": percentile_summary(adjacent_angles),
            "overlap_entry_adjacent_rotation_deg": float(
                adjacent_angles[certificate["overlap_start"] - start_source_frame - 1]
            ),
            "overlap_exit_adjacent_rotation_deg": float(
                adjacent_angles[certificate["overlap_end_inclusive"] - start_source_frame]
            ),
            "trajectory_length_m": float(
                np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()
            ),
        }
    )
    return source_indices, merged_w2c, certificate


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    prefix = args.prefix.resolve()
    suffix = args.suffix.resolve()
    output = args.output.resolve()
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"Stale staging directory exists: {staging}")

    source_cameras = load_cameras(source)
    source_indices, poses, certificate = build_merged_poses(
        source_cameras,
        load_cameras(prefix),
        load_cameras(suffix),
        args.start_source_frame,
        args.end_source_frame,
        normalize_first=not args.keep_gt_world,
    )

    rectified = staging / "rectified"
    rectified.mkdir(parents=True)
    cameras = []
    for output_index, (source_index, pose) in enumerate(zip(source_indices, poses)):
        source_camera = source_cameras[source_index]
        source_image = source / "rectified" / source_camera["image"]
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        link = rectified / f"aria_{output_index:05d}.png"
        link.symlink_to(source_image.resolve())
        camera = dict(source_camera)
        camera["T_camera_world"] = pose.tolist()
        camera["image"] = link.name
        camera["frame_index"] = output_index
        camera["source_frame_index"] = source_index
        camera["timestamp"] = float(source_index) / args.fps
        cameras.append(camera)

    trajectory_text = json.dumps({"cameras": cameras}, indent=2) + "\n"
    (staging / "trajectory.json").write_text(trajectory_text, encoding="utf-8")
    (staging / "trajectory_orb.json").write_text(trajectory_text, encoding="utf-8")
    stats = {
        "schema_version": 1,
        "method": "overlap-Slerp ORB-SLAM3 monocular rotations with 360DVO GT centers",
        "pose_source": "orbslam3_mono",
        "sparse_world_geometry": "not_exported_pose_stage",
        "source": str(source),
        "prefix_pose_segment": str(prefix),
        "suffix_pose_segment": str(suffix),
        "pose_fusion": "piecewise ORB-SLAM3 rotations with per-frame 360DVO GT camera centers",
        "coordinate_contract": (
            "one global rigid transform makes the first retained camera identity"
            if not args.keep_gt_world
            else "the source GT world is retained"
        ),
        "continuity_certificate": certificate,
    }
    (staging / "conversion_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    (staging / "README.md").write_text(
        "# 360DVO merged ORB-SLAM3 front-view poses\n\n"
        "ORB-SLAM3 visual rotations are merged across a reset using overlap Slerp. "
        "Every camera center comes directly from the corresponding 360DVO GT frame. "
        "This pose-stage dataset intentionally contains no sparse points.\n",
        encoding="utf-8",
    )
    staging.replace(output)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
