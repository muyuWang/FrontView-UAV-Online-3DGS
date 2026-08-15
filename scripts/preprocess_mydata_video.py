#!/usr/bin/env python3
"""Convert one UAV MP4 into an ORB-backed Online-3DGS dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "mydata"
DEFAULT_ORB_ROOT = Path("/home/wmy/workspace_vla/third_party/ORB_SLAM3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video",
        help="Video filename under data/mydata, or an explicit MP4 path.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit extracted frames for validation; zero processes the full video.",
    )
    parser.add_argument(
        "--horizontal-fov-deg",
        type=float,
        default=82.1,
        help="DJI Mini 4 Pro nominal horizontal field of view.",
    )
    parser.add_argument("--target-median-depth", type=float, default=10.0)
    parser.add_argument("--max-interpolation-gap-s", type=float, default=0.5)
    parser.add_argument("--min-orb-observations", type=int, default=3)
    parser.add_argument("--min-orb-found-ratio", type=float, default=0.25)
    parser.add_argument("--max-points-per-frame", type=int, default=5000)
    parser.add_argument("--min-supported-frame-fraction", type=float, default=0.50)
    parser.add_argument("--orb-root", type=Path, default=DEFAULT_ORB_ROOT)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_logged(command: list[str], log_path: Path, *, cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}; see {log_path}"
        )


def probe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration:format_tags=encoder,creation_time:stream=codec_name,width,height,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    if not payload.get("streams"):
        raise RuntimeError(f"No video stream found in {path}")
    return payload


def resolve_video(value: str, data_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = data_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    if candidate.suffix.lower() != ".mp4":
        raise ValueError(f"Expected an MP4 input: {candidate}")
    return candidate


def extraction_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(current.get(key) == value for key, value in expected.items())


def extract_frames(
    video: Path,
    staging: Path,
    args: argparse.Namespace,
    video_probe: dict[str, Any],
) -> list[Path]:
    frames = staging / "frames_720p"
    manifest_path = staging / "extraction.json"
    expected = {
        "source_video": str(video),
        "fps": float(args.fps),
        "width": int(args.width),
        "height": int(args.height),
        "max_frames": int(args.max_frames),
    }
    existing = sorted(frames.glob("aria_*.jpg")) if frames.is_dir() else []
    if extraction_matches(manifest_path, expected):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(existing) == int(manifest["frame_count"]):
            return existing
        raise RuntimeError(f"Incomplete extracted frame set in {frames}")
    if existing or manifest_path.exists():
        raise RuntimeError(
            f"Stale extraction exists in {staging}; inspect it before retrying"
        )

    frames.mkdir(parents=True, exist_ok=True)
    command = [
        args.ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        f"fps={args.fps:.12g},scale={args.width}:{args.height}:flags=lanczos",
        "-q:v",
        "2",
        "-start_number",
        "0",
    ]
    if args.max_frames > 0:
        command.extend(("-frames:v", str(args.max_frames)))
    command.append(str(frames / "aria_%06d.jpg"))
    run_logged(command, staging / "logs" / "extract_frames.log")
    extracted = sorted(frames.glob("aria_*.jpg"))
    if len(extracted) < 30:
        raise RuntimeError(f"Only {len(extracted)} frames were extracted from {video}")
    stream = video_probe["streams"][0]
    write_json(
        manifest_path,
        {
            **expected,
            "frame_count": len(extracted),
            "source_stream": stream,
            "source_format": video_probe.get("format", {}),
            "sampling": "uniform temporal sampling followed by Lanczos spatial resize",
            "image_format": "JPEG quality 2",
        },
    )
    return extracted


def focal_from_fov(width: int, horizontal_fov_deg: float) -> float:
    return width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))


def write_orb_settings(path: Path, args: argparse.Namespace, focal: float) -> None:
    text = f"""%YAML:1.0

File.version: \"1.0\"
Camera.type: \"PinHole\"

Camera1.fx: {focal:.9f}
Camera1.fy: {focal:.9f}
Camera1.cx: {args.width / 2.0:.9f}
Camera1.cy: {args.height / 2.0:.9f}
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera1.k3: 0.0
Camera.fps: {int(round(args.fps))}
Camera.RGB: 0
Camera.width: {args.width}
Camera.height: {args.height}

ORBextractor.nFeatures: 3000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 12
ORBextractor.minThFAST: 5

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""
    path.write_text(text, encoding="utf-8")


def prepare_orb_input(
    staging: Path, frames: list[Path], fps: float
) -> tuple[np.ndarray, Path, Path]:
    orb_input = staging / "orb_input"
    image_dir = orb_input / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    start_ns = 1_000_000_000
    step_ns = int(round(1.0e9 / fps))
    timestamps = start_ns + np.arange(len(frames), dtype=np.int64) * step_ns
    times_path = orb_input / "times.txt"
    times_path.write_text(
        "\n".join(str(int(value)) for value in timestamps) + "\n",
        encoding="utf-8",
    )
    for timestamp, frame in zip(timestamps, frames):
        destination = image_dir / f"{int(timestamp)}.png"
        if destination.is_symlink():
            continue
        if destination.exists():
            raise FileExistsError(destination)
        relative = os.path.relpath(frame, destination.parent)
        destination.symlink_to(relative)
    return timestamps, image_dir, times_path


def run_orb_slam(
    staging: Path,
    image_dir: Path,
    times_path: Path,
    scene: str,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    orb_root = args.orb_root.resolve()
    binary = orb_root / "Examples" / "Monocular" / "mono_tum_vi"
    vocabulary = orb_root / "Vocabulary" / "ORBvoc.txt"
    settings = staging / "orbslam3_monocular.yaml"
    trajectory = staging / "orb_output" / f"f_{scene}.txt"
    keyframes = staging / "orb_output" / f"kf_{scene}.txt"
    points = staging / "orb_output" / f"map_{scene}.txt"
    for required in (binary, vocabulary, settings):
        if not required.is_file():
            raise FileNotFoundError(required)
    if all(path.is_file() and path.stat().st_size > 128 for path in (trajectory, keyframes, points)):
        return trajectory, keyframes, points
    output = staging / "orb_output"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Incomplete ORB-SLAM3 output exists in {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_logged(
        [
            str(binary),
            str(vocabulary),
            str(settings),
            str(image_dir),
            str(times_path),
            scene,
        ],
        staging / "logs" / "orbslam3_monocular.log",
        cwd=output,
    )
    for required in (trajectory, keyframes, points):
        if not required.is_file() or required.stat().st_size <= 128:
            if required == points:
                patch = ROOT / "third_party_patches" / "orbslam3_mono_tum_vi_map_export.patch"
                raise RuntimeError(
                    "ORB-SLAM3 did not export persistent map points. Apply "
                    f"{patch} in {orb_root}, rebuild mono_tum_vi, and retry."
                )
            raise RuntimeError(f"ORB-SLAM3 did not produce a usable {required.name}")
    return trajectory, keyframes, points


def load_orb_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if rows.shape[1] != 8 or len(rows) < 8:
        raise RuntimeError(f"Invalid ORB trajectory shape {rows.shape}: {path}")
    timestamps = np.rint(rows[:, 0]).astype(np.int64)
    order = np.argsort(timestamps)
    rows = rows[order]
    timestamps = timestamps[order]
    unique = np.concatenate(([True], np.diff(timestamps) > 0))
    return timestamps[unique], rows[unique, 1:4], rows[unique, 4:8]


def select_contiguous_track(
    expected_timestamps: np.ndarray,
    pose_timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    max_gap_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    lookup = {int(value): index for index, value in enumerate(expected_timestamps)}
    matched = np.asarray(
        [lookup.get(int(value), -1) for value in pose_timestamps], dtype=np.int64
    )
    valid = matched >= 0
    matched = matched[valid]
    positions = positions[valid]
    quaternions = quaternions[valid]
    pose_timestamps = pose_timestamps[valid]
    if len(matched) < 8:
        raise RuntimeError("Fewer than eight ORB poses match extracted frame timestamps")
    order = np.argsort(matched)
    matched = matched[order]
    positions = positions[order]
    quaternions = quaternions[order]
    pose_timestamps = pose_timestamps[order]
    unique = np.concatenate(([True], np.diff(matched) > 0))
    matched = matched[unique]
    positions = positions[unique]
    quaternions = quaternions[unique]
    pose_timestamps = pose_timestamps[unique]

    split_positions = np.flatnonzero(np.diff(matched) > max_gap_frames) + 1
    segments = np.split(np.arange(len(matched)), split_positions)
    segment = max(
        segments,
        key=lambda indices: (len(indices), int(matched[indices[-1]] - matched[indices[0]])),
    )
    matched = matched[segment]
    positions = positions[segment]
    quaternions = quaternions[segment]
    pose_timestamps = pose_timestamps[segment]
    first = int(matched[0])
    last = int(matched[-1])
    if last - first + 1 < 30:
        raise RuntimeError("The longest continuous ORB segment has fewer than 30 frames")
    return matched, pose_timestamps, positions, quaternions, {
        "matched_pose_count": int(len(matched)),
        "first_source_frame": first,
        "last_source_frame_inclusive": last,
        "selected_frame_count": last - first + 1,
        "interpolated_frame_count": int(last - first + 1 - len(matched)),
        "max_observed_gap_frames": int(np.diff(matched).max(initial=0)),
        "discarded_matched_poses": int(valid.sum() - len(matched)),
    }


def interpolate_c2w(
    frame_indices: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.arange(frame_indices[0], frame_indices[-1] + 1, dtype=np.float64)
    source = frame_indices.astype(np.float64)
    translation = np.column_stack(
        [np.interp(query, source, positions[:, axis]) for axis in range(3)]
    )
    rotations = Slerp(source, Rotation.from_quat(quaternions))(query).as_matrix()
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], len(query), axis=0)
    c2w[:, :3, :3] = rotations
    c2w[:, :3, 3] = translation
    return query.astype(np.int64), c2w


def load_map_points(
    path: Path, min_observations: int, min_found_ratio: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if rows.shape[1] < 6:
        raise RuntimeError(f"Invalid ORB map point shape {rows.shape}: {path}")
    valid = (
        np.isfinite(rows[:, :6]).all(axis=1)
        & (rows[:, 4] >= min_observations)
        & (rows[:, 5] >= min_found_ratio)
    )
    selected = rows[valid]
    if len(selected) < 100:
        raise RuntimeError(
            f"Only {len(selected)} persistent ORB points pass the quality filter"
        )
    return selected[:, 1:4], selected[:, 0].astype(np.int64), {
        "raw_point_count": int(len(rows)),
        "filtered_point_count": int(len(selected)),
        "min_observations": int(min_observations),
        "min_found_ratio": float(min_found_ratio),
    }


def visible_point_indices(
    w2c: np.ndarray,
    points: np.ndarray,
    focal: float,
    width: int,
    height: int,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]
    depth = camera[:, 2]
    uv = np.empty((len(points), 2), dtype=np.float64)
    safe = np.maximum(depth, np.finfo(np.float64).eps)
    uv[:, 0] = focal * camera[:, 0] / safe + width / 2.0
    uv[:, 1] = focal * camera[:, 1] / safe + height / 2.0
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(depth)
        & (depth > min_depth)
        & (depth < max_depth)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < height)
    )
    return np.flatnonzero(valid), depth, uv


def estimate_depth_scale(
    canonical_c2w: np.ndarray,
    points: np.ndarray,
    focal: float,
    width: int,
    height: int,
    target_median_depth: float,
) -> tuple[float, float, int]:
    sample_count = min(64, len(canonical_c2w))
    frame_indices = np.unique(
        np.rint(np.linspace(0, len(canonical_c2w) - 1, sample_count)).astype(int)
    )
    samples = []
    for frame in frame_indices:
        w2c = np.linalg.inv(canonical_c2w[frame])
        visible, depth, _ = visible_point_indices(
            w2c, points, focal, width, height, 1.0e-5, float("inf")
        )
        if len(visible) > 4096:
            positions = np.linspace(0, len(visible) - 1, 4096).astype(int)
            visible = visible[positions]
        samples.append(depth[visible])
    valid_samples = [sample for sample in samples if len(sample)]
    if not valid_samples:
        raise RuntimeError("No filtered ORB point is visible from the selected trajectory")
    depths = np.concatenate(valid_samples)
    raw_median = float(np.median(depths))
    if not math.isfinite(raw_median) or raw_median <= 1.0e-8:
        raise RuntimeError(f"Invalid raw median visible depth: {raw_median}")
    return float(target_median_depth / raw_median), raw_median, int(len(depths))


def coverage_first_selection(
    visible: np.ndarray,
    depth: np.ndarray,
    uv: np.ndarray,
    maximum: int,
    width: int,
) -> np.ndarray:
    if len(visible) <= maximum:
        return np.sort(visible.astype(np.int64))
    order = np.argsort(depth[visible])
    cells = (
        (uv[visible, 1].astype(np.int64) // 16) * math.ceil(width / 16)
        + uv[visible, 0].astype(np.int64) // 16
    )
    _, first = np.unique(cells[order], return_index=True)
    selected_positions = order[np.sort(first)]
    selected = visible[selected_positions]
    if len(selected) < maximum:
        used = np.zeros(len(visible), dtype=bool)
        used[selected_positions] = True
        selected = np.concatenate((selected, visible[order[~used[order]]]))
    return np.sort(selected[:maximum].astype(np.int64))


def write_dataset(
    video: Path,
    output: Path,
    staging: Path,
    frames: list[Path],
    expected_timestamps: np.ndarray,
    trajectory_path: Path,
    keyframe_path: Path,
    map_path: Path,
    focal: float,
    args: argparse.Namespace,
    video_probe: dict[str, Any],
) -> dict[str, Any]:
    pose_ns, positions, quaternions = load_orb_trajectory(trajectory_path)
    max_gap_frames = max(1, int(round(args.max_interpolation_gap_s * args.fps)))
    frame_indices, _, positions, quaternions, tracking = select_contiguous_track(
        expected_timestamps,
        pose_ns,
        positions,
        quaternions,
        max_gap_frames,
    )
    selected_indices, c2w = interpolate_c2w(frame_indices, positions, quaternions)
    canonical_from_orb = np.linalg.inv(c2w[0])
    canonical_c2w = canonical_from_orb[None] @ c2w

    raw_points, point_ids, point_stats = load_map_points(
        map_path, args.min_orb_observations, args.min_orb_found_ratio
    )
    homogeneous = np.column_stack((raw_points, np.ones(len(raw_points))))
    canonical_points = (canonical_from_orb @ homogeneous.T).T[:, :3]
    scale, raw_median_depth, depth_sample_count = estimate_depth_scale(
        canonical_c2w,
        canonical_points,
        focal,
        args.width,
        args.height,
        args.target_median_depth,
    )
    canonical_c2w[:, :3, 3] *= scale
    canonical_points *= scale
    canonical_w2c = np.linalg.inv(canonical_c2w)

    rectified = staging / "rectified"
    point_dir = staging / "orb_point_clouds"
    id_dir = staging / "orb_point_ids"
    preprocess = staging / "preprocess"
    for directory in (rectified, point_dir, id_dir, preprocess):
        directory.mkdir(parents=True, exist_ok=True)
    np.save(preprocess / "global_sparse_points.npy", canonical_points.astype(np.float32))
    np.save(preprocess / "global_sparse_point_ids.npy", point_ids.astype(np.int64))

    cameras = []
    point_counts = []
    for output_index, (source_index, pose) in enumerate(
        zip(selected_indices, canonical_w2c)
    ):
        source_frame = frames[int(source_index)]
        image_name = f"aria_{output_index:05d}.jpg"
        destination = rectified / image_name
        if not destination.exists():
            destination.symlink_to(os.path.relpath(source_frame, destination.parent))
        visible, depths, uv = visible_point_indices(
            pose,
            canonical_points,
            focal,
            args.width,
            args.height,
            0.3,
            120.0,
        )
        selected = coverage_first_selection(
            visible,
            depths,
            uv,
            args.max_points_per_frame,
            args.width,
        )
        np.save(
            point_dir / f"point_cloud_{output_index}.npy",
            canonical_points[selected].astype(np.float32),
        )
        np.save(
            id_dir / f"point_ids_{output_index}.npy",
            point_ids[selected].astype(np.int64),
        )
        point_counts.append(int(len(selected)))
        cameras.append(
            {
                "T_camera_world": pose.tolist(),
                "image": image_name,
                "timestamp": float(expected_timestamps[int(source_index)]) / 1.0e9,
                "frame_index": output_index,
                "source_frame_index": int(source_index),
                "intrinsic": {
                    "fx": focal,
                    "fy": focal,
                    "cx": args.width / 2.0,
                    "cy": args.height / 2.0,
                },
                "width": args.width,
                "height": args.height,
                "focal": focal,
            }
        )

    counts = np.asarray(point_counts, dtype=np.int64)
    supported_fraction = float(np.mean(counts >= 10))
    if supported_fraction < args.min_supported_frame_fraction:
        raise RuntimeError(
            f"Only {supported_fraction:.3f} of selected frames have at least 10 "
            "persistent points"
        )
    trajectory = {"cameras": cameras}
    write_json(staging / "trajectory_orb.json", trajectory)
    write_json(staging / "trajectory.json", trajectory)
    (staging / "image_list.txt").write_text(
        "\n".join(f"rectified/{camera['image']}" for camera in cameras) + "\n",
        encoding="utf-8",
    )

    dataset_name = f"MyData-{video.stem}-ORBMono"
    calibration = {
        "fx": focal,
        "fy": focal,
        "cx": args.width / 2.0,
        "cy": args.height / 2.0,
        "width": args.width,
        "height": args.height,
        "near": 0.01,
        "far": 1000,
    }
    dataset = {
        "name": dataset_name,
        "type": "aria",
        "data_source": "orb",
        "dataset_path": str(output),
        "num_threads": 0,
        "begin_cutoff": 0,
        "end_cutoff": 0,
        "max_pts_num": 5000,
        "vignette": False,
        "use_vignette_type": "post-render",
        "Calibration": calibration,
    }
    base_config = {
        "inherit_from": "configs/aria/orb_tracking/aria_base.yaml",
        "Dataset": dataset,
        "Testset": dict(dataset),
        "Results": {
            "save_dir": str(ROOT / "Logs_mydata"),
            "save_gt": False,
            "save_exr": False,
            "save_mesh": False,
            "skip_eval": True,
        },
        "Mapper": {
            "use_multi_reso": False,
            "initialization_frames": 4,
            "optimization_iters": 30,
            "initialization_iters": 40,
            "post_refinement": {"max_steps": 1000, "opt_cam": False},
            "KFGraph": {"kf_interval": 1, "global_window_size": 4},
            "CameraOptimizer": {"pose_refine_init_steps": 0, "pose_opt_steps": 4},
        },
        "Model": {
            "extra_pts_num": 3200,
            "err_threshold": 0.05,
            "camera_scale_rescalar": 0.25,
            "scene_scale": 1.0,
        },
    }
    (staging / "config.yaml").write_text(
        yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8"
    )

    centers = canonical_c2w[:, :3, 3]
    stats = {
        "schema_version": 1,
        "dataset": "mydata",
        "scene": video.stem,
        "source_video": str(video),
        "source_probe": video_probe,
        "frame_sampling_fps": float(args.fps),
        "output_resolution": [args.width, args.height],
        "horizontal_fov_deg": float(args.horizontal_fov_deg),
        "intrinsic": calibration,
        "extracted_frame_count": len(frames),
        "frame_count": len(cameras),
        "tracking": tracking,
        "pose_source": "ORB-SLAM3 monocular largest continuous map",
        "sparse_world_geometry": "persistent ORB-SLAM3 map points",
        "orb_keyframe_trajectory": str(output / "orb_output" / keyframe_path.name),
        "scale_contract": {
            "metric_scale_observable": False,
            "method": "median visible persistent-point depth normalization",
            "target_median_depth_units": float(args.target_median_depth),
            "raw_median_visible_depth": raw_median_depth,
            "applied_scale": scale,
            "depth_sample_count": depth_sample_count,
        },
        "persistent_points": point_stats,
        "points_per_frame": {
            "min": int(counts.min()),
            "median": float(np.median(counts)),
            "mean": float(np.mean(counts)),
            "max": int(counts.max()),
            "supported_frame_fraction_ge_10": supported_fraction,
        },
        "trajectory_span_units": np.ptp(centers, axis=0).tolist(),
        "trajectory_length_units": float(
            np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()
        ),
        "output": str(output),
    }
    write_json(staging / "conversion_stats.json", stats)
    return stats


def validate_final(path: Path, video: Path) -> dict[str, Any] | None:
    stats_path = path / "conversion_stats.json"
    trajectory_path = path / "trajectory_orb.json"
    if not stats_path.is_file() or not trajectory_path.is_file():
        return None
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        count = int(stats["frame_count"])
        if Path(stats["source_video"]).resolve() != video.resolve():
            return None
        if len(trajectory.get("cameras", [])) != count:
            return None
        if len(list((path / "rectified").glob("aria_*.jpg"))) != count:
            return None
        if len(list((path / "orb_point_clouds").glob("point_cloud_*.npy"))) != count:
            return None
        if len(list((path / "orb_point_ids").glob("point_ids_*.npy"))) != count:
            return None
        return stats
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def main() -> int:
    args = parse_args()
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if not math.isclose(args.fps, round(args.fps), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("ORB-SLAM3 requires --fps to be an integer value")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Output dimensions must be positive")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if not 1.0 < args.horizontal_fov_deg < 179.0:
        raise ValueError("--horizontal-fov-deg must be in (1, 179)")
    if args.target_median_depth <= 0.0:
        raise ValueError("--target-median-depth must be positive")
    if args.max_points_per_frame <= 0:
        raise ValueError("--max-points-per-frame must be positive")
    if not 0.0 <= args.min_supported_frame_fraction <= 1.0:
        raise ValueError("--min-supported-frame-fraction must be in [0, 1]")

    data_root = args.data_root.expanduser().resolve()
    video = resolve_video(args.video, data_root)
    output = data_root / video.stem
    complete = validate_final(output, video)
    if complete is not None:
        print(f"Reusing complete dataset: {output}")
        print(json.dumps(complete, indent=2))
        return 0
    if output.exists():
        raise RuntimeError(f"Existing output is incomplete or mismatched: {output}")

    staging = output.with_name(output.name + ".staging")
    staging.mkdir(parents=True, exist_ok=True)
    video_probe = probe_video(video, args.ffprobe)
    frames = extract_frames(video, staging, args, video_probe)
    focal = focal_from_fov(args.width, args.horizontal_fov_deg)
    settings = staging / "orbslam3_monocular.yaml"
    write_orb_settings(settings, args, focal)
    timestamps, image_dir, times_path = prepare_orb_input(staging, frames, args.fps)
    trajectory_path, keyframe_path, map_path = run_orb_slam(
        staging, image_dir, times_path, video.stem, args
    )
    stats = write_dataset(
        video,
        output,
        staging,
        frames,
        timestamps,
        trajectory_path,
        keyframe_path,
        map_path,
        focal,
        args,
        video_probe,
    )
    staging.replace(output)
    print(json.dumps(stats, indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
