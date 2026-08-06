#!/usr/bin/env python3
"""Build corrected 360DVO inputs from windowed ORB-SLAM3 and persistent tracks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/home/wmy/anaconda3/envs/worldvln/bin/python")
DEFAULT_ORB_BINARY = Path(
    "/home/wmy/workspace_vla/third_party/ORB_SLAM3/Examples/Monocular/mono_tum_vi"
)
DEFAULT_ORB_VOCABULARY = Path(
    "/home/wmy/workspace_vla/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt"
)
DEFAULT_ORB_SETTINGS = ROOT / "configs" / "360dvo" / "orbslam3_grove_frontview.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--orb-binary", type=Path, default=DEFAULT_ORB_BINARY)
    parser.add_argument("--orb-vocabulary", type=Path, default=DEFAULT_ORB_VOCABULARY)
    parser.add_argument("--orb-settings", type=Path, default=DEFAULT_ORB_SETTINGS)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--window-stride", type=int, default=40)
    parser.add_argument("--minimum-tail-frames", type=int, default=100)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-features", type=int, default=2048)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--pair-gaps", default="1,2,4,8,16,32,64")
    parser.add_argument("--min-accepted-points", type=int, default=100)
    parser.add_argument("--min-epipolar-inlier-fraction", type=float, default=0.35)
    parser.add_argument("--temporal-support-points", type=int, default=32)
    parser.add_argument(
        "--pose-contract-support-points",
        type=int,
        default=8,
        help="Minimum persistent anchors per frame for the pose-contract path.",
    )
    parser.add_argument(
        "--min-temporal-supported-frame-fraction", type=float, default=0.50
    )
    parser.add_argument("--temporal-support-bins", type=int, default=10)
    parser.add_argument(
        "--min-supported-frame-fraction-per-bin", type=float, default=0.25
    )
    parser.add_argument("--min-passing-temporal-bin-fraction", type=float, default=0.80)
    parser.add_argument("--max-edge-fill-frames", type=int, default=60)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def plan_windows(
    frame_count: int,
    window_size: int,
    stride: int,
    minimum_tail_frames: int,
) -> list[tuple[int, int]]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if stride >= window_size:
        raise ValueError("window_stride must be smaller than window_size")
    if minimum_tail_frames <= 0 or minimum_tail_frames > window_size:
        raise ValueError("minimum_tail_frames must be in [1, window_size]")

    starts = list(range(0, frame_count, stride))
    while len(starts) > 1 and frame_count - starts[-1] < minimum_tail_frames:
        starts.pop()
    windows = []
    for number, start in enumerate(starts):
        if number == len(starts) - 1:
            length = frame_count - start
        else:
            length = min(window_size, frame_count - start)
        windows.append((start, length))
    return windows


def append_logged(
    command: list[str],
    log_path: Path,
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{iso_now()}] $ {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command exited with code {result.returncode}; see {log_path}"
        )


def source_cameras(source: Path) -> list[dict[str, Any]]:
    cameras = read_json(source / "trajectory_orb.json").get("cameras", [])
    if not cameras:
        raise RuntimeError(f"No cameras in {source / 'trajectory_orb.json'}")
    for camera in cameras:
        image = source / "rectified" / camera["image"]
        if not image.is_file():
            raise FileNotFoundError(image)
    return cameras


def valid_prepared_window(path: Path, source: Path, start: int, length: int) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
        return (
            Path(manifest["source"]).resolve() == source.resolve()
            and int(manifest["start_frame"]) == start
            and int(manifest["frame_count"]) == length
            and len(list((path / "images").glob("*.png"))) == length
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def valid_orb_trajectory(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 128:
        return False
    try:
        rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
        return len(rows) >= 8 and all(len(row) == 8 for row in rows)
    except OSError:
        return False


def valid_pose_segment(path: Path, source: Path, start: int, length: int) -> bool:
    stats_path = path / "conversion_stats.json"
    if not stats_path.is_file():
        return False
    try:
        stats = read_json(stats_path)
        first = int(stats["source_frame_start"])
        last = int(stats["source_frame_end_inclusive"])
        return (
            Path(stats["source"]).resolve() == source.resolve()
            and bool(stats["used_gt_centers"])
            and start <= first <= last < start + length
            and int(stats["frame_count"]) == last - first + 1
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def valid_merged_pose(path: Path, source: Path, frame_count: int) -> bool:
    stats_path = path / "conversion_stats.json"
    if not stats_path.is_file():
        return False
    try:
        stats = read_json(stats_path)
        return (
            Path(stats["source"]).resolve() == source.resolve()
            and int(stats["frame_count"]) == frame_count
            and stats["pose_source"] == "orbslam3_mono_windowed"
            and len(list((path / "rectified").glob("aria_*.png"))) == frame_count
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def validate_track_dataset(
    path: Path,
    source: Path,
    frame_count: int,
    min_accepted_points: int,
    min_epipolar_inlier_fraction: float,
) -> dict[str, Any]:
    stats_path = path / "conversion_stats.json"
    if not stats_path.is_file():
        raise RuntimeError(f"Missing learned-track statistics: {stats_path}")
    stats = read_json(stats_path)
    graph = stats.get("graph", {})
    score_matches = int(graph.get("score_filtered_matches", 0))
    geometry_matches = int(graph.get("geometry_filtered_matches", 0))
    inlier_fraction = geometry_matches / max(score_matches, 1)
    checks = {
        "source_matches": Path(stats["source"]).resolve() == source.resolve(),
        "frame_count_matches": int(stats["frame_count"]) == frame_count,
        "persistent_sparse_geometry": stats.get("sparse_world_geometry")
        == "persistent",
        "accepted_point_count": int(stats.get("accepted_point_count", 0)),
        "epipolar_inlier_fraction": inlier_fraction,
    }
    if not checks["source_matches"] or not checks["frame_count_matches"]:
        raise RuntimeError(f"Learned-track dataset identity mismatch: {checks}")
    if not checks["persistent_sparse_geometry"]:
        raise RuntimeError("Learned-track output does not declare persistent geometry")
    if checks["accepted_point_count"] < min_accepted_points:
        raise RuntimeError(
            f"Only {checks['accepted_point_count']} accepted tracks; "
            f"minimum is {min_accepted_points}"
        )
    if inlier_fraction < min_epipolar_inlier_fraction:
        raise RuntimeError(
            f"Epipolar inlier fraction {inlier_fraction:.4f} is below "
            f"{min_epipolar_inlier_fraction:.4f}"
        )
    return {"statistics": stats, "validation": checks}


def temporal_track_support_certificate(
    path: Path,
    frame_count: int,
    *,
    support_points: int,
    min_supported_frame_fraction: float,
    temporal_bins: int,
    min_supported_frame_fraction_per_bin: float,
    min_passing_temporal_bin_fraction: float,
) -> dict[str, Any]:
    if support_points <= 0:
        raise ValueError("temporal support point threshold must be positive")
    if temporal_bins <= 0:
        raise ValueError("temporal support bin count must be positive")
    for name, value in (
        ("min_supported_frame_fraction", min_supported_frame_fraction),
        (
            "min_supported_frame_fraction_per_bin",
            min_supported_frame_fraction_per_bin,
        ),
        ("min_passing_temporal_bin_fraction", min_passing_temporal_bin_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    point_paths = sorted((path / "orb_point_clouds").glob("point_cloud_*.npy"))
    if len(point_paths) != frame_count:
        raise RuntimeError(
            f"Persistent geometry has {len(point_paths)} frame files; "
            f"expected {frame_count}"
        )
    counts = np.asarray(
        [
            np.load(point_path, mmap_mode="r", allow_pickle=False).shape[0]
            for point_path in point_paths
        ],
        dtype=np.int64,
    )
    supported = counts >= support_points
    bin_fractions = [
        float(np.mean(chunk)) for chunk in np.array_split(supported, temporal_bins)
    ]
    passing_bins = sum(
        fraction >= min_supported_frame_fraction_per_bin for fraction in bin_fractions
    )
    required_passing_bins = int(
        np.ceil(temporal_bins * min_passing_temporal_bin_fraction)
    )
    supported_frame_fraction = float(np.mean(supported))
    accepted = (
        supported_frame_fraction >= min_supported_frame_fraction
        and passing_bins >= required_passing_bins
    )
    return {
        "accepted": accepted,
        "frame_count": frame_count,
        "support_points": support_points,
        "supported_frame_count": int(np.sum(supported)),
        "supported_frame_fraction": supported_frame_fraction,
        "min_supported_frame_fraction": min_supported_frame_fraction,
        "temporal_bins": temporal_bins,
        "supported_frame_fraction_per_bin": bin_fractions,
        "min_supported_frame_fraction_per_bin": (min_supported_frame_fraction_per_bin),
        "passing_temporal_bin_count": passing_bins,
        "required_passing_temporal_bin_count": required_passing_bins,
        "point_count": {
            "min": int(counts.min()),
            "median": float(np.median(counts)),
            "p95": float(np.percentile(counts, 95)),
            "max": int(counts.max()),
        },
    }


def source_fallback_selection(
    source: Path,
    frame_count: int,
    candidate_rejections: dict[str, str],
) -> tuple[Path, Path, dict[str, Any]]:
    del source, frame_count
    raise RuntimeError(
        "Unsafe source fallback is disabled because the original 360DVO converter "
        "uses uncertified frame-local ORB geometry. Candidate failures: "
        + json.dumps(candidate_rejections, sort_keys=True)
    )


def selection_status(validation: dict[str, Any]) -> dict[str, Any]:
    checks = validation["validation"]
    result = {
        "accepted_point_count": checks["accepted_point_count"],
        "epipolar_inlier_fraction": checks["epipolar_inlier_fraction"],
        "pose_selection": validation["pose_selection"],
    }
    if "selected_frame_count" in validation:
        result["selected_frame_count"] = validation["selected_frame_count"]
    for key in (
        "temporal_support_certificate",
        "pose_contract_certificate",
        "candidate_rejections",
    ):
        if key in validation:
            result[key] = validation[key]
    return result


def held_out_pose_accepted(statistics: dict[str, Any]) -> bool:
    certificate = statistics.get("epipolar_refinement_certificate")
    if certificate is None:
        return True
    if "held_out_accepted" in certificate:
        return bool(certificate["held_out_accepted"])
    return float(certificate["validation_inlier_fraction_gain"]) >= float(
        certificate["required_validation_gain"]
    )


def learned_track_command(
    source: Path,
    output: Path,
    feature_cache: Path,
    args: argparse.Namespace,
    *,
    cache_only: bool = False,
    cache_frame_offset: int = 0,
) -> list[str]:
    command = [
        str(args.python),
        str(ROOT / "scripts" / "build_panoair_learned_tracks.py"),
        "--source",
        str(source),
        "--output",
        str(output),
        "--feature-cache",
        str(feature_cache),
        "--cache-frame-offset",
        str(cache_frame_offset),
        "--trajectory-file",
        "trajectory.json",
        "--pose-source-label",
        "orbslam3_mono",
        "--start-frame",
        "0",
        "--num-frames",
        "0",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--max-features",
        str(args.max_features),
        "--feature-batch-size",
        str(args.feature_batch_size),
        "--pair-gaps",
        args.pair_gaps,
        "--max-epipolar-error-px",
        "1.5",
        "--min-track-length",
        "3",
        "--max-reprojection-error-px",
        "2.0",
        "--max-median-reprojection-error-px",
        "1.0",
        "--min-triangulation-angle-deg",
        "1.5",
        "--max-depth-m",
        "120",
        "--max-nearest-camera-distance-m",
        "60",
        "--max-position-std-m",
        "1.0",
        "--max-relative-position-std",
        "0.05",
    ]
    if cache_only:
        command.append("--cache-only")
    return command


def build_auto_pose_contract_tracks(
    source: Path,
    output_root: Path,
    cache_root: Path,
    frame_count: int,
    logs: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str, Any]]:
    """Select the pose convention before triangulating persistent tracks."""
    feature_cache = cache_root / f"{args.scene}_disklg{args.max_features}"
    pose_output = output_root / f"{args.scene}_auto_pose_contract"
    track_output = output_root / f"{args.scene}_auto_pose_contract_tracks"

    if not any((feature_cache / "matches").glob("*.npz")):
        append_logged(
            learned_track_command(
                source,
                output_root / f"{args.scene}_cache_only_unused",
                feature_cache,
                args,
                cache_only=True,
            ),
            logs / "build_pose_contract_image_cache.log",
            cwd=ROOT,
        )

    if not (pose_output / "conversion_stats.json").is_file():
        if pose_output.exists() or pose_output.is_symlink():
            raise RuntimeError(f"Incomplete pose-contract output: {pose_output}")
        append_logged(
            [
                str(args.python),
                str(ROOT / "scripts" / "select_360dvo_pose_contract.py"),
                "--source",
                str(source),
                "--feature-cache",
                str(feature_cache),
                "--output",
                str(pose_output),
            ],
            logs / "select_pose_contract.log",
            cwd=ROOT,
        )
    pose_statistics = read_json(pose_output / "conversion_stats.json")
    pose_certificate = pose_statistics.get("pose_contract_certificate", {})
    if not pose_certificate.get("accepted", False):
        raise RuntimeError("Pose-contract output lacks an accepted certificate")
    source_frame_count = int(
        pose_statistics.get("source_frame_count", pose_statistics["frame_count"])
    )
    if source_frame_count != frame_count:
        raise RuntimeError(
            "Pose-contract source frame count mismatch: "
            f"{source_frame_count} != {frame_count}"
        )
    coordinate_certificate = pose_statistics.get("coordinate_certificate", {})
    selected_frame_count = int(pose_statistics["frame_count"])
    cache_frame_offset = int(coordinate_certificate.get("source_image_start", 0))

    if not (track_output / "conversion_stats.json").is_file():
        if track_output.exists() or track_output.is_symlink():
            raise RuntimeError(f"Incomplete pose-contract track output: {track_output}")
        append_logged(
            learned_track_command(
                pose_output,
                track_output,
                feature_cache,
                args,
                cache_frame_offset=cache_frame_offset,
            ),
            logs / "build_pose_contract_persistent_tracks.log",
            cwd=ROOT,
        )
    validation = validate_track_dataset(
        track_output,
        pose_output,
        selected_frame_count,
        args.min_accepted_points,
        args.min_epipolar_inlier_fraction,
    )
    temporal_certificate = temporal_track_support_certificate(
        track_output,
        selected_frame_count,
        support_points=args.pose_contract_support_points,
        min_supported_frame_fraction=args.min_temporal_supported_frame_fraction,
        temporal_bins=args.temporal_support_bins,
        min_supported_frame_fraction_per_bin=(
            args.min_supported_frame_fraction_per_bin
        ),
        min_passing_temporal_bin_fraction=args.min_passing_temporal_bin_fraction,
    )
    if not temporal_certificate["accepted"]:
        raise RuntimeError(
            "Pose-contract persistent geometry lacks sequence-wide support: "
            f"frames={temporal_certificate['supported_frame_fraction']:.3f}, "
            "passing_bins="
            f"{temporal_certificate['passing_temporal_bin_count']}/"
            f"{temporal_certificate['temporal_bins']}"
        )
    validation["pose_selection"] = "held_out_image_pose_contract"
    validation["selected_frame_count"] = selected_frame_count
    validation["pose_contract_certificate"] = pose_certificate
    validation["temporal_support_certificate"] = temporal_certificate
    return pose_output, track_output, validation


def build_epipolar_calibrated_tracks(
    source: Path,
    work_dir: Path,
    output_root: Path,
    cache_root: Path,
    frame_count: int,
    logs: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str, Any]]:
    feature_cache = cache_root / f"{args.scene}_disklg{args.max_features}"
    pose_output = output_root / f"{args.scene}_orbmono_epipolar_gtcenter_pose"
    track_output = output_root / f"{args.scene}_orbmono_epipolar_gtcenter_tracks"

    if not (pose_output / "conversion_stats.json").is_file():
        if pose_output.exists() or pose_output.is_symlink():
            raise RuntimeError(
                f"Existing epipolar pose output is incomplete: {pose_output}"
            )
        append_logged(
            learned_track_command(
                source,
                work_dir / "cache_only_unused",
                feature_cache,
                args,
                cache_only=True,
            ),
            logs / "build_image_match_cache.log",
            cwd=ROOT,
        )
        append_logged(
            [
                str(args.python),
                str(ROOT / "scripts" / "refine_360dvo_rotation_prior.py"),
                "--source",
                str(source),
                "--orb-work-dir",
                str(work_dir),
                "--feature-cache",
                str(feature_cache),
                "--output",
                str(pose_output),
                "--min-validation-gain",
                "0.0",
                "--allow-rejected-initializer",
                "--search-deg",
                "45",
                "--seed",
                str(args.seed),
            ],
            logs / "refine_rotation_prior.log",
            cwd=ROOT,
        )

    if not (track_output / "conversion_stats.json").is_file():
        if track_output.exists() or track_output.is_symlink():
            raise RuntimeError(
                f"Existing epipolar track output is incomplete: {track_output}"
            )
        append_logged(
            learned_track_command(
                pose_output,
                track_output,
                feature_cache,
                args,
            ),
            logs / "build_epipolar_persistent_tracks.log",
            cwd=ROOT,
        )

    try:
        validation = validate_track_dataset(
            track_output,
            pose_output,
            frame_count,
            args.min_accepted_points,
            args.min_epipolar_inlier_fraction,
        )
        pose_statistics = read_json(pose_output / "conversion_stats.json")
        if not held_out_pose_accepted(pose_statistics):
            raise RuntimeError(
                "Constant rotation calibration is held-out rejected and may only "
                "initialize the spline stage"
            )
        validation["pose_selection"] = "held_out_epipolar_refined_source_prior"
        return pose_output, track_output, validation
    except RuntimeError as constant_error:
        spline_pose = (
            output_root / f"{args.scene}_orbmono_epipolar_spline_gtcenter_pose"
        )
        spline_tracks = (
            output_root / f"{args.scene}_orbmono_epipolar_spline_gtcenter_tracks"
        )
        try:
            if not (spline_pose / "conversion_stats.json").is_file():
                if spline_pose.exists() or spline_pose.is_symlink():
                    raise RuntimeError(
                        "Existing epipolar spline pose output is incomplete: "
                        f"{spline_pose}"
                    )
                append_logged(
                    [
                        str(args.python),
                        str(ROOT / "scripts" / "refine_360dvo_rotation_spline.py"),
                        "--source",
                        str(source),
                        "--initial-pose",
                        str(pose_output),
                        "--feature-cache",
                        str(feature_cache),
                        "--output",
                        str(spline_pose),
                        "--device",
                        args.device,
                        "--seed",
                        str(args.seed),
                    ],
                    logs / "refine_rotation_spline.log",
                    cwd=ROOT,
                )
            if not (spline_tracks / "conversion_stats.json").is_file():
                if spline_tracks.exists() or spline_tracks.is_symlink():
                    raise RuntimeError(
                        "Existing epipolar spline track output is incomplete: "
                        f"{spline_tracks}"
                    )
                append_logged(
                    learned_track_command(
                        spline_pose,
                        spline_tracks,
                        feature_cache,
                        args,
                    ),
                    logs / "build_epipolar_spline_persistent_tracks.log",
                    cwd=ROOT,
                )
            validation = validate_track_dataset(
                spline_tracks,
                spline_pose,
                frame_count,
                args.min_accepted_points,
                args.min_epipolar_inlier_fraction,
            )
            temporal_certificate = temporal_track_support_certificate(
                spline_tracks,
                frame_count,
                support_points=args.temporal_support_points,
                min_supported_frame_fraction=(
                    args.min_temporal_supported_frame_fraction
                ),
                temporal_bins=args.temporal_support_bins,
                min_supported_frame_fraction_per_bin=(
                    args.min_supported_frame_fraction_per_bin
                ),
                min_passing_temporal_bin_fraction=(
                    args.min_passing_temporal_bin_fraction
                ),
            )
            if not temporal_certificate["accepted"]:
                raise RuntimeError(
                    "Spline persistent geometry lacks sequence-wide support: "
                    f"frames={temporal_certificate['supported_frame_fraction']:.3f}, "
                    "passing_bins="
                    f"{temporal_certificate['passing_temporal_bin_count']}/"
                    f"{temporal_certificate['temporal_bins']}"
                )
            validation["pose_selection"] = "held_out_epipolar_rotation_spline"
            validation["constant_pose_rejection"] = str(constant_error)
            validation["temporal_support_certificate"] = temporal_certificate
            return spline_pose, spline_tracks, validation
        except RuntimeError as spline_error:
            raise RuntimeError(
                "All certified visual-pose candidates failed: "
                + json.dumps(
                    {
                        "constant_epipolar_candidate": str(constant_error),
                        "rotation_spline_candidate": str(spline_error),
                    },
                    sort_keys=True,
                )
            )


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    work_dir = args.work_root.resolve() / args.scene
    output_root = args.output_root.resolve()
    cache_root = args.cache_root.resolve()
    for required in (
        args.python,
        args.orb_binary,
        args.orb_vocabulary,
        args.orb_settings,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    cameras = source_cameras(source)
    windows = plan_windows(
        len(cameras),
        args.window_size,
        args.window_stride,
        args.minimum_tail_frames,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    logs = work_dir / "logs"
    status_path = work_dir / "preprocess_status.json"
    status: dict[str, Any] = {
        "schema_version": 1,
        "scene": args.scene,
        "source": str(source),
        "frame_count": len(cameras),
        "window_plan": [
            {"index": i, "start_frame": start, "frame_count": length}
            for i, (start, length) in enumerate(windows)
        ],
        "started_at": iso_now(),
        "status": "running",
    }
    atomic_write_json(status_path, status)

    try:
        selected_pose, selected_tracks, validation = build_auto_pose_contract_tracks(
            source,
            output_root,
            cache_root,
            len(cameras),
            logs,
            args,
        )
        status.update(
            {
                "status": "success",
                "finished_at": iso_now(),
                "merged_pose_dataset": str(selected_pose),
                "track_dataset": str(selected_tracks),
                "feature_cache": str(
                    cache_root / f"{args.scene}_disklg{args.max_features}"
                ),
                **selection_status(validation),
            }
        )
        atomic_write_json(status_path, status)
        print(json.dumps(status, indent=2), flush=True)
        return 0
    except RuntimeError as pose_contract_error:
        status["pose_contract_rejection"] = str(pose_contract_error)
        atomic_write_json(status_path, status)

    existing_epipolar_pose = (
        output_root / f"{args.scene}_orbmono_epipolar_gtcenter_pose"
    )
    existing_epipolar_tracks = (
        output_root / f"{args.scene}_orbmono_epipolar_gtcenter_tracks"
    )
    if (existing_epipolar_pose / "conversion_stats.json").is_file() and (
        existing_epipolar_tracks / "conversion_stats.json"
    ).is_file():
        selected_pose, selected_tracks, validation = build_epipolar_calibrated_tracks(
            source,
            work_dir,
            output_root,
            cache_root,
            len(cameras),
            logs,
            args,
        )
        status.update(
            {
                "status": "success",
                "finished_at": iso_now(),
                "merged_pose_dataset": str(selected_pose),
                "track_dataset": str(selected_tracks),
                "feature_cache": str(
                    cache_root / f"{args.scene}_disklg{args.max_features}"
                ),
                **selection_status(validation),
            }
        )
        atomic_write_json(status_path, status)
        print(json.dumps(status, indent=2), flush=True)
        return 0

    segment_paths: list[Path] = []
    skipped_windows: list[dict[str, Any]] = []
    for index, (start, length) in enumerate(windows):
        stem = f"window_{index:03d}_start_{start:05d}_len_{length:05d}"
        window_dir = work_dir / "windows" / stem
        orb_input = window_dir / "input"
        orb_output = window_dir / "orb"
        pose_output = window_dir / "pose_gtcenter"
        orb_name = f"{args.scene}_w{index:03d}_s{start:05d}"
        orb_trajectory = orb_output / f"f_{orb_name}.txt"

        if not valid_prepared_window(orb_input, source, start, length):
            if orb_input.exists():
                raise RuntimeError(f"Existing prepared window is invalid: {orb_input}")
            append_logged(
                [
                    str(args.python),
                    str(ROOT / "scripts" / "prepare_360dvo_orbslam3_mono.py"),
                    "--source",
                    str(source),
                    "--output",
                    str(orb_input),
                    "--start-frame",
                    str(start),
                    "--num-frames",
                    str(length),
                    "--fps",
                    str(args.fps),
                ],
                logs / f"{stem}_prepare.log",
                cwd=ROOT,
            )

        orb_output.mkdir(parents=True, exist_ok=True)
        orb_error = None
        if not valid_orb_trajectory(orb_trajectory):
            try:
                append_logged(
                    [
                        str(args.orb_binary),
                        str(args.orb_vocabulary),
                        str(args.orb_settings),
                        str(orb_input / "images"),
                        str(orb_input / "times.txt"),
                        orb_name,
                    ],
                    logs / f"{stem}_orb.log",
                    cwd=orb_output,
                )
            except RuntimeError as error:
                orb_error = str(error)
        if not valid_orb_trajectory(orb_trajectory):
            skipped_windows.append(
                {
                    "index": index,
                    "start_frame": start,
                    "frame_count": length,
                    "stage": "orb",
                    "error": orb_error or "ORB-SLAM3 produced fewer than eight poses",
                }
            )
            continue

        convert_error = None
        if not valid_pose_segment(pose_output, source, start, length):
            if pose_output.exists():
                convert_error = f"Existing pose segment is invalid: {pose_output}"
            else:
                try:
                    append_logged(
                        [
                            str(args.python),
                            str(ROOT / "scripts" / "convert_360dvo_orbslam3_mono.py"),
                            "--source",
                            str(source),
                            "--orb-input",
                            str(orb_input),
                            "--orb-trajectory",
                            str(orb_trajectory),
                            "--output",
                            str(pose_output),
                            "--alignment-threshold-m",
                            "0.75",
                            "--max-interpolation-gap-s",
                            "0.5",
                            "--seed",
                            str(args.seed),
                            "--keep-gt-world",
                            "--use-gt-centers",
                        ],
                        logs / f"{stem}_convert.log",
                        cwd=ROOT,
                    )
                except RuntimeError as error:
                    convert_error = str(error)
        if not valid_pose_segment(pose_output, source, start, length):
            skipped_windows.append(
                {
                    "index": index,
                    "start_frame": start,
                    "frame_count": length,
                    "stage": "convert",
                    "error": convert_error
                    or "Converted pose segment failed validation",
                }
            )
            continue
        segment_paths.append(pose_output)

    if not segment_paths:
        selected_pose, selected_tracks, validation = build_epipolar_calibrated_tracks(
            source,
            work_dir,
            output_root,
            cache_root,
            len(cameras),
            logs,
            args,
        )
        status.update(
            {
                "status": "success",
                "finished_at": iso_now(),
                "valid_pose_segment_count": 0,
                "skipped_windows": skipped_windows,
                "merged_pose_dataset": str(selected_pose),
                "track_dataset": str(selected_tracks),
                "feature_cache": str(
                    cache_root / f"{args.scene}_disklg{args.max_features}"
                ),
                **selection_status(validation),
                "orb_merge_failure": "No valid ORB pose segment",
            }
        )
        atomic_write_json(status_path, status)
        print(json.dumps(status, indent=2), flush=True)
        return 0
    status["valid_pose_segment_count"] = len(segment_paths)
    status["skipped_windows"] = skipped_windows
    atomic_write_json(status_path, status)

    merged_pose = output_root / f"{args.scene}_orbmono_windowed_gtcenter_pose"
    merge_error = None
    if not valid_merged_pose(merged_pose, source, len(cameras)):
        if merged_pose.exists():
            raise RuntimeError(
                f"Existing merged pose dataset is invalid: {merged_pose}"
            )
        command = [
            str(args.python),
            str(ROOT / "scripts" / "merge_360dvo_orbslam3_pose_windows.py"),
            "--source",
            str(source),
            "--output",
            str(merged_pose),
            "--blend-ramp-frames",
            "50",
            "--max-overlap-disagreement-deg",
            "15",
            "--max-gap-fill-frames",
            "60",
            "--max-gap-rotation-deg",
            "180",
            "--max-edge-fill-frames",
            str(args.max_edge_fill_frames),
            "--select-covering-segments",
            "--allow-source-rotation-prior-fallback",
            "--fps",
            str(args.fps),
        ]
        for segment_path in segment_paths:
            command.extend(("--segment", str(segment_path)))
        try:
            append_logged(command, logs / "merge_pose_windows.log", cwd=ROOT)
        except RuntimeError as error:
            merge_error = error

    if merge_error is not None:
        selected_pose, selected_tracks, validation = build_epipolar_calibrated_tracks(
            source,
            work_dir,
            output_root,
            cache_root,
            len(cameras),
            logs,
            args,
        )
        status.update(
            {
                "status": "success",
                "finished_at": iso_now(),
                "merged_pose_dataset": str(selected_pose),
                "track_dataset": str(selected_tracks),
                "feature_cache": str(
                    cache_root / f"{args.scene}_disklg{args.max_features}"
                ),
                **selection_status(validation),
                "orb_merge_failure": str(merge_error),
            }
        )
        atomic_write_json(status_path, status)
        print(json.dumps(status, indent=2), flush=True)
        return 0

    track_output = output_root / f"{args.scene}_orbmono_windowed_gtcenter_tracks"
    feature_cache = cache_root / f"{args.scene}_disklg{args.max_features}"
    if not (track_output / "conversion_stats.json").is_file():
        if track_output.exists():
            raise RuntimeError(
                f"Existing learned-track output is incomplete: {track_output}"
            )
        append_logged(
            learned_track_command(
                merged_pose,
                track_output,
                feature_cache,
                args,
            ),
            logs / "build_persistent_tracks.log",
            cwd=ROOT,
        )

    try:
        validation = validate_track_dataset(
            track_output,
            merged_pose,
            len(cameras),
            args.min_accepted_points,
            args.min_epipolar_inlier_fraction,
        )
        selected_pose = merged_pose
        selected_tracks = track_output
        pose_selection = "overlap_certified_orb_windows"
    except RuntimeError as initial_error:
        selected_pose, selected_tracks, validation = build_epipolar_calibrated_tracks(
            source,
            work_dir,
            output_root,
            cache_root,
            len(cameras),
            logs,
            args,
        )
        pose_selection = validation["pose_selection"]
        status["initial_track_rejection"] = str(initial_error)
    status.update(
        {
            "status": "success",
            "finished_at": iso_now(),
            "merged_pose_dataset": str(selected_pose),
            "track_dataset": str(selected_tracks),
            "feature_cache": str(feature_cache),
            **selection_status(validation),
            "pose_selection": pose_selection,
        }
    )
    atomic_write_json(status_path, status)
    print(json.dumps(status, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
