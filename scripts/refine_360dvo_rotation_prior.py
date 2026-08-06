#!/usr/bin/env python3
"""Refine one 360DVO camera-axis rotation with held-out epipolar evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation

try:
    from scripts.merge_360dvo_orbslam3_pose_windows import build_poses, c2w, summarize
except ModuleNotFoundError:
    from merge_360dvo_orbslam3_pose_windows import build_poses, c2w, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    initial = parser.add_mutually_exclusive_group(required=True)
    initial.add_argument("--initial-pose", type=Path)
    initial.add_argument(
        "--orb-work-dir",
        type=Path,
        help="Use every valid ORB pose window as a calibration candidate.",
    )
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--max-matches-per-pair", type=int, default=256)
    parser.add_argument("--min-match-score", type=float, default=0.15)
    parser.add_argument("--epipolar-threshold-px", type=float, default=1.5)
    parser.add_argument("--soft-temperature-px", type=float, default=0.25)
    parser.add_argument("--search-deg", type=float, default=20.0)
    parser.add_argument("--max-iterations", type=int, default=16)
    parser.add_argument("--min-validation-gain", type=float, default=0.005)
    parser.add_argument(
        "--allow-rejected-initializer",
        action="store_true",
        help="Write a held-out-rejected result only for a subsequent certified stage.",
    )
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def camera_model(cameras: list[dict]) -> np.ndarray:
    intrinsic = cameras[0]["intrinsic"]
    return np.asarray(
        [
            [intrinsic["fx"], 0.0, intrinsic["cx"]],
            [0.0, intrinsic["fy"], intrinsic["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def pose_from_rotation_center(rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = -rotation.T @ center
    return pose


def fundamental(
    rotation_i: np.ndarray,
    center_i: np.ndarray,
    rotation_j: np.ndarray,
    center_j: np.ndarray,
    k_inverse: np.ndarray,
) -> np.ndarray:
    pose_i = pose_from_rotation_center(rotation_i, center_i)
    pose_j = pose_from_rotation_center(rotation_j, center_j)
    relative = pose_j @ np.linalg.inv(pose_i)
    essential = skew(relative[:3, 3]) @ relative[:3, :3]
    return k_inverse.T @ essential @ k_inverse


def sampson_errors(
    matrix: np.ndarray, points_i: np.ndarray, points_j: np.ndarray
) -> np.ndarray:
    ones = np.ones((len(points_i), 1), dtype=np.float64)
    left = np.concatenate((points_i, ones), axis=1)
    right = np.concatenate((points_j, ones), axis=1)
    f_left = (matrix @ left.T).T
    ft_right = (matrix.T @ right.T).T
    numerator = np.sum(right * f_left, axis=1) ** 2
    denominator = (
        f_left[:, 0] ** 2
        + f_left[:, 1] ** 2
        + ft_right[:, 0] ** 2
        + ft_right[:, 1] ** 2
        + 1.0e-12
    )
    return np.sqrt(numerator / denominator)


def parse_pair(path: Path) -> tuple[int, int]:
    left, right = path.stem.split("_")
    return int(left), int(right)


def select_pair_paths(match_dir: Path, maximum: int) -> list[Path]:
    paths = sorted(match_dir.glob("*.npz"))
    if not paths:
        raise RuntimeError(f"No cached matches under {match_dir}")
    if maximum <= 0:
        raise ValueError("max-pairs must be positive")
    if len(paths) <= maximum:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=maximum, dtype=np.int64)
    return [paths[int(index)] for index in np.unique(indices)]


def orb_correction_candidates(
    source_cameras: list[dict], work_dir: Path
) -> list[tuple[str, np.ndarray]]:
    source_rotations = np.asarray([c2w(camera)[:3, :3] for camera in source_cameras])
    candidates = []
    all_corrections = []
    for trajectory_path in sorted(
        work_dir.glob("windows/*/pose_gtcenter/trajectory_orb.json")
    ):
        corrections = []
        for fallback, camera in enumerate(read_json(trajectory_path)["cameras"]):
            frame = int(
                camera.get("source_frame_index", camera.get("frame_index", fallback))
            )
            corrections.append(source_rotations[frame].T @ c2w(camera)[:3, :3])
        if not corrections:
            continue
        matrix = Rotation.from_matrix(np.stack(corrections)).mean().as_matrix()
        candidates.append((str(trajectory_path.parent), matrix))
        all_corrections.extend(corrections)
    if all_corrections:
        candidates.append(
            (
                "all_orb_observations_mean",
                Rotation.from_matrix(np.stack(all_corrections)).mean().as_matrix(),
            )
        )
    candidates.extend(
        [
            ("identity", np.eye(3)),
            ("axis_x_180", Rotation.from_euler("x", 180.0, degrees=True).as_matrix()),
            ("axis_y_180", Rotation.from_euler("y", 180.0, degrees=True).as_matrix()),
            ("axis_z_180", Rotation.from_euler("z", 180.0, degrees=True).as_matrix()),
        ]
    )
    unique = []
    for label, candidate in candidates:
        if any(
            np.degrees(Rotation.from_matrix(existing.T @ candidate).magnitude())
            < 1.0e-3
            for _, existing in unique
        ):
            continue
        unique.append((label, candidate))
    if not unique:
        raise RuntimeError(f"No rotation candidates found under {work_dir}")
    return unique


def load_samples(
    cache: Path,
    pair_paths: list[Path],
    min_score: float,
    max_matches: int,
) -> tuple[list[tuple[int, int, np.ndarray, np.ndarray]], list]:
    if max_matches <= 0:
        raise ValueError("max-matches-per-pair must be positive")
    needed_frames = sorted({value for path in pair_paths for value in parse_pair(path)})
    keypoints = {}
    for frame in needed_frames:
        with np.load(cache / "features" / f"{frame:05d}.npz") as payload:
            keypoints[frame] = payload["keypoints"].astype(np.float64)

    fit = []
    validation = []
    for path in pair_paths:
        frame_i, frame_j = parse_pair(path)
        with np.load(path) as payload:
            indices = payload["indices"]
            scores = payload["scores"].astype(np.float32)
        selected = np.flatnonzero(scores >= min_score)
        if len(selected) > max_matches:
            top = np.argpartition(scores[selected], -max_matches)[-max_matches:]
            selected = selected[top]
        if not len(selected):
            continue
        matches = indices[selected]
        sample = (
            frame_i,
            frame_j,
            keypoints[frame_i][matches[:, 0]],
            keypoints[frame_j][matches[:, 1]],
        )
        target = validation if (frame_i * 131 + frame_j * 17) % 5 == 0 else fit
        target.append(sample)
    if not fit or not validation:
        raise RuntimeError("Epipolar fit/validation split is empty")
    return fit, validation


def evaluate(
    correction: np.ndarray,
    source_rotations: np.ndarray,
    centers: np.ndarray,
    samples: list[tuple[int, int, np.ndarray, np.ndarray]],
    k_inverse: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict]:
    errors = []
    for frame_i, frame_j, points_i, points_j in samples:
        errors.append(
            sampson_errors(
                fundamental(
                    source_rotations[frame_i] @ correction,
                    centers[frame_i],
                    source_rotations[frame_j] @ correction,
                    centers[frame_j],
                    k_inverse,
                ),
                points_i,
                points_j,
            )
        )
    values = np.concatenate(errors)
    return values, {
        "match_count": len(values),
        "inlier_count": int(np.sum(values <= threshold)),
        "inlier_fraction": float(np.mean(values <= threshold)),
        "error_px": summarize(values),
    }


def write_pose_dataset(
    source: Path,
    output: Path,
    source_cameras: list[dict],
    rotations: np.ndarray,
    statistics: dict,
) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {output}")
    staging = output.with_name(output.name + ".staging")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Stale staging directory exists: {staging}")
    poses, coordinate = build_poses(source_cameras, rotations, normalize_first=True)
    rectified = staging / "rectified"
    rectified.mkdir(parents=True)
    cameras = []
    for index, (source_camera, pose) in enumerate(zip(source_cameras, poses)):
        image = source / "rectified" / source_camera["image"]
        link = rectified / f"aria_{index:05d}.png"
        link.symlink_to(image.resolve())
        camera = dict(source_camera)
        camera["T_camera_world"] = pose.tolist()
        camera["image"] = link.name
        camera["frame_index"] = index
        camera["source_frame_index"] = index
        cameras.append(camera)
    trajectory = json.dumps({"cameras": cameras}, indent=2) + "\n"
    (staging / "trajectory.json").write_text(trajectory, encoding="utf-8")
    (staging / "trajectory_orb.json").write_text(trajectory, encoding="utf-8")
    statistics["coordinate_certificate"] = coordinate
    (staging / "conversion_stats.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
    )
    (staging / "README.md").write_text(
        "# Epipolar-refined 360DVO camera-axis calibration\n\n"
        "A constant source-to-front-camera rotation is initialized by ORB-SLAM3 "
        "windows and refined on cached DISK-LightGlue correspondences. A disjoint "
        "held-out pair set must improve before this pose dataset is written. Camera "
        "centers remain the original 360DVO GT centers.\n",
        encoding="utf-8",
    )
    staging.replace(output)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    initial_pose = args.initial_pose.resolve() if args.initial_pose else None
    orb_work_dir = args.orb_work_dir.resolve() if args.orb_work_dir else None
    cache = args.feature_cache.resolve()
    output = args.output.resolve()
    source_cameras = read_json(source / "trajectory_orb.json")["cameras"]
    source_c2w = np.asarray([c2w(camera) for camera in source_cameras])
    source_rotations = source_c2w[:, :3, :3]
    centers = source_c2w[:, :3, 3]
    k_inverse = np.linalg.inv(camera_model(source_cameras))
    pair_paths = select_pair_paths(cache / "matches", args.max_pairs)
    fit, validation = load_samples(
        cache,
        pair_paths,
        args.min_match_score,
        args.max_matches_per_pair,
    )

    if initial_pose is not None:
        initial_stats = read_json(initial_pose / "conversion_stats.json")
        candidates = [
            (
                str(initial_pose),
                np.asarray(
                    initial_stats["continuity_certificate"][
                        "camera_axis_correction"
                    ],
                    dtype=np.float64,
                ),
            )
        ]
    else:
        assert orb_work_dir is not None
        candidates = orb_correction_candidates(source_cameras, orb_work_dir)

    def correction_objective(correction: np.ndarray) -> float:
        errors, _ = evaluate(
            correction,
            source_rotations,
            centers,
            fit,
            k_inverse,
            args.epipolar_threshold_px,
        )
        logits = np.clip(
            (errors - args.epipolar_threshold_px) / args.soft_temperature_px,
            -40.0,
            40.0,
        )
        return -float(np.mean(1.0 / (1.0 + np.exp(logits))))

    candidate_scores = [
        (correction_objective(candidate), label, candidate)
        for label, candidate in candidates
    ]
    candidate_scores.sort(key=lambda item: item[0])
    _, initial_label, initial_correction = candidate_scores[0]

    def corrected(parameters: np.ndarray) -> np.ndarray:
        delta = Rotation.from_rotvec(np.deg2rad(parameters)).as_matrix()
        return initial_correction @ delta

    def objective(parameters: np.ndarray) -> float:
        data_term = correction_objective(corrected(parameters))
        regularizer = 1.0e-4 * float(np.sum(np.square(parameters / args.search_deg)))
        return data_term + regularizer

    bounds = [(-args.search_deg, args.search_deg)] * 3
    global_result = differential_evolution(
        objective,
        bounds,
        seed=args.seed,
        maxiter=args.max_iterations,
        popsize=8,
        polish=False,
        workers=1,
        updating="immediate",
    )
    local_result = minimize(
        objective,
        global_result.x,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 80, "xtol": 1.0e-3, "ftol": 1.0e-5},
    )
    parameters = local_result.x if local_result.fun <= global_result.fun else global_result.x
    refined_correction = corrected(parameters)
    _, initial_fit = evaluate(
        initial_correction,
        source_rotations,
        centers,
        fit,
        k_inverse,
        args.epipolar_threshold_px,
    )
    _, refined_fit = evaluate(
        refined_correction,
        source_rotations,
        centers,
        fit,
        k_inverse,
        args.epipolar_threshold_px,
    )
    _, initial_validation = evaluate(
        initial_correction,
        source_rotations,
        centers,
        validation,
        k_inverse,
        args.epipolar_threshold_px,
    )
    _, refined_validation = evaluate(
        refined_correction,
        source_rotations,
        centers,
        validation,
        k_inverse,
        args.epipolar_threshold_px,
    )
    validation_gain = (
        refined_validation["inlier_fraction"]
        - initial_validation["inlier_fraction"]
    )
    certificate = {
        "initial_pose": str(initial_pose) if initial_pose is not None else None,
        "orb_work_dir": str(orb_work_dir) if orb_work_dir is not None else None,
        "initial_candidate_label": initial_label,
        "initial_candidate_count": len(candidate_scores),
        "initial_candidate_soft_scores": [
            {"label": label, "soft_objective": float(score)}
            for score, label, _ in candidate_scores
        ],
        "feature_cache": str(cache),
        "sampled_pair_count": len(pair_paths),
        "fit_pair_count": len(fit),
        "validation_pair_count": len(validation),
        "epipolar_threshold_px": args.epipolar_threshold_px,
        "initial_correction": initial_correction.tolist(),
        "delta_rotvec_deg": parameters.tolist(),
        "refined_correction": refined_correction.tolist(),
        "initial_fit": initial_fit,
        "refined_fit": refined_fit,
        "initial_validation": initial_validation,
        "refined_validation": refined_validation,
        "validation_inlier_fraction_gain": validation_gain,
        "required_validation_gain": args.min_validation_gain,
        "held_out_accepted": validation_gain >= args.min_validation_gain,
    }
    print(json.dumps(certificate, indent=2), flush=True)
    if validation_gain < args.min_validation_gain and not args.allow_rejected_initializer:
        raise RuntimeError(
            f"Held-out epipolar gain {validation_gain:.6f} is below "
            f"{args.min_validation_gain:.6f}"
        )

    statistics = {
        "schema_version": 1,
        "method": "held-out epipolar-refined ORB-calibrated source rotation prior",
        "pose_source": "orbslam3_mono_windowed_epipolar_refined",
        "sparse_world_geometry": "not_exported_pose_stage",
        "source": str(source),
        "frame_count": len(source_cameras),
        "pose_mode": "epipolar_refined_source_rotation_prior",
        "initializer_only": not certificate["held_out_accepted"],
        "epipolar_refinement_certificate": certificate,
    }
    write_pose_dataset(
        source,
        output,
        source_cameras,
        source_rotations @ refined_correction,
        statistics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
