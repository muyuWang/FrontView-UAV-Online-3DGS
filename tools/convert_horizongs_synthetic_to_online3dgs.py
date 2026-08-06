#!/usr/bin/env python3
"""Convert one HorizonGS synthetic RGB sequence into Online-3DGS-Monocular layout."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_airvln_to_online3dgs import (  # noqa: E402
    triangulate_pair,
    visible_points,
    voxel_downsample,
    write_points,
)
from convert_horizongs_real_to_online3dgs import (  # noqa: E402
    symlink_image,
    to_t_camera_world,
    write_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "HorizonGS" / "synthetic" / "citysample" / "street"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "Online3DGS_HorizonGS" / "city_street"
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "horizongs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--name", default="HorizonGS-city-street")
    parser.add_argument("--config-prefix", default="HorizonGS_city_street")
    parser.add_argument(
        "--pose-convention",
        choices=["nerf-opengl-c2w", "opencv-c2w", "opencv-w2c"],
        default="nerf-opengl-c2w",
        help="Convention of transform_matrix in HorizonGS transforms.json.",
    )
    parser.add_argument("--orb-features", type=int, default=4000)
    parser.add_argument("--pair-gaps", default="1,2,4,8")
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--max-reproj-error", type=float, default=8.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=1000.0)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--max-points-per-frame", type=int, default=5000)
    parser.add_argument("--frame-start", type=int, default=None, help="Inclusive source image index, e.g. 300.")
    parser.add_argument("--frame-end", type=int, default=None, help="Inclusive source image index, e.g. 349.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def frame_index(file_path: str) -> int:
    return int(Path(file_path).stem)


def load_frames(
    input_dir: Path,
    max_frames: int | None,
    frame_start: int | None,
    frame_end: int | None,
) -> tuple[Path, list[dict], list[str]]:
    scene_dir = input_dir.resolve().parent
    sequence = input_dir.name
    transforms_path = scene_dir / "transforms.json"
    data = json.load(open(transforms_path, "r", encoding="utf-8"))

    prefix = f"{sequence}/rgb/"
    frames = []
    missing = []
    for frame in data["frames"]:
        file_path = frame["file_path"]
        if not file_path.startswith(prefix):
            continue
        idx = frame_index(file_path)
        if frame_start is not None and idx < frame_start:
            continue
        if frame_end is not None and idx > frame_end:
            continue
        if (scene_dir / file_path).exists():
            frames.append(frame)
        else:
            missing.append(file_path)

    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("--max-frames must be >= 1")
        frames = frames[:max_frames]
    return scene_dir, frames, missing


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not (input_dir / "rgb").is_dir():
        raise FileNotFoundError(f"Expected RGB directory: {input_dir / 'rgb'}")
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output exists: {output_dir}. Use --force to overwrite.")
        shutil.rmtree(output_dir)

    if (
        args.frame_start is not None
        and args.frame_end is not None
        and args.frame_end < args.frame_start
    ):
        raise ValueError("--frame-end must be >= --frame-start")

    scene_dir, frames, missing = load_frames(
        input_dir,
        args.max_frames,
        args.frame_start,
        args.frame_end,
    )
    if not frames:
        raise RuntimeError(f"No existing RGB frames found under {input_dir}")

    first = frames[0]
    src_width = int(first["w"])
    src_height = int(first["h"])
    width = src_width - (src_width % 8)
    height = src_height - (src_height % 8)
    fx = float(first["fl_x"])
    fy = float(first["fl_y"])
    cx = float(first["cx"])
    cy = float(first["cy"])
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    rectified_dir = output_dir / "rectified"
    point_dir = output_dir / "orb_point_clouds"
    rectified_dir.mkdir(parents=True, exist_ok=True)
    point_dir.mkdir(parents=True, exist_ok=True)

    image_names: list[str] = []
    t_camera_worlds: list[np.ndarray] = []
    gray_images: list[np.ndarray] = []

    for i, frame in enumerate(frames):
        src = scene_dir / frame["file_path"]
        image_name = f"aria_{i:04d}{src.suffix}"
        symlink_image(src, rectified_dir / image_name)
        image_names.append(image_name)
        t_camera_worlds.append(
            to_t_camera_world(frame["transform_matrix"], args.pose_convention)
        )
        gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Could not read image: {src}")
        gray_images.append(gray[:height, :width])
        if i == 0 or (i + 1) % 250 == 0 or i + 1 == len(frames):
            print(f"Prepared RGB/poses: {i + 1}/{len(frames)}", flush=True)

    cameras = []
    for image_name, t_camera_world in zip(image_names, t_camera_worlds):
        cameras.append(
            {
                "T_camera_world": t_camera_world.astype(float).tolist(),
                "image": image_name,
                "intrinsic": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                "width": float(width),
                "height": float(height),
                "focal": float(fx),
            }
        )
    trajectory = {"cameras": cameras}
    for name in ["trajectory_orb.json", "trajectory.json"]:
        with open(output_dir / name, "w", encoding="utf-8") as f:
            json.dump(trajectory, f)

    with open(output_dir / "image_list.txt", "w", encoding="utf-8") as f:
        for image_name in image_names:
            f.write(f"rectified/{image_name}\n")

    orb = cv2.ORB_create(nfeatures=args.orb_features)
    keypoints = []
    descriptors = []
    for i, gray in enumerate(gray_images):
        kp, des = orb.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)
        if i == 0 or (i + 1) % 250 == 0 or i + 1 == len(gray_images):
            print(f"Detected ORB features: {i + 1}/{len(gray_images)}", flush=True)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pair_gaps = [int(x) for x in args.pair_gaps.split(",") if x.strip()]
    global_points = []
    pair_stats = []
    for i in range(len(frames)):
        for gap in pair_gaps:
            j = i + gap
            if j >= len(frames):
                continue
            center_i = np.linalg.inv(t_camera_worlds[i])[:3, 3]
            center_j = np.linalg.inv(t_camera_worlds[j])[:3, 3]
            baseline = float(np.linalg.norm(center_i - center_j))
            if baseline < 1e-4:
                continue
            pts = triangulate_pair(
                k,
                t_camera_worlds[i],
                t_camera_worlds[j],
                keypoints[i],
                keypoints[j],
                descriptors[i],
                descriptors[j],
                matcher,
                args.ratio,
                args.max_reproj_error,
                args.min_depth,
                args.max_depth,
            )
            if pts.shape[0] > 0:
                global_points.append(pts)
            pair_stats.append(
                {"i": i, "j": j, "baseline": baseline, "triangulated_valid": int(pts.shape[0])}
            )
        if i == 0 or (i + 1) % 250 == 0 or i + 1 == len(frames):
            point_count = sum(points.shape[0] for points in global_points)
            print(f"Triangulated pairs through frame {i + 1}/{len(frames)}; raw points={point_count}", flush=True)

    if global_points:
        global_points_arr = np.concatenate(global_points, axis=0)
        global_points_arr = voxel_downsample(global_points_arr, args.voxel_size)
    else:
        global_points_arr = np.empty((0, 3), dtype=np.float64)

    per_frame_counts = {}
    for i, t_camera_world in enumerate(t_camera_worlds):
        pts = visible_points(
            k,
            t_camera_world,
            global_points_arr,
            width,
            height,
            args.min_depth,
            args.max_depth,
            args.max_points_per_frame,
        )
        write_points(point_dir / f"point_cloud_{i}.txt", pts)
        per_frame_counts[str(i)] = int(pts.shape[0])
        if i == 0 or (i + 1) % 250 == 0 or i + 1 == len(t_camera_worlds):
            print(f"Wrote per-frame point clouds: {i + 1}/{len(t_camera_worlds)}", flush=True)

    config_path = write_config(
        args.config_dir.resolve(),
        output_dir,
        args.name,
        args.config_prefix,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
    )

    stats = {
        "source_scene_dir": str(scene_dir),
        "input_dir": str(input_dir),
        "sequence": input_dir.name,
        "output_dir": str(output_dir),
        "frame_count": len(frames),
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "source_file_paths": [frame["file_path"] for frame in frames],
        "missing_frame_count": len(missing),
        "missing_frames": missing,
        "source_width": src_width,
        "source_height": src_height,
        "used_width": width,
        "used_height": height,
        "pose_convention": args.pose_convention,
        "pose_note": "HorizonGS transform_matrix is converted to this repo's world-to-camera T_camera_world.",
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "pair_gaps": pair_gaps,
        "global_point_count": int(global_points_arr.shape[0]),
        "per_frame_point_count_min": int(min(per_frame_counts.values())) if per_frame_counts else 0,
        "per_frame_point_count_max": int(max(per_frame_counts.values())) if per_frame_counts else 0,
        "per_frame_point_count_mean": float(np.mean(list(per_frame_counts.values()))) if per_frame_counts else 0.0,
        "pair_stats": pair_stats,
    }
    with open(output_dir / "conversion_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Converted dataset: {output_dir}")
    print(f"Frames: {len(frames)} | missing skipped: {len(missing)}")
    print(f"Images: symlinked under {rectified_dir}")
    print(f"Image size used by config: {width}x{height} from source {src_width}x{src_height}")
    print(f"Global sparse points: {global_points_arr.shape[0]}")
    print(
        "Per-frame points: min={} mean={:.1f} max={}".format(
            stats["per_frame_point_count_min"],
            stats["per_frame_point_count_mean"],
            stats["per_frame_point_count_max"],
        )
    )
    print(f"Config: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
