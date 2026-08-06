#!/usr/bin/env python3
"""Convert a 360DVO frontview_pinhole scene into MODP Online-3DGS layout."""

# python \
#   tools/convert_360dvo_frontview_to_online3dgs.py \
#   --all-scenes \
#   --input-root data/360DVO \
#   --output-root data/Online3DGS_360DVO \
#   --config-dir configs/360dvo \
#   --pose-convention frontview-w2c-center


from __future__ import annotations

import argparse
import json
import math
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

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPO_ROOT / "data" / "360DVO" / "drone_racetrack" / "frontview_pinhole"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "Online3DGS_360DVO" / "drone_racetrack"
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "360DVO"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "Online3DGS_360DVO"
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "360dvo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help=(
            "Batch convert every <scene>/frontview_pinhole under --input-root "
            "to --output-root/<scene>."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="360DVO root used by --all-scenes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root used by --all-scenes.",
    )
    parser.add_argument(
        "--scenes",
        default=None,
        help="Comma-separated scene names to include in --all-scenes mode.",
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--name", default="360DVO-drone-racetrack")
    parser.add_argument("--config-prefix", default="360DVO_drone_racetrack")
    parser.add_argument(
        "--pose-convention",
        choices=["frontview-w2c-center", "opencv-c2w", "opencv-w2c"],
        default="frontview-w2c-center",
        help=(
            "Interpret trajectory.txt rows x y z qx qy qz qw. The default uses "
            "360DVO's world-to-camera quaternion, camera center, and virtual "
            "front-view camera axes. Legacy interpretations remain reproducible."
        ),
    )
    parser.add_argument("--orb-features", type=int, default=4000)
    parser.add_argument("--pair-gaps", default="1,2,4,8")
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--max-reproj-error", type=float, default=8.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=1000.0)
    parser.add_argument("--voxel-size", type=float, default=0.25)
    parser.add_argument("--max-points-per-frame", type=int, default=5000)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError("Quaternion has zero norm")
    x, y, z, w = q / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_row_to_t_camera_world(row: np.ndarray, convention: str) -> np.ndarray:
    rotation = quat_xyzw_to_rot(row[3:7])
    if convention == "frontview-w2c-center":
        frontview_axis = np.diag([1.0, -1.0, -1.0])
        world_to_camera = frontview_axis @ rotation
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = world_to_camera
        pose[:3, 3] = -world_to_camera @ row[:3]
        return pose
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = row[:3]
    if convention == "opencv-c2w":
        return np.linalg.inv(pose)
    if convention == "opencv-w2c":
        return pose
    raise ValueError(convention)


def symlink_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def load_camera_params(input_dir: Path) -> dict:
    path = input_dir / "camera_params.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.load(open(path, "r", encoding="utf-8"))
    camera = data["virtual_camera"]
    required = ["fx", "fy", "cx", "cy", "image_width", "image_height"]
    missing = [key for key in required if key not in camera]
    if missing:
        raise KeyError(f"Missing virtual_camera keys in {path}: {missing}")
    return data


def load_images(input_dir: Path, max_frames: int | None) -> list[Path]:
    image_dir = input_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        images = sorted(image_dir.glob("*.png"))
    if not images:
        raise RuntimeError(f"No images found under {image_dir}")
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        images = images[:max_frames]
    return images


def load_poses(input_dir: Path, count: int, convention: str) -> list[np.ndarray]:
    path = input_dir / "trajectory.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = np.loadtxt(path, dtype=np.float64)
    rows = np.atleast_2d(rows)
    if rows.shape[1] != 7:
        raise ValueError(f"Expected trajectory rows as x y z qx qy qz qw: {path}")
    if rows.shape[0] < count:
        raise ValueError(
            f"Trajectory has {rows.shape[0]} rows but {count} images were selected"
        )
    return [pose_row_to_t_camera_world(row, convention) for row in rows[:count]]


def scene_display_name(scene_name: str) -> str:
    return f"360DVO-{scene_name.replace('_', '-')}"


def scene_config_prefix(scene_name: str) -> str:
    return f"360DVO_{scene_name}"


def parse_pair_gaps(pair_gaps: str) -> list[int]:
    gaps = [int(x) for x in pair_gaps.split(",") if x.strip()]
    if not gaps:
        raise ValueError("--pair-gaps must contain at least one integer")
    if any(gap <= 0 for gap in gaps):
        raise ValueError("--pair-gaps values must be positive")
    return gaps


def discover_scene_inputs(
    input_root: Path, scene_names: str | None
) -> list[tuple[str, Path]]:
    requested = None
    if scene_names:
        requested = {name.strip() for name in scene_names.split(",") if name.strip()}
        if not requested:
            raise ValueError("--scenes was provided but no scene names were parsed")

    discovered: list[tuple[str, Path]] = []
    for scene_dir in sorted(input_root.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name.startswith("_"):
            continue
        if requested is not None and scene_dir.name not in requested:
            continue
        frontview_dir = scene_dir / "frontview_pinhole"
        if not frontview_dir.is_dir():
            continue
        discovered.append((scene_dir.name, frontview_dir))

    if requested is not None:
        found = {scene for scene, _ in discovered}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(
                "Requested scenes do not have frontview_pinhole directories: "
                + ", ".join(missing)
            )
    if not discovered:
        raise RuntimeError(
            f"No frontview_pinhole scene directories found under {input_root}"
        )
    return discovered


def output_data_is_complete(output_dir: Path, frame_count: int) -> bool:
    rectified_dir = output_dir / "rectified"
    point_dir = output_dir / "orb_point_clouds"
    rectified = sorted(rectified_dir.glob("aria_*")) if rectified_dir.is_dir() else []
    points = sorted(point_dir.glob("point_cloud_*.txt")) if point_dir.is_dir() else []
    if len(rectified) != frame_count or len(points) != frame_count:
        return False
    if not all(path.is_symlink() for path in rectified):
        return False
    required = [
        output_dir / "trajectory_orb.json",
        output_dir / "trajectory.json",
        output_dir / "image_list.txt",
        output_dir / "conversion_stats.json",
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        with open(output_dir / "conversion_stats.json", "r", encoding="utf-8") as f:
            stats = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return int(stats.get("frame_count", -1)) == frame_count


def write_config(
    config_dir: Path,
    dataset_dir: Path,
    name: str,
    config_prefix: str,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    max_points_per_frame: int,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{config_prefix}_orb.yaml"
    dataset_path = str(dataset_dir)
    dataset_block = f"""  name: "{name}"
  type: "aria"
  data_source: "orb"
  dataset_path: "{dataset_path}"
  num_threads: 0
  begin_cutoff: 0
  end_cutoff: 0
  max_pts_num: {max_points_per_frame}
  vignette: False
  use_vignette_type: "post-render"
  Calibration:
    fx: {fx:.6f}
    fy: {fy:.6f}
    cx: {cx:.6f}
    cy: {cy:.6f}
    width: {width}
    height: {height}
    near: 0.01
    far: 1000
"""
    config_path.write_text(
        f"""inherit_from: "configs/aria/orb_tracking/aria_base.yaml"

Dataset:
{dataset_block}
Testset:
{dataset_block}
Results:
  save_dir: "./Logs_360dvo"
  save_gt: False
  save_exr: False
  save_mesh: False
  skip_eval: True

Mapper:
  use_multi_reso: False
  initialization_frames: 4
  optimization_iters: 30
  initialization_iters: 40
  post_refinement:
    max_steps: 1000
    opt_cam: False
  KFGraph:
    kf_interval: 1
    global_window_size: 4
  CameraOptimizer:
    pose_refine_init_steps: 0
    pose_opt_steps: 4

Model:
  extra_pts_num: 3200
  err_threshold: 0.05
  camera_scale_rescalar: 0.25
  scene_scale: 1.0
""",
        encoding="utf-8",
    )
    return config_path


def convert_scene(
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path,
    name: str,
    config_prefix: str,
) -> dict:
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output exists: {output_dir}. Use --force to overwrite.")
        shutil.rmtree(output_dir)

    camera_data = load_camera_params(input_dir)
    camera = camera_data["virtual_camera"]
    width = int(camera["image_width"])
    height = int(camera["image_height"])
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    src_images = load_images(input_dir, args.max_frames)
    t_camera_worlds = load_poses(input_dir, len(src_images), args.pose_convention)
    pair_gaps = parse_pair_gaps(args.pair_gaps)

    rectified_dir = output_dir / "rectified"
    point_dir = output_dir / "orb_point_clouds"
    rectified_dir.mkdir(parents=True, exist_ok=True)
    point_dir.mkdir(parents=True, exist_ok=True)

    image_names: list[str] = []
    gray_images: list[np.ndarray] = []
    for idx, src in enumerate(src_images):
        image_name = f"aria_{idx:04d}{src.suffix.lower()}"
        symlink_image(src, rectified_dir / image_name)
        image_names.append(image_name)
        gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Could not read image: {src}")
        gray_images.append(gray[:height, :width])
        if idx == 0 or (idx + 1) % 200 == 0 or idx + 1 == len(src_images):
            print(f"Prepared images/poses: {idx + 1}/{len(src_images)}", flush=True)

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
    for trajectory_name in ["trajectory_orb.json", "trajectory.json"]:
        with open(output_dir / trajectory_name, "w", encoding="utf-8") as f:
            json.dump(trajectory, f)

    with open(output_dir / "image_list.txt", "w", encoding="utf-8") as f:
        for image_name in image_names:
            f.write(f"rectified/{image_name}\n")

    orb = cv2.ORB_create(nfeatures=args.orb_features)
    keypoints: list[list[cv2.KeyPoint]] = []
    descriptors: list[np.ndarray | None] = []
    for idx, gray in enumerate(gray_images):
        kp, des = orb.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)
        if idx == 0 or (idx + 1) % 200 == 0 or idx + 1 == len(gray_images):
            print(f"Detected ORB features: {idx + 1}/{len(gray_images)}", flush=True)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    global_points = []
    pair_stats = []
    for i in range(len(src_images)):
        for gap in pair_gaps:
            j = i + gap
            if j >= len(src_images):
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
                {
                    "i": i,
                    "j": j,
                    "baseline": baseline,
                    "triangulated_valid": int(pts.shape[0]),
                }
            )
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(src_images):
            point_count = sum(points.shape[0] for points in global_points)
            print(
                f"Triangulated pairs through frame {i + 1}/{len(src_images)}; "
                f"raw points={point_count}",
                flush=True,
            )

    if global_points:
        global_points_arr = np.concatenate(global_points, axis=0)
        raw_point_count = int(global_points_arr.shape[0])
        global_points_arr = voxel_downsample(global_points_arr, args.voxel_size)
    else:
        raw_point_count = 0
        global_points_arr = np.empty((0, 3), dtype=np.float64)

    per_frame_counts = {}
    for idx, t_camera_world in enumerate(t_camera_worlds):
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
        write_points(point_dir / f"point_cloud_{idx}.txt", pts)
        per_frame_counts[str(idx)] = int(pts.shape[0])
        if idx == 0 or (idx + 1) % 200 == 0 or idx + 1 == len(t_camera_worlds):
            print(
                f"Wrote per-frame point clouds: {idx + 1}/{len(t_camera_worlds)}",
                flush=True,
            )

    config_path = write_config(
        args.config_dir.resolve(),
        output_dir,
        name,
        config_prefix,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        args.max_points_per_frame,
    )

    pose_rows = np.loadtxt(input_dir / "trajectory.txt", dtype=np.float64)
    stats = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "frame_count": len(src_images),
        "source_images": [str(path) for path in src_images],
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "virtual_camera": camera,
        "pose_convention": args.pose_convention,
        "pose_format": "x y z qx qy qz qw",
        "pose_note": (
            "T_camera_world is world-to-camera. The default path interprets xyz "
            "as camera center, quaternion as world-to-camera rotation, and applies "
            "the virtual front-view axis diag(1,-1,-1)."
        ),
        "camera_center_min": pose_rows[: len(src_images), :3].min(axis=0).tolist(),
        "camera_center_max": pose_rows[: len(src_images), :3].max(axis=0).tolist(),
        "pair_gaps": pair_gaps,
        "orb_features": args.orb_features,
        "raw_triangulated_point_count": raw_point_count,
        "global_point_count": int(global_points_arr.shape[0]),
        "per_frame_point_count_min": (
            int(min(per_frame_counts.values())) if per_frame_counts else 0
        ),
        "per_frame_point_count_max": (
            int(max(per_frame_counts.values())) if per_frame_counts else 0
        ),
        "per_frame_point_count_mean": (
            float(np.mean(list(per_frame_counts.values()))) if per_frame_counts else 0.0
        ),
        "per_frame_counts": per_frame_counts,
        "pair_stats": pair_stats,
        "config_path": str(config_path),
    }
    with open(output_dir / "conversion_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Converted dataset: {output_dir}")
    print(f"Frames: {len(src_images)}")
    print(f"Images: symlinked under {rectified_dir}")
    print(f"Global sparse points: {global_points_arr.shape[0]}")
    print(
        "Per-frame points: min={} mean={:.1f} max={}".format(
            stats["per_frame_point_count_min"],
            stats["per_frame_point_count_mean"],
            stats["per_frame_point_count_max"],
        )
    )
    print(f"Config: {config_path}")
    return stats


def write_existing_scene_config(
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path,
    scene_name: str,
    frame_count: int,
) -> Path:
    camera_data = load_camera_params(input_dir)
    camera = camera_data["virtual_camera"]
    return write_config(
        args.config_dir.resolve(),
        output_dir,
        scene_display_name(scene_name),
        scene_config_prefix(scene_name),
        float(camera["fx"]),
        float(camera["fy"]),
        float(camera["cx"]),
        float(camera["cy"]),
        int(camera["image_width"]),
        int(camera["image_height"]),
        args.max_points_per_frame,
    )


def run_batch(args: argparse.Namespace) -> int:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    scenes = discover_scene_inputs(input_root, args.scenes)
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "config_dir": str(args.config_dir.resolve()),
        "pose_convention": args.pose_convention,
        "max_frames": args.max_frames,
        "scenes": [],
    }
    print(f"Batch converting {len(scenes)} scenes from {input_root} to {output_root}")

    for scene_idx, (scene_name, input_dir) in enumerate(scenes, start=1):
        output_dir = output_root / scene_name
        src_images = load_images(input_dir, args.max_frames)
        frame_count = len(src_images)
        print(
            f"[{scene_idx}/{len(scenes)}] scene={scene_name} frames={frame_count} "
            f"output={output_dir}",
            flush=True,
        )

        if output_dir.exists() and not args.force:
            if output_data_is_complete(output_dir, frame_count):
                config_path = write_existing_scene_config(
                    args,
                    input_dir,
                    output_dir,
                    scene_name,
                    frame_count,
                )
                print(
                    f"Skip existing complete scene: {scene_name}; config={config_path}",
                    flush=True,
                )
                summary["scenes"].append(
                    {
                        "scene": scene_name,
                        "status": "skipped_existing",
                        "input_dir": str(input_dir),
                        "output_dir": str(output_dir),
                        "frame_count": frame_count,
                        "config_path": str(config_path),
                    }
                )
                continue
            raise SystemExit(
                f"Output exists but is incomplete: {output_dir}. "
                "Use --force to rebuild it."
            )

        stats = convert_scene(
            args,
            input_dir,
            output_dir,
            scene_display_name(scene_name),
            scene_config_prefix(scene_name),
        )
        summary["scenes"].append(
            {
                "scene": scene_name,
                "status": "converted",
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "frame_count": stats["frame_count"],
                "config_path": stats["config_path"],
                "global_point_count": stats["global_point_count"],
                "per_frame_point_count_min": stats["per_frame_point_count_min"],
                "per_frame_point_count_mean": stats["per_frame_point_count_mean"],
                "per_frame_point_count_max": stats["per_frame_point_count_max"],
            }
        )

    summary_path = output_root / "batch_conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Batch summary: {summary_path}")
    print(
        "Batch done: converted={} skipped={}".format(
            sum(1 for item in summary["scenes"] if item["status"] == "converted"),
            sum(
                1 for item in summary["scenes"] if item["status"] == "skipped_existing"
            ),
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.all_scenes:
        return run_batch(args)

    convert_scene(
        args,
        args.input_dir.resolve(),
        args.output_dir.resolve(),
        args.name,
        args.config_prefix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
