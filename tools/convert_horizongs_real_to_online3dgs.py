#!/usr/bin/env python3
"""Convert one HorizonGS real camera sequence into Online-3DGS-Monocular layout."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_airvln_to_online3dgs import (  # noqa: E402
    triangulate_pair,
    project_points,
    voxel_downsample,
    write_points,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_DIR = REPO_ROOT / "data" / "HorizonGS" / "real" / "road"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "Online3DGS_HorizonGS" / "road_street1"
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "horizongs"
CAMERA_AXIS_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--camera", default="street_cam1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--name", default="HorizonGS-road-street1")
    parser.add_argument("--config-prefix", default="HorizonGS_road_street1")
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
    parser.add_argument("--voxel-size", type=float, default=0.25)
    parser.add_argument("--max-points-per-frame", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def to_t_camera_world(transform_matrix: list[list[float]], convention: str) -> np.ndarray:
    mat = np.array(transform_matrix, dtype=np.float64)
    if convention == "nerf-opengl-c2w":
        return np.linalg.inv(mat @ CAMERA_AXIS_FLIP)
    if convention == "opencv-c2w":
        return np.linalg.inv(mat)
    if convention == "opencv-w2c":
        return mat
    raise ValueError(convention)


def load_frames(scene_dir: Path, camera: str) -> tuple[list[dict], list[str]]:
    transforms_path = scene_dir / "transforms.json"
    data = json.load(open(transforms_path, "r", encoding="utf-8"))
    image_root = scene_dir / "images"
    frames = []
    missing = []
    for frame in data["frames"]:
        file_path = frame["file_path"]
        if not file_path.startswith(f"{camera}/"):
            continue
        if (image_root / file_path).exists():
            frames.append(frame)
        else:
            missing.append(file_path)
    return frames, missing


def symlink_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def visible_point_ids(
    k: np.ndarray,
    t_camera_world: np.ndarray,
    points_world: np.ndarray,
    width: int,
    height: int,
    min_depth: float,
    max_depth: float,
    max_points: int,
    grid_size: int = 16,
) -> np.ndarray:
    if points_world.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    uv, depth = project_points(k, t_camera_world, points_world)
    valid = (
        np.isfinite(uv).all(axis=1)
        & (depth > min_depth)
        & (depth < max_depth)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    ids = np.flatnonzero(valid)
    if max_points <= 0 or len(ids) <= max_points:
        return ids.astype(np.int64)

    order = np.argsort(depth[ids])
    columns = (width + grid_size - 1) // grid_size
    cells = (
        (uv[ids, 1].astype(np.int64) // grid_size) * columns
        + uv[ids, 0].astype(np.int64) // grid_size
    )
    _, first = np.unique(cells[order], return_index=True)
    selected_positions = order[np.sort(first)]
    if len(selected_positions) < max_points:
        used = np.zeros(len(ids), dtype=bool)
        used[selected_positions] = True
        selected_positions = np.concatenate(
            (selected_positions, order[~used[order]])
        )
    return np.sort(ids[selected_positions[:max_points]].astype(np.int64))


def color_global_points(
    k: np.ndarray,
    poses: list[np.ndarray],
    image_paths: list[Path],
    points: np.ndarray,
    width: int,
    height: int,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    colors = np.full((len(points), 3), 128, dtype=np.uint8)
    assigned = np.zeros(len(points), dtype=bool)
    for pose, image_path in zip(poses, image_paths):
        ids = visible_point_ids(
            k, pose, points, width, height, min_depth, max_depth, -1
        )
        ids = ids[~assigned[ids]]
        if not len(ids):
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        image = image[:height, :width]
        uv, _ = project_points(k, pose, points[ids])
        pixels = np.rint(uv).astype(np.int64)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
        colors[ids] = image[pixels[:, 1], pixels[:, 0], ::-1]
        assigned[ids] = True
        if assigned.all():
            break
    return colors


def write_total_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("point_id", "u4"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    vertices["point_id"] = np.arange(len(points), dtype=np.uint32)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


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
    max_points_per_frame: int = 5000,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{config_prefix}_orb.yaml"
    dataset_path = str(dataset_dir)
    config_path.write_text(
        f"""inherit_from: "configs/aria/orb_tracking/aria_base.yaml"

Dataset:
  name: "{name}"
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

Testset:
  name: "{name}"
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

Results:
  save_dir: "./Logs_horizongs"
  save_gt: False
  save_exr: False
  save_mesh: False
  skip_eval: True

Mapper:
  use_multi_reso: False
  initialization_frames: 4
  optimization_iters: 10
  initialization_iters: 10
  post_refinement:
    max_steps: 100
    opt_cam: False
  KFGraph:
    kf_interval: 3
    global_window_size: 2
  CameraOptimizer:
    pose_refine_init_steps: 0
    pose_opt_steps: 2

Model:
  camera_scale_rescalar: 0.25
  scene_scale: 1.0
""",
        encoding="utf-8",
    )
    return config_path


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output exists: {output_dir}. Use --force to overwrite.")
        shutil.rmtree(output_dir)

    frames, missing = load_frames(scene_dir, args.camera)
    if not frames:
        raise RuntimeError(f"No existing RGB frames found for camera {args.camera} under {scene_dir}")

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
    point_id_dir = output_dir / "orb_point_ids"
    preprocess_dir = output_dir / "preprocess"
    rectified_dir.mkdir(parents=True, exist_ok=True)
    point_dir.mkdir(parents=True, exist_ok=True)
    point_id_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    image_root = scene_dir / "images"
    image_names: list[str] = []
    image_paths: list[Path] = []
    t_camera_worlds: list[np.ndarray] = []
    gray_images: list[np.ndarray] = []

    for i, frame in enumerate(frames):
        src = image_root / frame["file_path"]
        image_name = f"aria_{i:04d}{src.suffix}"
        symlink_image(src, rectified_dir / image_name)
        image_names.append(image_name)
        image_paths.append(src)
        t_camera_worlds.append(
            to_t_camera_world(frame["transform_matrix"], args.pose_convention)
        )
        gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Could not read image: {src}")
        gray_images.append(gray[:height, :width])

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
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == len(gray_images):
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
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == len(frames):
            point_count = sum(points.shape[0] for points in global_points)
            print(f"Triangulated pairs through frame {i + 1}/{len(frames)}; raw points={point_count}", flush=True)

    if global_points:
        global_points_arr = np.concatenate(global_points, axis=0)
        global_points_arr = voxel_downsample(global_points_arr, args.voxel_size)
    else:
        global_points_arr = np.empty((0, 3), dtype=np.float64)

    global_points_arr = global_points_arr.astype(np.float32)
    np.save(preprocess_dir / "global_sparse_points.npy", global_points_arr)
    np.save(preprocess_dir / "global_horizongs_orb_points.npy", global_points_arr)
    point_colors = color_global_points(
        k,
        t_camera_worlds,
        image_paths,
        global_points_arr,
        width,
        height,
        args.min_depth,
        args.max_depth,
    )
    write_total_ply(
        output_dir / "initialization_horizongs_orb_persistent_total.ply",
        global_points_arr,
        point_colors,
    )

    per_frame_counts = {}
    for i, t_camera_world in enumerate(t_camera_worlds):
        point_ids = visible_point_ids(
            k,
            t_camera_world,
            global_points_arr,
            width,
            height,
            args.min_depth,
            args.max_depth,
            args.max_points_per_frame,
        )
        pts = global_points_arr[point_ids]
        write_points(point_dir / f"point_cloud_{i}.txt", pts)
        np.save(point_dir / f"point_cloud_{i}.npy", pts)
        np.save(point_id_dir / f"point_ids_{i}.npy", point_ids)
        per_frame_counts[str(i)] = int(pts.shape[0])

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
        args.max_points_per_frame,
    )

    stats = {
        "schema_version": 1,
        "method": (
            "fixed HorizonGS poses with ORB multi-view triangulation and persistent "
            "world point identities"
        ),
        "pose_source": "horizongs_fixed",
        "sparse_world_geometry": "persistent",
        "coordinate_contract": (
            "all pair triangulations use the unchanged HorizonGS world-to-camera poses; "
            "voxel fusion and per-frame visibility preserve one global point identity"
        ),
        "source_scene_dir": str(scene_dir),
        "camera": args.camera,
        "output_dir": str(output_dir),
        "frame_count": len(frames),
        "missing_frame_count": len(missing),
        "missing_frames": missing,
        "source_width": src_width,
        "source_height": src_height,
        "used_width": width,
        "used_height": height,
        "crop_note": "Images are symlinked; dataset loader crops to used_width/used_height at runtime.",
        "pose_convention": args.pose_convention,
        "pose_note": "HorizonGS transform_matrix is converted to this repo's world-to-camera T_camera_world.",
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "orb_features_per_frame": int(args.orb_features),
        "pair_gaps": pair_gaps,
        "lowe_ratio": float(args.ratio),
        "min_depth_m": float(args.min_depth),
        "max_depth_m": float(args.max_depth),
        "global_point_count": int(global_points_arr.shape[0]),
        "persistent_point_ids": True,
        "max_reprojection_error_px_sum_two_views": float(args.max_reproj_error),
        "voxel_size_m": float(args.voxel_size),
        "max_points_per_frame": int(args.max_points_per_frame),
        "visibility_grid_size_px": 16,
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
