#!/usr/bin/env python3
"""Convert a Pano360 openMVG scene to Online-3DGS front-view input."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "pano360"
DEFAULT_OUTPUT_ROOT = Path(
    "/data_0/wmy/workspace_vla/uavdata/pano360/Online3DGS_pano360"
)
DEFAULT_LINK_ROOT = REPO_ROOT / "data" / "Online3DGS_pano360"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene",
        help="Scene name under --input-root, or an absolute scene directory.",
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--link-root", type=Path, default=DEFAULT_LINK_ROOT)
    parser.add_argument(
        "--no-link",
        action="store_true",
        help="Do not create <link-root>/<scene> after successful conversion.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--hfov-deg", type=float, default=90.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument("--max-points-per-frame", type=int, default=5000)
    parser.add_argument("--min-depth", type=float, default=0.01)
    parser.add_argument("--max-depth", type=float, default=1000.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only process the first N registered frames (useful for smoke tests).",
    )
    parser.add_argument(
        "--skip-point-clouds",
        action="store_true",
        help="Write empty per-frame point clouds instead of converting openMVG tracks.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_scene(scene_arg: str, input_root: Path) -> tuple[str, Path]:
    requested = Path(scene_arg).expanduser()
    scene_dir = requested if requested.is_absolute() else input_root / requested
    scene_dir = scene_dir.resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")
    return scene_dir.name, scene_dir


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw, pitch, roll = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    ry = np.array(
        [[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0],
         [-math.sin(yaw), 0.0, math.cos(yaw)]],
        dtype=np.float64,
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(pitch), -math.sin(pitch)],
         [0.0, math.sin(pitch), math.cos(pitch)]],
        dtype=np.float64,
    )
    rz = np.array(
        [[math.cos(roll), -math.sin(roll), 0.0],
         [math.sin(roll), math.cos(roll), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return ry @ rx @ rz


def build_remap(
    src_width: int,
    src_height: int,
    width: int,
    height: int,
    hfov_deg: float,
    view_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    focal = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    cx, cy = width / 2.0, height / 2.0
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    rays = np.stack(((u - cx) / focal, (v - cy) / focal, np.ones_like(u)), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    # view_rotation maps virtual pinhole rays into the source spherical camera.
    rays = rays @ view_rotation.astype(np.float32).T
    longitude = np.arctan2(rays[..., 0], rays[..., 2])
    latitude = np.arcsin(np.clip(-rays[..., 1], -1.0, 1.0))
    map_x = np.mod((longitude / (2.0 * math.pi) + 0.5) * src_width, src_width)
    map_y = np.clip((0.5 - latitude / math.pi) * src_height, 0, src_height - 1)
    intrinsics = {"fx": focal, "fy": focal, "cx": cx, "cy": cy}
    return map_x.astype(np.float32), map_y.astype(np.float32), intrinsics


def load_openmvg(scene_dir: Path) -> tuple[dict[str, Any], Path]:
    sfm_path = scene_dir / "reconstruction" / "sfm_data.json"
    if not sfm_path.is_file():
        raise FileNotFoundError(f"Missing openMVG reconstruction: {sfm_path}")
    with sfm_path.open("r", encoding="utf-8") as handle:
        sfm = json.load(handle)
    images_dir = scene_dir / sfm.get("root_path", "images")
    if not images_dir.is_dir():
        images_dir = scene_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing source image directory: {images_dir}")
    return sfm, images_dir


def view_data(entry: dict[str, Any]) -> dict[str, Any]:
    return entry["value"]["ptr_wrapper"]["data"]


def registered_frames(
    sfm: dict[str, Any], images_dir: Path, max_frames: int | None
) -> tuple[list[dict[str, Any]], set[int]]:
    extrinsics = {int(item["key"]): item["value"] for item in sfm["extrinsics"]}
    frames = []
    missing_pose_ids = set()
    views = sorted(sfm["views"], key=lambda item: int(view_data(item)["id_view"]))
    for item in views:
        view = view_data(item)
        pose_id = int(view["id_pose"])
        if pose_id not in extrinsics:
            missing_pose_ids.add(pose_id)
            continue
        image_path = images_dir / view["local_path"] / view["filename"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        frames.append(
            {
                "view_id": int(view["id_view"]),
                "pose_id": pose_id,
                "source": image_path,
                "extrinsic": extrinsics[pose_id],
            }
        )
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        frames = frames[:max_frames]
    if not frames:
        raise RuntimeError("No registered images with valid openMVG poses")
    return frames, missing_pose_ids


def camera_pose(extrinsic: dict[str, Any], view_rotation: np.ndarray) -> np.ndarray:
    source_world_to_camera = np.asarray(extrinsic["rotation"], dtype=np.float64)
    center = np.asarray(extrinsic["center"], dtype=np.float64)
    world_to_pinhole = view_rotation.T @ source_world_to_camera
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = world_to_pinhole
    pose[:3, 3] = -world_to_pinhole @ center
    return pose


def write_point_clouds(
    sfm: dict[str, Any],
    frames: list[dict[str, Any]],
    poses: list[np.ndarray],
    intrinsics: dict[str, float],
    width: int,
    height: int,
    min_depth: float,
    max_depth: float,
    max_points: int,
    output_dir: Path,
    skip: bool,
) -> list[int]:
    points_dir = output_dir / "orb_point_clouds"
    ids_dir = output_dir / "orb_point_ids"
    points_dir.mkdir(parents=True)
    ids_dir.mkdir(parents=True)
    observations: dict[int, list[tuple[int, np.ndarray]]] = {
        frame["view_id"]: [] for frame in frames
    }
    if not skip:
        for item in sfm.get("structure", []):
            point_id = int(item["key"])
            point = np.asarray(item["value"]["X"], dtype=np.float64)
            for observation in item["value"].get("observations", []):
                view_id = int(observation["key"])
                if view_id in observations:
                    observations[view_id].append((point_id, point))

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    counts = []
    for index, (frame, pose) in enumerate(zip(frames, poses)):
        candidates = observations[frame["view_id"]]
        if candidates:
            ids = np.asarray([item[0] for item in candidates], dtype=np.int64)
            xyz = np.asarray([item[1] for item in candidates], dtype=np.float64)
            camera_xyz = xyz @ pose[:3, :3].T + pose[:3, 3]
            depth = camera_xyz[:, 2]
            u = fx * camera_xyz[:, 0] / depth + cx
            v = fy * camera_xyz[:, 1] / depth + cy
            keep = (
                (depth > min_depth) & (depth < max_depth)
                & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            )
            xyz, ids = xyz[keep], ids[keep]
            if max_points > 0 and len(xyz) > max_points:
                # Even selection is deterministic and avoids favoring one file segment.
                order = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
                xyz, ids = xyz[order], ids[order]
        else:
            xyz = np.empty((0, 3), dtype=np.float64)
            ids = np.empty((0,), dtype=np.int64)
        np.save(points_dir / f"point_cloud_{index}.npy", xyz.astype(np.float32))
        np.save(ids_dir / f"point_ids_{index}.npy", ids)
        counts.append(len(xyz))
    return counts


def write_config(
    write_dir: Path,
    dataset_dir: Path,
    scene_name: str,
    intrinsics: dict[str, float],
    width: int,
    height: int,
    max_points: int,
) -> None:
    block = f'''  name: "Pano360-{scene_name}-frontview"
  type: "aria"
  data_source: "orb"
  dataset_path: "{dataset_dir}"
  num_threads: 0
  begin_cutoff: 0
  end_cutoff: 0
  max_pts_num: {max_points}
  vignette: False
  use_vignette_type: "post-render"
  Calibration:
    fx: {intrinsics['fx']:.8f}
    fy: {intrinsics['fy']:.8f}
    cx: {intrinsics['cx']:.8f}
    cy: {intrinsics['cy']:.8f}
    width: {width}
    height: {height}
    near: 0.01
    far: 1000
'''
    text = f'''inherit_from: "configs/aria/orb_tracking/aria_base.yaml"

Dataset:
{block}
Testset:
{block}
Results:
  save_dir: "./Logs_pano360"
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
'''
    (write_dir / "config.yaml").write_text(text, encoding="utf-8")


def create_scene_link(
    output_dir: Path, link_root: Path, scene_name: str, force: bool
) -> Path:
    link_root.mkdir(parents=True, exist_ok=True)
    link_path = link_root / scene_name
    if link_path.is_symlink():
        current_target = link_path.resolve(strict=False)
        if current_target == output_dir:
            return link_path
        if not force:
            raise FileExistsError(
                f"Scene link already points elsewhere: {link_path} -> {current_target}; "
                "pass --force to replace the link"
            )
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(
            f"Cannot create scene link because a non-symlink exists: {link_path}"
        )
    link_path.symlink_to(output_dir, target_is_directory=True)
    return link_path


def create_downsampled_video(
    image_dir: Path, output_path: Path, fps: float, crf: int
) -> tuple[int, int]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create the H.264 preview video")
    first = cv2.imread(str(image_dir / "aria_00000.jpg"), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"Cannot read first video frame under {image_dir}")
    source_height, source_width = first.shape[:2]
    # H.264 yuv420p requires even dimensions. Floor to the nearest even value.
    video_width = max(2, 2 * (source_width // 3 // 2))
    video_height = max(2, 2 * (source_height // 3 // 2))
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(image_dir / "aria_%05d.jpg"),
        "-vf",
        f"scale={video_width}:{video_height}:flags=lanczos",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create a valid video: {output_path}")
    return video_width, video_height


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if not 0.0 < args.hfov_deg < 180.0:
        raise ValueError("--hfov-deg must be in (0, 180)")
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [0, 100]")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not 0 <= args.video_crf <= 51:
        raise ValueError("--video-crf must be in [0, 51]")

    scene_name, scene_dir = resolve_scene(args.scene, args.input_root.resolve())
    output_dir = (args.output_root / scene_name).resolve()
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"Output exists: {output_dir}; pass --force to replace it")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    sfm, images_dir = load_openmvg(scene_dir)
    frames, missing_pose_ids = registered_frames(sfm, images_dir, args.max_frames)
    first = cv2.imread(str(frames[0]["source"]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"OpenCV cannot read {frames[0]['source']}")
    src_height, src_width = first.shape[:2]
    view_rotation = rotation_matrix(args.yaw_deg, args.pitch_deg, args.roll_deg)
    map_x, map_y, intrinsics = build_remap(
        src_width, src_height, args.width, args.height, args.hfov_deg, view_rotation
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{scene_name}.tmp-", dir=output_dir.parent))
    try:
        rectified_dir = staging / "rectified"
        rectified_dir.mkdir()
        poses = []
        cameras = []
        image_list = []
        for index, frame in enumerate(frames):
            source = first if index == 0 else cv2.imread(str(frame["source"]), cv2.IMREAD_COLOR)
            if source is None or source.shape[:2] != (src_height, src_width):
                raise RuntimeError(f"Invalid or inconsistent source image: {frame['source']}")
            image = cv2.remap(
                source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
            )
            image_name = f"aria_{index:05d}.jpg"
            if not cv2.imwrite(
                str(rectified_dir / image_name), image,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            ):
                raise RuntimeError(f"Failed to write {rectified_dir / image_name}")
            pose = camera_pose(frame["extrinsic"], view_rotation)
            poses.append(pose)
            cameras.append(
                {
                    "T_camera_world": pose.tolist(),
                    "image": image_name,
                    "timestamp": index / args.fps,
                    "frame_index": index,
                    "source_view_id": frame["view_id"],
                    "source_image": frame["source"].name,
                    "intrinsic": intrinsics,
                    "width": args.width,
                    "height": args.height,
                    "focal": intrinsics["fx"],
                }
            )
            image_list.append(f"rectified/{image_name}")
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(frames):
                print(f"images: {index + 1}/{len(frames)}", flush=True)

        trajectory = json.dumps({"cameras": cameras}, indent=2)
        (staging / "trajectory_orb.json").write_text(trajectory + "\n", encoding="utf-8")
        (staging / "trajectory.json").write_text(trajectory + "\n", encoding="utf-8")
        (staging / "image_list.txt").write_text("\n".join(image_list) + "\n", encoding="utf-8")
        point_counts = write_point_clouds(
            sfm, frames, poses, intrinsics, args.width, args.height,
            args.min_depth, args.max_depth, args.max_points_per_frame,
            staging, args.skip_point_clouds,
        )
        video_name = "frontview_downsample3x_h264.mp4"
        video_width, video_height = create_downsampled_video(
            rectified_dir, staging / video_name, args.fps, args.video_crf
        )
        final_config_path = output_dir / "config.yaml"
        write_config(
            staging,
            output_dir,
            scene_name,
            intrinsics,
            args.width,
            args.height,
            args.max_points_per_frame,
        )
        stats = {
            "dataset": "Pano360",
            "scene": scene_name,
            "source": str(scene_dir),
            "source_projection": "equirectangular",
            "source_resolution": [src_width, src_height],
            "output_projection": "pinhole",
            "output_resolution": [args.width, args.height],
            "horizontal_fov_deg": args.hfov_deg,
            "yaw_pitch_roll_deg": [args.yaw_deg, args.pitch_deg, args.roll_deg],
            "intrinsic": intrinsics,
            "frame_count": len(frames),
            "unregistered_view_count": len(missing_pose_ids),
            "point_clouds_skipped": args.skip_point_clouds,
            "video": {
                "path": str(output_dir / video_name),
                "codec": "h264",
                "resolution": [video_width, video_height],
                "downsample_factor": 3,
                "fps": args.fps,
                "crf": args.video_crf,
                "temporary_frames_written": False,
            },
            "points_per_frame_min_mean_max": [
                int(min(point_counts)), float(np.mean(point_counts)), int(max(point_counts))
            ],
            "config": str(final_config_path),
        }
        (staging / "conversion_stats.json").write_text(
            json.dumps(stats, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "README.md").write_text(
            f"# Pano360 {scene_name} front view\n\n"
            f"Generated at {args.width}x{args.height}, HFOV {args.hfov_deg:g} deg. "
            "Images are distortion-free virtual pinhole views. Use `config.yaml` "
            "with Online-3DGS-Monocular. The 3x-downsampled H.264 preview is "
            f"`{video_name}`; no extra downsampled frames are stored.\n",
            encoding="utf-8",
        )
        del sfm
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    link_path = None
    if not args.no_link:
        link_path = create_scene_link(
            output_dir, args.link_root.resolve(), scene_name, args.force
        )

    print(f"done: {output_dir}")
    if link_path is not None:
        print(f"link: {link_path} -> {output_dir}")
    print(f"frames={len(frames)}, elapsed={time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
