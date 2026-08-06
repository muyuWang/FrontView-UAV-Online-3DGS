#!/usr/bin/env python3
"""Extract a virtual pinhole front view from a 360DVO scene directory."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - user environment guard
    raise SystemExit(
        "This script requires OpenCV. Run it with a Python environment that has "
        "cv2 installed, e.g. /home/wmy/anaconda3/envs/uavon/bin/python."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a 360DVO equirectangular scene to a virtual pinhole "
            "front-view image sequence."
        )
    )
    parser.add_argument(
        "scene_dir",
        type=Path,
        help="360DVO scene directory, e.g. .../360DVO/drone_racetrack.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <scene_dir>/frontview_pinhole.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--hfov-deg", type=float, default=90.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--preview-width", type=int, default=960)
    parser.add_argument("--preview-crf", type=int, default=30)
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Do not generate the compressed front-view preview mp4.",
    )
    return parser.parse_args()


def find_sequence_dir(scene_dir: Path) -> tuple[str, Path]:
    sequences_dir = scene_dir / "Sequences"
    if not sequences_dir.exists():
        raise FileNotFoundError(f"Missing Sequences directory: {sequences_dir}")

    preferred = sequences_dir / scene_dir.name
    if preferred.exists():
        return scene_dir.name, preferred

    candidates = sorted(
        path
        for path in sequences_dir.iterdir()
        if path.is_dir() and any(path.glob("*.jpg"))
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not infer scene image directory under {sequences_dir}. "
            f"Found {len(candidates)} candidates."
        )
    return candidates[0].name, candidates[0]


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    ry = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float32,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float32,
    )
    rz = np.array(
        [
            [math.cos(roll), -math.sin(roll), 0.0],
            [math.sin(roll), math.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return ry @ rx @ rz


def build_remap(
    src_width: int,
    src_height: int,
    out_width: int,
    out_height: int,
    hfov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    hfov = math.radians(hfov_deg)
    fx = out_width / (2.0 * math.tan(hfov / 2.0))
    fy = fx
    cx = out_width / 2.0
    cy = out_height / 2.0
    vfov = 2.0 * math.atan(out_height / (2.0 * fy))

    u, v = np.meshgrid(
        np.arange(out_width, dtype=np.float32),
        np.arange(out_height, dtype=np.float32),
    )
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x, dtype=np.float32)
    rays = np.stack([x, y, z], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    rot = rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    rays = rays @ rot.T
    lon = np.arctan2(rays[..., 0], rays[..., 2])
    lat = np.arcsin(np.clip(-rays[..., 1], -1.0, 1.0))

    map_x = ((lon / (2.0 * math.pi)) + 0.5) * src_width
    map_y = (0.5 - (lat / math.pi)) * src_height
    map_x = np.mod(map_x, src_width).astype(np.float32)
    map_y = np.clip(map_y, 0, src_height - 1).astype(np.float32)

    params = {
        "camera_model": "PINHOLE",
        "distortion_model": "none",
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
        "front_definition": (
            "source equirectangular center direction at yaw=0, pitch=0, roll=0"
        ),
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "image_width": out_width,
        "image_height": out_height,
        "horizontal_fov_deg": hfov_deg,
        "vertical_fov_deg": math.degrees(vfov),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
    }
    return map_x, map_y, params


def write_metadata(
    output_dir: Path,
    scene_dir: Path,
    scene_name: str,
    src_width: int,
    src_height: int,
    camera_params: dict,
    num_frames: int,
) -> None:
    gt_path = scene_dir / "GroundTruth" / f"{scene_name}.txt"
    if gt_path.exists():
        shutil.copy2(gt_path, output_dir / "trajectory.txt")

    payload = {
        "dataset": "360DVO",
        "scene": scene_name,
        "source_projection": "equirectangular",
        "source_image_width": src_width,
        "source_image_height": src_height,
        "virtual_camera": camera_params,
        "trajectory": {
            "path": "trajectory.txt",
            "source_path": str(gt_path),
            "format": "x y z qx qy qz qw",
            "note": (
                "Copied from 360DVO ground truth. Valid for this virtual camera "
                "under the fixed yaw/pitch/roll defined in virtual_camera."
            ),
        },
        "num_frames": num_frames,
    }
    with open(output_dir / "camera_params.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    readme = f"""# {scene_name} frontview_pinhole

Derived from a 360DVO equirectangular sequence.

- output images: images/*.jpg
- camera model: PINHOLE
- output resolution: {camera_params['image_width']}x{camera_params['image_height']}
- horizontal FOV: {camera_params['horizontal_fov_deg']:.6f} deg
- vertical FOV: {camera_params['vertical_fov_deg']:.6f} deg
- fx: {camera_params['fx']:.6f}
- fy: {camera_params['fy']:.6f}
- cx: {camera_params['cx']:.6f}
- cy: {camera_params['cy']:.6f}
- yaw/pitch/roll: {camera_params['yaw_deg']:.6f}, {camera_params['pitch_deg']:.6f}, {camera_params['roll_deg']:.6f} deg
- distortion: none

The original 360DVO frames are equirectangular 360-degree images, not native
pinhole captures. These intrinsics are virtual intrinsics chosen during
conversion.

trajectory.txt is copied from GroundTruth/{scene_name}.txt.
Row format: x y z qx qy qz qw.
"""
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)


def generate_preview(
    image_dir: Path,
    scene_dir: Path,
    output_width: int,
    output_height: int,
    preview_width: int,
    fps: float,
    crf: int,
) -> Path:
    preview_height = int(round(preview_width * output_height / output_width))
    if preview_height % 2:
        preview_height += 1
    preview_path = scene_dir / f"frontview_pinhole_preview_{preview_width}x{preview_height}.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        "1",
        "-i",
        str(image_dir / "%04d.jpg"),
        "-vf",
        f"scale={preview_width}:{preview_height}:flags=lanczos",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(preview_path),
    ]
    subprocess.run(cmd, check=True)
    return preview_path


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive.")
    if args.hfov_deg <= 0.0 or args.hfov_deg >= 180.0:
        raise ValueError("hfov-deg must be in (0, 180).")

    scene_dir = args.scene_dir.resolve()
    output_dir = (args.output_dir or (scene_dir / "frontview_pinhole")).resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    scene_name, sequence_dir = find_sequence_dir(scene_dir)
    src_files = sorted(sequence_dir.glob("*.jpg"))
    if not src_files:
        raise RuntimeError(f"No source jpg files found in {sequence_dir}")

    first = cv2.imread(str(src_files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read {src_files[0]}")
    src_height, src_width = first.shape[:2]

    map_x, map_y, camera_params = build_remap(
        src_width=src_width,
        src_height=src_height,
        out_width=args.width,
        out_height=args.height,
        hfov_deg=args.hfov_deg,
        yaw_deg=args.yaw_deg,
        pitch_deg=args.pitch_deg,
        roll_deg=args.roll_deg,
    )

    t0 = time.time()
    for idx, path in enumerate(src_files, start=1):
        img = first if idx == 1 else cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read {path}")
        rectified = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        out_path = image_dir / path.name
        ok = cv2.imwrite(
            str(out_path),
            rectified,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
        )
        if not ok:
            raise RuntimeError(f"Could not write {out_path}")
        if idx == 1 or idx % 50 == 0 or idx == len(src_files):
            print(f"converted {idx}/{len(src_files)}")

    with open(output_dir / "image_list.txt", "w", encoding="utf-8") as f:
        for path in src_files:
            f.write(f"images/{path.name}\n")
    write_metadata(
        output_dir=output_dir,
        scene_dir=scene_dir,
        scene_name=scene_name,
        src_width=src_width,
        src_height=src_height,
        camera_params=camera_params,
        num_frames=len(src_files),
    )

    preview_path = None
    if not args.skip_preview:
        preview_path = generate_preview(
            image_dir=image_dir,
            scene_dir=scene_dir,
            output_width=args.width,
            output_height=args.height,
            preview_width=args.preview_width,
            fps=args.fps,
            crf=args.preview_crf,
        )

    print(f"done in {time.time() - t0:.1f}s")
    print(f"output_dir={output_dir}")
    if preview_path is not None:
        print(f"preview={preview_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
