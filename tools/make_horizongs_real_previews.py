#!/usr/bin/env python3
"""Create downsampled MP4 previews for HorizonGS real image folders."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_ROOT = REPO_ROOT / "data" / "HorizonGS" / "real"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def find_image_dirs(real_root: Path) -> list[Path]:
    image_dirs = []
    for scene_dir in sorted([p for p in real_root.iterdir() if p.is_dir()]):
        images_root = scene_dir / "images"
        if not images_root.is_dir():
            continue
        for child in sorted([p for p in images_root.iterdir() if p.is_dir()]):
            image_dirs.append(child)
    return image_dirs


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS],
        key=natural_key,
    )


def even_size(width: int, height: int) -> tuple[int, int]:
    width = max(2, width)
    height = max(2, height)
    if width % 2:
        width -= 1
    if height % 2:
        height -= 1
    return width, height


def make_preview(
    image_dir: Path,
    fps: float,
    downsample: int,
    overwrite: bool,
    output_path: Path | None = None,
) -> Path | None:
    images = list_images(image_dir)
    if not images:
        print(f"Skip empty image dir: {image_dir}")
        return None

    output_path = output_path or image_dir.parent / f"{image_dir.name}.mp4"
    if output_path.exists() and not overwrite:
        print(f"Skip existing: {output_path}")
        return output_path

    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read first image: {images[0]}")

    src_h, src_w = first.shape[:2]
    out_w, out_h = even_size(src_w // downsample, src_h // downsample)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(
        prefix=f".{image_dir.name}_", suffix=".mp4", dir=str(image_dir.parent), delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{out_w}x{out_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        assert proc.stdin is not None
        for idx, path in enumerate(images):
            frame = first if idx == 0 else cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Could not read image: {path}")
            if frame.shape[:2] != (src_h, src_w):
                frame = cv2.resize(frame, (src_w, src_h), interpolation=cv2.INTER_AREA)
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            proc.stdin.write(frame.tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()

    stdout = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    ret = proc.wait()
    if ret != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg failed for {output_path} with code {ret}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    tmp_path.replace(output_path)

    print(
        f"Wrote {output_path} | frames={len(images)} | "
        f"{src_w}x{src_h} -> {out_w}x{out_h} | codec=h264",
        flush=True,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    real_root = args.real_root.resolve()
    if args.downsample <= 0:
        raise ValueError("--downsample must be > 0")
    if not real_root.is_dir():
        raise FileNotFoundError(real_root)

    image_dirs = find_image_dirs(real_root)
    if not image_dirs:
        raise RuntimeError(f"No image subdirectories found under {real_root}")

    print(f"Found {len(image_dirs)} image folders under {real_root}")
    for image_dir in image_dirs:
        make_preview(
            image_dir=image_dir,
            fps=args.fps,
            downsample=args.downsample,
            overwrite=not args.no_overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
