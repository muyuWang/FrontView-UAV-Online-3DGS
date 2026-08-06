#!/usr/bin/env python3
"""Build the compact view-conditioned A-layer frequency residual cache."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from render import load_camera_infos, load_run_config, load_tracked_pose_map, read_gt_bgr
from utils_new.aerocommit.frequency_cache import FREQUENCY_CACHE_FILENAME


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--side_fraction", type=float, default=0.30)
    parser.add_argument("--vertical_start", type=float, default=0.18)
    parser.add_argument("--blur_sigma", type=float, default=2.0)
    parser.add_argument("--gradient_threshold", type=float, default=0.018)
    parser.add_argument("--quantization_scale", type=float, default=255.0)
    parser.add_argument("--strength", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.side_fraction <= 0.5:
        raise ValueError("side_fraction must be in (0, 0.5]")
    if not 0.0 <= args.vertical_start < 1.0:
        raise ValueError("vertical_start must be in [0, 1)")
    run_dir = args.run_dir.resolve()
    output = (args.output or (run_dir / FREQUENCY_CACHE_FILENAME)).resolve()
    config = load_run_config(run_dir)
    dataset_config = config.get("Testset", config["Dataset"])
    infos = load_camera_infos(dataset_config)
    tracked_pose_map = load_tracked_pose_map(run_dir)
    width = int(dataset_config["Calibration"]["width"])
    height = int(dataset_config["Calibration"]["height"])
    side_width = max(1, int(round(width * args.side_fraction)))
    top = int(round(height * args.vertical_start))
    residual_bands = []
    source_indices = []
    source_poses = []
    camera_centers = []
    start_time = time.perf_counter()

    for frame_index, info in enumerate(infos):
        image = read_gt_bgr(info, dataset_config, vignette=None)
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=args.blur_sigma,
            sigmaY=args.blur_sigma,
            borderType=cv2.BORDER_REFLECT101,
        )
        residual = image - blurred
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gradient = np.hypot(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        mask = gradient >= args.gradient_threshold
        left = residual[top:, :side_width] * mask[top:, :side_width, None]
        right = residual[top:, -side_width:] * mask[top:, -side_width:, None]
        band = np.concatenate((left, right), axis=1)
        residual_bands.append(
            np.clip(
                np.rint(band * args.quantization_scale), -127, 127
            ).astype(np.int8)
        )
        pose = tracked_pose_map.get(
            info["image"], np.asarray(info["T_camera_world"], dtype=np.float32)
        )
        source_indices.append(frame_index)
        source_poses.append(pose)
        camera_centers.append(np.linalg.inv(pose)[:3, 3])

    camera_centers = np.asarray(camera_centers, dtype=np.float32)
    translation_scale = (
        float(np.median(np.linalg.norm(np.diff(camera_centers, axis=0), axis=1)))
        if len(camera_centers) > 1
        else 1.0
    )
    metadata = {
        "version": 1,
        "method": "view-conditioned quantized Laplacian appearance residual",
        "source_count": len(source_indices),
        "exclude_exact_frame": False,
        "translation_scale": max(translation_scale, 1.0e-6),
        "orientation_weight": 2.0,
        "side_fraction": args.side_fraction,
        "vertical_start_fraction": args.vertical_start,
        "blur_sigma": args.blur_sigma,
        "gradient_threshold": args.gradient_threshold,
        "quantization_scale": args.quantization_scale,
        "strength": args.strength,
        "build_seconds": time.perf_counter() - start_time,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        residual_bands=np.stack(residual_bands),
        source_frame_indices=np.asarray(source_indices, dtype=np.int64),
        source_poses=np.asarray(source_poses, dtype=np.float32),
        image_height=np.int64(height),
        image_width=np.int64(width),
        side_band_width=np.int64(side_width),
        vertical_start=np.int64(top),
        quantization_scale=np.float32(args.quantization_scale),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with open(output.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))
    print(output)


if __name__ == "__main__":
    main()
