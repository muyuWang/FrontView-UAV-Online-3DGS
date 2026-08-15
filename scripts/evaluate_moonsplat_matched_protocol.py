#!/usr/bin/env python3
"""Render a saved map on MoonSplat's matched keyframe evaluation protocol."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import render as render_tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def moon_resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS)
    )


def psnr(gt: np.ndarray, prediction: np.ndarray) -> float:
    mse = np.mean((gt.astype(np.float64) - prediction.astype(np.float64)) ** 2)
    return float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim(gt: np.ndarray, prediction: np.ndarray, device: str) -> float:
    x = torch.from_numpy(gt.copy()).permute(2, 0, 1).float()[None].to(device) / 255.0
    y = (
        torch.from_numpy(prediction.copy()).permute(2, 0, 1).float()[None].to(device)
        / 255.0
    )
    coords = torch.arange(11, dtype=torch.float32, device=device) - 5
    gaussian = torch.exp(-(coords.square()) / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    window = (gaussian[:, None] @ gaussian[None, :]).view(1, 1, 11, 11)
    window = window.expand(3, 1, 11, 11)
    mu_x = F.conv2d(x, window, padding=5, groups=3)
    mu_y = F.conv2d(y, window, padding=5, groups=3)
    sigma_x = F.conv2d(x * x, window, padding=5, groups=3) - mu_x.square()
    sigma_y = F.conv2d(y * y, window, padding=5, groups=3) - mu_y.square()
    sigma_xy = F.conv2d(x * y, window, padding=5, groups=3) - mu_x * mu_y
    value = ((2 * mu_x * mu_y + 0.01**2) * (2 * sigma_xy + 0.03**2)) / (
        (mu_x.square() + mu_y.square() + 0.01**2)
        * (sigma_x + sigma_y + 0.03**2)
    )
    return float(value.mean().item())


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def scaled_dataset_config(config: dict, width: int, height: int) -> dict:
    dataset = copy.deepcopy(config["Testset"] if "Testset" in config else config["Dataset"])
    calibration = dataset["Calibration"]
    source_width = float(calibration["width"])
    source_height = float(calibration["height"])
    scale_x = width / source_width
    scale_y = height / source_height
    calibration.update(
        {
            "width": width,
            "height": height,
            "fx": float(calibration["fx"]) * scale_x,
            "fy": float(calibration["fy"]) * scale_y,
            "cx": float(calibration["cx"]) * scale_x,
            "cy": float(calibration["cy"]) * scale_y,
        }
    )
    return dataset


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Output dimensions must be positive")
    render_tools.configure_cuda_arch(args.device)
    run_dir = args.run_dir.expanduser().resolve()
    mapping = json.loads(args.mapping.expanduser().resolve().read_text())
    source_indices = [int(value) for value in mapping["source_frame_indices"]]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = render_tools.load_run_config(run_dir)
    dataset = scaled_dataset_config(config, args.width, args.height)
    infos = render_tools.load_camera_infos(
        config["Testset"] if "Testset" in config else config["Dataset"]
    )
    if max(source_indices) >= len(infos):
        raise ValueError("Mapping contains a source index outside the trajectory")
    vignette = render_tools.load_vignette(run_dir, dataset)
    gaussians = render_tools.load_gaussians(run_dir, config, args.device, vignette)
    tracked = render_tools.load_tracked_camera_map(run_dir)

    render_dir = output_dir / "render"
    gt_dir = output_dir / "gt"
    if args.save_images:
        render_dir.mkdir(exist_ok=True)
        gt_dir.mkdir(exist_ok=True)

    rows = []
    for moon_index, source_index in enumerate(tqdm(source_indices, desc="matched_render")):
        info = infos[source_index]
        tracked_camera = tracked.get(info["image"], {})
        pose = tracked_camera.get(
            "pose", np.asarray(info["T_camera_world"], dtype=np.float32)
        )
        modalities = render_tools.render_pose_modalities(
            gaussians,
            pose,
            dataset,
            args.device,
            f"moon_matched_{moon_index:06d}",
            source_index,
            exposure_gain=tracked_camera.get("exposure_gain"),
        )
        prediction_bgr = modalities["rgb"]
        prediction = cv2.cvtColor(prediction_bgr, cv2.COLOR_BGR2RGB)
        gt_path = Path(dataset["dataset_path"]) / "rectified" / info["image"]
        gt = moon_resize(read_rgb(gt_path), args.width, args.height)
        row = {
            "moon_keyframe_index": moon_index,
            "source_frame_index": source_index,
            "source_image": info["image"],
            "psnr": psnr(gt, prediction),
            "ssim": ssim(gt, prediction, args.device),
        }
        rows.append(row)
        if args.save_images:
            stem = f"{moon_index:06d}.png"
            cv2.imwrite(str(render_dir / stem), prediction_bgr)
            cv2.imwrite(str(gt_dir / stem), cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))

    finite_psnr = [row["psnr"] for row in rows if math.isfinite(row["psnr"])]
    payload = {
        "protocol": {
            "frame_selection": "recovered MoonSplat keyframes mapped to source trajectory",
            "resolution": [args.width, args.height],
            "intrinsics": {
                key: dataset["Calibration"][key]
                for key in ("fx", "fy", "cx", "cy")
            },
            "psnr": "MoonSplat uint8 RGB per-frame MSE definition",
            "ssim": "MoonSplat 11x11 sigma=1.5 zero-padded RGB definition",
            "lpips": "not evaluated",
        },
        "run_dir": str(run_dir),
        "mapping": str(args.mapping.expanduser().resolve()),
        "frame_count": len(rows),
        "mean": {
            "psnr": float(np.mean(finite_psnr)),
            "ssim": float(np.mean([row["ssim"] for row in rows])),
        },
        "frames": rows,
    }
    metrics_path = output_dir / "metrics.json"
    atomic_json(metrics_path, payload)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in payload.items() if key != "frames"}, indent=2))
    print(f"Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
