#!/usr/bin/env python3
"""Recover MoonSplat keyframe source indices from saved renders and metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moon-results", type=Path, required=True)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--descriptor-width", type=int, default=128)
    parser.add_argument("--source-batch", type=int, default=4)
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def moon_resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(rgb)
    return np.asarray(image.resize((width, height), Image.Resampling.LANCZOS))


def descriptor(rgb: np.ndarray, width: int) -> np.ndarray:
    height = int(round(rgb.shape[0] * width / rgb.shape[1]))
    small = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    values = np.concatenate((gray.reshape(-1), dx.reshape(-1), dy.reshape(-1)))
    values -= values.mean()
    norm = np.linalg.norm(values)
    if norm <= 1.0e-12:
        raise ValueError("Degenerate image descriptor")
    return values / norm


def pairwise_descriptor_cost(
    render_paths: list[Path], source_paths: list[Path], width: int, device: str
) -> np.ndarray:
    render = np.stack([descriptor(read_rgb(path), width) for path in render_paths])
    source = np.stack([descriptor(read_rgb(path), width) for path in source_paths])
    render_tensor = torch.from_numpy(render).to(device=device, dtype=torch.float32)
    source_tensor = torch.from_numpy(source).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        cost = 2.0 - 2.0 * render_tensor @ source_tensor.T
    return cost.cpu().numpy()


def psnr(gt: np.ndarray, render: np.ndarray) -> float:
    mse = np.mean((gt.astype(np.float64) - render.astype(np.float64)) ** 2)
    return float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim(gt: np.ndarray, render: np.ndarray, device: str) -> float:
    x = torch.from_numpy(gt.copy()).permute(2, 0, 1).float()[None].to(device) / 255.0
    y = (
        torch.from_numpy(render.copy()).permute(2, 0, 1).float()[None].to(device)
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


def pairwise_mse_signature_costs(
    render_paths: list[Path],
    source_paths: list[Path],
    recorded_rows: list[dict],
    device: str,
    source_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    renders = np.stack([read_rgb(path) for path in render_paths]).astype(np.float32)
    height, width = renders.shape[1:3]
    render_tensor = torch.from_numpy(renders).to(device).flatten(1)
    pairwise_mse = np.empty((len(render_paths), len(source_paths)), dtype=np.float32)
    with torch.inference_mode():
        for begin in range(0, len(source_paths), source_batch):
            batch_paths = source_paths[begin : begin + source_batch]
            batch = np.stack(
                [moon_resize(read_rgb(path), width, height) for path in batch_paths]
            ).astype(np.float32)
            source_tensor = torch.from_numpy(batch).to(device).flatten(1)
            mse = (
                (render_tensor[None, :, :] - source_tensor[:, None, :])
                .square()
                .mean(dim=2)
            )
            pairwise_mse[:, begin : begin + len(batch_paths)] = (
                mse.transpose(0, 1).cpu().numpy()
            )
    target_mse = np.asarray(
        [255.0**2 / (10.0 ** (float(row["psnr"]) / 10.0)) for row in recorded_rows],
        dtype=np.float64,
    )
    costs = np.abs(pairwise_mse.astype(np.float64) - target_mse[:, None])
    return costs, pairwise_mse


def monotonic_path(costs: np.ndarray) -> np.ndarray:
    rows, columns = costs.shape
    previous = costs[0].copy()
    backpointers = np.full((rows, columns), -1, dtype=np.int32)
    for row in range(1, rows):
        prefix_cost = np.full(columns, np.inf, dtype=np.float64)
        prefix_index = np.full(columns, -1, dtype=np.int32)
        best_cost = np.inf
        best_index = -1
        for column in range(columns):
            prefix_cost[column] = best_cost
            prefix_index[column] = best_index
            if previous[column] < best_cost:
                best_cost = previous[column]
                best_index = column
        current = costs[row] + prefix_cost
        backpointers[row] = prefix_index
        previous = current
    if not np.any(np.isfinite(previous)):
        raise RuntimeError("No strictly monotonic source-frame path")
    path = np.empty(rows, dtype=np.int32)
    path[-1] = int(np.argmin(previous))
    for row in range(rows - 1, 0, -1):
        path[row - 1] = backpointers[row, path[row]]
    if np.any(np.diff(path) <= 0):
        raise RuntimeError("Recovered path is not strictly increasing")
    return path


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.descriptor_width <= 0 or args.source_batch <= 0:
        raise ValueError("Descriptor width and source batch must be positive")
    moon_results = args.moon_results.expanduser().resolve()
    source_mapping = json.loads(
        args.source_mapping.expanduser().resolve().read_text(encoding="utf-8")
    )
    source_paths = [Path(row["source_image"]).resolve() for row in source_mapping]
    render_paths = sorted((moon_results / "eval" / "render").glob("*.png"))
    metrics = json.loads((moon_results / "eval" / "metrics.json").read_text())
    recorded_rows = metrics["frames"]
    if len(render_paths) != len(recorded_rows):
        raise RuntimeError("MoonSplat render and metric counts differ")

    descriptor_costs = pairwise_descriptor_cost(
        render_paths, source_paths, args.descriptor_width, args.device
    )
    exact_costs, pairwise_mse = pairwise_mse_signature_costs(
        render_paths,
        source_paths,
        recorded_rows,
        args.device,
        args.source_batch,
    )
    path = monotonic_path(exact_costs)

    rows = []
    for moon_index, source_index in enumerate(path):
        render = read_rgb(render_paths[moon_index])
        gt = moon_resize(
            read_rgb(source_paths[int(source_index)]), render.shape[1], render.shape[0]
        )
        recovered_psnr = psnr(gt, render)
        recovered_ssim = ssim(gt, render, args.device)
        descriptor_rank = int(
            np.flatnonzero(np.argsort(descriptor_costs[moon_index]) == source_index)[0]
            + 1
        )
        signature_rank = int(
            np.flatnonzero(np.argsort(exact_costs[moon_index]) == source_index)[0] + 1
        )
        rows.append(
            {
                "moon_keyframe_index": moon_index,
                "source_frame_index": int(source_index),
                "source_image": str(source_paths[int(source_index)]),
                "gap_from_previous": (
                    None if moon_index == 0 else int(source_index - path[moon_index - 1])
                ),
                "descriptor_rank": descriptor_rank,
                "descriptor_cost": float(descriptor_costs[moon_index, source_index]),
                "mse_signature_rank": signature_rank,
                "mse_signature_error": float(exact_costs[moon_index, source_index]),
                "recovered_mse": float(pairwise_mse[moon_index, source_index]),
                "moon_recorded_psnr": float(recorded_rows[moon_index]["psnr"]),
                "recovered_gt_psnr": recovered_psnr,
                "psnr_abs_error": abs(
                    recovered_psnr - float(recorded_rows[moon_index]["psnr"])
                ),
                "moon_recorded_ssim": float(recorded_rows[moon_index]["ssim"]),
                "recovered_gt_ssim": recovered_ssim,
                "ssim_abs_error": abs(
                    recovered_ssim - float(recorded_rows[moon_index]["ssim"])
                ),
            }
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": "MoonSplat render-to-source metric-signature recovery with strict monotonic order",
        "moon_results": str(moon_results),
        "source_mapping": str(args.source_mapping.expanduser().resolve()),
        "frame_count": len(rows),
        "source_frame_indices": [int(value) for value in path],
        "source_index_min": int(path.min()),
        "source_index_max": int(path.max()),
        "descriptor_rank_max": max(row["descriptor_rank"] for row in rows),
        "descriptor_rank_mean": float(
            np.mean([row["descriptor_rank"] for row in rows])
        ),
        "mse_signature_rank_max": max(row["mse_signature_rank"] for row in rows),
        "mse_signature_error_max": max(row["mse_signature_error"] for row in rows),
        "psnr_abs_error_max": max(row["psnr_abs_error"] for row in rows),
        "psnr_abs_error_mean": float(
            np.mean([row["psnr_abs_error"] for row in rows])
        ),
        "ssim_abs_error_max": max(row["ssim_abs_error"] for row in rows),
        "ssim_abs_error_mean": float(
            np.mean([row["ssim_abs_error"] for row in rows])
        ),
        "rows": rows,
    }
    atomic_json(output, summary)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    print(f"Mapping: {output}")
    print(f"Rows: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
