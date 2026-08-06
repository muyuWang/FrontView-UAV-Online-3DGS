#!/usr/bin/env python3
"""Measure a side-by-side render-vs-GT video with a reproducible metric."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def psnr_for_pixels(render_float, gt_float):
    mse = float(np.mean((render_float - gt_float) ** 2))
    return 100.0 if mse == 0.0 else float(10.0 * np.log10(255.0**2 / mse))


def metric_row(
    render_bgr,
    gt_bgr,
    ssim_scale,
    nonblack_threshold,
    edge_gradient_threshold,
):
    render_float = render_bgr.astype(np.float32)
    gt_float = gt_bgr.astype(np.float32)
    psnr = psnr_for_pixels(render_float, gt_float)

    render_small = cv2.resize(
        render_bgr,
        None,
        fx=ssim_scale,
        fy=ssim_scale,
        interpolation=cv2.INTER_AREA,
    )
    gt_small = cv2.resize(
        gt_bgr,
        None,
        fx=ssim_scale,
        fy=ssim_scale,
        interpolation=cv2.INTER_AREA,
    )
    ssim = structural_similarity(
        render_small,
        gt_small,
        channel_axis=2,
        data_range=255,
    )
    coverage = 100.0 * np.mean(np.max(render_bgr, axis=2) > nonblack_threshold)
    bottom_start = render_bgr.shape[0] // 2
    bottom_half_psnr = psnr_for_pixels(
        render_float[bottom_start:], gt_float[bottom_start:]
    )
    width = render_bgr.shape[1]
    third = width // 3
    side_band = max(1, int(round(width * 0.30)))
    left_psnr = psnr_for_pixels(
        render_float[:, :third], gt_float[:, :third]
    )
    center_psnr = psnr_for_pixels(
        render_float[:, third : width - third],
        gt_float[:, third : width - third],
    )
    right_psnr = psnr_for_pixels(
        render_float[:, width - third :], gt_float[:, width - third :]
    )
    side_render = np.concatenate(
        (render_float[:, :side_band], render_float[:, -side_band:]), axis=1
    )
    side_gt = np.concatenate(
        (gt_float[:, :side_band], gt_float[:, -side_band:]), axis=1
    )
    side_band_psnr = psnr_for_pixels(side_render, side_gt)

    near_top = int(round(render_bgr.shape[0] * 0.25))
    near_side_mask = np.zeros(render_bgr.shape[:2], dtype=np.bool_)
    near_side_mask[near_top:, :side_band] = True
    near_side_mask[near_top:, -side_band:] = True
    near_side_psnr = psnr_for_pixels(
        render_float[near_side_mask], gt_float[near_side_mask]
    )

    # Ignore the overlaid video labels when measuring texture sharpness.
    detail_top = min(64, max(render_bgr.shape[0] - 1, 0))
    detail_render = render_float[detail_top:]
    detail_gt = gt_float[detail_top:]
    gt_gray = cv2.cvtColor(detail_gt, cv2.COLOR_BGR2GRAY)
    render_gray = cv2.cvtColor(detail_render, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gt_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gt_gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mask = np.hypot(grad_x, grad_y) >= edge_gradient_threshold
    edge_psnr = (
        psnr_for_pixels(detail_render[edge_mask], detail_gt[edge_mask])
        if np.any(edge_mask)
        else psnr_for_pixels(detail_render, detail_gt)
    )
    render_laplacian_variance = float(cv2.Laplacian(render_gray, cv2.CV_32F).var())
    gt_laplacian_variance = float(cv2.Laplacian(gt_gray, cv2.CV_32F).var())
    detail_near_side_mask = near_side_mask[detail_top:]
    near_side_edge_mask = edge_mask & detail_near_side_mask
    near_side_edge_psnr = (
        psnr_for_pixels(
            detail_render[near_side_edge_mask], detail_gt[near_side_edge_mask]
        )
        if np.any(near_side_edge_mask)
        else near_side_psnr
    )
    render_laplacian = cv2.Laplacian(render_gray, cv2.CV_32F)
    gt_laplacian = cv2.Laplacian(gt_gray, cv2.CV_32F)
    near_side_render_laplacian_variance = float(
        render_laplacian[detail_near_side_mask].var()
    )
    near_side_gt_laplacian_variance = float(
        gt_laplacian[detail_near_side_mask].var()
    )
    return {
        "psnr": psnr,
        "ssim_downscaled": float(ssim),
        "nonblack_coverage_percent": float(coverage),
        "bottom_half_psnr": bottom_half_psnr,
        "left_third_psnr": left_psnr,
        "center_third_psnr": center_psnr,
        "right_third_psnr": right_psnr,
        "side_band_psnr": side_band_psnr,
        "near_side_psnr": near_side_psnr,
        "edge_psnr": edge_psnr,
        "near_side_edge_psnr": near_side_edge_psnr,
        "edge_pixel_percent": float(100.0 * edge_mask.mean()),
        "render_laplacian_variance": render_laplacian_variance,
        "gt_laplacian_variance": gt_laplacian_variance,
        "near_side_render_laplacian_variance": near_side_render_laplacian_variance,
        "near_side_gt_laplacian_variance": near_side_gt_laplacian_variance,
    }


def mean_rows(rows):
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key != "frame_index"
    }


def evaluate_video(
    video_path,
    ssim_scale=0.25,
    nonblack_threshold=8,
    edge_gradient_threshold=24.0,
):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    rows = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[1] % 2 != 0:
            raise ValueError(
                f"Expected an even-width render-vs-GT frame, got {frame.shape[1]}"
            )
        half_width = frame.shape[1] // 2
        row = metric_row(
            frame[:, :half_width],
            frame[:, half_width:],
            ssim_scale,
            nonblack_threshold,
            edge_gradient_threshold,
        )
        row["frame_index"] = len(rows)
        rows.append(row)
    capture.release()

    if not rows:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    return {
        "definition": {
            "source": "decoded H.264 side-by-side render_vs_gt.mp4",
            "psnr_data_range": 255,
            "ssim_scale": float(ssim_scale),
            "nonblack_threshold": f"max_bgr_uint8 > {nonblack_threshold}",
            "detail_label_rows_ignored": 64,
            "near_side_region": "bottom 75% of left/right 30% image bands",
            "edge_mask": (
                "GT grayscale 3x3 Sobel magnitude >= "
                f"{edge_gradient_threshold:g}"
            ),
            "labels_included": True,
        },
        "frame_count": len(rows),
        "mean": mean_rows(rows),
        "first_20": mean_rows(rows[:20]),
        "last_8": mean_rows(rows[-8:]),
        "frames": rows,
    }


def read_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_final_progressive_state(path):
    if not path.exists():
        return {}
    last_line = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if last_line is None:
        return {}
    stats = json.loads(last_line)
    return {
        "P": stats.get("num_active_P", 0),
        "M": stats.get("num_active_M", 0),
        "S": stats.get("num_active_S", 0),
        "A": stats.get("num_active_A", 0),
        "visible_roots": stats.get("num_visible_roots", 0),
        "optimized_roots": stats.get("num_optimized_roots", 0),
        "frozen_roots": stats.get("num_frozen_roots", 0),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the render and GT halves of render_vs_gt.mp4."
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ssim_scale", type=float, default=0.25)
    parser.add_argument("--nonblack_threshold", type=int, default=8)
    parser.add_argument("--edge_gradient_threshold", type=float, default=24.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.ssim_scale <= 1.0:
        raise ValueError("ssim_scale must be in (0, 1].")
    if not 0 <= args.nonblack_threshold <= 255:
        raise ValueError("nonblack_threshold must be in [0, 255].")
    if args.edge_gradient_threshold < 0.0:
        raise ValueError("edge_gradient_threshold must be non-negative.")

    run_dir = args.run_dir.resolve()
    video_path = (args.video or (run_dir / "videos" / "render_vs_gt.mp4")).resolve()
    output_path = (args.output or (run_dir / "validation_metrics.json")).resolve()
    results = read_json(run_dir / "results.json")
    payload = {
        "run": run_dir.name,
        "video": str(video_path.relative_to(run_dir)),
        "reconstruction": {
            "online_reconstruction_seconds": results.get("online_recon_time"),
            "num_processed_frames": results.get("num_processed_frames"),
            "num_gaussians": results.get("num_gaussians"),
            "num_keyframes": results.get("num_keyframes"),
            "progressive_runtime": results.get("progressive_runtime", {}),
        },
        "final_progressive_state": read_final_progressive_state(
            run_dir / "progressive_stats.jsonl"
        ),
        "trajectory_metrics": evaluate_video(
            video_path,
            ssim_scale=args.ssim_scale,
            nonblack_threshold=args.nonblack_threshold,
            edge_gradient_threshold=args.edge_gradient_threshold,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(json.dumps(payload["trajectory_metrics"]["mean"], indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
