#!/usr/bin/env python3
"""Precompute ORB-scale-aligned Depth Anything V2 inverse depth maps."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import least_squares
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from utils_new.aerocommit.sparse_track_geometry import zbuffer_sparse_tracks


def robust_affine_inverse_depth(prediction, metric_depth, max_depth):
    valid = (
        np.isfinite(prediction)
        & (prediction > 0.0)
        & np.isfinite(metric_depth)
        & (metric_depth > 0.1)
        & (metric_depth <= max_depth)
    )
    x = prediction[valid].astype(np.float64)
    y = (1.0 / metric_depth[valid]).astype(np.float64)
    if len(x) < 100:
        raise ValueError("Insufficient ORB support for inverse-depth alignment")
    design = np.stack((x, np.ones_like(x)), axis=1)
    initial = np.linalg.lstsq(design, y, rcond=None)[0]
    result = least_squares(
        lambda parameters: parameters[0] * x + parameters[1] - y,
        initial,
        loss="huber",
        f_scale=0.005,
        max_nfev=100,
    )
    aligned_depth = 1.0 / np.maximum(result.x[0] * x + result.x[1], 1.0e-5)
    abs_rel = np.abs(aligned_depth - metric_depth[valid]) / metric_depth[valid]
    return result.x, {
        "support": int(len(x)),
        "median_abs_rel": float(np.median(abs_rel)),
        "p90_abs_rel": float(np.percentile(abs_rel, 90)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model", default="depth-anything/Depth-Anything-V2-Small-hf"
    )
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--alignment_max_depth", type=float, default=60.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--prediction_type",
        choices=("auto", "relative_inverse", "metric_depth"),
        default="auto",
    )
    args = parser.parse_args()
    prediction_type = args.prediction_type
    if prediction_type == "auto":
        prediction_type = (
            "metric_depth" if "Metric" in args.model else "relative_inverse"
        )

    root = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = json.load(open(root / "trajectory_orb.json", encoding="utf-8"))[
        "cameras"
    ]
    if args.max_frames > 0:
        cameras = cameras[: args.max_frames]

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model)
    model.to(args.device).eval()
    records = []
    start_time = time.perf_counter()
    for frame_id, camera in enumerate(cameras):
        image = Image.open(root / "rectified" / camera["image"]).convert("RGB")
        inputs = {
            name: value.to(args.device)
            for name, value in processor(images=image, return_tensors="pt").items()
        }
        with torch.inference_mode():
            prediction = model(**inputs).predicted_depth
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(int(camera["height"]), int(camera["width"])),
                mode="bicubic",
                align_corners=False,
            )[0, 0]
        prediction_numpy = prediction.detach().cpu().numpy()
        relative_inverse_depth = (
            1.0 / np.maximum(prediction_numpy, 1.0e-5)
            if prediction_type == "metric_depth"
            else prediction_numpy
        )
        points = np.loadtxt(
            root / "orb_point_clouds" / "point_cloud_{}.txt".format(frame_id),
            dtype=np.float32,
        )
        intrinsics = np.asarray(
            [
                [camera["intrinsic"]["fx"], 0.0, camera["intrinsic"]["cx"]],
                [0.0, camera["intrinsic"]["fy"], camera["intrinsic"]["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        observations = zbuffer_sparse_tracks(
            points,
            np.asarray(camera["T_camera_world"], dtype=np.float64),
            intrinsics,
            int(camera["width"]),
            int(camera["height"]),
        )
        sparse_prediction = relative_inverse_depth.reshape(-1)[
            observations.pixel_indices
        ]
        coefficients, metrics = robust_affine_inverse_depth(
            sparse_prediction,
            observations.depths,
            args.alignment_max_depth,
        )
        aligned_inverse_depth = (
            coefficients[0] * relative_inverse_depth + coefficients[1]
        )
        valid = np.isfinite(aligned_inverse_depth) & (aligned_inverse_depth > 1.0e-5)
        aligned_depth = np.zeros_like(relative_inverse_depth, dtype=np.float32)
        aligned_depth[valid] = np.clip(
            1.0 / aligned_inverse_depth[valid], 0.1, args.alignment_max_depth
        )
        output_path = output_dir / "depth_{:04d}.npz".format(frame_id)
        np.savez_compressed(
            output_path,
            depth=aligned_depth.astype(np.float16),
            coefficients=coefficients.astype(np.float32),
            median_abs_rel=np.float32(metrics["median_abs_rel"]),
        )
        metrics.update(
            {
                "frame_id": frame_id,
                "coefficients": coefficients.tolist(),
                "path": str(output_path),
            }
        )
        records.append(metrics)
        print(
            "frame {:04d}: support {} median_abs_rel {:.3f}".format(
                frame_id, metrics["support"], metrics["median_abs_rel"]
            ),
            flush=True,
        )

    summary = {
        "model": args.model,
        "prediction_type": prediction_type,
        "frames": len(records),
        "elapsed_seconds": time.perf_counter() - start_time,
        "median_alignment_abs_rel": float(
            np.median([record["median_abs_rel"] for record in records])
        ),
        "records": records,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}))


if __name__ == "__main__":
    main()
