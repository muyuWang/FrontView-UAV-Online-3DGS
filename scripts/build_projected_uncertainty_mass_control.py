#!/usr/bin/env python3
"""Build a fixed-checkpoint control for uncertain projected Gaussian mass."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils_new.frontview_far_field import adaptive_log_depth_responsibility


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", default="0,48,120,192,240")
    parser.add_argument("--cell-px", type=float, default=12.0)
    parser.add_argument(
        "--mode",
        choices=("scale_cap", "ray_reanchor"),
        default="scale_cap",
    )
    parser.add_argument(
        "--adaptive-far-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    frame_ids = [int(value) for value in args.frames.split(",") if value.strip()]
    if not frame_ids:
        raise ValueError("At least one diagnostic frame is required")
    if args.cell_px <= 0.0:
        raise ValueError("cell-px must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)

    ply = PlyData.read(str(run_dir / "point_cloud.ply"))
    vertex = ply["vertex"]
    rows = vertex.data.copy()
    names = set(rows.dtype.names or ())
    required = {
        "x",
        "y",
        "z",
        "scale_0",
        "scale_1",
        "scale_2",
        "metric_confidence",
    }
    if not required.issubset(names):
        raise ValueError(f"PLY is missing fields: {sorted(required - names)}")

    with (run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        import yaml

        config = yaml.safe_load(handle)
    calibration = config["Dataset"]["Calibration"]
    fx = float(calibration["fx"])
    fy = float(calibration["fy"])
    cx = float(calibration["cx"])
    cy = float(calibration["cy"])
    width = int(calibration["width"])
    height = int(calibration["height"])

    with (run_dir / "tracked_info.json").open("r", encoding="utf-8") as handle:
        cameras = json.load(handle)["cameras"]
    if min(frame_ids) < 0 or max(frame_ids) >= len(cameras):
        raise ValueError("Requested frames are outside tracked_info.json")

    means = np.stack([rows[name] for name in ("x", "y", "z")], axis=1).astype(
        np.float64
    )
    scales = np.exp(
        np.stack(
            [rows[name] for name in ("scale_0", "scale_1", "scale_2")],
            axis=1,
        ).astype(np.float64)
    )
    max_scale = scales.max(axis=1)
    max_radius = np.zeros(len(rows), dtype=np.float64)
    reference_depth = np.zeros(len(rows), dtype=np.float64)
    reference_center = np.zeros((len(rows), 3), dtype=np.float64)
    visible_count = np.zeros(len(rows), dtype=np.int16)
    focal = max(fx, fy)
    for frame_id in frame_ids:
        camera = cameras[frame_id]
        pose = np.asarray(camera.get("pose") or camera["raw_pose"], dtype=np.float64)
        center = -pose[:3, :3].T @ pose[:3, 3]
        camera_points = means @ pose[:3, :3].T + pose[:3, 3]
        depth = camera_points[:, 2]
        u = fx * camera_points[:, 0] / np.maximum(depth, 1.0e-8) + cx
        v = fy * camera_points[:, 1] / np.maximum(depth, 1.0e-8) + cy
        visible = (
            (depth > 0.0)
            & (u >= 0.0)
            & (u < width)
            & (v >= 0.0)
            & (v < height)
        )
        if args.adaptive_far_only:
            far, _ = adaptive_log_depth_responsibility(depth, visible)
            visible &= far
        radius = focal * max_scale / np.maximum(depth, 1.0e-8)
        better = visible & (radius > max_radius)
        reference_depth[better] = depth[better]
        reference_center[better] = center
        max_radius = np.maximum(max_radius, np.where(visible, radius, 0.0))
        visible_count += visible

    confidence = np.clip(
        np.asarray(rows["metric_confidence"], dtype=np.float64), 0.0, 1.0
    )
    uncertainty_radius = max_radius * np.sqrt(np.maximum(1.0 - confidence, 0.0))
    scale_factor = np.minimum(
        1.0,
        float(args.cell_px) / np.maximum(uncertainty_radius, 1.0e-8),
    )
    affected = (visible_count > 0) & (scale_factor < 1.0)
    if args.mode == "scale_cap":
        geometric_factor = scale_factor
    else:
        geometric_factor = np.reciprocal(
            np.maximum(scale_factor, np.finfo(np.float32).tiny)
        )
        rays = means - reference_center
        rows["x"][affected] = (
            reference_center[affected, 0]
            + geometric_factor[affected] * rays[affected, 0]
        ).astype(np.float32)
        rows["y"][affected] = (
            reference_center[affected, 1]
            + geometric_factor[affected] * rays[affected, 1]
        ).astype(np.float32)
        rows["z"][affected] = (
            reference_center[affected, 2]
            + geometric_factor[affected] * rays[affected, 2]
        ).astype(np.float32)
    log_factor = np.log(
        np.maximum(geometric_factor, np.finfo(np.float32).tiny)
    )
    for name in ("scale_0", "scale_1", "scale_2"):
        rows[name][affected] += log_factor[affected].astype(np.float32)

    PlyData(
        [PlyElement.describe(rows, "vertex")],
        text=ply.text,
        byte_order=ply.byte_order,
    ).write(str(output_dir / "point_cloud.ply"))
    for filename in (
        "config.yaml",
        "tracked_info.json",
        "frontview_directional_layer.pt",
    ):
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)

    summary = {
        "source_run": str(run_dir),
        "frames": frame_ids,
        "cell_px": float(args.cell_px),
        "mode": args.mode,
        "adaptive_far_only": bool(args.adaptive_far_only),
        "gaussians": int(len(rows)),
        "visible_gaussians": int(np.count_nonzero(visible_count)),
        "affected_gaussians": int(np.count_nonzero(affected)),
        "affected_fraction": float(np.mean(affected)),
        "mean_affected_confidence": (
            float(np.mean(confidence[affected])) if np.any(affected) else None
        ),
        "mean_scale_factor": (
            float(np.mean(scale_factor[affected])) if np.any(affected) else None
        ),
        "min_scale_factor": (
            float(np.min(scale_factor[affected])) if np.any(affected) else None
        ),
        "mean_geometric_factor": (
            float(np.mean(geometric_factor[affected])) if np.any(affected) else None
        ),
        "max_geometric_factor": (
            float(np.max(geometric_factor[affected])) if np.any(affected) else None
        ),
        "mean_reference_depth_m": (
            float(np.mean(reference_depth[affected])) if np.any(affected) else None
        ),
        "constraint": "max_view_radius * sqrt(1 - metric_confidence) <= cell_px",
    }
    with (output_dir / "control_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
