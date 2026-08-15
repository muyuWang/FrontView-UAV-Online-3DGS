#!/usr/bin/env python3
"""Count finite-depth Mountains Gaussians in known error bands."""

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--begin", type=int, default=545)
    parser.add_argument("--end", type=int, default=619)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    ply = PlyData.read(run_dir / "point_cloud.ply")["vertex"]
    means = np.stack((ply["x"], ply["y"], ply["z"]), axis=1).astype(np.float64)
    properties = {item.name for item in ply.properties}
    metric = (
        np.asarray(ply["metric_confidence"], dtype=np.float64)
        if "metric_confidence" in properties
        else np.ones(len(means), dtype=np.float64)
    )
    with (run_dir / "tracked_info.json").open("r", encoding="utf-8") as handle:
        tracked = json.load(handle)
    cameras = tracked["cameras"]
    if not 0 <= args.begin <= args.end < len(cameras):
        raise ValueError("Requested frame range is outside tracked cameras")

    rows = []
    for frame_id in range(args.begin, args.end + 1):
        pose = np.asarray(
            cameras[frame_id].get("pose") or cameras[frame_id]["raw_pose"],
            dtype=np.float64,
        )
        camera_points = means @ pose[:3, :3].T + pose[:3, 3]
        depth = camera_points[:, 2]
        visible = np.isfinite(depth) & (depth > 0.0)
        far = visible & (depth > 250.0)
        error_band = visible & (depth >= 324.0) & (depth <= 339.0)
        metric_rows = metric > 0.0
        rows.append(
            {
                "frame_id": frame_id,
                "visible": int(visible.sum()),
                "depth_gt_250": int(far.sum()),
                "depth_324_339": int(error_band.sum()),
                "metric_depth_gt_250": int((far & metric_rows).sum()),
                "metric_depth_324_339": int((error_band & metric_rows).sum()),
                "proxy_depth_gt_250": int((far & ~metric_rows).sum()),
                "proxy_depth_324_339": int((error_band & ~metric_rows).sum()),
            }
        )
    keys = [key for key in rows[0] if key != "frame_id"]
    payload = {
        "run_dir": str(run_dir),
        "frame_range": [args.begin, args.end],
        "gaussians": len(means),
        "metric_gaussians": int((metric > 0.0).sum()),
        "proxy_gaussians": int((metric <= 0.0).sum()),
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in keys},
        "max": {key: int(max(row[key] for row in rows)) for key in keys},
        "frames": rows,
    }
    output = args.output or (run_dir / "far_gaussian_diagnostics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({"output": str(output), "mean": payload["mean"]}, indent=2))


if __name__ == "__main__":
    main()
