#!/usr/bin/env python3
"""Materialize a reindexed frame slice without changing its canonical world."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def link(destination: Path, source: Path) -> None:
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    trajectory_path = source / "trajectory_orb.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    cameras = trajectory["cameras"]
    end = min(len(cameras), args.start_frame + args.num_frames)
    selected = cameras[args.start_frame:end]
    if len(selected) != args.num_frames:
        raise ValueError("Requested slice extends beyond the source dataset")
    if args.output.exists():
        if not args.force:
            raise FileExistsError(args.output)
        shutil.rmtree(args.output)
    staging = args.output.with_name(args.output.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    for relative in ("rectified", "orb_point_clouds", "orb_point_ids", "preprocess"):
        (staging / relative).mkdir(parents=True, exist_ok=True)

    output_cameras = []
    counts = []
    source_indices = []
    for output_index, camera in enumerate(selected):
        source_index = int(camera.get("frame_index", args.start_frame + output_index))
        source_indices.append(source_index)
        image_name = f"aria_{output_index:05d}.png"
        updated = dict(camera)
        updated["image"] = image_name
        updated["frame_index"] = output_index
        updated["source_frame_index"] = source_index
        output_cameras.append(updated)
        link(staging / "rectified" / image_name, source / "rectified" / camera["image"])
        source_stem = Path(camera["image"]).stem.rsplit("_", 1)[-1]
        point_source = source / "orb_point_clouds" / f"point_cloud_{int(source_stem)}.npy"
        id_source = source / "orb_point_ids" / f"point_ids_{int(source_stem)}.npy"
        if not point_source.is_file() or not id_source.is_file():
            raise FileNotFoundError(f"Missing source point rows for {camera['image']}")
        link(
            staging / "orb_point_clouds" / f"point_cloud_{output_index}.npy",
            point_source,
        )
        link(staging / "orb_point_ids" / f"point_ids_{output_index}.npy", id_source)
        counts.append(len(np.load(point_source, mmap_mode="r")))

    payload = json.dumps({"cameras": output_cameras}, indent=2) + "\n"
    (staging / "trajectory.json").write_text(payload, encoding="utf-8")
    (staging / "trajectory_orb.json").write_text(payload, encoding="utf-8")
    np.save(staging / "preprocess" / "source_frame_indices.npy", source_indices)
    for name in ("global_colmap_points_sim3.npy", "global_sparse_points.npy"):
        candidate = source / "preprocess" / name
        if candidate.is_file():
            link(staging / "preprocess" / name, candidate)
    model = source / "preprocess" / "colmap_model"
    if model.is_dir():
        link(staging / "preprocess" / "colmap_model", model)

    stats_path = source / "conversion_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    stats.update(
        {
            "method": str(stats.get("method", "canonical input")) + "; reindexed frame slice",
            "source_dataset": str(source),
            "removed_prefix_frames": int(args.start_frame),
            "source_frame_start": int(source_indices[0]),
            "source_frame_end_inclusive": int(source_indices[-1]),
            "frame_count": len(output_cameras),
            "per_frame_points_min": int(min(counts)),
            "per_frame_points_median": float(np.median(counts)),
            "per_frame_points_mean": float(np.mean(counts)),
            "per_frame_points_max": int(max(counts)),
        }
    )
    (staging / "conversion_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    (staging / "README.md").write_text(
        "# Reindexed MODP dataset slice\n\n"
        f"Frames 0..{args.start_frame - 1} are omitted. This slice preserves the source "
        "canonical world and reindexes images, point files, IDs, and trajectory rows.\n",
        encoding="utf-8",
    )
    staging.replace(args.output)
    print(
        f"Wrote {args.output}: frames={len(output_cameras)} "
        f"points/frame median={np.median(counts):.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
