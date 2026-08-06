#!/usr/bin/env python3
"""Restore exact per-frame COLMAP point IDs from the canonical global point table."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pycolmap


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset.resolve()
    point_dir = dataset / "orb_point_clouds"
    id_dir = dataset / "orb_point_ids"
    global_path = dataset / "preprocess" / "global_colmap_points_sim3.npy"
    model_path = dataset / "preprocess" / "colmap_model"
    if not point_dir.is_dir() or not global_path.is_file() or not model_path.is_dir():
        raise FileNotFoundError("Canonical point clouds/global table/COLMAP model missing")

    global_points = np.load(global_path).astype("<f4", copy=False)
    reconstruction = pycolmap.Reconstruction(model_path)
    model_ids = set(int(value) for value in reconstruction.point3D_ids())
    if global_points.shape != (len(model_ids), 3):
        raise RuntimeError(
            "Global point rows ({}) do not match COLMAP IDs ({})".format(
                len(global_points), len(model_ids)
            )
        )
    identity_lookup = {}
    duplicate_coordinates = 0
    for row, point in enumerate(np.ascontiguousarray(global_points)):
        key = point.tobytes()
        if key in identity_lookup:
            duplicate_coordinates += 1
        else:
            identity_lookup[key] = int(row)
    point_paths = sorted(
        point_dir.glob("point_cloud_*.npy"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    frame_limit = len(point_paths) if args.frames <= 0 else int(args.frames)
    if frame_limit > len(point_paths):
        raise ValueError("Requested more frames than available point clouds")
    id_dir.mkdir(exist_ok=True)

    frame_identities = []
    verified = 0
    rows = 0
    for frame_id, point_path in enumerate(point_paths[:frame_limit]):
        if int(point_path.stem.rsplit("_", 1)[1]) != frame_id:
            raise RuntimeError("Point-cloud frame numbering is not contiguous")
        stored_points = np.load(point_path).astype("<f4", copy=False)
        try:
            identities = np.asarray(
                [
                    identity_lookup[point.tobytes()]
                    for point in np.ascontiguousarray(stored_points)
                ],
                dtype=np.int64,
            )
        except KeyError as error:
            raise RuntimeError(
                "Frame {} contains a point outside the exact canonical global table".format(
                    frame_id
                )
            ) from error
        frame_identities.append(identities)
        rows += len(identities)
        output = id_dir / "point_ids_{}.npy".format(frame_id)
        if output.is_file():
            existing = np.load(output).astype(np.int64, copy=False)
            if not np.array_equal(existing, identities):
                raise RuntimeError(
                    "Existing canonical identity sidecar disagrees at frame {}".format(
                        frame_id
                    )
                )
            verified += 1

    generated = 0
    for frame_id, identities in enumerate(frame_identities):
        output = id_dir / "point_ids_{}.npy".format(frame_id)
        if output.is_file():
            continue
        if not args.dry_run:
            temporary = output.with_suffix(".tmp.npy")
            np.save(temporary, identities)
            os.replace(temporary, output)
        generated += 1

    payload = {
        "schema_version": 1,
        "dataset": str(dataset),
        "frames": frame_limit,
        "global_point_count": int(len(global_points)),
        "canonical_identity_count": int(len(identity_lookup)),
        "duplicate_coordinate_rows_collapsed": int(duplicate_coordinates),
        "verified_existing_frames": verified,
        "generated_missing_frames": generated,
        "validated_point_rows": rows,
        "identity_method": "exact canonical float32 coordinate identity",
        "all_frame_points_found_in_global_table": True,
        "nearest_neighbor_used": False,
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        (id_dir / "restore_stats.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
