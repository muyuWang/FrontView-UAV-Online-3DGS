#!/usr/bin/env python3
"""Export canonical initialization points with their COLMAP colors as PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap
from plyfile import PlyData, PlyElement


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def fit_similarity(source, target):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    covariance = centered_source.T @ centered_target / len(source)
    left, singular, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    variance = np.mean(np.sum(centered_source**2, axis=1))
    scale = float(np.sum(singular) / variance)
    translation = target_center - scale * (rotation @ source_center)
    transformed = (scale * (rotation @ source.T)).T + translation
    residuals = np.linalg.norm(transformed - target, axis=1)
    return scale, rotation, translation, residuals


def write_ply(path, xyz, rgb, row_ids, colmap_ids, errors, track_lengths):
    vertices = np.empty(
        len(xyz),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("canonical_row_id", "u4"),
            ("colmap_point3d_id", "u4"),
            ("reprojection_error_px", "f4"),
            ("track_length", "u2"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    vertices["canonical_row_id"] = row_ids
    vertices["colmap_point3d_id"] = colmap_ids
    vertices["reprojection_error_px"] = errors
    vertices["track_length"] = np.minimum(track_lengths, np.iinfo(np.uint16).max)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def summarize(xyz, errors, track_lengths):
    return {
        "point_count": int(len(xyz)),
        "bounds_min": xyz.min(axis=0).astype(float).tolist(),
        "bounds_max": xyz.max(axis=0).astype(float).tolist(),
        "reprojection_error_px": {
            "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95)),
            "max": float(np.max(errors)),
        },
        "track_length": {
            "median": float(np.median(track_lengths)),
            "p95": float(np.percentile(track_lengths, 95)),
            "max": int(np.max(track_lengths)),
        },
    }


def main():
    args = parse_args()
    dataset = args.dataset.resolve()
    global_path = dataset / "preprocess" / "global_colmap_points_sim3.npy"
    model_path = dataset / "preprocess" / "colmap_model"
    id_dir = dataset / "orb_point_ids"
    if not global_path.is_file() or not model_path.is_dir():
        raise FileNotFoundError("Canonical COLMAP points or model are missing")

    canonical = np.load(global_path).astype(np.float32, copy=False)
    reconstruction = pycolmap.Reconstruction(model_path)
    colmap_ids = np.asarray(list(reconstruction.point3D_ids()), dtype=np.uint32)
    if canonical.shape != (len(colmap_ids), 3):
        raise RuntimeError("Canonical rows do not match the COLMAP point count")
    points = [reconstruction.point3D(int(point_id)) for point_id in colmap_ids]
    source_xyz = np.asarray([point.xyz for point in points], dtype=np.float64)
    colors = np.asarray([point.color for point in points], dtype=np.uint8)
    errors = np.asarray([point.error for point in points], dtype=np.float32)
    track_lengths = np.asarray(
        [len(point.track.elements) for point in points], dtype=np.uint32
    )

    scale, rotation, translation, fit_residuals = fit_similarity(
        source_xyz, canonical.astype(np.float64)
    )
    maximum_residual = float(np.max(fit_residuals))
    if maximum_residual > 1.0e-3:
        raise RuntimeError(
            "COLMAP row order does not match canonical points; fit residual is {} m".format(
                maximum_residual
            )
        )

    output_dir = args.output_dir.resolve()
    all_rows = np.arange(len(canonical), dtype=np.uint32)
    global_output = output_dir / "initialization_global_colmap_canonical.ply"
    write_ply(
        global_output,
        canonical,
        colors,
        all_rows,
        colmap_ids,
        errors,
        track_lengths,
    )

    selected_rows = all_rows
    visible_output = None
    if args.frames > 0:
        frame_rows = []
        for frame_id in range(int(args.frames)):
            path = id_dir / "point_ids_{}.npy".format(frame_id)
            if not path.is_file():
                raise FileNotFoundError("Missing point identity sidecar: {}".format(path))
            frame_rows.append(np.load(path).astype(np.int64, copy=False))
        selected_rows = np.unique(np.concatenate(frame_rows)).astype(np.uint32)
        if selected_rows.size and int(selected_rows.max()) >= len(canonical):
            raise ValueError("A frame sidecar references an invalid canonical row")
        visible_output = output_dir / "initialization_visible_first_{:04d}_frames.ply".format(
            int(args.frames)
        )
        write_ply(
            visible_output,
            canonical[selected_rows],
            colors[selected_rows],
            selected_rows,
            colmap_ids[selected_rows],
            errors[selected_rows],
            track_lengths[selected_rows],
        )

    payload = {
        "schema_version": 1,
        "dataset": str(dataset),
        "coordinate_source": str(global_path),
        "color_source": str(model_path),
        "row_alignment": {
            "method": "verified global similarity with fixed row correspondence",
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "residual_median_m": float(np.median(fit_residuals)),
            "residual_p95_m": float(np.percentile(fit_residuals, 95)),
            "residual_max_m": maximum_residual,
        },
        "global": {"path": str(global_output), **summarize(canonical, errors, track_lengths)},
    }
    if visible_output is not None:
        payload["visible_frames"] = {
            "frames": int(args.frames),
            "path": str(visible_output),
            **summarize(
                canonical[selected_rows], errors[selected_rows], track_lengths[selected_rows]
            ),
        }
    stats_path = output_dir / "initialization_point_cloud_stats.json"
    stats_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
