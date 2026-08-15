#!/usr/bin/env python3
"""Archive causal dropout-episode anchors for fixed-checkpoint evaluation."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils_new.frontview_directional_layer import FrontViewDirectionalLayer


class EpisodeCamera:
    def __init__(self, frame_id, image, info, sparse_count):
        self.cam_idx = int(frame_id)
        self._image = image
        pose = info.get("pose") or info["raw_pose"]
        self._pose = torch.as_tensor(pose, dtype=torch.float32)
        self._intrinsics = torch.tensor(
            [
                [float(info["fx"]), 0.0, float(info["cx"])],
                [0.0, float(info["fy"]), float(info["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self._sparse_count = int(sparse_count)
        self.exposure_gain = float(info.get("exposure_gain", 1.0))

    def get_pts(self):
        return range(self._sparse_count)

    def get_gt_image(self, _scale):
        return self._image

    def get_pose(self):
        return self._pose

    def get_int_mat(self, _scale):
        return self._intrinsics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--short-episode-budget", type=int, default=2)
    parser.add_argument("--long-episode-budget", type=int, default=24)
    parser.add_argument("--long-episode-min-frames", type=int, default=24)
    parser.add_argument("--dense-begin", type=int, default=None)
    parser.add_argument("--dense-end", type=int, default=None)
    parser.add_argument("--dense-budget", type=int, default=24)
    parser.add_argument(
        "--episode-geometry-gate-mode",
        choices=("metric_transmittance", "uncertainty_mass"),
        default=None,
    )
    parser.add_argument("--episode-uncertainty-cell-px", type=float, default=None)
    parser.add_argument("--episode-blend-weight", type=float, default=None)
    parser.add_argument(
        "--episode-boundary-taper",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--episode-source-fusion",
        choices=("first", "mean", "causal_crossfade"),
        default=None,
    )
    return parser.parse_args()


def sparse_count(point_path):
    if not point_path.exists():
        return 0
    return int(len(np.load(point_path, mmap_mode="r")))


def episode_ranges(counts, threshold):
    ranges = []
    begin = None
    for frame_id, count in enumerate(counts + [threshold]):
        dropout = count < threshold
        if dropout and begin is None:
            begin = frame_id
        elif not dropout and begin is not None:
            ranges.append((begin, frame_id))
            begin = None
    return ranges


def load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1])).float() / 255.0


def link_or_copy(source, destination):
    try:
        os.symlink(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def main():
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if min(args.short_episode_budget, args.long_episode_budget) < 1:
        raise ValueError("Episode budgets must be positive")
    if args.long_episode_min_frames < 1:
        raise ValueError("long-episode-min-frames must be positive")
    if (
        args.episode_geometry_gate_mode == "uncertainty_mass"
        and args.episode_uncertainty_cell_px is None
    ):
        raise ValueError("uncertainty_mass episodes require a cell size")
    if args.episode_uncertainty_cell_px is not None and args.episode_uncertainty_cell_px <= 0:
        raise ValueError("episode-uncertainty-cell-px must be positive")
    if args.episode_blend_weight is not None and not 0.0 <= args.episode_blend_weight <= 1.0:
        raise ValueError("episode-blend-weight must be in [0, 1]")
    output_dir.mkdir(parents=True, exist_ok=False)

    payload = torch.load(run_dir / "frontview_directional_layer.pt", map_location="cpu")
    config = dict(payload["config"])
    threshold = int(config["sparse_point_threshold"])
    with (run_dir / "tracked_info.json").open("r", encoding="utf-8") as handle:
        infos = json.load(handle)["cameras"]
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        import yaml

        dataset_dir = Path(yaml.safe_load(handle)["Dataset"]["dataset_path"])
    points_dir = dataset_dir / "orb_point_clouds"
    image_dir = dataset_dir / "rectified"
    counts = [
        sparse_count(points_dir / f"point_cloud_{frame_id}.npy")
        for frame_id in range(len(infos))
    ]

    anchors = []
    episodes = []
    ownership_profile = {
        key: value
        for key, value in (
            ("geometry_gate_mode", args.episode_geometry_gate_mode),
            ("uncertainty_cell_px", args.episode_uncertainty_cell_px),
            ("blend_weight", args.episode_blend_weight),
            ("boundary_taper", args.episode_boundary_taper),
            ("source_fusion", args.episode_source_fusion),
        )
        if value is not None
    }
    if (args.dense_begin is None) != (args.dense_end is None):
        raise ValueError("dense-begin and dense-end must be provided together")
    ranges = episode_ranges(counts, threshold)
    if args.dense_begin is not None:
        if not 0 <= args.dense_begin <= args.dense_end < len(infos):
            raise ValueError("Dense archive range is outside tracked_info.json")
        ranges = [(args.dense_begin, args.dense_end + 1)]
    for begin, end in ranges:
        length = end - begin
        if args.dense_begin is not None:
            budget = args.dense_budget
        else:
            budget = (
                args.long_episode_budget
                if length >= args.long_episode_min_frames
                else args.short_episode_budget
            )
        episode_config = {
            **config,
            "anchor_selection_mode": "ordered_ward",
            "max_anchors": int(budget),
            "min_anchors": min(int(config["min_anchors"]), int(budget)),
        }
        layer = FrontViewDirectionalLayer(episode_config)
        for frame_id in range(begin, end):
            info = infos[frame_id]
            image = load_rgb(image_dir / info["name"])
            observed_count = 0 if args.dense_begin is not None else counts[frame_id]
            layer.observe(EpisodeCamera(frame_id, image, info, observed_count))
        episode_anchors = []
        for anchor in layer.anchors:
            scoped_anchor = dict(anchor)
            scoped_anchor["support_begin_frame"] = int(begin)
            scoped_anchor["support_end_frame"] = int(end - 1)
            if ownership_profile:
                scoped_anchor["ownership_profile"] = dict(ownership_profile)
            episode_anchors.append(scoped_anchor)
        anchors.extend(episode_anchors)
        episodes.append(
            {
                "begin": begin,
                "end": end - 1,
                "frames": length,
                "budget": budget,
                "anchors": [int(anchor["frame_id"]) for anchor in episode_anchors],
            }
        )
    if args.dense_begin is not None:
        anchors.extend(
            anchor
            for anchor in payload["anchors"]
            if int(anchor["frame_id"]) > int(args.dense_end)
        )

    payload["anchors"] = anchors
    payload["stats"] = {
        **payload.get("stats", {}),
        "archived_episode_count": len(episodes),
        "archived_anchor_count": len(anchors),
    }
    payload["stream_state"] = {
        "pose_center_count": 0,
        "pose_center_mean": torch.zeros(3, dtype=torch.float64),
        "pose_center_m2": 0.0,
        "dropout_episode_active": False,
    }
    torch.save(payload, output_dir / "frontview_directional_layer.pt")
    for filename in ("point_cloud.ply", "config.yaml", "tracked_info.json"):
        link_or_copy(run_dir / filename, output_dir / filename)

    summary = {
        "source_run": str(run_dir),
        "sparse_point_threshold": threshold,
        "short_episode_budget": args.short_episode_budget,
        "long_episode_budget": args.long_episode_budget,
        "long_episode_min_frames": args.long_episode_min_frames,
        "dense_begin": args.dense_begin,
        "dense_end": args.dense_end,
        "dense_budget": args.dense_budget,
        "ownership_profile": ownership_profile,
        "episode_count": len(episodes),
        "anchor_count": len(anchors),
        "anchor_frames": [int(anchor["frame_id"]) for anchor in anchors],
        "episodes": episodes,
    }
    with (output_dir / "archive_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
