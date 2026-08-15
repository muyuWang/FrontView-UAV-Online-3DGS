#!/usr/bin/env python3
"""Causally replay a run into a fixed-memory directional anchor coreset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils_new.frontview_directional_layer import FrontViewDirectionalLayer


class ReplayCamera:
    def __init__(self, row, image, points):
        self.cam_idx = int(row["uid"])
        self._pose = torch.as_tensor(row.get("pose", row["raw_pose"])).float()
        self._image = image
        self._points = points
        self.exposure_gain = float(row.get("exposure_gain", 1.0))
        self._intrinsics = torch.tensor(
            [
                [float(row["fx"]), 0.0, float(row["cx"])],
                [0.0, float(row["fy"]), float(row["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

    def get_pts(self):
        return self._points

    def get_gt_image(self, level=0):
        if level != 0:
            raise ValueError("Directional replay only supports level zero")
        return self._image

    def get_pose(self):
        return self._pose

    def get_int_mat(self, level=0):
        if level != 0:
            raise ValueError("Directional replay only supports level zero")
        return self._intrinsics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--anchor-selection-mode",
        choices=(
            "interval_fifo",
            "streaming_kcenter",
            "ordered_ward",
            "episode_ordered_ward",
            "episode_bridge_ward",
        ),
        default="streaming_kcenter",
    )
    parser.add_argument("--max-anchors", type=int, default=None)
    return parser.parse_args()


def point_count(dataset_path, frame_id):
    path = dataset_path / "orb_point_clouds" / f"point_cloud_{frame_id}.npy"
    if not path.is_file():
        path = dataset_path / "orb_point_clouds" / f"point_cloud_{frame_id}.txt"
        if not path.is_file():
            return 0
        points = np.loadtxt(path, ndmin=2)
    else:
        points = np.load(path, mmap_mode="r")
    return int(len(points))


def main():
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with (run_dir / "tracked_info.json").open("r", encoding="utf-8") as handle:
        tracked = json.load(handle)["cameras"]
    dataset = config.get("Testset", config["Dataset"])
    dataset_path = Path(dataset["dataset_path"]).expanduser().resolve()
    layer_config = dict(config["FrontViewDirectionalLayer"])
    layer_config.update(
        anchor_selection_mode=args.anchor_selection_mode,
        pose_score_mode="rendered_inverse_depth",
        warp_mode="se3_fallback",
        source_fusion="first",
        warp_depth_control="aligned",
    )
    if args.max_anchors is not None:
        layer_config["max_anchors"] = int(args.max_anchors)
        layer_config["min_anchors"] = min(
            int(layer_config["min_anchors"]), int(args.max_anchors)
        )
    layer = FrontViewDirectionalLayer(layer_config)

    for row in tracked:
        frame_id = int(row["uid"])
        count = point_count(dataset_path, frame_id)
        image = None
        if count < int(layer_config["sparse_point_threshold"]):
            image_path = dataset_path / "rectified" / row["name"]
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(image_path)
            image = torch.from_numpy(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
                / 255.0
            )
        layer.observe(ReplayCamera(row, image, range(count)))

    if not layer.activate(True):
        raise RuntimeError("Replay did not retain enough directional anchors")
    output.parent.mkdir(parents=True, exist_ok=True)
    layer.save(output)
    summary = layer.summary()
    summary["anchor_frame_ids"] = [
        int(anchor["frame_id"]) for anchor in layer.anchors
    ]
    summary_path = output.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Sidecar: {output}")
    print(f"Summary: {summary_path}")
    print(f"Anchors: {summary['anchor_frame_ids']}")


if __name__ == "__main__":
    main()
