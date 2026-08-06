#!/usr/bin/env python3
"""CPU-only synthetic smoke test for P -> M -> S -> A -> S transitions."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils_new.progressive_mapping import ProgressiveManager
from utils_new.progressive_mapping.config import validate_progressive_config


class SyntheticCamera:
    def __init__(self, frame_id, image, depth):
        self.cam_idx = frame_id
        self._image = image
        self._depth = depth
        self._pose = torch.eye(4)
        self._intrinsics = torch.tensor(
            [[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]]
        )
        self.near = 0.1
        self.far = 100.0

    def get_gt_image(self, level=0):
        return self._image

    def get_sparse_depth(self, level=0):
        return self._depth

    def get_pose(self):
        return self._pose

    def get_int_mat(self, level=0):
        return self._intrinsics

    def get_height(self, level=0):
        return self._image.shape[0]

    def get_width(self, level=0):
        return self._image.shape[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = args.output or tempfile.mkdtemp(prefix="progressive_smoke_")
    os.makedirs(output, exist_ok=True)
    cfg = validate_progressive_config(
        {
            "enabled": True,
            "debug": False,
            "patch_stride": 64,
            "max_new_anchors_per_keyframe": 1,
            "association_radius_px": 32.0,
            "association_feature_threshold": 1.0,
            "promotion_min_observations": 2,
            "promotion_min_best_weight": 0.0,
            "promotion_max_normalized_entropy": 1.0,
            "promotion_max_relative_std": 10.0,
            "promotion_min_parallax_deg": 0.0,
            "promotion_max_match_error": 2.0,
            "refine_min_observations": 0,
            "refine_min_projected_radius_px": 0.0,
            "refine_min_residual": 0.0,
            "refine_min_confidence": 0.0,
        }
    )
    manager = ProgressiveManager(cfg, gaussian_model=None, output_dir=output)
    y, x = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64), indexing="ij")
    image = torch.stack((x, y, 0.5 + 0.2 * torch.sin(9 * x)), dim=-1)
    depth = torch.full((64, 64), 4.0)
    stable = {
        "render": torch.zeros_like(image),
        "opacity": torch.zeros((64, 64, 1)),
        "diff": torch.zeros((64, 64)),
    }
    first = manager.process_frame(SyntheticCamera(0, image, depth), stable, True)
    second = manager.process_frame(SyntheticCamera(1, image, depth), stable, True)
    assert first.num_new_P == 1
    assert second.num_active_S == 9, second.to_dict()
    root_id = manager.registry.root_nodes()[0].node_id
    manager.archive_root(root_id, 2)
    assert manager.archive_store.get(root_id).means_fp16.dtype == torch.float16
    manager.reactivate_root(root_id)
    export_path = os.path.join(output, "synthetic_full_map.pt")
    manager.export_full_progressive_map(export_path)
    assert os.path.exists(export_path)
    print(json.dumps({"output": output, "frame0": first.to_dict(), "frame1": second.to_dict()}))


if __name__ == "__main__":
    main()
