"""View-conditioned appearance carriers for frequency-faithful rendering."""

import json
from pathlib import Path

import numpy as np
import torch

from utils_new.tool_utils import rgb_to_sh


DETAIL_FILENAME = "view_conditioned_detail.npz"


def camera_center_and_forward(pose):
    """Return the camera center and forward axis for a world-to-camera pose."""
    pose = np.asarray(pose, dtype=np.float32)
    camera_to_world = np.linalg.inv(pose)
    return camera_to_world[:3, 3], camera_to_world[:3, 2]


class ViewConditionedDetailStore:
    """Immutable per-source surface details selected by target camera pose."""

    def __init__(
        self,
        means,
        scales,
        colors,
        opacities,
        source_offsets,
        source_frame_indices,
        source_poses,
        metadata=None,
        device="cuda:0",
        sh_degree=0,
    ):
        self.device = torch.device(device)
        self.sh_degree = int(sh_degree)
        self.metadata = dict(metadata or {})
        self.source_offsets = np.asarray(source_offsets, dtype=np.int64)
        self.source_frame_indices = np.asarray(source_frame_indices, dtype=np.int64)
        self.source_poses = np.asarray(source_poses, dtype=np.float32)
        self.source_centers = []
        self.source_forwards = []
        for pose in self.source_poses:
            center, forward = camera_center_and_forward(pose)
            self.source_centers.append(center)
            self.source_forwards.append(forward)
        self.source_centers = np.asarray(self.source_centers, dtype=np.float32)
        self.source_forwards = np.asarray(self.source_forwards, dtype=np.float32)

        count = int(np.asarray(means).shape[0])
        if self.source_offsets.shape != (len(self.source_frame_indices) + 1,):
            raise ValueError("source_offsets must contain one boundary per source")
        if self.source_offsets[0] != 0 or self.source_offsets[-1] != count:
            raise ValueError("source_offsets do not cover all appearance carriers")
        if self.source_poses.shape != (len(self.source_frame_indices), 4, 4):
            raise ValueError("source_poses must have shape [num_sources, 4, 4]")

        self.means = torch.as_tensor(means, dtype=torch.float32, device=self.device)
        self.scales = torch.as_tensor(scales, dtype=torch.float32, device=self.device)
        self.colors = torch.as_tensor(colors, dtype=torch.float32, device=self.device)
        self.opacities = torch.as_tensor(
            opacities, dtype=torch.float32, device=self.device
        ).reshape(-1)
        if self.means.shape != (count, 3) or self.scales.shape != (count, 3):
            raise ValueError("means and scales must have shape [N, 3]")
        if self.colors.shape != (count, 3) or self.opacities.shape != (count,):
            raise ValueError("colors and opacities have incompatible shapes")

    @classmethod
    def load(cls, path, device="cuda:0", sh_degree=0):
        path = Path(path)
        with np.load(path, allow_pickle=False) as payload:
            metadata_raw = payload.get("metadata_json")
            metadata = (
                json.loads(str(metadata_raw.item()))
                if metadata_raw is not None
                else {}
            )
            return cls(
                means=payload["means"],
                scales=payload["scales"],
                colors=payload["colors"],
                opacities=payload["opacities"],
                source_offsets=payload["source_offsets"],
                source_frame_indices=payload["source_frame_indices"],
                source_poses=payload["source_poses"],
                metadata=metadata,
                device=device,
                sh_degree=sh_degree,
            )

    @property
    def gaussian_count(self):
        return int(self.means.shape[0])

    @property
    def source_count(self):
        return int(len(self.source_frame_indices))

    def nearest_sources(
        self,
        target_pose,
        target_frame_index=None,
        exclude_exact=None,
        count=1,
    ):
        if self.source_count == 0 or int(count) <= 0:
            return np.empty((0,), dtype=np.int64)
        target_center, target_forward = camera_center_and_forward(target_pose)
        translation_scale = max(
            float(self.metadata.get("translation_scale", 1.0)), 1.0e-6
        )
        orientation_weight = max(
            float(self.metadata.get("orientation_weight", 2.0)), 0.0
        )
        translation = np.linalg.norm(
            self.source_centers - target_center[None, :], axis=1
        ) / translation_scale
        angular = 1.0 - np.clip(
            self.source_forwards @ target_forward, -1.0, 1.0
        )
        score = translation + orientation_weight * angular
        if exclude_exact is None:
            exclude_exact = bool(self.metadata.get("exclude_exact_frame", True))
        if exclude_exact and target_frame_index is not None:
            score = score.copy()
            score[self.source_frame_indices == int(target_frame_index)] = np.inf
        valid = np.flatnonzero(np.isfinite(score))
        if valid.size == 0:
            return np.empty((0,), dtype=np.int64)
        order = valid[np.argsort(score[valid], kind="stable")]
        return order[: min(int(count), order.size)]

    @torch.no_grad()
    def external_splats_for_pose(
        self,
        target_pose,
        target_frame_index=None,
        exclude_exact=None,
        source_count=1,
    ):
        selected = self.nearest_sources(
            target_pose,
            target_frame_index=target_frame_index,
            exclude_exact=exclude_exact,
            count=source_count,
        )
        if selected.size == 0:
            return None
        row_indices = []
        for source in selected:
            start = int(self.source_offsets[source])
            end = int(self.source_offsets[source + 1])
            if end > start:
                row_indices.append(
                    torch.arange(start, end, device=self.device, dtype=torch.long)
                )
        if not row_indices:
            return None
        rows = torch.cat(row_indices)
        colors = torch.clamp(self.colors[rows], 0.0, 1.0)
        coefficients = (self.sh_degree + 1) ** 2
        shs = torch.zeros(
            (rows.numel(), coefficients, 3),
            dtype=torch.float32,
            device=self.device,
        )
        shs[:, 0] = rgb_to_sh(colors)
        quats = torch.zeros(
            (rows.numel(), 4), dtype=torch.float32, device=self.device
        )
        quats[:, 0] = 1.0
        return {
            "means": self.means[rows].detach(),
            "scales": self.scales[rows].detach(),
            "quats": quats.detach(),
            "opacities": self.opacities[rows].detach(),
            "shs": shs.detach(),
        }
