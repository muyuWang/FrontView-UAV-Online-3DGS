"""Compact view-conditioned Laplacian residual cache for the appearance layer."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils_new.aerocommit.view_detail import camera_center_and_forward


FREQUENCY_CACHE_FILENAME = "frequency_residual_cache.npz"


class FrequencyResidualCache:
    def __init__(
        self,
        residual_bands,
        source_frame_indices,
        source_poses,
        image_height,
        image_width,
        side_band_width,
        vertical_start,
        quantization_scale,
        metadata=None,
    ):
        self.residual_bands = np.asarray(residual_bands, dtype=np.int8)
        self.source_frame_indices = np.asarray(source_frame_indices, dtype=np.int64)
        self.source_poses = np.asarray(source_poses, dtype=np.float32)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.side_band_width = int(side_band_width)
        self.vertical_start = int(vertical_start)
        self.quantization_scale = float(quantization_scale)
        self.metadata = dict(metadata or {})
        if self.residual_bands.shape != (
            len(self.source_frame_indices),
            self.image_height - self.vertical_start,
            2 * self.side_band_width,
            3,
        ):
            raise ValueError("frequency residual bands have an incompatible shape")
        if self.source_poses.shape != (len(self.source_frame_indices), 4, 4):
            raise ValueError("frequency cache poses have an incompatible shape")
        centers = []
        forwards = []
        for pose in self.source_poses:
            center, forward = camera_center_and_forward(pose)
            centers.append(center)
            forwards.append(forward)
        self.source_centers = np.asarray(centers, dtype=np.float32)
        self.source_forwards = np.asarray(forwards, dtype=np.float32)

    def _full_residual(self, source, device, dtype):
        band = torch.as_tensor(
            self.residual_bands[source].astype(np.float32)
            / self.quantization_scale,
            device=device,
            dtype=dtype,
        )
        residual = torch.zeros(
            (self.image_height, self.image_width, 3),
            device=device,
            dtype=dtype,
        )
        residual[
            self.vertical_start :, : self.side_band_width
        ] = band[:, : self.side_band_width]
        residual[
            self.vertical_start :, -self.side_band_width :
        ] = band[:, self.side_band_width :]
        return residual

    def _warp_from_source(
        self,
        residual,
        source,
        target_pose,
        target_depth,
        target_intrinsics,
    ):
        """Transport a source residual through the stable map's target depth."""
        device = residual.device
        dtype = residual.dtype
        depth = torch.as_tensor(target_depth, device=device, dtype=dtype).squeeze()
        if depth.shape != (self.image_height, self.image_width):
            raise ValueError("target depth has an incompatible shape")
        intrinsics = torch.as_tensor(
            target_intrinsics, device=device, dtype=dtype
        )
        if intrinsics.shape != (3, 3):
            raise ValueError("target intrinsics must have shape [3, 3]")
        target_w2c = torch.as_tensor(target_pose, device=device, dtype=dtype)
        source_w2c = torch.as_tensor(
            self.source_poses[source], device=device, dtype=dtype
        )
        if target_w2c.shape != (4, 4):
            raise ValueError("target pose must have shape [4, 4]")

        relative = source_w2c @ torch.linalg.inv(target_w2c)
        rows = torch.arange(self.image_height, device=device, dtype=dtype) + 0.5
        columns = torch.arange(self.image_width, device=device, dtype=dtype) + 0.5
        v, u = torch.meshgrid(rows, columns, indexing="ij")
        x = (u - intrinsics[0, 2]) * depth / intrinsics[0, 0]
        y = (v - intrinsics[1, 2]) * depth / intrinsics[1, 1]

        source_x = (
            relative[0, 0] * x
            + relative[0, 1] * y
            + relative[0, 2] * depth
            + relative[0, 3]
        )
        source_y = (
            relative[1, 0] * x
            + relative[1, 1] * y
            + relative[1, 2] * depth
            + relative[1, 3]
        )
        source_z = (
            relative[2, 0] * x
            + relative[2, 1] * y
            + relative[2, 2] * depth
            + relative[2, 3]
        )
        safe_z = torch.clamp(source_z, min=torch.finfo(dtype).eps)
        source_u = intrinsics[0, 0] * source_x / safe_z + intrinsics[0, 2]
        source_v = intrinsics[1, 1] * source_y / safe_z + intrinsics[1, 2]
        grid = torch.stack(
            (
                2.0 * source_u / self.image_width - 1.0,
                2.0 * source_v / self.image_height - 1.0,
            ),
            dim=-1,
        ).unsqueeze(0)
        warped = F.grid_sample(
            residual.permute(2, 0, 1).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0].permute(1, 2, 0)
        valid = (
            torch.isfinite(depth)
            & (depth > 0.0)
            & torch.isfinite(source_z)
            & (source_z > 0.0)
            & (source_u >= 0.5)
            & (source_u <= self.image_width - 0.5)
            & (source_v >= 0.5)
            & (source_v <= self.image_height - 0.5)
        )
        return warped * valid.unsqueeze(-1)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata_raw = payload.get("metadata_json")
            metadata = (
                json.loads(str(metadata_raw.item()))
                if metadata_raw is not None
                else {}
            )
            return cls(
                residual_bands=payload["residual_bands"],
                source_frame_indices=payload["source_frame_indices"],
                source_poses=payload["source_poses"],
                image_height=int(payload["image_height"]),
                image_width=int(payload["image_width"]),
                side_band_width=int(payload["side_band_width"]),
                vertical_start=int(payload["vertical_start"]),
                quantization_scale=float(payload["quantization_scale"]),
                metadata=metadata,
            )

    @property
    def source_count(self):
        return int(len(self.source_frame_indices))

    def nearest_source(
        self, target_pose, target_frame_index=None, exclude_exact=None
    ):
        if self.source_count == 0:
            return None
        target_center, target_forward = camera_center_and_forward(target_pose)
        translation_scale = max(
            float(self.metadata.get("translation_scale", 1.0)), 1.0e-6
        )
        orientation_weight = max(
            float(self.metadata.get("orientation_weight", 2.0)), 0.0
        )
        score = np.linalg.norm(
            self.source_centers - target_center[None], axis=1
        ) / translation_scale
        score += orientation_weight * (
            1.0
            - np.clip(self.source_forwards @ target_forward, -1.0, 1.0)
        )
        if exclude_exact is None:
            exclude_exact = bool(self.metadata.get("exclude_exact_frame", False))
        if exclude_exact and target_frame_index is not None:
            score = score.copy()
            score[self.source_frame_indices == int(target_frame_index)] = np.inf
        source = int(np.argmin(score))
        return source if np.isfinite(score[source]) else None

    def residual_for_pose(
        self,
        target_pose,
        target_frame_index=None,
        exclude_exact=None,
        device="cuda:0",
        dtype=torch.float32,
        target_depth=None,
        target_intrinsics=None,
        warp_to_target=False,
    ):
        source = self.nearest_source(
            target_pose,
            target_frame_index=target_frame_index,
            exclude_exact=exclude_exact,
        )
        if source is None:
            return None
        residual = self._full_residual(source, device, dtype)
        if warp_to_target:
            if target_depth is None or target_intrinsics is None:
                raise ValueError(
                    "target depth and intrinsics are required for residual warping"
                )
            residual = self._warp_from_source(
                residual,
                source,
                target_pose,
                target_depth,
                target_intrinsics,
            )
        return residual
