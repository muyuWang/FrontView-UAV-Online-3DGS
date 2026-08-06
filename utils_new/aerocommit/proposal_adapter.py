"""Convert dense host proposal batches into bounded local candidate groups."""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .types import GaussianProposalBatch


class PatchDescriptorExtractor:
    def __init__(self, patch_size: int = 8, descriptor_resize: int = 4):
        self.patch_size = int(patch_size)
        self.descriptor_resize = int(descriptor_resize)

    def describe(self, image: torch.Tensor, uvs: np.ndarray):
        if len(uvs) == 0:
            size = self.descriptor_resize**2 * 3
            return (
                np.empty((0, size), dtype=np.float32),
                np.empty((0, self.descriptor_resize, self.descriptor_resize), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Expected an HxWx3 image")
        device = image.device
        centers = torch.as_tensor(uvs, device=device, dtype=torch.float32)
        height, width = image.shape[:2]
        half = self.patch_size // 2
        x0 = torch.clamp(
            torch.round(centers[:, 0]).long() - half,
            min=0,
            max=max(0, width - self.patch_size),
        )
        y0 = torch.clamp(
            torch.round(centers[:, 1]).long() - half,
            min=0,
            max=max(0, height - self.patch_size),
        )
        offsets = torch.arange(self.patch_size, device=device)
        xs = x0[:, None] + offsets[None, :]
        ys = y0[:, None] + offsets[None, :]
        patches = image[ys[:, :, None], xs[:, None, :]].permute(0, 3, 1, 2)
        resized = F.interpolate(
            patches,
            size=(self.descriptor_resize, self.descriptor_resize),
            mode="bilinear",
            align_corners=False,
        )
        flat = resized.permute(0, 2, 3, 1).reshape(len(uvs), -1)
        descriptors = flat - flat.mean(dim=1, keepdim=True)
        descriptors = descriptors / torch.clamp(
            torch.linalg.norm(descriptors, dim=1, keepdim=True), min=1.0e-8
        )
        gray = resized.mean(dim=1)
        colors = patches.mean(dim=(2, 3))
        return (
            descriptors.detach().cpu().numpy().astype(np.float32, copy=False),
            gray.detach().cpu().numpy().astype(np.float32, copy=False),
            colors.detach().cpu().numpy().astype(np.float32, copy=False),
        )


def _group_priority(batch: GaussianProposalBatch, indices: np.ndarray, width: int, config):
    center_u = float(np.median(batch.uv[indices, 0]))
    lateral = abs(center_u - 0.5 * width) / max(0.5 * width, 1.0)
    coverage = float(np.mean(batch.coverage_scores[indices]))
    residual = float(np.mean(batch.residual_scores[indices]))
    sparse_fraction = float(np.mean(batch.sparse_depth_valid[indices]))
    priority = coverage + residual + 0.20 * sparse_fraction
    priority += float(config["side_priority_weight"]) * lateral
    return priority, lateral


def group_host_proposals(
    batch: GaussianProposalBatch, image_width: int, config: Dict[str, object]
) -> List[Tuple[GaussianProposalBatch, float, float]]:
    """Group image/depth-local proposals and reserve capacity for side bands."""
    if len(batch) == 0:
        return []
    cell = float(config["candidate_group_cell_px"])
    depth_bin = float(config["candidate_group_log_depth_bin"])
    log_depth = np.log(np.maximum(batch.depths, 1.0e-8))
    keys = np.stack(
        (
            np.floor(batch.uv[:, 0] / cell),
            np.floor(batch.uv[:, 1] / cell),
            np.floor(log_depth / depth_bin),
        ),
        axis=1,
    ).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    starts = np.concatenate(
        ([0], np.flatnonzero(np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)) + 1)
    )
    ends = np.concatenate((starts[1:], [len(order)]))
    max_rows = int(config["max_proposals_per_candidate"])
    grouped_indices = []
    for start, end in zip(starts, ends):
        indices = order[start:end]
        if len(indices) > max_rows:
            local_score = (
                batch.coverage_scores[indices]
                + batch.residual_scores[indices]
                + 0.10 * batch.sparse_depth_valid[indices]
            )
            indices = indices[np.argsort(local_score)[-max_rows:]]
        priority, lateral = _group_priority(batch, indices, image_width, config)
        grouped_indices.append((indices, priority, lateral))

    limit = int(config["max_candidate_groups_per_frame"])
    if len(grouped_indices) <= limit:
        selected = sorted(grouped_indices, key=lambda item: item[1], reverse=True)
        return [(batch.select(item[0]), item[1], item[2]) for item in selected]
    side_start = float(config["side_band_start"])
    side = [item for item in grouped_indices if item[2] >= side_start]
    center = [item for item in grouped_indices if item[2] < side_start]
    side.sort(key=lambda item: item[1], reverse=True)
    center.sort(key=lambda item: item[1], reverse=True)
    side_quota = min(len(side), int(round(limit * float(config["side_quota_fraction"]))))
    selected = side[:side_quota] + center[: limit - side_quota]
    if len(selected) < limit:
        selected += side[side_quota : side_quota + limit - len(selected)]
    selected.sort(key=lambda item: item[1], reverse=True)
    return [(batch.select(item[0]), item[1], item[2]) for item in selected]


def representative_uvs(groups: Sequence[Tuple[GaussianProposalBatch, float, float]]):
    return np.stack(
        [np.median(group[0].uv, axis=0) for group in groups], axis=0
    ).astype(np.float32)
