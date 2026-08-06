"""Recoverable group-level FP16 CPU archive for AeroCommit."""

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ArchivedGaussianGroup:
    archive_id: int
    source_group_id: int
    level: int
    params: Dict[str, torch.Tensor]
    bbox_min: torch.Tensor
    bbox_max: torch.Tensor
    last_seen_frame: int
    metadata: Dict[str, object]

    @property
    def count(self):
        return int(self.params["means"].shape[0])

    @property
    def bytes(self):
        return sum(value.numel() * value.element_size() for value in self.params.values())


class ArchiveStore:
    def __init__(self, archive_dir: Optional[str] = None):
        self.archive_dir = archive_dir
        self.groups: Dict[int, ArchivedGaussianGroup] = {}
        self._next_archive_id = 0
        if archive_dir:
            os.makedirs(archive_dir, exist_ok=True)

    def archive(self, source_group_id, level, params, last_seen_frame, metadata=None):
        archive_id = self._next_archive_id
        self._next_archive_id += 1
        cpu_params = {
            name: value.detach().to(device="cpu", dtype=torch.float16).contiguous()
            for name, value in params.items()
            if name in ("means", "scales", "quats", "opacities", "sh0", "shN")
        }
        means = cpu_params["means"].float()
        record = ArchivedGaussianGroup(
            archive_id=archive_id,
            source_group_id=int(source_group_id),
            level=int(level),
            params=cpu_params,
            bbox_min=means.amin(dim=0),
            bbox_max=means.amax(dim=0),
            last_seen_frame=int(last_seen_frame),
            metadata=dict(metadata or {}),
        )
        self.groups[archive_id] = record
        if self.archive_dir:
            torch.save(record, os.path.join(self.archive_dir, "group_{:06d}.pt".format(archive_id)))
        return archive_id

    def restore_params(self, archive_id, device, dtype=torch.float32):
        record = self.groups[int(archive_id)]
        return {
            name: value.to(device=device, dtype=dtype)
            for name, value in record.params.items()
        }

    def remove(self, archive_id):
        return self.groups.pop(int(archive_id))

    @property
    def cpu_bytes(self):
        return sum(record.bytes for record in self.groups.values())

    @property
    def gaussian_count(self):
        return sum(record.count for record in self.groups.values())
