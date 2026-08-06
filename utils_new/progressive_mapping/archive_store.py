"""FP16 CPU archive for inactive surface detail."""

import os
from typing import Dict, List, Optional

import torch

from .types import ArchiveDetail


class ArchiveStore:
    """Store detached surface parameters on CPU and optionally persist each root."""

    def __init__(self, archive_dir: Optional[str] = None):
        self.archive_dir = archive_dir
        self.details: Dict[int, ArchiveDetail] = {}
        if archive_dir is not None:
            os.makedirs(archive_dir, exist_ok=True)

    def archive(
        self,
        node_id: int,
        child_node_ids: List[int],
        params: Dict[str, torch.Tensor],
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        metadata: Optional[Dict[str, object]] = None,
    ) -> str:
        required = {"means", "scales", "quats", "opacities", "sh0", "shN"}
        if set(params) < required:
            raise ValueError("Archive parameters are missing {}".format(sorted(required - set(params))))
        cpu = {name: params[name].detach().to(device="cpu", dtype=torch.float16).contiguous() for name in required}
        detail = ArchiveDetail(
            node_id=node_id,
            child_node_ids=list(child_node_ids),
            means_fp16=cpu["means"],
            scales_fp16=cpu["scales"],
            quats_fp16=cpu["quats"],
            opacities_fp16=cpu["opacities"],
            sh0_fp16=cpu["sh0"],
            shN_fp16=cpu["shN"],
            parent_id=node_id,
            bbox_min=bbox_min.detach().cpu(),
            bbox_max=bbox_max.detach().cpu(),
            metadata=dict(metadata or {}),
        )
        self.details[node_id] = detail
        handle = "memory://{}".format(node_id)
        if self.archive_dir is not None:
            path = os.path.join(self.archive_dir, "root_{:08d}.pt".format(node_id))
            torch.save(detail, path)
            handle = path
        return handle

    def get(self, node_id: int) -> ArchiveDetail:
        if node_id in self.details:
            return self.details[node_id]
        if self.archive_dir is None:
            raise KeyError(node_id)
        path = os.path.join(self.archive_dir, "root_{:08d}.pt".format(node_id))
        if not os.path.exists(path):
            raise KeyError(node_id)
        detail = torch.load(path, map_location="cpu")
        self.details[node_id] = detail
        return detail

    def restore(self, node_id: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        return {
            name: value.to(device=device, dtype=dtype)
            for name, value in self.get(node_id).tensor_dict(dtype=dtype).items()
        }

    @property
    def cpu_bytes(self) -> int:
        total = 0
        for detail in self.details.values():
            for tensor in (
                detail.means_fp16,
                detail.scales_fp16,
                detail.quats_fp16,
                detail.opacities_fp16,
                detail.sh0_fp16,
                detail.shN_fp16,
            ):
                total += tensor.numel() * tensor.element_size()
        return total
