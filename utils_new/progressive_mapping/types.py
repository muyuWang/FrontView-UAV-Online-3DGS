"""Lightweight state records used by progressive mapping."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch


class NodeState(str, Enum):
    """The four states in the single-level progressive hierarchy."""

    PROJECTIVE = "PROJECTIVE"
    METRIC = "METRIC"
    SURFACE = "SURFACE"
    ARCHIVED = "ARCHIVED"


@dataclass
class Observation:
    frame_id: int
    uv: torch.Tensor
    patch_bbox: torch.Tensor
    descriptor: torch.Tensor
    mean_color: torch.Tensor
    appearance_grid: torch.Tensor
    depth_prior: float
    depth_valid: bool
    depth_uncertainty: float
    gradient_score: float
    residual_score: float
    depth_support: int = 0


@dataclass
class ProjectiveAnchor:
    anchor_id: int
    reference_frame_id: int
    reference_pose: torch.Tensor
    reference_intrinsics: torch.Tensor
    uv: torch.Tensor
    patch_size_px: torch.Tensor
    descriptor: torch.Tensor
    mean_color: torch.Tensor
    appearance_grid: torch.Tensor
    inverse_depth_modes: torch.Tensor
    mode_log_weights: torch.Tensor
    reference_depth_prior: float = 0.0
    reference_depth_valid: bool = False
    reference_depth_uncertainty: float = float("inf")
    reference_surface_normal: Optional[torch.Tensor] = None
    reference_surface_confidence: float = 0.0
    reference_surface_support: int = 0
    observation_count: int = 1
    valid_update_count: int = 0
    last_seen_frame: int = 0
    max_parallax_rad: float = 0.0
    # A new anchor has not earned static confidence yet. The pruning grace
    # period gives it time to accumulate observations before quality ranking.
    static_confidence: float = 0.0
    posterior_mean: float = 0.0
    posterior_variance: float = 0.0
    posterior_entropy: float = 1.0
    best_error_ema: float = 1.0
    commitment_score: float = 0.0


@dataclass
class GaussianTreeNode:
    node_id: int
    state: NodeState
    parent_id: Optional[int]
    children_ids: List[int]
    source_anchor_ids: List[int]
    world_center: torch.Tensor
    world_bbox_min: torch.Tensor
    world_bbox_max: torch.Tensor
    metric_gaussian_id: Optional[int]
    surface_gaussian_ids: List[int]
    archive_handle: Optional[str]
    observation_count: int
    last_seen_frame: int
    residual_ema: float
    projected_radius_ema: float
    confidence: float
    root_rotation: torch.Tensor
    root_scale: torch.Tensor
    root_color: torch.Tensor
    appearance_grid: Optional[torch.Tensor]
    descriptor: torch.Tensor
    support_keyframes: List[int] = field(default_factory=list)
    metric_depth_update_count: int = 0
    metric_depth_reject_count: int = 0
    metric_center_innovation_ema: float = 0.0


@dataclass
class ArchiveDetail:
    node_id: int
    child_node_ids: List[int]
    means_fp16: torch.Tensor
    scales_fp16: torch.Tensor
    quats_fp16: torch.Tensor
    opacities_fp16: torch.Tensor
    sh0_fp16: torch.Tensor
    shN_fp16: torch.Tensor
    parent_id: Optional[int]
    bbox_min: torch.Tensor
    bbox_max: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    def tensor_dict(self, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
        """Return raw Gaussian parameters in the model parameter domain."""
        return {
            "means": self.means_fp16.to(dtype=dtype),
            "scales": self.scales_fp16.to(dtype=dtype),
            "quats": self.quats_fp16.to(dtype=dtype),
            "opacities": self.opacities_fp16.to(dtype=dtype),
            "sh0": self.sh0_fp16.to(dtype=dtype),
            "shN": self.shN_fp16.to(dtype=dtype),
        }


@dataclass
class ProgressiveFrameStats:
    frame_id: int
    num_observations: int = 0
    num_near_observations: int = 0
    num_observations_matched_to_P: int = 0
    num_new_P: int = 0
    num_near_new_P: int = 0
    num_new_P_with_plane: int = 0
    num_pruned_P: int = 0
    num_active_P: int = 0
    num_promoted_P_to_M: int = 0
    num_promoted_plane_P_to_M: int = 0
    num_merged_into_existing_M: int = 0
    num_metric_depth_corrections: int = 0
    num_metric_depth_correction_rejections: int = 0
    num_metric_waiting_for_depth: int = 0
    num_active_M: int = 0
    num_refined_M_to_S: int = 0
    num_new_surface_gaussians: int = 0
    num_active_S: int = 0
    num_archived_S_to_A: int = 0
    num_active_A: int = 0
    num_reactivated_A: int = 0
    num_proxy_splats: int = 0
    num_total_active_gaussians: int = 0
    num_clamped_S_scales: int = 0
    num_clamped_S_opacities: int = 0
    num_visible_roots: int = 0
    num_optimized_roots: int = 0
    num_frozen_roots: int = 0
    progressive_process_seconds: float = 0.0
    gpu_memory_allocated: int = 0
    gpu_memory_reserved: int = 0
    cpu_archive_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
