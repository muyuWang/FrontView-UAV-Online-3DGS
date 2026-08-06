"""Data exchanged between the host proposal generator and AeroCommit."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


IndexLike = Optional[Union[np.ndarray, Sequence[int]]]


@dataclass
class GaussianProposalBatch:
    """Ephemeral output of MODP's proposal generator.

    The arrays intentionally live outside ``torch.nn.Parameter``. Creating a
    batch must not mutate HashBlock, a Gaussian group, or an optimizer.
    """

    source_frame_id: int
    level: int
    uv: np.ndarray
    patch_bboxes: np.ndarray
    depths: np.ndarray
    inverse_depths: np.ndarray
    world_points: np.ndarray
    log_scales: np.ndarray
    colors: np.ndarray
    residual_scores: np.ndarray
    coverage_scores: np.ndarray
    sparse_depth_valid: np.ndarray
    view_scale_size: float
    cover_sizes: Optional[np.ndarray] = None
    view_directions: Optional[np.ndarray] = None
    stable_depths: Optional[np.ndarray] = None
    depth_confidences: Optional[np.ndarray] = None
    multiview_support_scores: Optional[np.ndarray] = None
    frequency_scores: Optional[np.ndarray] = None
    track_ids: Optional[np.ndarray] = None
    source_kinds: Optional[np.ndarray] = None
    responsibility_parent_uids: Optional[np.ndarray] = None
    responsibility_levels: Optional[np.ndarray] = None
    responsibility_sectors: Optional[np.ndarray] = None
    create_new_group: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        count = int(self.world_points.shape[0])
        row_fields = (
            "uv",
            "patch_bboxes",
            "depths",
            "inverse_depths",
            "log_scales",
            "colors",
            "residual_scores",
            "coverage_scores",
            "sparse_depth_valid",
        )
        for name in row_fields:
            value = np.asarray(getattr(self, name))
            if value.shape[0] != count:
                raise ValueError(
                    "Proposal field {} has {} rows, expected {}".format(
                        name, value.shape[0], count
                    )
                )
            setattr(self, name, value)
        if self.depth_confidences is None:
            self.depth_confidences = self.sparse_depth_valid.astype(np.float32)
        else:
            self.depth_confidences = np.asarray(
                self.depth_confidences, dtype=np.float32
            )
            if self.depth_confidences.shape != (count,):
                raise ValueError("Proposal depth confidence has the wrong shape")
        if self.multiview_support_scores is None:
            self.multiview_support_scores = np.zeros((count,), dtype=np.float32)
        else:
            self.multiview_support_scores = np.asarray(
                self.multiview_support_scores, dtype=np.float32
            )
            if self.multiview_support_scores.shape != (count,):
                raise ValueError("Proposal multiview support score has the wrong shape")
            if np.any(~np.isfinite(self.multiview_support_scores)):
                raise ValueError("Proposal multiview support scores must be finite")
        if self.cover_sizes is None:
            self.cover_sizes = np.full(
                (count,), float(self.view_scale_size), dtype=np.float32
            )
        else:
            self.cover_sizes = np.asarray(self.cover_sizes, dtype=np.float32)
            if self.cover_sizes.shape != (count,):
                raise ValueError("Proposal cover sizes have the wrong shape")
            if np.any(~np.isfinite(self.cover_sizes)) or np.any(
                self.cover_sizes <= 0.0
            ):
                raise ValueError("Proposal cover sizes must be finite and positive")
        if self.view_directions is None:
            self.view_directions = np.zeros((count, 3), dtype=np.float32)
        else:
            self.view_directions = np.asarray(
                self.view_directions, dtype=np.float32
            )
            if self.view_directions.shape != (count, 3):
                raise ValueError("Proposal view directions have the wrong shape")
            if np.any(~np.isfinite(self.view_directions)):
                raise ValueError("Proposal view directions must be finite")
        if self.stable_depths is None:
            self.stable_depths = np.full((count,), np.nan, dtype=np.float32)
        else:
            self.stable_depths = np.asarray(self.stable_depths, dtype=np.float32)
            if self.stable_depths.shape != (count,):
                raise ValueError("Proposal stable depth has the wrong shape")
        if self.frequency_scores is None:
            self.frequency_scores = np.zeros((count,), dtype=np.float32)
        else:
            self.frequency_scores = np.asarray(
                self.frequency_scores, dtype=np.float32
            )
            if self.frequency_scores.shape != (count,):
                raise ValueError("Proposal frequency score has the wrong shape")
        if self.track_ids is None:
            self.track_ids = np.full((count,), -1, dtype=np.int64)
        else:
            self.track_ids = np.asarray(self.track_ids, dtype=np.int64)
            if self.track_ids.shape != (count,):
                raise ValueError("Proposal track IDs have the wrong shape")
        if self.source_kinds is None:
            self.source_kinds = np.full((count,), "unknown", dtype="U32")
        else:
            self.source_kinds = np.asarray(self.source_kinds, dtype="U32")
            if self.source_kinds.shape != (count,):
                raise ValueError("Proposal source kinds have the wrong shape")
        if self.responsibility_parent_uids is None:
            self.responsibility_parent_uids = np.full((count,), -1, dtype=np.int64)
        else:
            self.responsibility_parent_uids = np.asarray(
                self.responsibility_parent_uids, dtype=np.int64
            )
            if self.responsibility_parent_uids.shape != (count,):
                raise ValueError("Proposal responsibility parent UIDs have the wrong shape")
        if self.responsibility_levels is None:
            self.responsibility_levels = np.zeros((count,), dtype=np.int16)
        else:
            self.responsibility_levels = np.asarray(
                self.responsibility_levels, dtype=np.int16
            )
            if self.responsibility_levels.shape != (count,):
                raise ValueError("Proposal responsibility levels have the wrong shape")
        if self.responsibility_sectors is None:
            self.responsibility_sectors = np.full((count,), -1, dtype=np.int16)
        else:
            self.responsibility_sectors = np.asarray(
                self.responsibility_sectors, dtype=np.int16
            )
            if self.responsibility_sectors.shape != (count,):
                raise ValueError("Proposal responsibility sectors have the wrong shape")
        self.world_points = np.asarray(self.world_points)

    def __len__(self) -> int:
        return int(self.world_points.shape[0])

    def select(self, indices: IndexLike) -> "GaussianProposalBatch":
        if indices is None:
            selection = np.arange(len(self), dtype=np.int64)
        else:
            selection = np.asarray(indices)
            if selection.dtype == np.bool_:
                if selection.shape != (len(self),):
                    raise ValueError("Boolean proposal selection has the wrong shape")
            else:
                selection = selection.astype(np.int64, copy=False)
        return GaussianProposalBatch(
            source_frame_id=self.source_frame_id,
            level=self.level,
            uv=self.uv[selection].copy(),
            patch_bboxes=self.patch_bboxes[selection].copy(),
            depths=self.depths[selection].copy(),
            inverse_depths=self.inverse_depths[selection].copy(),
            world_points=self.world_points[selection].copy(),
            log_scales=self.log_scales[selection].copy(),
            colors=self.colors[selection].copy(),
            residual_scores=self.residual_scores[selection].copy(),
            coverage_scores=self.coverage_scores[selection].copy(),
            sparse_depth_valid=self.sparse_depth_valid[selection].copy(),
            cover_sizes=self.cover_sizes[selection].copy(),
            view_directions=self.view_directions[selection].copy(),
            stable_depths=self.stable_depths[selection].copy(),
            depth_confidences=self.depth_confidences[selection].copy(),
            multiview_support_scores=self.multiview_support_scores[selection].copy(),
            frequency_scores=self.frequency_scores[selection].copy(),
            track_ids=self.track_ids[selection].copy(),
            source_kinds=self.source_kinds[selection].copy(),
            responsibility_parent_uids=self.responsibility_parent_uids[selection].copy(),
            responsibility_levels=self.responsibility_levels[selection].copy(),
            responsibility_sectors=self.responsibility_sectors[selection].copy(),
            view_scale_size=self.view_scale_size,
            create_new_group=self.create_new_group,
            metadata=dict(self.metadata),
        )

    @classmethod
    def concatenate(
        cls, batches: Sequence["GaussianProposalBatch"], source_frame_id: int
    ) -> "GaussianProposalBatch":
        batches = [batch for batch in batches if len(batch) > 0]
        if not batches:
            raise ValueError("At least one non-empty proposal batch is required")
        level = batches[0].level
        if any(batch.level != level for batch in batches):
            raise ValueError("Cannot concatenate proposal batches from different levels")
        fields = (
            "uv",
            "patch_bboxes",
            "depths",
            "inverse_depths",
            "world_points",
            "log_scales",
            "colors",
            "residual_scores",
            "coverage_scores",
            "sparse_depth_valid",
            "cover_sizes",
            "view_directions",
            "stable_depths",
            "depth_confidences",
            "multiview_support_scores",
            "frequency_scores",
            "track_ids",
            "source_kinds",
            "responsibility_parent_uids",
            "responsibility_levels",
            "responsibility_sectors",
        )
        values = {
            name: np.concatenate([getattr(batch, name) for batch in batches], axis=0)
            for name in fields
        }
        return cls(
            source_frame_id=int(source_frame_id),
            level=level,
            view_scale_size=float(np.median([batch.view_scale_size for batch in batches])),
            create_new_group=any(batch.create_new_group for batch in batches),
            metadata={"source_batches": len(batches)},
            **values,
        )


class CandidateStatus(str, Enum):
    WAITING = "WAITING"
    READY_FOR_RISK = "READY_FOR_RISK"
    COMMITTED = "COMMITTED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class SupportEdge:
    frame_id: int
    world_to_camera: np.ndarray
    intrinsics: np.ndarray
    uv: np.ndarray
    descriptor: np.ndarray
    gray_patch: np.ndarray
    world_point: np.ndarray
    depth: float
    log_scale: float
    color: np.ndarray
    association_error: float
    photometric_residual: float
    parallax_rad: float
    pose_covariance: np.ndarray
    robust_weight: float = 1.0
    linearization_rho: float = 0.0
    pose_version: int = 0


@dataclass
class CandidateRecord:
    candidate_id: int
    reference_frame_id: int
    reference_pose: np.ndarray
    reference_K: np.ndarray
    reference_uv: np.ndarray
    patch_bbox: np.ndarray
    reference_gray_patch: np.ndarray
    reference_descriptor: np.ndarray
    mean_color: np.ndarray
    rho_mean: float
    rho_variance: float
    depth_prior: float
    created_frame: int
    last_seen_frame: int
    proposal_batch: GaussianProposalBatch
    original_proposal_batch: Optional[GaussianProposalBatch] = None
    support_edges: List[SupportEdge] = field(default_factory=list)
    association_error_ema: float = 0.0
    residual_mad_ema: float = 0.0
    stable_residual_ema: float = 0.0
    parallax_max_rad: float = 0.0
    coverage_score: float = 0.0
    priority_score: float = 0.0
    lateral_score: float = 0.0
    representative_world_point: Optional[np.ndarray] = None
    fused_proposal_count: int = 0
    observation_count: int = 1
    status: CandidateStatus = CandidateStatus.WAITING
    last_risk: float = float("inf")
    last_information: float = 0.0
    refined_rho: Optional[float] = None
    commit_loss_before: Optional[float] = None
    commit_loss_after: Optional[float] = None

    @property
    def age(self) -> int:
        return max(0, self.last_seen_frame - self.created_frame)

    @property
    def support_count(self) -> int:
        return len(self.support_edges)

    @property
    def frequency_score(self) -> float:
        if len(self.proposal_batch.frequency_scores) == 0:
            return 0.0
        return float(np.mean(self.proposal_batch.frequency_scores))


@dataclass
class RiskResult:
    candidate_ids: np.ndarray
    commitment_risk: np.ndarray
    information: np.ndarray
    residual_sigma: np.ndarray
    pose_projected_uncertainty: np.ndarray


@dataclass
class AeroCommitFrameStats:
    frame_id: int
    frame_total_ms: float = 0.0
    proposal_ms: float = 0.0
    candidate_association_ms: float = 0.0
    risk_gate_ms: float = 0.0
    commit_refinement_ms: float = 0.0
    detail_refinement_ms: float = 0.0
    archive_transfer_ms: float = 0.0
    num_raw_proposals: int = 0
    num_proposal_groups: int = 0
    num_candidate_matches: int = 0
    num_new_candidates: int = 0
    num_waiting_candidates: int = 0
    num_risk_evaluations: int = 0
    num_committed_candidates: int = 0
    num_fast_path_gaussians: int = 0
    num_depth_confidence_fast_path_gaussians: int = 0
    num_frequency_deferred_gaussians: int = 0
    num_frequency_probation_gaussians: int = 0
    num_filtered_depthcov_candidates: int = 0
    num_committed_gaussians: int = 0
    num_rejected: int = 0
    num_expired: int = 0
    num_detail_splits: int = 0
    num_side_detail_splits: int = 0
    num_fused_proposals: int = 0
    depth_correction_abs_rel_mean: float = 0.0
    depth_correction_abs_rel_p95: float = 0.0
    risk_histogram_edges: List[float] = field(default_factory=list)
    risk_histogram_counts: List[int] = field(default_factory=list)
    risk_min: float = 0.0
    risk_median: float = 0.0
    risk_p95: float = 0.0
    risk_max: float = 0.0
    num_active_gaussians: int = 0
    num_trainable_gaussians: int = 0
    num_archived_gaussians: int = 0
    parameter_bytes: int = 0
    gradient_bytes: int = 0
    optimizer_bytes: int = 0
    candidate_bytes: int = 0
    archive_cpu_bytes: int = 0
    cuda_memory_allocated: int = 0
    cuda_memory_reserved: int = 0
    cuda_peak_allocated: int = 0


@dataclass(frozen=True)
class CommitResult:
    """Result of the only operation allowed to mutate the permanent map."""

    source_frame_id: int
    proposed: int
    selected: int
    committed: int
    group_id: Optional[int]
    committed_indices: np.ndarray
