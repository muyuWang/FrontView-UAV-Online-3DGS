"""End-to-end AeroCommit proposal admission, refinement, and map control."""

import math
import os
import time
from typing import Dict, List

import numpy as np
import torch

from utils_new.logging_utils import Log

from .active_budget import ActiveBudgetManager
from .admission_policy import AdmissionPolicy
from .archive_store import ArchiveStore
from .association import project_world
from .candidate_bank import CandidateBank
from .commit_refiner import CommitRefiner
from .detail_refiner import DetailRefiner
from .metrics import MetricsRecorder
from .npo_lite import NPOLiteEvaluator
from .pose_uncertainty import PoseUncertaintyProvider
from .proposal_adapter import (
    PatchDescriptorExtractor,
    group_host_proposals,
    representative_uvs,
)
from .types import (
    AeroCommitFrameStats,
    CandidateStatus,
    GaussianProposalBatch,
)


def split_fast_path_masks(
    proposals,
    config,
    use_sparse_fast_path,
    use_depthcov_fast_path,
):
    """Split proposals into permanent, reversible-impact, and deferred paths."""

    count = len(proposals)
    sparse = proposals.sparse_depth_valid.astype(np.bool_)
    confidence = proposals.depth_confidences
    high_frequency = proposals.frequency_scores > float(
        config["frequency_gate_score_threshold"]
    )
    base_threshold = float(config["trusted_depth_confidence_threshold"])
    frequency_threshold = float(
        config["trusted_frequency_depth_confidence_threshold"]
    )

    trusted = np.zeros((count,), dtype=np.bool_)
    if use_sparse_fast_path:
        sparse_deferred = (
            high_frequency
            if config["frequency_gate_enabled"] and config["frequency_gate_sparse"]
            else np.zeros((count,), dtype=np.bool_)
        )
        trusted |= np.logical_and(sparse, ~sparse_deferred)

    base_depth_trusted = np.logical_and(~sparse, confidence >= base_threshold)
    depth_trusted = np.zeros((count,), dtype=np.bool_)
    if use_depthcov_fast_path:
        if config["frequency_gate_enabled"]:
            per_proposal_threshold = np.where(
                high_frequency,
                frequency_threshold,
                base_threshold,
            )
            depth_trusted = np.logical_and(
                ~sparse, confidence >= per_proposal_threshold
            )
        else:
            depth_trusted = base_depth_trusted.copy()
        trusted |= depth_trusted

    probation = np.zeros((count,), dtype=np.bool_)
    if config["frequency_probation_enabled"] and use_depthcov_fast_path:
        probation = np.logical_and.reduce(
            (
                high_frequency,
                base_depth_trusted,
                confidence < frequency_threshold,
            )
        )
        trusted &= ~probation
        depth_trusted &= ~probation

    base_trusted = np.logical_or(
        np.logical_and(sparse, use_sparse_fast_path),
        np.logical_and(base_depth_trusted, use_depthcov_fast_path),
    )
    deferred = np.logical_and(base_trusted, ~(trusted | probation))
    return trusted, depth_trusted, probation, deferred


def frequency_probation_opacities(proposals, config):
    """Map depth confidence to bounded initial influence for probation points."""

    minimum = float(config["frequency_probation_initial_opacity"])
    maximum = float(config["frequency_probation_max_opacity"])
    lower_confidence = float(config["trusted_depth_confidence_threshold"])
    upper_confidence = float(
        config["trusted_frequency_depth_confidence_threshold"]
    )
    span = max(upper_confidence - lower_confidence, 1.0e-8)
    responsibility = np.clip(
        (proposals.depth_confidences - lower_confidence) / span, 0.0, 1.0
    )
    return (minimum + responsibility * (maximum - minimum)).astype(np.float32)


def split_candidate_masks(
    proposals, excluded, allow_depthcov_candidates, stable_depth_ratio=0.0
):
    """Select proposals allowed to enter the reversible candidate bank."""

    candidate = ~np.asarray(excluded, dtype=np.bool_)
    filtered_depthcov = np.zeros((len(proposals),), dtype=np.bool_)
    if not allow_depthcov_candidates:
        filtered_depthcov = np.logical_and(
            candidate, ~proposals.sparse_depth_valid.astype(np.bool_)
        )
        candidate &= proposals.sparse_depth_valid.astype(np.bool_)
    elif float(stable_depth_ratio) > 0.0:
        stable_depths = np.asarray(proposals.stable_depths, dtype=np.float32)
        depthcov = ~proposals.sparse_depth_valid.astype(np.bool_)
        stable_valid = np.isfinite(stable_depths) & (stable_depths > 0.0)
        relative_error = np.abs(proposals.depths - stable_depths) / np.maximum(
            proposals.depths, 1.0e-8
        )
        inconsistent = np.logical_and.reduce(
            (
                candidate,
                depthcov,
                stable_valid,
                relative_error > float(stable_depth_ratio),
            )
        )
        filtered_depthcov |= inconsistent
        candidate &= ~inconsistent
    return candidate, filtered_depthcov


def select_budgeted_fast_path_indices(proposals, budget, frequency_fraction):
    """Retain high-frequency evidence while preserving deterministic coverage."""
    count = len(proposals)
    budget = int(budget)
    if budget <= 0 or count <= budget:
        return np.arange(count, dtype=np.int64)

    priority_count = min(
        budget,
        max(0, int(round(budget * float(frequency_fraction)))),
    )
    priority_score = (
        proposals.frequency_scores.astype(np.float64)
        + 0.5 * proposals.residual_scores.astype(np.float64)
    )
    ranked = np.argsort(-priority_score, kind="stable")
    priority = ranked[:priority_count]

    remaining_mask = np.ones((count,), dtype=np.bool_)
    remaining_mask[priority] = False
    remaining = np.flatnonzero(remaining_mask)
    coverage_count = budget - len(priority)
    if coverage_count > 0:
        positions = np.linspace(
            0, len(remaining) - 1, coverage_count, dtype=np.int64
        )
        selected = np.concatenate((priority, remaining[positions]))
    else:
        selected = priority
    return np.sort(selected.astype(np.int64, copy=False))


def select_budgeted_bootstrap_indices(proposals, budget, frequency_fraction):
    """Bound bootstrap growth while preferring measured sparse geometry."""
    count = len(proposals)
    budget = int(budget)
    if budget <= 0 or count <= budget:
        return np.arange(count, dtype=np.int64)

    sparse_indices = np.flatnonzero(proposals.sparse_depth_valid)
    if len(sparse_indices) >= budget:
        selected = select_budgeted_fast_path_indices(
            proposals.select(sparse_indices), budget, frequency_fraction
        )
        return np.sort(sparse_indices[selected])

    remaining_budget = budget - len(sparse_indices)
    dense_indices = np.flatnonzero(~proposals.sparse_depth_valid)
    selected_dense = select_budgeted_fast_path_indices(
        proposals.select(dense_indices), remaining_budget, frequency_fraction
    )
    return np.sort(np.concatenate((sparse_indices, dense_indices[selected_dense])))


class AeroCommitManager:
    def __init__(self, config, gaussian_model, output_dir, device="cuda:0"):
        self.config = config
        self.gaussian_model = gaussian_model
        self.output_dir = output_dir
        self.device = device
        admission = config["admission"]
        self.pose_uncertainty = PoseUncertaintyProvider(admission)
        self.candidate_bank = CandidateBank(admission, self.pose_uncertainty)
        self.descriptor_extractor = PatchDescriptorExtractor(
            admission["patch_size"], admission["descriptor_resize"]
        )
        self.npo_evaluator = NPOLiteEvaluator(admission, device=device)
        self.admission_policy = AdmissionPolicy(admission, self.npo_evaluator)
        self.commit_refiner = CommitRefiner(
            config["commit_refinement"], device=device
        )
        self.detail_refiner = DetailRefiner(config["detail_refinement"])
        self.budget_manager = ActiveBudgetManager(config["budget"])
        archive_dir = config["archive"]["archive_directory"]
        if archive_dir is None:
            archive_dir = os.path.join(output_dir, "aerocommit_archive")
        self.archive_store = ArchiveStore(archive_dir if config["archive"]["enabled"] else None)
        self.metrics = MetricsRecorder(output_dir)
        self.active_group_metadata: Dict[int, Dict[str, object]] = {}
        self.restored_group_to_archive: Dict[int, int] = {}
        self.total_risk_evaluations = 0
        self.total_committed_candidates = 0
        self.total_committed_gaussians = 0
        self._candidate_commit_group_id = None
        self._candidate_commit_group_events = 0
        self._fast_path_group_id = None
        self._fast_path_group_events = 0

    @property
    def mode(self):
        return self.config["mode"]

    def _record_group(self, group_id, batch, frame_id):
        if group_id is None or len(batch) == 0:
            return
        group_id = int(group_id)
        bbox_min = np.min(batch.world_points, axis=0).astype(np.float32)
        bbox_max = np.max(batch.world_points, axis=0).astype(np.float32)
        existing = self.active_group_metadata.get(group_id)
        if existing is not None:
            existing["bbox_min"] = np.minimum(existing["bbox_min"], bbox_min)
            existing["bbox_max"] = np.maximum(existing["bbox_max"], bbox_max)
            existing["last_seen_frame"] = int(frame_id)
            return
        self.active_group_metadata[group_id] = {
            "level": int(batch.level),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "last_seen_frame": int(frame_id),
            "created_frame": int(frame_id),
        }

    def _commit_fast_path(self, batch, frame_id, initial_opacity):
        group_frames = int(
            self.config["admission"]["fast_path_group_frames"]
        )
        if group_frames <= 0:
            return self.gaussian_model.commit_proposals(
                batch,
                initial_opacity=initial_opacity,
                target_group_id=0,
            )

        target_group_id = None
        force_new_group = True
        if (
            self._fast_path_group_id in self.gaussian_model.valid_groups
            and self._fast_path_group_events < group_frames
        ):
            target_group_id = self._fast_path_group_id
            force_new_group = False
        result = self.gaussian_model.commit_proposals(
            batch,
            initial_opacity=initial_opacity,
            force_new_group=force_new_group,
            target_group_id=target_group_id,
        )
        if result.group_id is not None:
            if target_group_id is None:
                self._fast_path_group_id = result.group_id
                self._fast_path_group_events = 1
            else:
                self._fast_path_group_events += 1
            self._record_group(result.group_id, batch, frame_id)
        return result

    def _update_group_visibility(self, cam, frame_id):
        if not self.active_group_metadata:
            return
        pose = cam.get_raw_pose().detach().cpu().numpy()
        K = cam.get_int_mat(0).detach().cpu().numpy()
        width, height = cam.get_width(0), cam.get_height(0)
        group_ids = list(self.active_group_metadata)
        centers = np.stack(
            [
                0.5
                * (
                    self.active_group_metadata[group_id]["bbox_min"]
                    + self.active_group_metadata[group_id]["bbox_max"]
                )
                for group_id in group_ids
            ],
            axis=0,
        )
        uv, depth = project_world(centers, pose, K)
        for group_id, point, point_depth in zip(group_ids, uv, depth):
            visible = (
                point_depth > cam.near
                and point_depth < cam.far
                and -32 <= point[0] < width + 32
                and -32 <= point[1] < height + 32
            )
            if visible:
                self.active_group_metadata[group_id]["last_seen_frame"] = int(frame_id)

    def _reactivate_visible_archives(self, cam, frame_id):
        if not (
            self.mode == "aerocommit_mvp"
            and self.config["archive"]["enabled"]
            and self.config["archive"]["enable_reactivation"]
        ):
            return 0, 0.0
        start = time.perf_counter()
        pose = cam.get_raw_pose().detach().cpu().numpy()
        K = cam.get_int_mat(0).detach().cpu().numpy()
        width, height = cam.get_width(0), cam.get_height(0)
        restored = 0
        for archive_id, record in list(self.archive_store.groups.items()):
            center = (0.5 * (record.bbox_min + record.bbox_max)).numpy()[None]
            uv, depth = project_world(center, pose, K)
            if not (
                depth[0] > cam.near
                and depth[0] < cam.far
                and 0 <= uv[0, 0] < width
                and 0 <= uv[0, 1] < height
            ):
                continue
            params = self.archive_store.restore_params(
                archive_id, self.gaussian_model.device
            )
            authority = getattr(
                self.gaussian_model, "worldtest_certificate_authority", None
            )
            admission_certificate = None
            if authority is not None:
                certificate_ids = record.metadata.get("certificate_ids", [])
                if not certificate_ids:
                    authority.bypass_count += 1
                    raise RuntimeError(
                        "Archived WorldTest group has no admission provenance"
                    )
                admission_certificate = authority.issued[certificate_ids[0]]
            group_id = self.gaussian_model.restore_gaussian_group(
                params,
                level=record.level,
                optimize=True,
                admission_certificate=admission_certificate,
            )
            self.active_group_metadata[group_id] = {
                "level": record.level,
                "bbox_min": record.bbox_min.numpy(),
                "bbox_max": record.bbox_max.numpy(),
                "last_seen_frame": int(frame_id),
                "created_frame": int(frame_id),
            }
            self.archive_store.remove(archive_id)
            restored += record.count
        return restored, (time.perf_counter() - start) * 1000.0

    def restore_all_archives_for_refinement(self, freeze_geometry=True):
        """Materialize the full map so archived appearance participates in replay.

        Online archiving removes old groups and their optimizer state. Final
        trajectory replay must restore those groups before constructing fresh
        optimizers; otherwise only the active tail is optimized and the archived
        majority is merged back after optimization has already finished.
        """

        start = time.perf_counter()
        restored_groups = 0
        restored_gaussians = 0
        for archive_id, record in list(self.archive_store.groups.items()):
            params = self.archive_store.restore_params(
                archive_id, self.gaussian_model.device
            )
            authority = getattr(
                self.gaussian_model, "worldtest_certificate_authority", None
            )
            admission_certificate = None
            if authority is not None:
                certificate_ids = record.metadata.get("certificate_ids", [])
                if not certificate_ids:
                    authority.bypass_count += 1
                    raise RuntimeError(
                        "Archived WorldTest group has no admission provenance"
                    )
                admission_certificate = authority.issued[certificate_ids[0]]
            group_id = self.gaussian_model.restore_gaussian_group(
                params,
                level=record.level,
                optimize=True,
                admission_certificate=admission_certificate,
            )
            if freeze_geometry:
                self.gaussian_model.freeze_group_geometry(group_id)
            self.active_group_metadata[group_id] = {
                "level": record.level,
                "bbox_min": record.bbox_min.numpy(),
                "bbox_max": record.bbox_max.numpy(),
                "last_seen_frame": int(record.last_seen_frame),
                "created_frame": int(record.metadata.get("created_frame", 0)),
            }
            restored_groups += 1
            restored_gaussians += record.count
            self.archive_store.remove(archive_id)
        return {
            "groups": restored_groups,
            "gaussians": restored_gaussians,
            "seconds": time.perf_counter() - start,
        }

    def _enforce_budget(self, cam, frame_id):
        if not (
            self.mode == "aerocommit_mvp" and self.config["budget"]["enabled"]
        ):
            return 0.0
        start = time.perf_counter()
        self._update_group_visibility(cam, frame_id)
        memory = self.budget_manager.measure(self.gaussian_model)
        current_group_ids = set(self.gaussian_model.current_gaussian_group.values())
        archive_after = int(self.config["budget"]["archive_after_unseen_frames"])
        candidates = []
        for group_id, metadata in self.active_group_metadata.items():
            if group_id in current_group_ids or group_id not in self.gaussian_model.valid_groups:
                continue
            unseen = frame_id - int(metadata["last_seen_frame"])
            timed_out = unseen >= archive_after
            if timed_out or self.budget_manager.over_budget(
                self.gaussian_model.get_num_gaussians, memory
            ):
                candidates.append((unseen, group_id))
        candidates.sort(reverse=True)
        for _, group_id in candidates:
            metadata = self.active_group_metadata[group_id]
            params = self.gaussian_model.export_group(group_id)
            authority = getattr(
                self.gaussian_model, "worldtest_certificate_authority", None
            )
            certificate_ids = sorted(
                self.gaussian_model.worldtest_group_certificates.get(group_id, ())
            )
            if authority is not None and not certificate_ids:
                authority.bypass_count += 1
                raise RuntimeError(
                    "WorldTest archive refuses an uncertified active group"
                )
            self.archive_store.archive(
                group_id,
                metadata["level"],
                params,
                metadata["last_seen_frame"],
                metadata={
                    "created_frame": metadata["created_frame"],
                    "certificate_ids": certificate_ids,
                },
            )
            self.gaussian_model.remove_group(group_id, metadata["level"])
            del self.active_group_metadata[group_id]
            memory = self.budget_manager.measure(self.gaussian_model)
            if not self.budget_manager.over_budget(
                self.gaussian_model.get_num_gaussians, memory
            ):
                break
        if self.budget_manager.over_budget(
            self.gaussian_model.get_num_gaussians, memory
        ):
            Log(
                "Active trainable map remains over budget after archive candidates were exhausted",
                tag="AeroCommit",
            )
        return (time.perf_counter() - start) * 1000.0

    def _finalize_stats(self, stats, start_time):
        memory = self.budget_manager.measure(self.gaussian_model)
        stats.num_waiting_candidates = self.candidate_bank.waiting_count
        stats.num_active_gaussians = self.gaussian_model.get_num_gaussians
        stats.num_trainable_gaussians = stats.num_active_gaussians
        stats.num_archived_gaussians = self.archive_store.gaussian_count
        stats.parameter_bytes = memory.parameter_bytes
        stats.gradient_bytes = memory.gradient_bytes
        stats.optimizer_bytes = memory.optimizer_bytes
        stats.candidate_bytes = self.candidate_bank.estimated_bytes
        stats.archive_cpu_bytes = self.archive_store.cpu_bytes
        if torch.cuda.is_available():
            stats.cuda_memory_allocated = torch.cuda.memory_allocated(self.device)
            stats.cuda_memory_reserved = torch.cuda.memory_reserved(self.device)
            stats.cuda_peak_allocated = torch.cuda.max_memory_allocated(self.device)
        stats.frame_total_ms = (time.perf_counter() - start_time) * 1000.0
        self.metrics.record(stats)
        return stats

    def process_proposals(
        self, cam, proposals: GaussianProposalBatch, is_key_frame: bool, proposal_ms=0.0
    ):
        start = time.perf_counter()
        frame_id = int(cam.cam_idx)
        stats = AeroCommitFrameStats(
            frame_id=frame_id,
            proposal_ms=float(proposal_ms),
            num_raw_proposals=len(proposals),
        )
        _, reactivation_ms = self._reactivate_visible_archives(cam, frame_id)
        stats.archive_transfer_ms += reactivation_ms
        if self.mode == "baseline":
            result = self.gaussian_model.commit_proposals(proposals)
            stats.num_committed_gaussians = result.committed
            self.total_committed_gaussians += result.committed
            stats.archive_transfer_ms += self._enforce_budget(cam, frame_id)
            return self._finalize_stats(stats, start)

        fast_path_opacity = float(
            self.config["admission"]["fast_path_initial_opacity"]
        )
        if frame_id < int(self.config["bootstrap_frames"]):
            bootstrap_indices = select_budgeted_bootstrap_indices(
                proposals,
                self.config["admission"]["fast_path_max_gaussians_per_frame"],
                self.config["admission"]["fast_path_frequency_fraction"],
            )
            bootstrap = proposals.select(bootstrap_indices)
            result = self._commit_fast_path(
                bootstrap,
                frame_id,
                initial_opacity=fast_path_opacity,
            )
            committed_sparse = bootstrap.sparse_depth_valid[
                result.committed_indices
            ]
            stats.num_fast_path_gaussians = result.committed
            stats.num_depth_confidence_fast_path_gaussians = int(
                np.count_nonzero(~committed_sparse)
            )
            stats.num_committed_gaussians = result.committed
            self.total_committed_gaussians += result.committed
            stats.archive_transfer_ms += self._enforce_budget(cam, frame_id)
            return self._finalize_stats(stats, start)

        use_sparse_fast_path = bool(
            self.config["admission"]["trusted_sparse_fast_path"]
        )
        use_depthcov_fast_path = bool(
            self.config["admission"]["trusted_depthcov_fast_path"]
        )
        excluded_from_candidates = np.zeros((len(proposals),), dtype=np.bool_)
        if (use_sparse_fast_path or use_depthcov_fast_path) and len(proposals):
            (
                trusted_mask,
                depth_confident_mask,
                probation_mask,
                deferred_mask,
            ) = split_fast_path_masks(
                proposals,
                self.config["admission"],
                use_sparse_fast_path,
                use_depthcov_fast_path,
            )
            stats.num_frequency_deferred_gaussians = int(
                np.count_nonzero(deferred_mask)
            )
            fast_path_budget = int(
                self.config["admission"]["fast_path_max_gaussians_per_frame"]
            )
            remaining_fast_path_budget = fast_path_budget
            if np.any(trusted_mask):
                trusted_global_indices = np.flatnonzero(trusted_mask)
                trusted = proposals.select(trusted_global_indices)
                selected_indices = select_budgeted_fast_path_indices(
                    trusted,
                    remaining_fast_path_budget,
                    self.config["admission"]["fast_path_frequency_fraction"],
                )
                trusted = trusted.select(selected_indices)
                trusted_depth_mask = depth_confident_mask[
                    trusted_global_indices[selected_indices]
                ]
                trusted_result = self._commit_fast_path(
                    trusted,
                    frame_id,
                    initial_opacity=fast_path_opacity,
                )
                stats.num_fast_path_gaussians = trusted_result.committed
                stats.num_depth_confidence_fast_path_gaussians = int(
                    np.count_nonzero(
                        trusted_depth_mask[trusted_result.committed_indices]
                    )
                )
                stats.num_committed_gaussians += trusted_result.committed
                self.total_committed_gaussians += trusted_result.committed
                if remaining_fast_path_budget > 0:
                    remaining_fast_path_budget = max(
                        0,
                        remaining_fast_path_budget - trusted_result.selected,
                    )
            if np.any(probation_mask) and (
                fast_path_budget <= 0 or remaining_fast_path_budget > 0
            ):
                probation = proposals.select(probation_mask)
                selected_indices = select_budgeted_fast_path_indices(
                    probation,
                    remaining_fast_path_budget,
                    self.config["admission"]["fast_path_frequency_fraction"],
                )
                probation = probation.select(selected_indices)
                probation_result = self._commit_fast_path(
                    probation,
                    frame_id,
                    initial_opacity=frequency_probation_opacities(
                        probation, self.config["admission"]
                    ),
                )
                stats.num_frequency_probation_gaussians = probation_result.committed
                stats.num_fast_path_gaussians += probation_result.committed
                stats.num_depth_confidence_fast_path_gaussians += (
                    probation_result.committed
                )
                stats.num_committed_gaussians += probation_result.committed
                self.total_committed_gaussians += probation_result.committed
            excluded_from_candidates = trusted_mask | probation_mask

        candidate_mask, filtered_depthcov_mask = split_candidate_masks(
            proposals,
            excluded_from_candidates,
            self.config["admission"]["allow_depthcov_candidates"],
            self.config["admission"]["depthcov_candidate_stable_depth_ratio"],
        )
        stats.num_filtered_depthcov_candidates = int(
            np.count_nonzero(filtered_depthcov_mask)
        )
        proposals = proposals.select(candidate_mask)
        if len(proposals) and (use_sparse_fast_path or use_depthcov_fast_path):
            occupied = self.gaussian_model.hash_block.getOccupy(
                proposals.world_points,
                proposals.colors,
                proposals.view_scale_size,
            )
            proposals = proposals.select(~occupied)
        if len(proposals) == 0:
            stats.archive_transfer_ms += self._enforce_budget(cam, frame_id)
            return self._finalize_stats(stats, start)

        association_start = time.perf_counter()
        groups = group_host_proposals(
            proposals, cam.get_width(proposals.level), self.config["admission"]
        )
        if self.config["admission"]["policy"] == "immediate":
            # Immediate is a regression control: sample the CandidateBank path
            # without paying to instantiate and destroy every possible group.
            groups = groups[: min(64, len(groups))]
        stats.num_proposal_groups = len(groups)
        if groups:
            descriptors, gray_patches, mean_colors = self.descriptor_extractor.describe(
                cam.get_gt_image(proposals.level), representative_uvs(groups)
            )
            matched, new = self.candidate_bank.associate_and_update(
                groups,
                descriptors,
                gray_patches,
                mean_colors,
                cam.get_raw_pose().detach().cpu().numpy(),
                cam.get_int_mat(proposals.level).detach().cpu().numpy(),
                frame_id,
                cam.get_width(proposals.level),
                cam.get_height(proposals.level),
            )
        else:
            matched, new = [], []
        stats.num_candidate_matches = len(matched)
        stats.num_new_candidates = len(new)
        expired = self.candidate_bank.expire(frame_id)
        stats.num_expired = len(expired)
        stats.candidate_association_ms = (
            time.perf_counter() - association_start
        ) * 1000.0

        if self.config["admission"]["policy"] == "immediate":
            result = self.gaussian_model.commit_proposals(proposals)
            self.candidate_bank.remove(
                set(matched) | set(new), CandidateStatus.COMMITTED
            )
            stats.num_committed_candidates = len(set(matched) | set(new))
            stats.num_committed_gaussians = result.committed
            self.total_committed_candidates += stats.num_committed_candidates
            self.total_committed_gaussians += result.committed
            return self._finalize_stats(stats, start)

        should_gate = (
            is_key_frame
            and frame_id % int(self.config["admission"]["gate_interval"]) == 0
        )
        if not should_gate:
            stats.archive_transfer_ms += self._enforce_budget(cam, frame_id)
            return self._finalize_stats(stats, start)

        risk_start = time.perf_counter()
        ready = self.candidate_bank.ready_candidates(frame_id)
        limit = min(
            int(self.config["admission"]["max_risk_candidates_per_keyframe"]),
            int(self.config["commit_refinement"]["max_commits_per_keyframe"]),
        )
        evaluated = ready[:limit]
        decisions = self.admission_policy.evaluate(evaluated)
        stats.num_risk_evaluations = len(evaluated)
        self.total_risk_evaluations += len(evaluated)
        stats.risk_gate_ms = (time.perf_counter() - risk_start) * 1000.0
        if decisions:
            risk_values = np.asarray([decision.score for decision in decisions])
            finite_risk = risk_values[np.isfinite(risk_values)]
            edges = np.asarray(
                [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0, 2.0, 5.0, 1.0e30]
            )
            stats.risk_histogram_edges = edges.tolist()
            stats.risk_histogram_counts = np.histogram(
                finite_risk, bins=edges
            )[0].astype(int).tolist()
            if len(finite_risk):
                stats.risk_min = float(np.min(finite_risk))
                stats.risk_median = float(np.median(finite_risk))
                stats.risk_p95 = float(np.percentile(finite_risk, 95))
                stats.risk_max = float(np.max(finite_risk))
        commit_ids = [decision.candidate_id for decision in decisions if decision.commit]
        committed_candidates = self.candidate_bank.remove(
            commit_ids, CandidateStatus.COMMITTED
        )
        if committed_candidates:
            refine_start = time.perf_counter()
            if self.mode == "aerocommit_mvp":
                commit_batches, _, _ = self.commit_refiner.refine(committed_candidates)
            else:
                commit_batches = [candidate.proposal_batch for candidate in committed_candidates]
            stats.commit_refinement_ms = (
                time.perf_counter() - refine_start
            ) * 1000.0
            corrections = np.asarray(
                [
                    abs((1.0 / max(candidate.refined_rho, 1.0e-8)) - candidate.depth_prior)
                    / max(candidate.depth_prior, 1.0e-8)
                    for candidate in committed_candidates
                    if candidate.refined_rho is not None
                ],
                dtype=np.float32,
            )
            if len(corrections):
                stats.depth_correction_abs_rel_mean = float(np.mean(corrections))
                stats.depth_correction_abs_rel_p95 = float(np.percentile(corrections, 95))
            stats.num_fused_proposals = int(
                sum(candidate.fused_proposal_count for candidate in committed_candidates)
            )
            detail_start = time.perf_counter()
            if self.mode == "aerocommit_mvp":
                commit_batches, split_count, _ = self.detail_refiner.refine(
                    committed_candidates, commit_batches
                )
            else:
                split_count = 0
            stats.detail_refinement_ms = (
                time.perf_counter() - detail_start
            ) * 1000.0
            stats.num_detail_splits = split_count
            stats.num_side_detail_splits = sum(
                int(batch.metadata.get("detail_split", False))
                and candidate.lateral_score
                >= float(self.config["admission"]["side_band_start"])
                for candidate, batch in zip(committed_candidates, commit_batches)
            )
            combined = GaussianProposalBatch.concatenate(commit_batches, frame_id)
            force_new_group = bool(
                self.config["commit_refinement"]["force_new_group"]
            )
            target_group_id = None
            chunk_size = int(
                self.config["commit_refinement"].get("group_chunk_keyframes", 1)
            )
            if (
                force_new_group
                and chunk_size > 1
                and self._candidate_commit_group_id in self.gaussian_model.valid_groups
                and self._candidate_commit_group_events < chunk_size
            ):
                target_group_id = self._candidate_commit_group_id
                force_new_group = False
            result = self.gaussian_model.commit_proposals(
                combined,
                initial_opacity=float(
                    self.config["commit_refinement"]["initial_opacity"]
                ),
                force_new_group=force_new_group,
                target_group_id=target_group_id,
            )
            if result.group_id is not None and chunk_size > 1:
                if target_group_id is None:
                    self._candidate_commit_group_id = result.group_id
                    self._candidate_commit_group_events = 1
                else:
                    self._candidate_commit_group_events += 1
            self._record_group(result.group_id, combined, frame_id)
            stats.num_committed_candidates = len(committed_candidates)
            stats.num_committed_gaussians += result.committed
            self.total_committed_candidates += len(committed_candidates)
            self.total_committed_gaussians += result.committed
        stats.archive_transfer_ms += self._enforce_budget(cam, frame_id)
        Log(
            "frame {} raw/groups/wait/risk/commit-gs {}/{}/{}/{}/{}".format(
                frame_id,
                stats.num_raw_proposals,
                stats.num_proposal_groups,
                self.candidate_bank.waiting_count,
                stats.num_risk_evaluations,
                stats.num_committed_gaussians,
            ),
            tag="AeroCommit",
        )
        return self._finalize_stats(stats, start)

    def export_full_map(self, path):
        parts: List[Dict[str, torch.Tensor]] = []
        active = self.gaussian_model.export_raw_splats()
        if active:
            parts.append(active)
        parts.extend(
            {
                name: value.float()
                for name, value in record.params.items()
            }
            for record in self.archive_store.groups.values()
        )
        if not parts:
            raise RuntimeError("No active or archived Gaussians are available for export")
        names = ("means", "scales", "quats", "opacities", "sh0", "shN")
        combined = {
            name: torch.cat([part[name].cpu() for part in parts], dim=0)
            for name in names
        }
        self.gaussian_model.save_raw_splats_as_ply(combined, path)

    def finalize(self):
        return self.metrics.summary(
            {
                "mode": self.mode,
                "admission_policy": self.config["admission"]["policy"],
                "total_risk_evaluations": self.total_risk_evaluations,
                "total_committed_candidates": self.total_committed_candidates,
                "total_committed_gaussians": self.total_committed_gaussians,
            }
        )
