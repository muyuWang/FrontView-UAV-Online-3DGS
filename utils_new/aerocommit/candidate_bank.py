"""CPU-resident ephemeral candidate bank for causal Gaussian admission."""

from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from .association import (
    build_projected_bins,
    candidate_world_point,
    neighboring_candidate_ids,
    parallax_angle,
)
from .types import (
    CandidateRecord,
    CandidateStatus,
    GaussianProposalBatch,
    SupportEdge,
)


class CandidateBank:
    def __init__(self, config, pose_uncertainty_provider):
        self.config = config
        self.pose_uncertainty_provider = pose_uncertainty_provider
        self.candidates: Dict[int, CandidateRecord] = {}
        self._next_candidate_id = 0

    def _minimum_support(self, candidate):
        if (
            self.config["frequency_candidate_enabled"]
            and candidate.frequency_score
            >= float(self.config["frequency_candidate_score_threshold"])
        ):
            return int(self.config["frequency_candidate_min_support"])
        return int(self.config["min_support"])

    @staticmethod
    def _representative(batch: GaussianProposalBatch):
        return {
            "uv": np.median(batch.uv, axis=0).astype(np.float32),
            "world_point": np.median(batch.world_points, axis=0).astype(np.float32),
            "depth": float(np.median(batch.depths)),
            "rho": float(np.median(batch.inverse_depths)),
            "log_scale": float(np.median(batch.log_scales)),
            "color": np.mean(batch.colors, axis=0).astype(np.float32),
            "residual": float(np.median(batch.residual_scores)),
            "coverage": float(np.mean(batch.coverage_scores)),
        }

    def _support_edge(
        self,
        batch,
        descriptor,
        gray_patch,
        pose,
        K,
        frame_id,
        association_error,
        reference_pose,
    ):
        representative = self._representative(batch)
        angle = parallax_angle(
            representative["world_point"], reference_pose, pose
        )
        return SupportEdge(
            frame_id=int(frame_id),
            world_to_camera=np.asarray(pose, dtype=np.float32).copy(),
            intrinsics=np.asarray(K, dtype=np.float32).copy(),
            uv=representative["uv"],
            descriptor=np.asarray(descriptor, dtype=np.float32).copy(),
            gray_patch=np.asarray(gray_patch, dtype=np.float32).copy(),
            world_point=representative["world_point"],
            depth=representative["depth"],
            log_scale=representative["log_scale"],
            color=representative["color"],
            association_error=float(association_error),
            photometric_residual=representative["residual"],
            parallax_rad=angle,
            pose_covariance=self.pose_uncertainty_provider.get_covariance(frame_id),
            linearization_rho=representative["rho"],
        )

    def _retain_representative_supports(
        self, candidate: CandidateRecord, new_edge: SupportEdge
    ):
        by_frame = {edge.frame_id: edge for edge in candidate.support_edges}
        by_frame[new_edge.frame_id] = new_edge
        edges = list(by_frame.values())
        limit = int(self.config["max_support_edges"])
        if len(edges) <= limit:
            candidate.support_edges = sorted(edges, key=lambda edge: edge.frame_id)
            return
        selected: List[SupportEdge] = []

        def add(edge):
            if edge.frame_id not in {item.frame_id for item in selected}:
                selected.append(edge)

        reference = min(
            edges,
            key=lambda edge: abs(edge.frame_id - candidate.reference_frame_id),
        )
        add(reference)
        add(max(edges, key=lambda edge: edge.parallax_rad))
        add(max(edges, key=lambda edge: edge.frame_id))
        add(min(edges, key=lambda edge: edge.photometric_residual))
        if len(selected) < limit:
            remaining = sorted(
                edges,
                key=lambda edge: (
                    -edge.parallax_rad,
                    edge.photometric_residual,
                    -edge.frame_id,
                ),
            )
            for edge in remaining:
                add(edge)
                if len(selected) == limit:
                    break
        candidate.support_edges = sorted(selected[:limit], key=lambda edge: edge.frame_id)

    def _create_candidate(
        self,
        batch,
        descriptor,
        gray_patch,
        mean_color,
        pose,
        K,
        frame_id,
        priority,
        lateral,
    ):
        representative = self._representative(batch)
        candidate_id = self._next_candidate_id
        self._next_candidate_id += 1
        candidate = CandidateRecord(
            candidate_id=candidate_id,
            reference_frame_id=int(frame_id),
            reference_pose=np.asarray(pose, dtype=np.float32).copy(),
            reference_K=np.asarray(K, dtype=np.float32).copy(),
            reference_uv=representative["uv"],
            patch_bbox=np.median(batch.patch_bboxes, axis=0).astype(np.float32),
            reference_gray_patch=np.asarray(gray_patch, dtype=np.float32).copy(),
            reference_descriptor=np.asarray(descriptor, dtype=np.float32).copy(),
            mean_color=np.asarray(mean_color, dtype=np.float32).copy(),
            rho_mean=representative["rho"],
            rho_variance=0.0,
            depth_prior=representative["depth"],
            created_frame=int(frame_id),
            last_seen_frame=int(frame_id),
            proposal_batch=batch,
            coverage_score=representative["coverage"],
            priority_score=float(priority),
            lateral_score=float(lateral),
            stable_residual_ema=representative["residual"],
            representative_world_point=representative["world_point"],
        )
        edge = self._support_edge(
            batch,
            descriptor,
            gray_patch,
            pose,
            K,
            frame_id,
            0.0,
            pose,
        )
        candidate.support_edges = [edge]
        self.candidates[candidate_id] = candidate
        return candidate_id

    def _fuse_support_proposals(self, candidate, batch):
        """Keep a bounded, spatially distinct union of causal observations."""
        if not self.config["fuse_support_proposals"] or len(batch) == 0:
            return
        target_depth = 1.0 / max(candidate.rho_mean, 1.0e-8)
        relative_depth = np.abs(batch.depths - target_depth) / max(target_depth, 1.0e-8)
        compatible = batch.select(
            relative_depth <= float(self.config["fusion_relative_depth_threshold"])
        )
        if len(compatible) == 0:
            return
        fused = GaussianProposalBatch.concatenate(
            (candidate.proposal_batch, compatible), candidate.reference_frame_id
        )
        linear_scale = float(np.exp(np.median(fused.log_scales)))
        voxel_size = max(
            1.0e-6,
            linear_scale * float(self.config["fusion_voxel_scale_ratio"]),
        )
        score = (
            fused.coverage_scores
            + fused.residual_scores
            + 0.1 * fused.sparse_depth_valid.astype(np.float32)
        )
        order = np.argsort(score)[::-1]
        base_capacity = int(self.config["max_fused_proposals_per_candidate"])
        side_multiplier = 1.0 + (
            float(self.config["side_fusion_capacity_multiplier"]) - 1.0
        ) * candidate.lateral_score
        capacity = max(1, int(round(base_capacity * side_multiplier)))
        keys = np.floor(fused.world_points / voxel_size).astype(np.int64)
        selected = []
        occupied = set()
        for index in order:
            key = tuple(keys[index].tolist())
            if key in occupied:
                continue
            occupied.add(key)
            selected.append(int(index))
            if len(selected) >= capacity:
                break
        previous_count = len(candidate.proposal_batch)
        candidate.proposal_batch = fused.select(np.asarray(selected, dtype=np.int64))
        candidate.fused_proposal_count += max(
            0, len(candidate.proposal_batch) - previous_count
        )
        candidate.representative_world_point = np.median(
            candidate.proposal_batch.world_points, axis=0
        ).astype(np.float32)

    def _update_commit_snapshot(self, candidate, batch):
        if self.config["fuse_support_proposals"]:
            self._fuse_support_proposals(candidate, batch)
            return
        if self.config["commit_snapshot_policy"] != "latest_consistent":
            return
        if candidate.original_proposal_batch is None:
            candidate.original_proposal_batch = candidate.proposal_batch.select(None)
        candidate.proposal_batch = batch.select(None)
        candidate.representative_world_point = np.median(
            candidate.proposal_batch.world_points, axis=0
        ).astype(np.float32)

    def _update_candidate(
        self,
        candidate,
        batch,
        descriptor,
        gray_patch,
        mean_color,
        pose,
        K,
        frame_id,
        association_error,
        priority,
        lateral,
    ):
        alpha = float(self.config["candidate_ema"])
        descriptor_alpha = float(self.config["descriptor_ema"])
        representative = self._representative(batch)
        innovation = representative["rho"] - candidate.rho_mean
        candidate.rho_mean += alpha * innovation
        candidate.rho_variance = (1.0 - alpha) * candidate.rho_variance + alpha * innovation**2
        candidate.association_error_ema = (
            (1.0 - alpha) * candidate.association_error_ema
            + alpha * float(association_error)
        )
        residual_delta = abs(
            representative["residual"] - candidate.stable_residual_ema
        )
        candidate.residual_mad_ema = (
            (1.0 - alpha) * candidate.residual_mad_ema + alpha * residual_delta
        )
        candidate.stable_residual_ema = (
            (1.0 - alpha) * candidate.stable_residual_ema
            + alpha * representative["residual"]
        )
        updated_descriptor = (
            (1.0 - descriptor_alpha) * candidate.reference_descriptor
            + descriptor_alpha * np.asarray(descriptor)
        )
        candidate.reference_descriptor = updated_descriptor / max(
            np.linalg.norm(updated_descriptor), 1.0e-8
        )
        candidate.mean_color = (
            (1.0 - alpha) * candidate.mean_color + alpha * np.asarray(mean_color)
        )
        candidate.coverage_score = max(
            candidate.coverage_score, representative["coverage"]
        )
        candidate.priority_score = max(candidate.priority_score, float(priority))
        candidate.lateral_score = max(candidate.lateral_score, float(lateral))
        candidate.last_seen_frame = int(frame_id)
        candidate.observation_count += 1
        self._update_commit_snapshot(candidate, batch)
        edge = self._support_edge(
            batch,
            descriptor,
            gray_patch,
            pose,
            K,
            frame_id,
            association_error,
            candidate.reference_pose,
        )
        candidate.parallax_max_rad = max(candidate.parallax_max_rad, edge.parallax_rad)
        self._retain_representative_supports(candidate, edge)
        if candidate.support_count >= self._minimum_support(candidate):
            candidate.status = CandidateStatus.READY_FOR_RISK

    def associate_and_update(
        self,
        groups: Sequence[Tuple[GaussianProposalBatch, float, float]],
        descriptors: np.ndarray,
        gray_patches: np.ndarray,
        mean_colors: np.ndarray,
        world_to_camera: np.ndarray,
        K: np.ndarray,
        frame_id: int,
        width: int,
        height: int,
    ):
        active = [
            candidate
            for candidate in self.candidates.values()
            if candidate.status
            in (CandidateStatus.WAITING, CandidateStatus.READY_FOR_RISK)
        ]
        radius = float(self.config["association_radius_px"])
        bins, projected, projected_depth = build_projected_bins(
            active, world_to_camera, K, width, height, radius
        )
        claimed: Set[int] = set()
        matched_ids: List[int] = []
        new_ids: List[int] = []
        descriptor_threshold = float(
            self.config["association_descriptor_threshold"]
        )
        relative_depth_threshold = float(
            self.config["association_relative_depth_threshold"]
        )
        for index, (batch, priority, lateral) in enumerate(groups):
            representative = self._representative(batch)
            best = None
            for candidate_id in neighboring_candidate_ids(
                bins, representative["uv"], radius
            ):
                if candidate_id in claimed:
                    continue
                candidate = self.candidates[candidate_id]
                pixel_error = float(
                    np.linalg.norm(projected[candidate_id] - representative["uv"])
                )
                if pixel_error > radius:
                    continue
                descriptor_error = float(
                    1.0 - np.dot(
                        candidate.reference_descriptor, descriptors[index]
                    )
                )
                if descriptor_error > descriptor_threshold:
                    continue
                depth_error = abs(
                    projected_depth[candidate_id] - representative["depth"]
                ) / max(representative["depth"], 1.0e-6)
                if depth_error > relative_depth_threshold:
                    continue
                score = (
                    pixel_error / max(radius, 1.0e-6)
                    + descriptor_error / max(descriptor_threshold, 1.0e-6)
                    + depth_error / max(relative_depth_threshold, 1.0e-6)
                ) / 3.0
                if best is None or score < best[0]:
                    best = (score, candidate_id)
            if best is None:
                if len(self.candidates) < int(self.config["max_candidate_bank_size"]):
                    new_ids.append(
                        self._create_candidate(
                            batch,
                            descriptors[index],
                            gray_patches[index],
                            mean_colors[index],
                            world_to_camera,
                            K,
                            frame_id,
                            priority,
                            lateral,
                        )
                    )
                continue
            association_error, candidate_id = best
            claimed.add(candidate_id)
            matched_ids.append(candidate_id)
            self._update_candidate(
                self.candidates[candidate_id],
                batch,
                descriptors[index],
                gray_patches[index],
                mean_colors[index],
                world_to_camera,
                K,
                frame_id,
                association_error,
                priority,
                lateral,
            )
        return matched_ids, new_ids

    def expire(self, current_frame: int):
        max_age = int(self.config["candidate_max_age"])
        expired = []
        for candidate_id, candidate in list(self.candidates.items()):
            if current_frame - candidate.last_seen_frame > max_age:
                candidate.status = CandidateStatus.EXPIRED
                expired.append(candidate_id)
                del self.candidates[candidate_id]
        return expired

    def ready_candidates(self, current_frame: int) -> List[CandidateRecord]:
        result = []
        for candidate in self.candidates.values():
            if candidate.status != CandidateStatus.READY_FOR_RISK:
                continue
            visibility_recency = max(0, current_frame - candidate.last_seen_frame)
            if visibility_recency > 1:
                continue
            association_confidence = max(0.0, 1.0 - candidate.association_error_ema)
            ambiguity = 1.0 + min(
                1.0,
                candidate.rho_variance / max(candidate.rho_mean**2, 1.0e-8),
            )
            candidate.priority_score = (
                candidate.coverage_score
                * association_confidence
                * ambiguity
                * (1.0 + 0.5 * candidate.lateral_score)
            )
            result.append(candidate)
        return sorted(result, key=lambda item: item.priority_score, reverse=True)

    def remove(self, candidate_ids: Iterable[int], status: CandidateStatus):
        removed = []
        for candidate_id in candidate_ids:
            candidate = self.candidates.pop(int(candidate_id), None)
            if candidate is not None:
                candidate.status = status
                removed.append(candidate)
        return removed

    @property
    def waiting_count(self):
        return len(self.candidates)

    @property
    def estimated_bytes(self):
        total = 0
        for candidate in self.candidates.values():
            batch = candidate.proposal_batch
            for name in (
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
                "stable_depths",
                "depth_confidences",
            ):
                total += getattr(batch, name).nbytes
            if candidate.original_proposal_batch is not None:
                original = candidate.original_proposal_batch
                for name in (
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
                    "stable_depths",
                    "depth_confidences",
                ):
                    total += getattr(original, name).nbytes
            total += candidate.reference_descriptor.nbytes
            total += candidate.reference_gray_patch.nbytes
            for edge in candidate.support_edges:
                total += sum(
                    value.nbytes
                    for value in (
                        edge.world_to_camera,
                        edge.intrinsics,
                        edge.uv,
                        edge.descriptor,
                        edge.gray_patch,
                        edge.world_point,
                        edge.color,
                        edge.pose_covariance,
                    )
                )
        return total
