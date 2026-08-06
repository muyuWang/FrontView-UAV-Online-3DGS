"""Stable residual driven commit-time coarse-to-fine detail splitting."""

from typing import List, Sequence, Tuple

import numpy as np

from .types import CandidateRecord, GaussianProposalBatch


class DetailRefiner:
    def __init__(self, config):
        self.config = config

    def _score(self, candidate, batch):
        focal = 0.5 * (candidate.reference_K[0, 0] + candidate.reference_K[1, 1])
        scale = float(np.exp(np.median(batch.log_scales)))
        depth = float(np.median(batch.depths))
        radius = scale * focal / max(depth, 1.0e-8)
        confidence = max(0.0, 1.0 - candidate.association_error_ema)
        score = radius * candidate.stable_residual_ema * confidence
        score *= 1.0 + (
            float(self.config["side_score_boost"]) - 1.0
        ) * candidate.lateral_score
        return score, radius

    def refine(
        self,
        candidates: Sequence[CandidateRecord],
        batches: Sequence[GaussianProposalBatch],
    ) -> Tuple[List[GaussianProposalBatch], int, int]:
        if not self.config["enabled"]:
            return list(batches), 0, 0
        eligible = []
        for index, (candidate, batch) in enumerate(zip(candidates, batches)):
            score, radius = self._score(candidate, batch)
            if (
                candidate.support_count >= int(self.config["refine_min_views"])
                and radius >= float(self.config["refine_min_projected_radius_px"])
                and candidate.stable_residual_ema
                >= float(self.config["refine_min_stable_residual"])
            ):
                eligible.append((score, index))
        eligible.sort(reverse=True)
        split_indices = {
            index
            for _, index in eligible[: int(self.config["max_splits_per_keyframe"])]
        }
        output = []
        added = 0
        for index, (candidate, batch) in enumerate(zip(candidates, batches)):
            if index not in split_indices:
                output.append(batch)
                continue
            scale_ratio = float(self.config["child_scale_ratio"])
            camera_to_world = np.linalg.inv(candidate.reference_pose)
            tangent_x = camera_to_world[:3, 0]
            tangent_y = camera_to_world[:3, 1]
            linear_scale = np.exp(batch.log_scales.reshape(-1))
            signs = np.asarray(
                [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)],
                dtype=np.float32,
            )
            means = []
            for sign_x, sign_y in signs:
                offset = (
                    sign_x * linear_scale[:, None] * tangent_x[None]
                    + sign_y * linear_scale[:, None] * tangent_y[None]
                ) * 0.35
                means.append(batch.world_points + offset)
            repeat = len(signs)
            split = GaussianProposalBatch(
                source_frame_id=batch.source_frame_id,
                level=batch.level,
                uv=np.tile(batch.uv, (repeat, 1)),
                patch_bboxes=np.tile(batch.patch_bboxes, (repeat, 1)),
                depths=np.tile(batch.depths, repeat),
                inverse_depths=np.tile(batch.inverse_depths, repeat),
                world_points=np.concatenate(means, axis=0).astype(np.float32),
                log_scales=np.tile(
                    batch.log_scales + np.log(scale_ratio), (repeat, 1)
                ).astype(np.float32),
                colors=np.tile(batch.colors, (repeat, 1)),
                residual_scores=np.tile(batch.residual_scores, repeat),
                coverage_scores=np.tile(batch.coverage_scores, repeat),
                sparse_depth_valid=np.tile(batch.sparse_depth_valid, repeat),
                depth_confidences=np.tile(batch.depth_confidences, repeat),
                frequency_scores=np.tile(batch.frequency_scores, repeat),
                view_scale_size=batch.view_scale_size,
                create_new_group=batch.create_new_group,
                metadata={**batch.metadata, "detail_split": True},
            )
            output.append(split)
            added += len(split) - len(batch)
        return output, len(split_indices), added
