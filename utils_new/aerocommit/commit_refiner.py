"""Lightweight causal multi-view geometry refinement at commit time."""

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .types import CandidateRecord, GaussianProposalBatch


class CommitRefiner:
    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.device = torch.device(device)

    def refine(
        self, candidates: Sequence[CandidateRecord]
    ) -> Tuple[List[GaussianProposalBatch], float, float]:
        if not candidates:
            return [], 0.0, 0.0
        if not self.config["enabled"] or int(self.config["iterations"]) <= 0:
            return [candidate.proposal_batch for candidate in candidates], 0.0, 0.0
        device = self.device
        batch = len(candidates)
        max_edges = max(candidate.support_count for candidate in candidates)
        reference_uv = torch.zeros((batch, 2), device=device)
        reference_pose = torch.zeros((batch, 4, 4), device=device)
        reference_K = torch.zeros((batch, 3, 3), device=device)
        initial_rho = torch.zeros((batch,), device=device)
        support_pose = torch.eye(4, device=device).repeat(batch, max_edges, 1, 1)
        support_K = torch.eye(3, device=device).repeat(batch, max_edges, 1, 1)
        observed_uv = torch.zeros((batch, max_edges, 2), device=device)
        weights = torch.zeros((batch, max_edges), device=device)
        for index, candidate in enumerate(candidates):
            reference_uv[index] = torch.as_tensor(candidate.reference_uv, device=device)
            reference_pose[index] = torch.as_tensor(candidate.reference_pose, device=device)
            reference_K[index] = torch.as_tensor(candidate.reference_K, device=device)
            initial_rho[index] = candidate.rho_mean
            for edge_index, edge in enumerate(candidate.support_edges):
                support_pose[index, edge_index] = torch.as_tensor(
                    edge.world_to_camera, device=device
                )
                support_K[index, edge_index] = torch.as_tensor(
                    edge.intrinsics, device=device
                )
                observed_uv[index, edge_index] = torch.as_tensor(edge.uv, device=device)
                weights[index, edge_index] = max(
                    0.05, 1.0 - float(edge.association_error)
                )

        log_rho = torch.nn.Parameter(torch.log(torch.clamp(initial_rho, min=1.0e-8)))
        optimizer = torch.optim.Adam(
            [log_rho], lr=float(self.config["learning_rate"])
        )

        def reprojection_loss():
            rho = torch.exp(log_rho)
            pixels = torch.cat(
                (reference_uv, torch.ones((batch, 1), device=device)), dim=1
            )
            rays = torch.linalg.solve(reference_K, pixels.unsqueeze(-1)).squeeze(-1)
            camera_points = rays / rho[:, None]
            homogeneous = torch.cat(
                (camera_points, torch.ones((batch, 1), device=device)), dim=1
            )
            world = torch.einsum(
                "bij,bj->bi", torch.linalg.inv(reference_pose), homogeneous
            )
            camera = torch.einsum("beij,bj->bei", support_pose, world)[..., :3]
            screen = torch.einsum("beij,bej->bei", support_K, camera)
            projected = screen[..., :2] / torch.clamp(
                screen[..., 2:3], min=1.0e-6
            )
            focal = 0.5 * (support_K[..., 0, 0] + support_K[..., 1, 1])
            normalized_error = (projected - observed_uv) / torch.clamp(
                focal[..., None], min=1.0
            )
            per_edge = F.smooth_l1_loss(
                normalized_error,
                torch.zeros_like(normalized_error),
                beta=0.01,
                reduction="none",
            ).sum(dim=-1)
            return (per_edge * weights).sum() / torch.clamp(weights.sum(), min=1.0)

        loss_before = float(reprojection_loss().detach().item())
        for _ in range(int(self.config["iterations"])):
            optimizer.zero_grad(set_to_none=True)
            loss = reprojection_loss()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                correction = float(self.config.get("max_depth_correction_ratio", 0.20))
                lower = torch.log(
                    torch.clamp(initial_rho / (1.0 + correction), min=1.0e-8)
                )
                upper = torch.log(
                    torch.clamp(initial_rho / max(1.0 - correction, 1.0e-3), min=1.0e-8)
                )
                log_rho.clamp_(lower, upper)
        loss_after = float(reprojection_loss().detach().item())
        refined_rho = torch.exp(log_rho).detach().cpu().numpy()
        correction = float(self.config.get("max_depth_correction_ratio", 0.20))
        for index, candidate in enumerate(candidates):
            refined_depth = 1.0 / max(float(refined_rho[index]), 1.0e-8)
            lower_depth = candidate.depth_prior * (1.0 - correction)
            upper_depth = candidate.depth_prior * (1.0 + correction)
            refined_depth = float(np.clip(refined_depth, lower_depth, upper_depth))
            refined_rho[index] = 1.0 / max(refined_depth, 1.0e-8)

        refined_batches = []
        for candidate, rho in zip(candidates, refined_rho):
            candidate.refined_rho = float(rho)
            candidate.commit_loss_before = loss_before
            candidate.commit_loss_after = loss_after
            refined_depth = 1.0 / max(float(rho), 1.0e-8)
            depth_ratio = refined_depth / max(candidate.depth_prior, 1.0e-8)
            batch_proposals = candidate.proposal_batch.select(None)
            pose = candidate.reference_pose
            homogeneous = np.concatenate(
                (
                    batch_proposals.world_points,
                    np.ones((len(batch_proposals), 1), dtype=np.float32),
                ),
                axis=1,
            )
            camera_points = homogeneous @ pose.T
            camera_points[:, :3] *= depth_ratio
            camera_to_world = np.linalg.inv(candidate.reference_pose)
            world = camera_points @ camera_to_world.T
            batch_proposals.world_points = world[:, :3].astype(np.float32)
            batch_proposals.depths = (
                batch_proposals.depths * depth_ratio
            ).astype(np.float32)
            batch_proposals.inverse_depths = 1.0 / np.maximum(
                batch_proposals.depths, 1.0e-8
            )
            batch_proposals.log_scales = (
                batch_proposals.log_scales + np.log(max(depth_ratio, 1.0e-8))
            ).astype(np.float32)
            support_colors = np.stack(
                [edge.color for edge in candidate.support_edges], axis=0
            )
            fused_color = np.median(support_colors, axis=0).astype(np.float32)
            source_color = np.median(batch_proposals.colors, axis=0)
            strength = float(self.config.get("color_fusion_strength", 0.35))
            color_delta = strength * (fused_color - source_color)
            batch_proposals.colors = np.clip(
                batch_proposals.colors + color_delta[None], 0.0, 1.0
            ).astype(np.float32)
            refined_batches.append(batch_proposals)
        return refined_batches, loss_before, loss_after
