"""Batched empirical nuisance-profiled inverse-depth commitment risk."""

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .types import CandidateRecord, RiskResult


def _skew(points: torch.Tensor) -> torch.Tensor:
    x, y, z = points.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        (zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1
    ).reshape(*points.shape[:-1], 3, 3)


def _world_from_inverse_depth(uv, rho, poses, intrinsics):
    pixels = torch.cat((uv, torch.ones_like(uv[:, :1])), dim=1)
    rays = torch.linalg.solve(intrinsics, pixels.unsqueeze(-1)).squeeze(-1)
    camera_points = rays / torch.clamp(rho[:, None], min=1.0e-8)
    homogeneous = torch.cat(
        (camera_points, torch.ones_like(camera_points[:, :1])), dim=1
    )
    camera_to_world = torch.linalg.inv(poses)
    return torch.einsum("bij,bj->bi", camera_to_world, homogeneous)[:, :3]


def _project_world(world_points, poses, intrinsics):
    batch, edges = poses.shape[:2]
    homogeneous = torch.cat(
        (world_points, torch.ones_like(world_points[:, :1])), dim=1
    )
    camera = torch.einsum("beij,bj->bei", poses, homogeneous)[..., :3]
    screen = torch.einsum("beij,bej->bei", intrinsics, camera)
    uv = screen[..., :2] / torch.clamp(screen[..., 2:3], min=1.0e-8)
    return uv, camera


class NPOLiteEvaluator:
    """Evaluate top-K candidates without invoking the full Gaussian renderer."""

    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.device = torch.device(device)

    def evaluate(self, candidates: Sequence[CandidateRecord]) -> RiskResult:
        if not candidates:
            empty = np.empty((0,), dtype=np.float32)
            return RiskResult(
                candidate_ids=np.empty((0,), dtype=np.int64),
                commitment_risk=empty,
                information=empty,
                residual_sigma=empty,
                pose_projected_uncertainty=empty,
            )
        device = self.device
        dtype = torch.float32
        batch = len(candidates)
        max_edges = int(self.config["max_support_edges"])
        patch_size = candidates[0].reference_gray_patch.shape[-1]

        reference_uv = torch.zeros((batch, 2), device=device, dtype=dtype)
        reference_rho = torch.zeros((batch,), device=device, dtype=dtype)
        reference_pose = torch.zeros((batch, 4, 4), device=device, dtype=dtype)
        reference_K = torch.zeros((batch, 3, 3), device=device, dtype=dtype)
        reference_gray = torch.zeros(
            (batch, patch_size, patch_size), device=device, dtype=dtype
        )
        poses = torch.eye(4, device=device, dtype=dtype).repeat(batch, max_edges, 1, 1)
        intrinsics = torch.eye(3, device=device, dtype=dtype).repeat(
            batch, max_edges, 1, 1
        )
        observed_gray = torch.zeros(
            (batch, max_edges, patch_size, patch_size), device=device, dtype=dtype
        )
        covariances = torch.eye(6, device=device, dtype=dtype).repeat(
            batch, max_edges, 1, 1
        )
        valid = torch.zeros((batch, max_edges), device=device, dtype=torch.bool)
        association = torch.zeros((batch,), device=device, dtype=dtype)
        ids = []
        for batch_index, candidate in enumerate(candidates):
            ids.append(candidate.candidate_id)
            reference_uv[batch_index] = torch.as_tensor(
                candidate.reference_uv, device=device, dtype=dtype
            )
            reference_rho[batch_index] = candidate.rho_mean
            reference_pose[batch_index] = torch.as_tensor(
                candidate.reference_pose, device=device, dtype=dtype
            )
            reference_K[batch_index] = torch.as_tensor(
                candidate.reference_K, device=device, dtype=dtype
            )
            reference_gray[batch_index] = torch.as_tensor(
                candidate.reference_gray_patch, device=device, dtype=dtype
            )
            association[batch_index] = candidate.association_error_ema
            for edge_index, edge in enumerate(candidate.support_edges[:max_edges]):
                poses[batch_index, edge_index] = torch.as_tensor(
                    edge.world_to_camera, device=device, dtype=dtype
                )
                intrinsics[batch_index, edge_index] = torch.as_tensor(
                    edge.intrinsics, device=device, dtype=dtype
                )
                observed_gray[batch_index, edge_index] = torch.as_tensor(
                    edge.gray_patch, device=device, dtype=dtype
                )
                covariances[batch_index, edge_index] = torch.as_tensor(
                    edge.pose_covariance, device=device, dtype=dtype
                )
                valid[batch_index, edge_index] = True

        relative_step = float(self.config["finite_difference_rho_relative_step"])
        rho_step = torch.clamp(reference_rho.abs() * relative_step, min=1.0e-6)
        world = _world_from_inverse_depth(
            reference_uv, reference_rho, reference_pose, reference_K
        )
        world_plus = _world_from_inverse_depth(
            reference_uv, reference_rho + rho_step, reference_pose, reference_K
        )
        world_minus = _world_from_inverse_depth(
            reference_uv,
            torch.clamp(reference_rho - rho_step, min=1.0e-8),
            reference_pose,
            reference_K,
        )
        uv, camera = _project_world(world, poses, intrinsics)
        uv_plus, _ = _project_world(world_plus, poses, intrinsics)
        uv_minus, _ = _project_world(world_minus, poses, intrinsics)
        duv_drho = (uv_plus - uv_minus) / (2.0 * rho_step[:, None, None])

        z = torch.clamp(camera[..., 2], min=1.0e-6)
        fx = intrinsics[..., 0, 0]
        fy = intrinsics[..., 1, 1]
        projection_jacobian = torch.zeros(
            (batch, max_edges, 2, 3), device=device, dtype=dtype
        )
        projection_jacobian[..., 0, 0] = fx / z
        projection_jacobian[..., 0, 2] = -fx * camera[..., 0] / z.square()
        projection_jacobian[..., 1, 1] = fy / z
        projection_jacobian[..., 1, 2] = -fy * camera[..., 1] / z.square()
        se3_jacobian = torch.cat(
            (
                torch.eye(3, device=device, dtype=dtype)
                .reshape(1, 1, 3, 3)
                .expand(batch, max_edges, -1, -1),
                -_skew(camera),
            ),
            dim=-1,
        )
        uv_pose_jacobian = projection_jacobian @ se3_jacobian

        gray = observed_gray
        grad_x = F.pad(0.5 * (gray[..., 2:] - gray[..., :-2]), (1, 1, 0, 0))
        grad_y = F.pad(0.5 * (gray[..., 2:, :] - gray[..., :-2, :]), (0, 0, 1, 1))
        gradients = torch.stack((grad_x, grad_y), dim=-1).reshape(
            batch, max_edges, -1, 2
        )
        residual = (gray - reference_gray[:, None]).reshape(batch, max_edges, -1)
        sample_count = min(
            int(self.config["max_patch_samples"]), gradients.shape[2]
        )
        gradient_strength = torch.linalg.norm(gradients, dim=-1)
        sample_indices = torch.topk(
            gradient_strength, k=sample_count, dim=2, largest=True
        ).indices
        gather_gradient = sample_indices[..., None].expand(-1, -1, -1, 2)
        gradients = torch.gather(gradients, 2, gather_gradient)
        residual = torch.gather(residual, 2, sample_indices)

        J_rho = torch.einsum("besd,bed->bes", gradients, duv_drho)
        J_pose = torch.einsum("besd,bedk->besk", gradients, uv_pose_jacobian)
        huber_delta = float(self.config["huber_delta"])
        weights = torch.where(
            residual.abs() <= huber_delta,
            torch.ones_like(residual),
            huber_delta / torch.clamp(residual.abs(), min=1.0e-8),
        )
        weights = weights * valid[..., None]
        H_rr = torch.sum(weights * J_rho.square(), dim=2)
        H_rx = torch.einsum("bes,bes,besk->bek", weights, J_rho, J_pose)
        H_xx = torch.einsum("bes,besj,besk->bejk", weights, J_pose, J_pose)
        eps = float(self.config["lambda_num"])
        identity = torch.eye(6, device=device, dtype=dtype)
        pose_prior = torch.linalg.inv(covariances + eps * identity)
        system = H_xx + pose_prior + eps * identity
        solved = torch.linalg.solve(system, H_rx.unsqueeze(-1)).squeeze(-1)
        profiled = H_rr - torch.sum(H_rx * solved, dim=-1)
        information = torch.sum(torch.clamp(profiled, min=0.0) * valid, dim=1)

        pose_variance = torch.einsum(
            "besj,bejk,besk->bes", J_pose, covariances, J_pose
        )
        valid_samples = torch.clamp(valid.sum(dim=1) * sample_count, min=1)
        pose_uncertainty = torch.sum(pose_variance * valid[..., None], dim=(1, 2))
        pose_uncertainty = pose_uncertainty / valid_samples
        mean_jrho2 = torch.sum(J_rho.square() * valid[..., None], dim=(1, 2))
        mean_jrho2 = mean_jrho2 / valid_samples

        residual_sigma = torch.zeros((batch,), device=device, dtype=dtype)
        for batch_index in range(batch):
            values = residual[batch_index][valid[batch_index]].reshape(-1)
            median = torch.median(values)
            residual_sigma[batch_index] = 1.4826 * torch.median(
                torch.abs(values - median)
            )
        mu_lower = torch.clamp(
            information - float(self.config["curvature_margin"]),
            min=float(self.config["min_information"]),
        )
        g_upper = (
            float(self.config["noise_weight"])
            * residual_sigma
            * torch.sqrt(mean_jrho2 + 1.0e-12)
            + float(self.config["association_weight"]) * association
            + float(self.config["pose_weight"])
            * torch.sqrt(torch.clamp(pose_uncertainty, min=0.0) + 1.0e-12)
        )
        risk = g_upper / (mu_lower + 1.0e-12)
        return RiskResult(
            candidate_ids=np.asarray(ids, dtype=np.int64),
            commitment_risk=risk.detach().cpu().numpy(),
            information=information.detach().cpu().numpy(),
            residual_sigma=residual_sigma.detach().cpu().numpy(),
            pose_projected_uncertainty=pose_uncertainty.detach().cpu().numpy(),
        )
