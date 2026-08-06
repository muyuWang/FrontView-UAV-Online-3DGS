"""Evidence-conditioned spherical-harmonic appearance levels."""

from __future__ import annotations

import math

import torch


def camera_centers_from_viewmats(viewmats: torch.Tensor) -> torch.Tensor:
    """Return world-space camera centers from world-to-camera matrices."""

    rotations = viewmats[..., :3, :3]
    translations = viewmats[..., :3, 3]
    return -torch.matmul(
        rotations.transpose(-1, -2), translations.unsqueeze(-1)
    ).squeeze(-1)


def _quantiles(values: torch.Tensor) -> dict:
    if values.numel() == 0:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    values = values.float()
    result = torch.quantile(
        values,
        torch.tensor([0.1, 0.5, 0.9], device=values.device),
    )
    return {
        "p10": float(result[0].item()),
        "p50": float(result[1].item()),
        "p90": float(result[2].item()),
    }


def _histogram(values: torch.Tensor, boundaries) -> dict:
    boundaries = torch.as_tensor(boundaries, device=values.device, dtype=torch.float32)
    if boundaries.ndim != 1 or boundaries.numel() < 2:
        raise ValueError("Histogram boundaries must contain at least two values")
    bucket_ids = torch.bucketize(values.float(), boundaries[1:-1])
    counts = torch.bincount(bucket_ids, minlength=boundaries.numel() - 1)
    labels = [
        "[{:.6g},{:.6g})".format(float(left), float(right))
        for left, right in zip(boundaries[:-1].tolist(), boundaries[1:].tolist())
    ]
    return {label: int(count.item()) for label, count in zip(labels, counts)}


class AppearanceLODEvidence:
    """Accumulate bounded per-Gaussian evidence from packed gsplat projections."""

    def __init__(self, gaussian_count: int, device, dtype=torch.float32):
        self.gaussian_count = int(gaussian_count)
        self.device = torch.device(device)
        self.dtype = dtype
        self.view_count = torch.zeros(
            self.gaussian_count, device=self.device, dtype=torch.int32
        )
        self.direction_sum = torch.zeros(
            (self.gaussian_count, 3), device=self.device, dtype=self.dtype
        )
        self.radius_sum = torch.zeros(
            self.gaussian_count, device=self.device, dtype=self.dtype
        )

    @torch.no_grad()
    def observe(
        self,
        projection_info: dict,
        means: torch.Tensor,
        viewmats: torch.Tensor,
    ) -> int:
        gaussian_ids = projection_info.get("gaussian_ids")
        camera_ids = projection_info.get("camera_ids")
        radii = projection_info.get("radii")
        depths = projection_info.get("depths")
        if gaussian_ids is None or camera_ids is None or radii is None:
            raise ValueError(
                "Packed gsplat projection info is required for appearance LOD"
            )

        gaussian_ids = gaussian_ids.reshape(-1).long()
        camera_ids = camera_ids.reshape(-1).long()
        if radii.ndim > 1:
            radii = radii.amax(dim=-1)
        radii = radii.reshape(-1).to(self.dtype)
        valid = (
            (gaussian_ids >= 0)
            & (gaussian_ids < self.gaussian_count)
            & (camera_ids >= 0)
            & (camera_ids < viewmats.shape[0])
            & torch.isfinite(radii)
            & (radii > 0)
        )
        if depths is not None:
            depths = depths.reshape(-1)
            valid &= torch.isfinite(depths) & (depths > 0)
        if not torch.any(valid):
            return 0

        gaussian_ids = gaussian_ids[valid]
        camera_ids = camera_ids[valid]
        radii = radii[valid]
        centers = camera_centers_from_viewmats(viewmats)[camera_ids]
        directions = torch.nn.functional.normalize(
            centers - means[gaussian_ids], dim=-1, eps=1.0e-8
        ).to(self.dtype)

        ones = torch.ones_like(gaussian_ids, dtype=self.view_count.dtype)
        self.view_count.index_add_(0, gaussian_ids, ones)
        self.direction_sum.index_add_(0, gaussian_ids, directions)
        self.radius_sum.index_add_(0, gaussian_ids, radii)
        return int(gaussian_ids.numel())

    @torch.no_grad()
    def measurements(self):
        count = self.view_count.to(self.dtype)
        denominator = count.clamp_min(1.0)
        mean_radius = self.radius_sum / denominator
        mean_resultant = torch.linalg.norm(self.direction_sum, dim=-1) / denominator
        angular_dispersion = torch.clamp(1.0 - mean_resultant, min=0.0, max=1.0)
        score = (
            torch.log1p(count)
            * torch.log1p(mean_radius)
            * torch.sqrt(angular_dispersion + 1.0e-8)
        )
        return count, mean_radius, angular_dispersion, score

    @staticmethod
    def _top_fraction_mask(
        eligible: torch.Tensor, score: torch.Tensor, fraction: float
    ) -> torch.Tensor:
        fraction = min(max(float(fraction), 0.0), 1.0)
        indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
        if fraction >= 1.0 or indices.numel() == 0:
            return eligible.clone()
        keep = min(indices.numel(), int(math.floor(fraction * eligible.numel())))
        result = torch.zeros_like(eligible)
        if keep <= 0:
            return result
        selected = torch.topk(score[indices], k=keep, sorted=False).indices
        result[indices[selected]] = True
        return result

    @torch.no_grad()
    def select_degrees(self, config: dict):
        count, radius, dispersion, score = self.measurements()
        sh1 = (
            (count >= int(config.get("min_views_sh1", 3)))
            & (radius >= float(config.get("min_mean_radius_sh1", 1.0)))
            & (dispersion >= float(config.get("min_angular_dispersion_sh1", 1.0e-4)))
        )
        sh1 = self._top_fraction_mask(
            sh1, score, float(config.get("max_sh1_fraction", 1.0))
        )
        sh2 = (
            sh1
            & (count >= int(config.get("min_views_sh2", 6)))
            & (radius >= float(config.get("min_mean_radius_sh2", 2.0)))
            & (dispersion >= float(config.get("min_angular_dispersion_sh2", 5.0e-4)))
        )
        sh2 = self._top_fraction_mask(
            sh2, score, float(config.get("max_sh2_fraction", 0.35))
        )

        degrees = torch.zeros(
            self.gaussian_count, device=self.device, dtype=torch.uint8
        )
        degrees[sh1] = 1
        degrees[sh2] = 2
        selection_mode = str(config.get("selection_mode", "evidence"))
        if selection_mode == "shuffled":
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(config.get("shuffle_seed", 42)))
            degrees = degrees[
                torch.randperm(
                    self.gaussian_count, generator=generator, device=self.device
                )
            ]
        elif selection_mode != "evidence":
            raise ValueError(
                "appearance_lod.selection_mode must be evidence or shuffled"
            )

        visible = count > 0
        stats = {
            "selection_mode": selection_mode,
            "gaussian_count": self.gaussian_count,
            "observed_gaussians": int(visible.sum().item()),
            "degree_counts": {
                "sh0": int((degrees == 0).sum().item()),
                "sh1": int((degrees == 1).sum().item()),
                "sh2": int((degrees == 2).sum().item()),
            },
            "view_count_quantiles": _quantiles(count[visible]),
            "mean_radius_quantiles": _quantiles(radius[visible]),
            "angular_dispersion_quantiles": _quantiles(dispersion[visible]),
            "score_quantiles": _quantiles(score[visible]),
            "view_count_histogram": _histogram(
                count[visible], [0, 2, 4, 8, 16, 32, 64, 128, 4096]
            ),
            "mean_radius_histogram": _histogram(
                radius[visible], [0, 1, 2, 4, 8, 16, 32, 64, 4096]
            ),
            "angular_dispersion_histogram": _histogram(
                dispersion[visible],
                [0, 1.0e-4, 5.0e-4, 1.0e-3, 5.0e-3, 1.0e-2, 5.0e-2, 0.2, 1.000001],
            ),
        }
        return degrees, stats
