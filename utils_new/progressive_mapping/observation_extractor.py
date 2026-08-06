"""Patch observation extraction from stable-map coverage and image residuals."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .types import Observation


class ObservationExtractor:
    """Extract a bounded grid of RGB patch observations without learned features."""

    def __init__(self, config: Dict[str, object]):
        self.patch_size = int(config["patch_size"])
        self.patch_stride = int(config["patch_stride"])
        self.descriptor_resize = int(config["descriptor_resize"])
        self.appearance_grid_size = int(config["appearance_grid_size"])
        self.max_observations = int(config["max_observations_per_frame"])
        self.near_depth = float(config["near_observation_depth_m"])
        self.near_fraction = float(config["near_observation_fraction"])
        self.opacity_threshold = float(config["candidate_opacity_threshold"])
        self.residual_threshold = float(config["candidate_residual_threshold"])
        self.min_gradient = float(config["candidate_min_gradient"])

    @staticmethod
    def _as_scalar_map(value: torch.Tensor, height: int, width: int) -> torch.Tensor:
        value = value.squeeze()
        if value.shape != (height, width):
            raise ValueError("Expected a {}x{} scalar map, got {}".format(height, width, tuple(value.shape)))
        return value

    def _patch_bounds(self, uv: torch.Tensor, height: int, width: int) -> Tuple[int, int, int, int]:
        half = self.patch_size // 2
        center_x = int(round(float(uv[0].item())))
        center_y = int(round(float(uv[1].item())))
        x0 = max(0, min(width - 1, center_x - half))
        y0 = max(0, min(height - 1, center_y - half))
        x1 = max(x0 + 1, min(width, x0 + self.patch_size))
        y1 = max(y0 + 1, min(height, y0 + self.patch_size))
        x0 = max(0, x1 - self.patch_size)
        y0 = max(0, y1 - self.patch_size)
        return x0, y0, x1, y1

    def describe_patch(self, image: torch.Tensor, uv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a 4x4 RGB descriptor and mean color for a centered patch."""
        descriptors, mean_colors = self.describe_patches(image, uv.reshape(1, 2))
        return descriptors[0], mean_colors[0]

    def describe_patches(
        self, image: torch.Tensor, uvs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Describe projected patches in one interpolation batch."""
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("image must have HxWx3 layout")
        if uvs.ndim != 2 or uvs.shape[1] != 2:
            raise ValueError("uvs must have shape Nx2")
        if uvs.shape[0] == 0:
            descriptor_size = self.descriptor_resize * self.descriptor_resize * 3
            return (
                image.new_empty((0, descriptor_size)),
                image.new_empty((0, 3)),
            )
        height, width = image.shape[:2]
        half = self.patch_size // 2
        center_x = torch.round(uvs[:, 0]).to(torch.long)
        center_y = torch.round(uvs[:, 1]).to(torch.long)
        x0 = torch.clamp(center_x - half, min=0, max=width - 1)
        y0 = torch.clamp(center_y - half, min=0, max=height - 1)
        x1 = torch.clamp(x0 + self.patch_size, max=width)
        y1 = torch.clamp(y0 + self.patch_size, max=height)
        x0 = torch.clamp(x1 - self.patch_size, min=0)
        y0 = torch.clamp(y1 - self.patch_size, min=0)
        patches = self._gather_patches(image, x0, y0).permute(0, 3, 1, 2)
        resized = F.interpolate(
            patches,
            size=(self.descriptor_resize, self.descriptor_resize),
            mode="bilinear",
            align_corners=False,
        )
        descriptors = resized.permute(0, 2, 3, 1).reshape(uvs.shape[0], -1)
        descriptors = descriptors - descriptors.mean(dim=1, keepdim=True)
        descriptors = descriptors / torch.clamp(
            torch.linalg.norm(descriptors, dim=1, keepdim=True), min=1.0e-8
        )
        mean_colors = patches.mean(dim=(2, 3))
        return descriptors, mean_colors

    def appearance_grid(self, image: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """Return an absolute-color grid used to initialize spatial S appearance."""
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("image must have HxWx3 layout")
        height, width = image.shape[:2]
        x0, y0, x1, y1 = self._patch_bounds(uv, height, width)
        patch = image[y0:y1, x0:x1].permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(
            patch,
            size=(self.appearance_grid_size, self.appearance_grid_size),
            mode="bilinear",
            align_corners=False,
        )[0]
        return resized.permute(1, 2, 0)

    @staticmethod
    def _gradient_map(image: torch.Tensor) -> torch.Tensor:
        gray = image.mean(dim=-1)
        dx = torch.zeros_like(gray)
        dy = torch.zeros_like(gray)
        dx[:, 1:] = torch.abs(gray[:, 1:] - gray[:, :-1])
        dy[1:, :] = torch.abs(gray[1:, :] - gray[:-1, :])
        return torch.sqrt(dx.square() + dy.square())

    def _patch_depth_prior(
        self,
        depth: torch.Tensor,
        bounds: Tuple[int, int, int, int],
        global_median: float,
        global_uncertainty: float,
    ) -> Tuple[float, bool, float]:
        x0, y0, x1, y1 = bounds
        patch_depth = depth[y0:y1, x0:x1]
        valid_patch_depth = patch_depth[
            torch.isfinite(patch_depth) & (patch_depth > 0)
        ]
        if valid_patch_depth.numel() == 0:
            return global_median, False, global_uncertainty
        median = valid_patch_depth.median()
        uncertainty = torch.median(torch.abs(valid_patch_depth - median))
        return float(median.item()), True, float(uncertainty.item())

    def _batched_patch_bounds(
        self, xs: torch.Tensor, ys: torch.Tensor, height: int, width: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match _patch_bounds for a batch of integer candidate pixels."""
        half = self.patch_size // 2
        center_x = torch.round(xs.to(torch.float32) + 0.5).to(torch.long)
        center_y = torch.round(ys.to(torch.float32) + 0.5).to(torch.long)
        x0 = torch.clamp(center_x - half, min=0, max=width - 1)
        y0 = torch.clamp(center_y - half, min=0, max=height - 1)
        x1 = torch.clamp(x0 + self.patch_size, max=width)
        y1 = torch.clamp(y0 + self.patch_size, max=height)
        x0 = torch.clamp(x1 - self.patch_size, min=0)
        y0 = torch.clamp(y1 - self.patch_size, min=0)
        return x0, y0, x1, y1

    def _gather_patches(
        self, image: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor
    ) -> torch.Tensor:
        offsets = torch.arange(self.patch_size, device=image.device)
        xs = x0[:, None] + offsets[None, :]
        ys = y0[:, None] + offsets[None, :]
        return image[ys[:, :, None], xs[:, None, :]]

    def extract(
        self,
        frame_id: int,
        image: torch.Tensor,
        sparse_depth: torch.Tensor,
        render_opacity: torch.Tensor,
        residual: torch.Tensor,
        optional_invalid_mask: Optional[torch.Tensor] = None,
        optional_sky_mask: Optional[torch.Tensor] = None,
        optional_dynamic_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Observation], torch.Tensor]:
        """Extract observations and return the exact candidate mask used for spawning."""
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("image must have HxWx3 layout")
        height, width = image.shape[:2]
        opacity = self._as_scalar_map(render_opacity, height, width)
        residual_map = self._as_scalar_map(residual, height, width)
        depth = self._as_scalar_map(sparse_depth, height, width)
        invalid = torch.zeros((height, width), dtype=torch.bool, device=image.device)
        for mask in (optional_invalid_mask, optional_sky_mask, optional_dynamic_mask):
            if mask is not None:
                invalid |= self._as_scalar_map(mask, height, width).bool()
        invalid |= ~torch.isfinite(image).all(dim=-1)
        invalid |= image.abs().sum(dim=-1) <= 1.0e-8
        candidate_mask = ((opacity < self.opacity_threshold) | (residual_map > self.residual_threshold)) & ~invalid
        gradient = self._gradient_map(image)

        valid_depths = depth[torch.isfinite(depth) & (depth > 0)]
        if valid_depths.numel() > 0:
            global_median = float(valid_depths.median().item())
            q25 = float(torch.quantile(valid_depths, 0.25).item())
            q75 = float(torch.quantile(valid_depths, 0.75).item())
            global_uncertainty = max(1.0e-6, 0.5 * (q75 - q25))
        else:
            global_median = 0.0
            global_uncertainty = float("inf")

        stride = self.patch_stride
        grid_height = (height + stride - 1) // stride
        grid_width = (width + stride - 1) // stride
        padded_height = grid_height * stride
        padded_width = grid_width * stride

        def cells(value: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
            padded = F.pad(
                value,
                (0, padded_width - width, 0, padded_height - height),
                value=pad_value,
            )
            return padded.reshape(grid_height, stride, grid_width, stride).permute(
                0, 2, 1, 3
            )

        candidate_cells = cells(candidate_mask.float())
        valid_cells = cells(torch.ones_like(candidate_mask, dtype=image.dtype))
        cell_counts = torch.clamp(valid_cells.sum(dim=(2, 3)), min=1.0)
        ratios = candidate_cells.sum(dim=(2, 3)) / cell_counts
        score_cells = candidate_cells * (
            cells(gradient) + cells(residual_map) + ratios[:, :, None, None]
        )
        flat_indices = score_cells.reshape(grid_height, grid_width, -1).argmax(dim=-1)
        cell_y = torch.arange(grid_height, device=image.device)[:, None]
        cell_x = torch.arange(grid_width, device=image.device)[None, :]
        ys = cell_y * stride + torch.div(flat_indices, stride, rounding_mode="floor")
        xs = cell_x * stride + flat_indices.remainder(stride)
        inside = (ys < height) & (xs < width) & (ratios > 0.0)
        ys = ys[inside]
        xs = xs[inside]
        ratios = ratios[inside]
        gradient_scores = gradient[ys, xs]
        residual_scores = residual_map[ys, xs]
        eligible = (gradient_scores >= self.min_gradient) | (
            residual_scores >= self.residual_threshold
        )
        ys = ys[eligible]
        xs = xs[eligible]
        ratios = ratios[eligible]
        gradient_scores = gradient_scores[eligible]
        residual_scores = residual_scores[eligible]

        if xs.numel() == 0:
            return [], candidate_mask

        x0, y0, x1, y1 = self._batched_patch_bounds(xs, ys, height, width)
        depth_patches = self._gather_patches(depth, x0, y0).reshape(xs.numel(), -1)
        valid_depth = torch.isfinite(depth_patches) & (depth_patches > 0)
        has_depth = valid_depth.any(dim=1)
        depth_for_median = depth_patches.masked_fill(~valid_depth, float("nan"))
        patch_median = torch.nanmedian(depth_for_median, dim=1).values
        patch_uncertainty = torch.nanmedian(
            torch.abs(depth_for_median - patch_median[:, None]), dim=1
        ).values
        depth_priors = torch.where(
            has_depth,
            patch_median,
            torch.full_like(patch_median, global_median),
        )
        depth_uncertainties = torch.where(
            has_depth,
            patch_uncertainty,
            torch.full_like(patch_uncertainty, global_uncertainty),
        )

        rank_order = torch.argsort(
            ratios + gradient_scores + residual_scores, descending=True
        )
        near_mask = has_depth & (depth_priors > 0.0) & (
            depth_priors <= self.near_depth
        )
        near_ranked = rank_order[near_mask[rank_order]]
        other_ranked = rank_order[~near_mask[rank_order]]
        near_quota = min(
            near_ranked.numel(), int(round(self.max_observations * self.near_fraction))
        )
        selected = torch.cat(
            (
                near_ranked[:near_quota],
                other_ranked[: self.max_observations - near_quota],
            )
        )
        if selected.numel() < self.max_observations:
            selected = torch.cat(
                (
                    selected,
                    near_ranked[
                        near_quota : near_quota
                        + self.max_observations
                        - selected.numel()
                    ],
                )
            )
        selected = selected[depth_priors[selected] > 0.0]
        if selected.numel() == 0:
            return [], candidate_mask

        xs = xs[selected]
        ys = ys[selected]
        x0 = x0[selected]
        y0 = y0[selected]
        x1 = x1[selected]
        y1 = y1[selected]
        image_patches = self._gather_patches(image, x0, y0).permute(0, 3, 1, 2)
        descriptor_images = F.interpolate(
            image_patches,
            size=(self.descriptor_resize, self.descriptor_resize),
            mode="bilinear",
            align_corners=False,
        )
        descriptors = descriptor_images.permute(0, 2, 3, 1).reshape(
            selected.numel(), -1
        )
        descriptors = descriptors - descriptors.mean(dim=1, keepdim=True)
        descriptors = descriptors / torch.clamp(
            torch.linalg.norm(descriptors, dim=1, keepdim=True), min=1.0e-8
        )
        mean_colors = image_patches.mean(dim=(2, 3))
        appearance_grids = F.interpolate(
            image_patches,
            size=(self.appearance_grid_size, self.appearance_grid_size),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)

        depth_prior_values = depth_priors[selected].tolist()
        depth_valid_values = has_depth[selected].tolist()
        depth_uncertainty_values = depth_uncertainties[selected].tolist()
        depth_support_values = valid_depth[selected].sum(dim=1).tolist()
        gradient_values = gradient_scores[selected].tolist()
        residual_values = residual_scores[selected].tolist()
        observations: List[Observation] = []
        for index in range(selected.numel()):
            observations.append(
                Observation(
                    frame_id=frame_id,
                    uv=torch.stack((xs[index], ys[index])).to(image.dtype) + 0.5,
                    patch_bbox=torch.stack(
                        (x0[index], y0[index], x1[index], y1[index])
                    ),
                    descriptor=descriptors[index],
                    mean_color=mean_colors[index],
                    appearance_grid=appearance_grids[index],
                    depth_prior=depth_prior_values[index],
                    depth_valid=depth_valid_values[index],
                    depth_uncertainty=depth_uncertainty_values[index],
                    gradient_score=gradient_values[index],
                    residual_score=residual_values[index],
                    depth_support=depth_support_values[index],
                )
            )
        return observations, candidate_mask
