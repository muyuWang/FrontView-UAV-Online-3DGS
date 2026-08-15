"""Packed one-pass rasterization with per-Gaussian SH degree routing."""

from __future__ import annotations

import math

import torch
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    spherical_harmonics,
)


def heterogeneous_sh_rasterization(
    means,
    quats,
    scales,
    opacities,
    sh_coefficients,
    sh_degrees,
    viewmats,
    Ks,
    width,
    height,
    *,
    metric_confidences=None,
    appearance_confidences=None,
    uncertainty_confidences=None,
    uncertainty_cell_px=None,
    base_degree=2,
    target_degree=3,
    probe_inactive=False,
    near_plane=0.01,
    far_plane=1.0e10,
    radius_clip=0.0,
    eps2d=0.3,
    tile_size=16,
    backgrounds=None,
    render_mode="RGB",
    rasterize_mode="classic",
):
    """Render mixed SH degrees after one packed projection and one tile blend.

    The target band is evaluated for promoted rows and, when requested, for
    inactive probe rows. Probe rows keep zero target coefficients but expose
    their counterfactual target-band gradient to TGBR.
    """

    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError("heterogeneous SH rasterization expects means [N, 3]")
    if viewmats.ndim != 3 or Ks.ndim != 3:
        raise ValueError("heterogeneous SH rasterization expects camera batches")
    if sh_degrees.shape != means.shape[:1]:
        raise ValueError("SH degree rows must align with Gaussian rows")
    if (
        metric_confidences is not None
        and metric_confidences.shape != means.shape[:1]
    ):
        raise ValueError("Metric confidence rows must align with Gaussian rows")
    if (
        uncertainty_confidences is not None
        and uncertainty_confidences.shape != means.shape[:1]
    ):
        raise ValueError("Uncertainty confidence rows must align with Gaussian rows")
    if (
        appearance_confidences is not None
        and appearance_confidences.shape != means.shape[:1]
    ):
        raise ValueError("Appearance confidence rows must align with Gaussian rows")
    if uncertainty_cell_px is not None and float(uncertainty_cell_px) <= 0.0:
        raise ValueError("uncertainty_cell_px must be positive")
    if uncertainty_cell_px is not None and metric_confidences is None:
        raise ValueError("Projected uncertainty mass requires metric confidences")
    if render_mode not in {"RGB", "D", "ED", "RGB+D", "RGB+ED"}:
        raise ValueError("Unsupported render mode")
    if int(base_degree) >= int(target_degree):
        raise ValueError("base_degree must be lower than target_degree")
    if sh_coefficients.shape != (
        means.shape[0],
        (int(target_degree) + 1) ** 2,
        3,
    ):
        raise ValueError("SH coefficients do not match the target degree")

    projection = fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=True,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=False,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model="pinhole",
        opacities=opacities,
    )
    (
        batch_ids,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        compensations,
    ) = projection

    packed_opacities = opacities[gaussian_ids]
    if compensations is not None:
        packed_opacities = packed_opacities * compensations

    camera_centers = torch.inverse(viewmats)[:, :3, 3]
    directions = means[gaussian_ids] - camera_centers[camera_ids]
    packed_degrees = sh_degrees[gaussian_ids]
    promoted_mask = packed_degrees >= int(target_degree)
    target_mask = (
        torch.ones_like(promoted_mask) if probe_inactive else promoted_mask
    )
    base_mask = ~target_mask
    base_indices = torch.nonzero(base_mask, as_tuple=False).reshape(-1)
    target_indices = torch.nonzero(target_mask, as_tuple=False).reshape(-1)

    packed_colors = means.new_zeros((gaussian_ids.numel(), 3))
    if base_indices.numel():
        base_gaussian_ids = gaussian_ids[base_indices]
        base_coefficient_count = (int(base_degree) + 1) ** 2
        base_colors = spherical_harmonics(
            int(base_degree),
            directions[base_indices],
            sh_coefficients[base_gaussian_ids, :base_coefficient_count],
        )
        packed_colors = packed_colors.index_copy(0, base_indices, base_colors)
    if target_indices.numel():
        target_gaussian_ids = gaussian_ids[target_indices]
        target_colors = spherical_harmonics(
            int(target_degree),
            directions[target_indices],
            sh_coefficients[target_gaussian_ids],
        )
        packed_colors = packed_colors.index_copy(0, target_indices, target_colors)
    packed_colors = torch.clamp_min(packed_colors + 0.5, 0.0)

    num_cameras = viewmats.shape[0]
    metric_depth = metric_confidences is not None and render_mode in {
        "RGB+D",
        "RGB+ED",
        "D",
        "ED",
    }
    if metric_depth:
        packed_metric = torch.clamp(
            metric_confidences[gaussian_ids].to(packed_colors), 0.0, 1.0
        )
    appearance_depth_enabled = (
        metric_depth and appearance_confidences is not None
    )
    if appearance_depth_enabled:
        packed_appearance = torch.clamp(
            appearance_confidences[gaussian_ids].to(packed_colors), 0.0, 1.0
        )
    uncertainty_mass_enabled = metric_depth and uncertainty_cell_px is not None
    if uncertainty_mass_enabled:
        packed_uncertainty_confidence = torch.clamp(
            (
                uncertainty_confidences
                if uncertainty_confidences is not None
                else metric_confidences
            )[gaussian_ids].to(packed_colors),
            0.0,
            1.0,
        )
        projected_radius = radii.to(packed_colors)
        if projected_radius.ndim > 1:
            projected_radius = torch.amax(projected_radius, dim=-1)
        uncertainty_radius = projected_radius * torch.sqrt(
            torch.clamp(1.0 - packed_uncertainty_confidence, min=0.0)
        )
        packed_uncertainty = (
            uncertainty_radius > float(uncertainty_cell_px)
        ).to(packed_colors)
    if render_mode in {"RGB+D", "RGB+ED"}:
        if metric_depth:
            channels = (
                packed_colors,
                depths[:, None],
                depths[:, None] * packed_metric[:, None],
                packed_metric[:, None],
            )
            if appearance_depth_enabled:
                channels += (
                    depths[:, None] * packed_appearance[:, None],
                    packed_appearance[:, None],
                )
            if uncertainty_mass_enabled:
                channels += (packed_uncertainty[:, None],)
            packed_colors = torch.cat(channels, dim=-1)
        else:
            packed_colors = torch.cat((packed_colors, depths[:, None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                (
                    backgrounds,
                    torch.zeros(
                        (
                            num_cameras,
                            (
                                3
                                + 2 * int(appearance_depth_enabled)
                                + int(uncertainty_mass_enabled)
                            )
                            if metric_depth else 1,
                        ),
                        device=backgrounds.device,
                        dtype=backgrounds.dtype,
                    ),
                ),
                dim=-1,
            )
    elif render_mode in {"D", "ED"}:
        if metric_depth:
            channels = (depths, depths * packed_metric, packed_metric)
            if appearance_depth_enabled:
                channels += (depths * packed_appearance, packed_appearance)
            if uncertainty_mass_enabled:
                channels += (packed_uncertainty,)
            packed_colors = torch.stack(channels, dim=-1)
        else:
            packed_colors = depths[:, None]
        if backgrounds is not None:
            backgrounds = torch.zeros(
                (
                    num_cameras,
                    (
                        3
                        + 2 * int(appearance_depth_enabled)
                        + int(uncertainty_mass_enabled)
                    )
                    if metric_depth else 1,
                ),
                device=backgrounds.device,
                dtype=backgrounds.dtype,
            )

    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    image_ids = batch_ids * num_cameras + camera_ids
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=True,
        n_images=num_cameras,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(
        isect_ids, num_cameras, tile_width, tile_height
    ).reshape(num_cameras, tile_height, tile_width)
    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        packed_colors,
        packed_opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=True,
        absgrad=False,
    )
    appearance_depth = None
    appearance_mass = None
    uncertainty_mass = None
    if metric_depth:
        prefix_channels = 3 if render_mode in {"RGB+D", "RGB+ED"} else 0
        full_numerator = render_colors[..., prefix_channels : prefix_channels + 1]
        metric_numerator = render_colors[
            ..., prefix_channels + 1 : prefix_channels + 2
        ]
        metric_mass = render_colors[
            ..., prefix_channels + 2 : prefix_channels + 3
        ]
        next_channel = prefix_channels + 3
        if appearance_depth_enabled:
            appearance_numerator = render_colors[
                ..., next_channel : next_channel + 1
            ]
            appearance_mass = render_colors[
                ..., next_channel + 1 : next_channel + 2
            ]
            appearance_depth = appearance_numerator / appearance_mass.clamp_min(
                1.0e-10
            )
            next_channel += 2
        if uncertainty_mass_enabled:
            uncertainty_mass = render_colors[
                ..., next_channel : next_channel + 1
            ]
        metric_expected = metric_numerator / metric_mass.clamp_min(1.0e-10)
        full_depth = full_numerator
        if render_mode in {"ED", "RGB+ED"}:
            full_depth = full_depth / render_alphas.clamp_min(1.0e-10)
        if render_mode in {"RGB+D", "RGB+ED"}:
            render_colors = torch.cat(
                (
                    render_colors[..., :prefix_channels],
                    full_depth,
                    metric_expected,
                ),
                dim=-1,
            )
        else:
            render_colors = torch.cat((full_depth, metric_expected), dim=-1)
    elif render_mode in {"ED", "RGB+ED"}:
        render_colors = torch.cat(
            (
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp_min(1.0e-10),
            ),
            dim=-1,
        )

    route_stats = {
        "packed_rows": int(gaussian_ids.numel()),
        "base_rows": base_mask.sum().detach(),
        "promoted_target_rows": promoted_mask.sum().detach(),
        "probe_rows": ((~promoted_mask) & target_mask).sum().detach(),
        "target_rows": target_mask.sum().detach(),
        "skipped_target_band_rows": base_mask.sum().detach(),
    }
    meta = {
        "batch_ids": batch_ids,
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": packed_opacities,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_batches": 1,
        "n_cameras": num_cameras,
        "heterogeneous_sh": route_stats,
        "metric_depth_mass": metric_mass if metric_depth else None,
        "appearance_depth": appearance_depth,
        "appearance_depth_mass": appearance_mass,
        "uncertainty_mass": uncertainty_mass if metric_depth else None,
    }
    return render_colors, render_alphas, meta
