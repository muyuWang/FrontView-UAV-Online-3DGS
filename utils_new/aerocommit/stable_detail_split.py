"""Multi-view selection and parameter-preserving stable-map detail splits."""

import math
from typing import Mapping

import torch


def stable_detail_split_scores(
    image: torch.Tensor,
    projection_info: Mapping[str, torch.Tensor],
    config,
    rendered: torch.Tensor | None = None,
):
    """Score packed visible Gaussians whose large footprints cover side detail."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Stable detail selection expects an HxWx3 image")
    gaussian_ids = projection_info.get("gaussian_ids")
    if gaussian_ids is None:
        raise ValueError("Stable detail selection requires packed Gaussian IDs")
    means2d = projection_info["means2d"]
    radii = projection_info["radii"]
    depths = projection_info["depths"].reshape(-1)
    if radii.ndim > 1:
        radii = radii.amax(dim=-1)
    radii = radii.reshape(-1).to(image.dtype)
    gaussian_ids = gaussian_ids.reshape(-1).long()

    height, width = image.shape[:2]
    rgb_weights = image.new_tensor((0.299, 0.587, 0.114))
    gray = (image * rgb_weights).sum(dim=-1)
    gradient = torch.zeros_like(gray)
    gradient[:, 1:] += torch.abs(gray[:, 1:] - gray[:, :-1])
    gradient[1:, :] += torch.abs(gray[1:, :] - gray[:-1, :])

    x = torch.floor(means2d[:, 0]).long()
    y = torch.floor(means2d[:, 1]).long()
    valid_pixel = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    sampled_gradient = torch.zeros_like(radii)
    sampled_gradient[valid_pixel] = gradient[y[valid_pixel], x[valid_pixel]]
    sampled_residual = torch.ones_like(radii)
    residual_threshold = float(config.get("residual_threshold", 0.0))
    if rendered is not None:
        if rendered.shape != image.shape:
            raise ValueError("Stable detail render must match the GT image")
        residual = torch.abs(image - rendered).mean(dim=-1)
        sampled_residual.zero_()
        sampled_residual[valid_pixel] = residual[y[valid_pixel], x[valid_pixel]]
    lateral = torch.abs(means2d[:, 0] - 0.5 * width) / max(0.5 * width, 1.0)

    eligible = (
        valid_pixel
        & (depths > 0.0)
        & (depths <= float(config.get("near_depth_m", 60.0)))
        & (radii >= float(config.get("min_projected_radius_px", 6.0)))
        & (sampled_gradient >= float(config.get("gradient_threshold", 0.04)))
        & (sampled_residual >= residual_threshold)
        & (lateral >= float(config.get("side_start", 0.45)))
    )
    score = radii * sampled_gradient * sampled_residual * (1.0 + lateral)
    score = torch.where(eligible, score, torch.full_like(score, -torch.inf))
    return gaussian_ids, score


def split_gaussian_parameters(
    params,
    selected_indices: torch.Tensor,
    tangent_x: torch.Tensor,
    tangent_y: torch.Tensor,
    scale_ratio: float = 0.55,
    offset_fraction: float = 0.35,
):
    """Replace selected parents by four opacity-preserving tangent-plane children."""
    selected = selected_indices.reshape(-1).long()
    if selected.numel() == 0:
        return {name: value.detach().clone() for name, value in params.items()}
    count = params["means"].shape[0]
    if selected.min() < 0 or selected.max() >= count:
        raise ValueError("Stable detail split index is out of range")
    if tangent_x.shape != (len(selected), 3) or tangent_y.shape != (len(selected), 3):
        raise ValueError("Each stable detail parent requires two camera tangents")

    keep = torch.ones((count,), device=selected.device, dtype=torch.bool)
    keep[selected] = False
    signs = params["means"].new_tensor(
        ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
    )
    parent_scale = torch.exp(params["scales"][selected]).amax(dim=1)
    offsets = (
        signs[:, None, 0:1] * tangent_x[None]
        + signs[:, None, 1:2] * tangent_y[None]
    ) * parent_scale[None, :, None] * float(offset_fraction)
    child_means = (params["means"][selected][None] + offsets).reshape(-1, 3)

    output = {}
    repeat = len(signs)
    for name, value in params.items():
        if name == "means":
            children = child_means
        elif name == "scales":
            children = (
                value[selected][None] + math.log(float(scale_ratio))
            ).expand(repeat, -1, -1).reshape(-1, value.shape[1])
        elif name == "opacities":
            parent_alpha = torch.sigmoid(value[selected])
            child_alpha = 1.0 - torch.pow(1.0 - parent_alpha, 1.0 / repeat)
            child_logits = torch.logit(child_alpha.clamp(1.0e-4, 1.0 - 1.0e-4))
            children = child_logits[None].expand(repeat, -1).reshape(-1)
        else:
            children = value[selected][None].expand(
                (repeat,) + value[selected].shape
            ).reshape((-1,) + value.shape[1:])
        output[name] = torch.cat((value[keep], children), dim=0)
    return output
