"""Cross-fitted sparse-world calibration for forward-view depth proposals."""

from copy import deepcopy

import torch


DEFAULT_FRONT_VIEW_DEPTH_TRANSPORT_CONFIG = {
    "enabled": False,
    "apply_correction": True,
    "calibration_fraction": 0.10,
    "min_training_anchors": 32,
    "min_calibration_anchors": 8,
    "neighbors": 8,
    "clip_quantiles": [0.10, 0.90],
    "shuffle_residual_locations": False,
    "split_seed": 42,
}


def validate_front_view_depth_transport_config(config=None):
    merged = deepcopy(DEFAULT_FRONT_VIEW_DEPTH_TRANSPORT_CONFIG)
    if config is not None:
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError(
                "Unknown FrontViewDepthTransport options: {}".format(
                    sorted(unknown)
                )
            )
        merged.update(config)
    for key in ("enabled", "apply_correction", "shuffle_residual_locations"):
        if not isinstance(merged[key], bool):
            raise TypeError("FrontViewDepthTransport.{} must be boolean".format(key))
    fraction = float(merged["calibration_fraction"])
    if not 0.0 < fraction < 0.5:
        raise ValueError(
            "FrontViewDepthTransport.calibration_fraction must be in (0, 0.5)"
        )
    for key in ("min_training_anchors", "min_calibration_anchors", "neighbors"):
        if not isinstance(merged[key], int) or merged[key] <= 0:
            raise ValueError("FrontViewDepthTransport.{} must be positive".format(key))
    quantiles = [float(value) for value in merged["clip_quantiles"]]
    if len(quantiles) != 2 or not 0.0 <= quantiles[0] < quantiles[1] <= 1.0:
        raise ValueError(
            "FrontViewDepthTransport.clip_quantiles must be two increasing values"
        )
    if not isinstance(merged["split_seed"], int):
        raise TypeError("FrontViewDepthTransport.split_seed must be an integer")
    return merged


def split_depth_anchors(
    count: int,
    calibration_fraction: float,
    min_training_anchors: int,
    min_calibration_anchors: int,
    *,
    seed: int,
    device,
):
    """Split anchors before GP conditioning so calibration residuals are held out."""

    count = int(count)
    all_indices = torch.arange(count, device=device, dtype=torch.long)
    if count < int(min_training_anchors) + int(min_calibration_anchors):
        return all_indices, all_indices[:0]
    calibration_count = max(
        int(min_calibration_anchors), int(round(count * float(calibration_fraction)))
    )
    calibration_count = min(calibration_count, count - int(min_training_anchors))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    order = torch.randperm(count, generator=generator, device=device)
    calibration = torch.sort(order[:calibration_count]).values
    training = torch.sort(order[calibration_count:]).values
    return training, calibration


def _sample_rgb(image: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Depth-transport image must have shape HxWx3")
    height, width = image.shape[:2]
    x = torch.clamp(coords[:, 0].long(), 0, width - 1)
    y = torch.clamp(coords[:, 1].long(), 0, height - 1)
    return image[y, x]


def transport_candidate_depths(
    candidate_coords: torch.Tensor,
    candidate_depths: torch.Tensor,
    calibration_coords: torch.Tensor,
    calibration_pred_depths: torch.Tensor,
    calibration_true_depths: torch.Tensor,
    image: torch.Tensor,
    *,
    neighbors: int,
    clip_quantiles,
    shuffle_residual_locations: bool = False,
    seed: int = 42,
):
    """Transport candidate depth using out-of-sample landmark residuals."""

    if candidate_coords.ndim != 2 or candidate_coords.shape[1] != 2:
        raise ValueError("Candidate coordinates must have shape Nx2")
    if calibration_coords.ndim != 2 or calibration_coords.shape[1] != 2:
        raise ValueError("Calibration coordinates must have shape Mx2")
    if candidate_depths.shape != (len(candidate_coords),):
        raise ValueError("Candidate depths must match candidate coordinates")
    if calibration_pred_depths.shape != (len(calibration_coords),):
        raise ValueError("Predicted calibration depths must match coordinates")
    if calibration_true_depths.shape != (len(calibration_coords),):
        raise ValueError("True calibration depths must match coordinates")
    if len(candidate_coords) == 0 or len(calibration_coords) == 0:
        return candidate_depths, torch.zeros_like(candidate_depths)

    eps = torch.finfo(candidate_depths.dtype).eps
    residuals = torch.log(torch.clamp(calibration_true_depths, min=eps)) - torch.log(
        torch.clamp(calibration_pred_depths, min=eps)
    )
    low, high = (float(value) for value in clip_quantiles)
    bounds = torch.quantile(residuals, torch.tensor([low, high], device=residuals.device))
    residuals = torch.clamp(residuals, bounds[0], bounds[1])
    if shuffle_residual_locations and len(residuals) > 1:
        generator = torch.Generator(device=residuals.device)
        generator.manual_seed(int(seed))
        residuals = residuals[
            torch.randperm(len(residuals), generator=generator, device=residuals.device)
        ]

    calibration_distance = torch.cdist(calibration_coords, calibration_coords)
    calibration_distance.fill_diagonal_(float("inf"))
    sigma_pixel = torch.median(torch.min(calibration_distance, dim=1).values)
    sigma_pixel = torch.clamp(sigma_pixel, min=1.0)

    calibration_color = _sample_rgb(image, calibration_coords)
    color_distance = torch.cdist(calibration_color, calibration_color, p=1)
    color_distance.fill_diagonal_(float("inf"))
    sigma_color = torch.median(torch.min(color_distance, dim=1).values)
    sigma_color = torch.clamp(sigma_color, min=1.0 / 255.0)

    candidate_color = _sample_rgb(image, candidate_coords)
    distances = torch.cdist(candidate_coords, calibration_coords)
    neighbor_count = min(int(neighbors), len(calibration_coords))
    nearest_distance, nearest = torch.topk(
        distances, neighbor_count, dim=1, largest=False, sorted=False
    )
    nearest_color = calibration_color[nearest]
    color_delta = torch.sum(
        torch.abs(candidate_color[:, None, :] - nearest_color), dim=-1
    )
    weights = torch.exp(
        -0.5 * (nearest_distance / sigma_pixel).square()
        - color_delta / sigma_color
    )
    corrections = torch.sum(weights * residuals[nearest], dim=1) / torch.clamp(
        torch.sum(weights, dim=1), min=eps
    )
    return candidate_depths * torch.exp(corrections), corrections
