"""Separate projective appearance support from metric depth responsibility."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import torch


DEFAULT_CAUSAL_DUAL_RESPONSIBILITY_CONFIG = {
    "enabled": False,
    "depthcov_confidence_mode": "posterior",
    "finite_depth_certificate_enabled": False,
    "finite_depth_certificate_scope": "all_unverified",
    "finite_depth_pixel_sigma": 1.0,
    "finite_depth_preserve_appearance_ownership": True,
    "directional_use_metric_depth": True,
    "geometry_use_metric_depth": True,
    "export_metric_depth": True,
    "minimum_metric_opacity": 1.0e-6,
}


def validate_causal_dual_responsibility_config(config=None):
    result = deepcopy(DEFAULT_CAUSAL_DUAL_RESPONSIBILITY_CONFIG)
    if config is not None:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(
                "Unknown CausalDualResponsibility options: {}".format(
                    sorted(unknown)
                )
            )
        result.update(config)
    for key in (
        "enabled",
        "finite_depth_certificate_enabled",
        "finite_depth_preserve_appearance_ownership",
        "directional_use_metric_depth",
        "geometry_use_metric_depth",
        "export_metric_depth",
    ):
        if not isinstance(result[key], bool):
            raise TypeError(
                "CausalDualResponsibility.{} must be boolean".format(key)
            )
    if result["depthcov_confidence_mode"] not in ("posterior", "binary"):
        raise ValueError(
            "CausalDualResponsibility.depthcov_confidence_mode must be "
            "posterior or binary"
        )
    if result["finite_depth_certificate_scope"] not in (
        "depthcov",
        "all_unverified",
    ):
        raise ValueError(
            "CausalDualResponsibility.finite_depth_certificate_scope must be "
            "depthcov or all_unverified"
        )
    pixel_sigma = float(result["finite_depth_pixel_sigma"])
    if not math.isfinite(pixel_sigma) or pixel_sigma <= 0.0:
        raise ValueError(
            "CausalDualResponsibility.finite_depth_pixel_sigma must be positive"
        )
    result["finite_depth_pixel_sigma"] = pixel_sigma
    minimum_opacity = float(result["minimum_metric_opacity"])
    if not math.isfinite(minimum_opacity) or minimum_opacity < 0.0:
        raise ValueError(
            "CausalDualResponsibility.minimum_metric_opacity must be nonnegative"
        )
    result["minimum_metric_opacity"] = minimum_opacity
    return result


def _sample_image(image, uv):
    image = torch.as_tensor(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Certificate images must have shape HxWx3")
    height, width = image.shape[:2]
    grid = torch.stack(
        (
            2.0 * uv[:, 0] / float(width) - 1.0,
            2.0 * uv[:, 1] / float(height) - 1.0,
        ),
        dim=1,
    ).reshape(1, -1, 1, 2)
    sampled = torch.nn.functional.grid_sample(
        image.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[0, :, :, 0].T


def _blurred_exposure_normalized_image(camera, level, device, dtype):
    image = camera.get_gt_image(level).to(device=device, dtype=dtype)
    image = image / max(float(camera.exposure_gain), 1.0e-8)
    return torch.nn.functional.avg_pool2d(
        image.permute(2, 0, 1).unsqueeze(0),
        kernel_size=3,
        stride=1,
        padding=1,
    )[0].permute(1, 2, 0)


@torch.no_grad()
def causal_finite_depth_certificates(
    camera,
    reference_cameras,
    pixels,
    depths,
    level,
    *,
    pixel_sigma=1.0,
):
    """Certify finite depth against the infinite-ray explanation.

    A candidate is observable when its finite-depth SE(3) projection separates
    from its rotation-only projection by more than image quantization noise.
    Metric responsibility is retained only to the extent that the finite-depth
    projection explains past appearance better. Unobservable rows abstain and
    preserve their prior confidence.
    """

    pixels = torch.as_tensor(pixels)
    depths = torch.as_tensor(
        depths, device=pixels.device, dtype=pixels.dtype
    ).reshape(-1)
    count = int(pixels.shape[0])
    if pixels.shape != (count, 2) or depths.shape != (count,):
        raise ValueError("Certificate pixels and depths must align")
    if count == 0:
        empty = torch.empty(0, device=pixels.device, dtype=torch.float32)
        return {
            "certificate": empty,
            "observability": empty,
            "finite_support": empty,
            "valid_views": torch.empty(
                0, device=pixels.device, dtype=torch.int32
            ),
        }
    if not math.isfinite(float(pixel_sigma)) or float(pixel_sigma) <= 0.0:
        raise ValueError("Certificate pixel sigma must be positive")
    references = list(reference_cameras or ())
    ones = torch.ones(count, device=pixels.device, dtype=pixels.dtype)
    if not references:
        return {
            "certificate": ones,
            "observability": torch.zeros_like(ones),
            "finite_support": torch.zeros_like(ones),
            "valid_views": torch.zeros(
                count, device=pixels.device, dtype=torch.int32
            ),
        }

    current_pose = camera.get_raw_pose().detach().to(pixels)
    current_intrinsics = camera.get_int_mat(level).detach().to(pixels)
    homogeneous = torch.cat((pixels, ones[:, None]), dim=1)
    camera_rays = homogeneous @ torch.linalg.inv(current_intrinsics.T)
    finite_local = camera_rays * depths[:, None]
    finite_world = torch.cat((finite_local, ones[:, None]), dim=1)
    finite_world = finite_world @ torch.linalg.inv(current_pose.T)
    world_rays = camera_rays @ torch.linalg.inv(current_pose[:3, :3].T)

    current_image = _blurred_exposure_normalized_image(
        camera, level, pixels.device, pixels.dtype
    )
    current_color = _sample_image(current_image, pixels)
    maximum_observability = torch.zeros_like(depths)
    maximum_support = torch.zeros_like(depths)
    valid_views = torch.zeros(count, device=pixels.device, dtype=torch.int32)
    sigma = float(pixel_sigma)

    for reference in references:
        pose = reference.get_raw_pose().detach().to(pixels)
        intrinsics = reference.get_int_mat(level).detach().to(pixels)
        finite_camera = finite_world @ pose.T
        finite_z = finite_camera[:, 2]
        finite_projection = finite_camera[:, :3] @ intrinsics.T
        finite_uv = finite_projection[:, :2] / torch.clamp(
            finite_z[:, None], min=1.0e-8
        )

        infinite_camera = world_rays @ pose[:3, :3].T
        infinite_z = infinite_camera[:, 2]
        infinite_projection = infinite_camera @ intrinsics.T
        infinite_uv = infinite_projection[:, :2] / torch.clamp(
            infinite_z[:, None], min=1.0e-8
        )

        width = float(reference.get_width(level))
        height = float(reference.get_height(level))
        valid = (
            (finite_z > float(reference.near))
            & (finite_z < float(reference.far))
            & (infinite_z > 0.0)
            & (finite_uv[:, 0] >= 0.0)
            & (finite_uv[:, 0] < width)
            & (finite_uv[:, 1] >= 0.0)
            & (finite_uv[:, 1] < height)
            & (infinite_uv[:, 0] >= 0.0)
            & (infinite_uv[:, 0] < width)
            & (infinite_uv[:, 1] >= 0.0)
            & (infinite_uv[:, 1] < height)
        )
        reference_image = _blurred_exposure_normalized_image(
            reference, level, pixels.device, pixels.dtype
        )
        finite_color = _sample_image(reference_image, finite_uv)
        infinite_color = _sample_image(reference_image, infinite_uv)
        finite_error = torch.mean(torch.abs(current_color - finite_color), dim=1)
        infinite_error = torch.mean(
            torch.abs(current_color - infinite_color), dim=1
        )
        displacement = torch.linalg.vector_norm(
            finite_uv - infinite_uv, dim=1
        )
        observability = 1.0 - torch.exp(
            -0.5 * (displacement / sigma).square()
        )
        finite_advantage = torch.clamp(
            (infinite_error - finite_error)
            / torch.clamp(infinite_error, min=1.0e-6),
            0.0,
            1.0,
        )
        observability = torch.where(
            valid, observability, torch.zeros_like(observability)
        )
        support = observability * torch.where(
            valid, finite_advantage, torch.zeros_like(finite_advantage)
        )
        maximum_observability = torch.maximum(
            maximum_observability, observability
        )
        maximum_support = torch.maximum(maximum_support, support)
        valid_views += valid.to(torch.int32)

    certificate = torch.clamp(
        1.0 - maximum_observability + maximum_support, 0.0, 1.0
    )
    certificate = torch.where(valid_views > 0, certificate, ones)
    return {
        "certificate": certificate,
        "observability": maximum_observability,
        "finite_support": maximum_support,
        "valid_views": valid_views,
    }


def proposal_metric_confidences(
    source_kinds,
    depth_confidences,
    depth_fallback,
    config,
):
    """Assign metric mass without discarding appearance-only proposals.

    Persistent sparse observations and parallax-certified tracks carry unit
    metric mass. DepthCov rows carry their posterior confidence, while rows
    admitted only by the unresolved-depth fallback carry no metric mass.
    """

    config = validate_causal_dual_responsibility_config(config)
    source_kinds = np.asarray(source_kinds, dtype="U32").reshape(-1)
    depth_confidences = np.asarray(depth_confidences, dtype=np.float32).reshape(-1)
    depth_fallback = np.asarray(depth_fallback, dtype=np.bool_).reshape(-1)
    if not (
        len(source_kinds) == len(depth_confidences) == len(depth_fallback)
    ):
        raise ValueError("Dual-responsibility proposal fields must align")
    confidence = np.ones((len(source_kinds),), dtype=np.float32)
    depthcov = np.char.startswith(source_kinds, "depthcov")
    if config["depthcov_confidence_mode"] == "posterior":
        confidence[depthcov] = np.clip(
            depth_confidences[depthcov], 0.0, 1.0
        )
    confidence[depth_fallback] = 0.0
    confidence[source_kinds == "tracked_metric"] = 1.0
    confidence[source_kinds == "sparse"] = 1.0
    return confidence


def geometry_decision_render(render_package):
    """Return the base GS image used by mapping and geometry decisions."""

    return render_package.get("geometry_render", render_package["render"])


def nearest_unique_replacement_positions(candidate_uv, fallback_positions, tracked_uv):
    """Greedily bind ranked tracks to distinct existing fallback slots."""

    candidate_uv = torch.as_tensor(candidate_uv)
    fallback_positions = torch.as_tensor(
        fallback_positions, device=candidate_uv.device, dtype=torch.long
    ).reshape(-1)
    tracked_uv = torch.as_tensor(
        tracked_uv, device=candidate_uv.device, dtype=candidate_uv.dtype
    ).reshape(-1, 2)
    if candidate_uv.ndim != 2 or candidate_uv.shape[1] != 2:
        raise ValueError("Candidate pixels must have shape [N, 2]")
    if fallback_positions.numel() and bool(
        torch.any(
            (fallback_positions < 0)
            | (fallback_positions >= candidate_uv.shape[0])
        )
    ):
        raise ValueError("Fallback positions are outside the candidate array")
    count = min(int(fallback_positions.numel()), int(tracked_uv.shape[0]))
    if count == 0:
        return torch.empty(0, device=candidate_uv.device, dtype=torch.long)
    available = fallback_positions.clone()
    assigned = []
    for tracked_row in tracked_uv[:count]:
        distances = torch.sum(
            (candidate_uv[available] - tracked_row.reshape(1, 2)) ** 2,
            dim=1,
        )
        nearest = int(torch.argmin(distances).item())
        assigned.append(available[nearest])
        available = torch.cat((available[:nearest], available[nearest + 1 :]))
    return torch.stack(assigned)
