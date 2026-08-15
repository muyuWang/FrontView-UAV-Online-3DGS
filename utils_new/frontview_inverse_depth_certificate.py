"""Causal multi-view inverse-depth certificates for front-view Gaussian births."""

import math
from copy import deepcopy

import torch
import torch.nn.functional as F

from utils_new.camera_utils import unproject_pts_tensor


DEFAULT_FRONT_VIEW_INVERSE_DEPTH_CERTIFICATE_CONFIG = {
    "enabled": False,
    "hypothesis_source": "local_prior",
    "support_neighbors": 8,
    "shuffle_support_binding": False,
    "reference_frames": 3,
    "history_frames": 24,
    "hypotheses": 9,
    "prior_span_stds": 6.0,
    "minimum_log_depth_span": 1.0,
    "patch_radius_px": 2,
    "photometric_cost": "centered_l1",
    "view_aggregation": "mean",
    "view_consistency_chi2": 3.841458820694124,
    "support_confidence_chi2": 3.841458820694124,
    "mode_nll_margin_min": 0.0,
    "leave_one_out_consistency": False,
    "photometric_temperature": 0.04,
    "information_gain_min": 0.20,
    "posterior_std_ratio_max": 0.85,
    "minimum_valid_views": 1,
    "conflict_nll_margin": 1.0,
    "uncertified_policy": "projective_reject_conflict",
    "shuffle_evidence": False,
    "shuffle_seed": 42,
}


def validate_front_view_inverse_depth_certificate_config(config=None):
    merged = deepcopy(DEFAULT_FRONT_VIEW_INVERSE_DEPTH_CERTIFICATE_CONFIG)
    if config is not None:
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError(
                "Unknown FrontViewInverseDepthCertificate options: {}".format(
                    sorted(unknown)
                )
            )
        merged.update(config)
    for key in (
        "enabled",
        "shuffle_evidence",
        "shuffle_support_binding",
        "leave_one_out_consistency",
    ):
        if not isinstance(merged[key], bool):
            raise TypeError(
                "FrontViewInverseDepthCertificate.{} must be boolean".format(key)
            )
    if merged["hypothesis_source"] not in (
        "local_prior",
        "track_support",
        "causal_frustum",
    ):
        raise ValueError(
            "hypothesis_source must be local_prior, track_support, or causal_frustum"
        )
    if merged["photometric_cost"] not in ("centered_l1", "zncc"):
        raise ValueError("photometric_cost must be centered_l1 or zncc")
    if merged["view_aggregation"] not in ("mean", "median", "consensus"):
        raise ValueError("view_aggregation must be mean, median, or consensus")
    for key in (
        "reference_frames",
        "history_frames",
        "hypotheses",
        "patch_radius_px",
        "minimum_valid_views",
        "support_neighbors",
    ):
        if not isinstance(merged[key], int) or merged[key] <= 0:
            raise ValueError(
                "FrontViewInverseDepthCertificate.{} must be positive".format(key)
            )
    if merged["hypotheses"] < 3 or merged["hypotheses"] % 2 == 0:
        raise ValueError(
            "FrontViewInverseDepthCertificate.hypotheses must be odd and at least 3"
        )
    if merged["minimum_valid_views"] > merged["reference_frames"]:
        raise ValueError("minimum_valid_views cannot exceed reference_frames")
    if merged["reference_frames"] > merged["history_frames"]:
        raise ValueError("reference_frames cannot exceed history_frames")
    for key in (
        "prior_span_stds",
        "minimum_log_depth_span",
        "photometric_temperature",
        "conflict_nll_margin",
        "view_consistency_chi2",
        "support_confidence_chi2",
    ):
        if not math.isfinite(float(merged[key])) or float(merged[key]) <= 0.0:
            raise ValueError(
                "FrontViewInverseDepthCertificate.{} must be positive".format(key)
            )
    if (
        not math.isfinite(float(merged["mode_nll_margin_min"]))
        or float(merged["mode_nll_margin_min"]) < 0.0
    ):
        raise ValueError("mode_nll_margin_min must be finite and nonnegative")
    for key in ("information_gain_min", "posterior_std_ratio_max"):
        value = float(merged[key])
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(
                "FrontViewInverseDepthCertificate.{} must lie in (0, 1)".format(key)
            )
    if not isinstance(merged["shuffle_seed"], int):
        raise TypeError("shuffle_seed must be an integer")
    if merged["uncertified_policy"] not in (
        "existing_route",
        "projective",
        "reject",
        "projective_reject_conflict",
    ):
        raise ValueError(
            "uncertified_policy must be existing_route, projective, reject, or "
            "projective_reject_conflict"
        )
    return merged


def track_supported_inverse_depth_hypotheses(
    depths,
    log_depth_stds,
    count,
    confidence_chi2=3.841458820694124,
):
    """Construct a metric inverse-depth support from certified tracks.

    A track contributes its propagated confidence interval in inverse depth.
    Robust plotting-position quantiles bound the shared search domain, so one
    accidental extreme track cannot set the depth gauge for every query ray.
    No scene-distance threshold or sequence-length statistic is used.
    """

    depths = torch.as_tensor(depths).reshape(-1).float()
    log_depth_stds = torch.as_tensor(
        log_depth_stds, device=depths.device, dtype=depths.dtype
    ).reshape(-1)
    count = int(count)
    confidence_chi2 = float(confidence_chi2)
    if count < 3:
        raise ValueError("Track-supported inverse depth needs at least 3 hypotheses")
    if len(depths) != len(log_depth_stds):
        raise ValueError("Track depths and uncertainties must align")
    if confidence_chi2 <= 0.0 or not math.isfinite(confidence_chi2):
        raise ValueError("Track support confidence chi-square must be positive")
    finite = (
        torch.isfinite(depths)
        & torch.isfinite(log_depth_stds)
        & (depths > 0.0)
        & (log_depth_stds > 0.0)
    )
    if not bool(finite.any().item()):
        return torch.empty(0, device=depths.device, dtype=depths.dtype)
    inverse = torch.reciprocal(depths[finite])
    inverse_std = inverse * log_depth_stds[finite]
    radius = math.sqrt(confidence_chi2)
    eps = torch.finfo(inverse.dtype).eps
    lower_samples = torch.clamp(inverse - radius * inverse_std, min=eps)
    upper_samples = inverse + radius * inverse_std
    lower_q = 1.0 / float(count + 1)
    upper_q = float(count) / float(count + 1)
    lower = torch.quantile(lower_samples, lower_q)
    upper = torch.quantile(upper_samples, upper_q)
    if not bool(torch.isfinite(lower).item() and torch.isfinite(upper).item()):
        return torch.empty(0, device=depths.device, dtype=depths.dtype)
    if bool((upper <= lower).item()):
        return lower.reshape(1)
    return torch.linspace(lower, upper, count, device=depths.device, dtype=depths.dtype)


def locally_track_supported_inverse_depth_hypotheses(
    query_uv,
    support_uv,
    support_depths,
    support_log_depth_stds,
    count,
    *,
    neighbors=8,
    confidence_chi2=3.841458820694124,
    shuffle_binding=False,
    seed=42,
):
    """Build one metric search interval per ray from nearby certified tracks."""

    query_uv = torch.as_tensor(query_uv).reshape(-1, 2).float()
    support_uv = torch.as_tensor(
        support_uv, device=query_uv.device, dtype=query_uv.dtype
    ).reshape(-1, 2)
    support_depths = torch.as_tensor(
        support_depths, device=query_uv.device, dtype=query_uv.dtype
    ).reshape(-1)
    support_log_depth_stds = torch.as_tensor(
        support_log_depth_stds, device=query_uv.device, dtype=query_uv.dtype
    ).reshape(-1)
    count = int(count)
    neighbors = int(neighbors)
    if count < 3 or neighbors < 1:
        raise ValueError("Local track support needs positive neighbors and >=3 hypotheses")
    if not (
        len(support_uv) == len(support_depths) == len(support_log_depth_stds)
    ):
        raise ValueError("Local track support arrays must align")
    finite = (
        torch.isfinite(support_uv).all(dim=1)
        & torch.isfinite(support_depths)
        & torch.isfinite(support_log_depth_stds)
        & (support_depths > 0.0)
        & (support_log_depth_stds > 0.0)
    )
    support_uv = support_uv[finite]
    support_depths = support_depths[finite]
    support_log_depth_stds = support_log_depth_stds[finite]
    if not len(query_uv) or not len(support_uv):
        return torch.empty(
            (0, len(query_uv)), device=query_uv.device, dtype=query_uv.dtype
        )
    if bool(shuffle_binding) and len(support_depths) > 1:
        generator = torch.Generator(device=query_uv.device)
        generator.manual_seed(int(seed))
        permutation = torch.randperm(
            len(support_depths), generator=generator, device=query_uv.device
        )
        support_depths = support_depths[permutation]
        support_log_depth_stds = support_log_depth_stds[permutation]
    local_count = min(neighbors, len(support_uv))
    nearest = torch.topk(
        torch.cdist(query_uv, support_uv),
        k=local_count,
        dim=1,
        largest=False,
    ).indices
    inverse = torch.reciprocal(support_depths[nearest])
    inverse_std = inverse * support_log_depth_stds[nearest]
    radius = math.sqrt(float(confidence_chi2))
    eps = torch.finfo(inverse.dtype).eps
    lower_samples = torch.clamp(inverse - radius * inverse_std, min=eps)
    upper_samples = inverse + radius * inverse_std
    lower_q = 1.0 / float(local_count + 1)
    upper_q = float(local_count) / float(local_count + 1)
    lower = torch.quantile(lower_samples, lower_q, dim=1)
    upper = torch.quantile(upper_samples, upper_q, dim=1)
    alpha = torch.linspace(
        0.0, 1.0, count, device=query_uv.device, dtype=query_uv.dtype
    )
    return lower[None] + alpha[:, None] * torch.clamp(
        upper - lower, min=eps
    )[None]


def causal_frustum_inverse_depth_hypotheses(
    support_depths,
    support_log_depth_stds,
    count,
    *,
    far_depth,
    confidence_chi2=3.841458820694124,
):
    """Cover every causally plausible farther surface in inverse depth.

    The far endpoint comes from the camera frustum. The near endpoint is a
    plotting-position envelope of the current metric tracks. Tracks therefore
    define only the finite search support, never a depth prediction for a ray.
    """

    depths = torch.as_tensor(support_depths).reshape(-1).float()
    log_stds = torch.as_tensor(
        support_log_depth_stds, device=depths.device, dtype=depths.dtype
    ).reshape(-1)
    count = int(count)
    far_depth = float(far_depth)
    if count < 3:
        raise ValueError("Causal-frustum inverse depth needs at least 3 hypotheses")
    if len(depths) != len(log_stds):
        raise ValueError("Support depths and uncertainties must align")
    if not math.isfinite(far_depth) or far_depth <= 0.0:
        raise ValueError("Camera far depth must be positive")
    finite = (
        torch.isfinite(depths)
        & torch.isfinite(log_stds)
        & (depths > 0.0)
        & (log_stds > 0.0)
    )
    if not bool(finite.any().item()):
        return torch.empty(0, device=depths.device, dtype=depths.dtype)
    inverse = torch.reciprocal(depths[finite])
    inverse_std = inverse * log_stds[finite]
    radius = math.sqrt(float(confidence_chi2))
    upper_samples = inverse + radius * inverse_std
    upper_quantile = float(count) / float(count + 1)
    upper = torch.quantile(upper_samples, upper_quantile)
    lower = torch.as_tensor(
        1.0 / far_depth, device=depths.device, dtype=depths.dtype
    )
    if not bool(torch.isfinite(upper).item()) or bool((upper <= lower).item()):
        return torch.empty(0, device=depths.device, dtype=depths.dtype)
    return torch.linspace(lower, upper, count, device=depths.device, dtype=depths.dtype)


def _sample_patches(image, uv, offsets):
    height, width = image.shape[:2]
    locations = uv[:, None, :] + offsets[None, :, :]
    grid = torch.empty_like(locations)
    grid[..., 0] = 2.0 * locations[..., 0] / float(width) - 1.0
    grid[..., 1] = 2.0 * locations[..., 1] / float(height) - 1.0
    sampled = F.grid_sample(
        image.permute(2, 0, 1).unsqueeze(0),
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[0].permute(1, 2, 0)


def _project_hypotheses(pixels, depths, current_camera, reference_camera, level):
    hypothesis_count, row_count = depths.shape
    tiled_pixels = pixels.repeat(hypothesis_count, 1)
    world = unproject_pts_tensor(
        tiled_pixels,
        depths.reshape(-1),
        current_camera.get_int_mat(level).to(tiled_pixels),
        current_camera.get_raw_pose().detach().to(tiled_pixels),
    )
    pose = reference_camera.get_raw_pose().detach().to(world)
    camera_points = world @ pose[:3, :3].T + pose[:3, 3]
    z = camera_points[:, 2]
    projected = camera_points @ reference_camera.get_int_mat(level).to(world).T
    uv = projected[:, :2] / torch.clamp(z[:, None], min=1.0e-8)
    uv = uv.reshape(hypothesis_count, row_count, 2)
    z = z.reshape(hypothesis_count, row_count)
    return uv, z


def _fisher_ranked_references(camera, reference_cameras, pixels, level, count):
    """Select baselines with maximum inverse-depth disparity information."""

    references = list(reference_cameras)
    if not references:
        return [], torch.zeros(
            pixels.shape[0], device=pixels.device, dtype=pixels.dtype
        )
    pose = camera.get_raw_pose().detach().to(pixels)
    intrinsic = camera.get_int_mat(level).detach().to(pixels)
    homogeneous = torch.cat((pixels, torch.ones_like(pixels[:, :1])), dim=1)
    camera_rays = homogeneous @ torch.linalg.inv(intrinsic).T
    world_rays = F.normalize(camera_rays @ pose[:3, :3], dim=1, eps=1.0e-8)
    current_center = -pose[:3, :3].T @ pose[:3, 3]
    per_reference = []
    for reference in references:
        reference_pose = reference.get_raw_pose().detach().to(pixels)
        center = -reference_pose[:3, :3].T @ reference_pose[:3, 3]
        baseline = center - current_center
        parallel = torch.sum(world_rays * baseline[None, :], dim=1)
        perpendicular_squared = torch.clamp(
            torch.sum(baseline.square()) - parallel.square(), min=0.0
        )
        per_reference.append(perpendicular_squared)
    information = torch.stack(per_reference, dim=0)
    global_scores = torch.median(information, dim=1).values
    selected_indices = torch.topk(
        global_scores, k=min(int(count), len(references)), largest=True
    ).indices
    selected = [references[int(index)] for index in selected_indices.tolist()]
    return selected, torch.amax(information[selected_indices], dim=0)


@torch.no_grad()
def causal_inverse_depth_posterior(
    camera,
    reference_cameras,
    pixels,
    prior_depths,
    prior_log_stds,
    level,
    config,
    *,
    support_depths=None,
    support_log_depth_stds=None,
    support_uv=None,
):
    """Fuse a DepthCov prior with causal patch likelihoods in log-depth.

    The hypothesis support is defined by each row's own prior uncertainty, with
    a minimum multiplicative span so overconfident priors cannot hide a distant
    mode. Certification uses entropy reduction and posterior concentration; it
    has no metric near/far threshold.
    """

    cfg = validate_front_view_inverse_depth_certificate_config(config)
    rows = int(pixels.shape[0])
    device = pixels.device
    dtype = prior_depths.dtype
    empty_bool = torch.zeros(rows, device=device, dtype=torch.bool)
    empty_float = torch.zeros(rows, device=device, dtype=dtype)
    empty_int = torch.zeros(rows, device=device, dtype=torch.int32)
    empty_inf = torch.full((rows,), float("inf"), device=device, dtype=dtype)
    if rows == 0 or not reference_cameras:
        return {
            "depths": prior_depths.clone(),
            "posterior_log_stds": prior_log_stds.clone(),
            "certified": empty_bool,
            "conflicted": empty_bool.clone(),
            "information_gain": empty_float,
            "valid_views": empty_int.clone(),
            "posterior_shift": empty_float.clone(),
            "baseline_information": empty_float.clone(),
            "consensus_support": empty_int.clone(),
            "consensus_pairwise_chi2": empty_inf.clone(),
            "mode_nll_margin": empty_float.clone(),
            "leave_one_out_views": empty_int.clone(),
            "leave_one_out_chi2": empty_inf.clone(),
        }

    reference_cameras, baseline_information = _fisher_ranked_references(
        camera,
        list(reference_cameras)[-int(cfg["history_frames"]):],
        pixels,
        level,
        int(cfg["reference_frames"]),
    )
    hypothesis_count = int(cfg["hypotheses"])
    log_prior_depth = torch.log(torch.clamp(prior_depths, min=1.0e-8))
    if cfg["hypothesis_source"] in ("track_support", "causal_frustum"):
        if (
            support_depths is None
            or support_log_depth_stds is None
            or support_uv is None
        ):
            support_inverse = torch.empty(0, device=device, dtype=dtype)
        else:
            support_depths_tensor = torch.as_tensor(
                support_depths, device=device, dtype=dtype
            )
            support_stds_tensor = torch.as_tensor(
                support_log_depth_stds, device=device, dtype=dtype
            )
            if cfg["hypothesis_source"] == "causal_frustum":
                shared_inverse = causal_frustum_inverse_depth_hypotheses(
                    support_depths_tensor,
                    support_stds_tensor,
                    hypothesis_count,
                    far_depth=float(camera.far),
                    confidence_chi2=float(cfg["support_confidence_chi2"]),
                )
                support_inverse = (
                    shared_inverse[:, None].expand(-1, rows)
                    if shared_inverse.numel()
                    else shared_inverse
                )
            else:
                support_inverse = locally_track_supported_inverse_depth_hypotheses(
                    pixels,
                    torch.as_tensor(support_uv, device=device, dtype=dtype),
                    support_depths_tensor,
                    support_stds_tensor,
                    hypothesis_count,
                    neighbors=int(cfg["support_neighbors"]),
                    confidence_chi2=float(cfg["support_confidence_chi2"]),
                    shuffle_binding=bool(cfg["shuffle_support_binding"]),
                    seed=int(cfg["shuffle_seed"]) + int(camera.cam_idx),
                ).to(device=device, dtype=dtype)
        if support_inverse.shape != (hypothesis_count, rows):
            return {
                "depths": prior_depths.clone(),
                "posterior_log_stds": prior_log_stds.clone(),
                "certified": empty_bool,
                "conflicted": empty_bool.clone(),
                "information_gain": empty_float,
                "valid_views": empty_int.clone(),
                "posterior_shift": empty_float.clone(),
                "baseline_information": empty_float.clone(),
                "consensus_support": empty_int.clone(),
                "consensus_pairwise_chi2": empty_inf.clone(),
                "mode_nll_margin": empty_float.clone(),
                "leave_one_out_views": empty_int.clone(),
                "leave_one_out_chi2": empty_inf.clone(),
            }
        inverse_depth_hypotheses = support_inverse
        log_prior = torch.zeros_like(inverse_depth_hypotheses)
    else:
        span = torch.maximum(
            prior_log_stds * float(cfg["prior_span_stds"]),
            torch.full_like(prior_log_stds, float(cfg["minimum_log_depth_span"])),
        )
        offsets_1d = torch.linspace(
            -1.0, 1.0, hypothesis_count, device=device, dtype=dtype
        )
        prior_inverse_depth = torch.reciprocal(
            torch.clamp(prior_depths, min=1.0e-8)
        )
        inverse_depth_half_width = prior_inverse_depth * torch.tanh(span)
        inverse_depth_hypotheses = (
            prior_inverse_depth[None, :]
            + offsets_1d[:, None] * inverse_depth_half_width[None, :]
        )
        normalized_offsets = (
            inverse_depth_hypotheses - prior_inverse_depth[None, :]
        ) / torch.clamp(inverse_depth_half_width[None, :], min=1.0e-8)
        log_prior = -0.5 * normalized_offsets.square()
    depth_hypotheses = torch.reciprocal(
        torch.clamp(inverse_depth_hypotheses, min=1.0e-8)
    )
    log_hypotheses = torch.log(depth_hypotheses)
    prior_probability = torch.softmax(log_prior, dim=0)
    prior_log_mean = torch.sum(prior_probability * log_hypotheses, dim=0)
    prior_log_variance = torch.sum(
        prior_probability * (log_hypotheses - prior_log_mean[None]).square(), dim=0
    )
    log_likelihood = torch.zeros_like(log_prior)
    valid_views = torch.zeros(rows, device=device, dtype=torch.int32)

    patch_radius = int(cfg["patch_radius_px"])
    axis = torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    patch_offsets = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    current_image = camera.get_gt_image(level).to(device=device, dtype=dtype)
    current_image = current_image / max(float(camera.exposure_gain), 1.0e-8)
    current_patches = _sample_patches(current_image, pixels, patch_offsets)
    temperature = float(cfg["photometric_temperature"])

    per_view_costs = []
    per_view_valid = []
    for reference in reference_cameras:
        uv, z = _project_hypotheses(
            pixels, depth_hypotheses, camera, reference, level
        )
        width = int(reference.get_width(level))
        height = int(reference.get_height(level))
        margin = float(patch_radius + 1)
        valid = (
            (z > float(reference.near))
            & (z < float(reference.far))
            & (uv[..., 0] >= margin)
            & (uv[..., 0] < float(width) - margin)
            & (uv[..., 1] >= margin)
            & (uv[..., 1] < float(height) - margin)
        )
        reference_image = reference.get_gt_image(level).to(device=device, dtype=dtype)
        reference_image = reference_image / max(float(reference.exposure_gain), 1.0e-8)
        reference_patches = _sample_patches(
            reference_image, uv.reshape(-1, 2), patch_offsets
        ).reshape(hypothesis_count, rows, -1, 3)
        centered_current = current_patches - current_patches.mean(dim=1, keepdim=True)
        centered_reference = reference_patches - reference_patches.mean(
            dim=2, keepdim=True
        )
        if cfg["photometric_cost"] == "zncc":
            current_vector = centered_current.reshape(rows, -1)
            reference_vector = centered_reference.reshape(
                hypothesis_count, rows, -1
            )
            current_norm = torch.linalg.vector_norm(
                current_vector, dim=-1
            ).clamp_min(1.0e-8)
            reference_norm = torch.linalg.vector_norm(
                reference_vector, dim=-1
            ).clamp_min(1.0e-8)
            correlation = torch.sum(
                reference_vector * current_vector[None], dim=-1
            ) / (reference_norm * current_norm[None])
            cost = 0.5 * (1.0 - torch.clamp(correlation, -1.0, 1.0))
        else:
            cost = torch.mean(
                torch.abs(centered_reference - centered_current[None, :, :, :]),
                dim=(2, 3),
            )
        per_view_costs.append(cost)
        per_view_valid.append(valid)

    costs = torch.stack(per_view_costs, dim=0)
    valid = torch.stack(per_view_valid, dim=0)
    if cfg["shuffle_evidence"] and rows > 1:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(cfg["shuffle_seed"]) + int(camera.cam_idx))
        permutation = torch.randperm(rows, generator=generator, device=device)
        costs = costs[:, :, permutation]
        valid = valid[:, :, permutation]

    row_view_valid = valid.any(dim=1)
    valid_views = row_view_valid.sum(dim=0).to(torch.int32)
    valid_float = valid.to(costs.dtype)
    consensus_support = torch.zeros(rows, device=device, dtype=torch.int32)
    consensus_pairwise_chi2 = torch.full(
        (rows,), float("inf"), device=device, dtype=dtype
    )
    if cfg["view_aggregation"] == "consensus":
        per_view_log_posterior = log_prior[None] + torch.where(
            valid,
            -costs / temperature,
            torch.full_like(costs, -1.0e9),
        )
        per_view_posterior = torch.softmax(per_view_log_posterior, dim=1)
        per_view_mean = torch.sum(
            per_view_posterior * log_hypotheses[None], dim=1
        )
        per_view_variance = torch.sum(
            per_view_posterior
            * (log_hypotheses[None] - per_view_mean[:, None]).square(),
            dim=1,
        )
        step_floor = torch.median(
            torch.abs(log_hypotheses[1:] - log_hypotheses[:-1]), dim=0
        ).values
        best_score = torch.full(
            (rows,), -float("inf"), device=device, dtype=dtype
        )
        robust_cost = torch.full(
            (hypothesis_count, rows), float("inf"), device=device, dtype=dtype
        )
        prior_entropy_for_pair = -torch.sum(
            prior_probability
            * torch.log(torch.clamp(prior_probability, min=1.0e-12)),
            dim=0,
        )
        for left in range(len(reference_cameras)):
            for right in range(left + 1, len(reference_cameras)):
                pair_valid = row_view_valid[left] & row_view_valid[right]
                variance = (
                    per_view_variance[left]
                    + per_view_variance[right]
                    + step_floor.square()
                )
                statistic = (
                    per_view_mean[left] - per_view_mean[right]
                ).square() / torch.clamp(variance, min=1.0e-8)
                compatible = pair_valid & (
                    statistic <= float(cfg["view_consistency_chi2"])
                )
                pair_cost = 0.5 * (costs[left] + costs[right])
                pair_posterior = torch.softmax(
                    log_prior - pair_cost / temperature, dim=0
                )
                pair_entropy = -torch.sum(
                    pair_posterior
                    * torch.log(torch.clamp(pair_posterior, min=1.0e-12)),
                    dim=0,
                )
                pair_score = (
                    prior_entropy_for_pair - pair_entropy - 0.01 * statistic
                )
                update = compatible & (pair_score > best_score)
                best_score = torch.where(update, pair_score, best_score)
                consensus_pairwise_chi2 = torch.where(
                    update, statistic, consensus_pairwise_chi2
                )
                robust_cost = torch.where(update[None], pair_cost, robust_cost)
                consensus_support = torch.where(
                    update,
                    torch.full_like(consensus_support, 2),
                    consensus_support,
                )
    elif cfg["view_aggregation"] == "median":
        robust_cost = torch.nanmedian(
            torch.where(valid, costs, torch.full_like(costs, float("nan"))),
            dim=0,
        ).values
    else:
        robust_cost = torch.sum(
            torch.where(valid, costs, torch.zeros_like(costs)), dim=0
        )
        robust_cost = robust_cost / torch.clamp(
            valid_float.sum(dim=0), min=1.0
        )
    robust_cost = torch.where(
        valid.any(dim=0), robust_cost, torch.full_like(robust_cost, float("inf"))
    )
    hypothesis_valid = valid.any(dim=0)
    log_likelihood = torch.where(
        hypothesis_valid,
        -robust_cost / temperature,
        torch.full_like(robust_cost, -1.0e9),
    )
    log_posterior = log_prior + log_likelihood
    posterior = torch.softmax(log_posterior, dim=0)
    posterior_inverse_depth = torch.sum(
        posterior * inverse_depth_hypotheses, dim=0
    )
    posterior_depth = torch.reciprocal(
        torch.clamp(posterior_inverse_depth, min=1.0e-8)
    )
    posterior_log_mean = torch.sum(posterior * log_hypotheses, dim=0)
    posterior_variance = torch.sum(
        posterior * (log_hypotheses - posterior_log_mean[None, :]).square(),
        dim=0,
    )
    posterior_std = torch.sqrt(torch.clamp(posterior_variance, min=0.0))
    prior_entropy = -torch.sum(
        prior_probability * torch.log(torch.clamp(prior_probability, min=1.0e-12)),
        dim=0,
    )
    posterior_entropy = -torch.sum(
        posterior * torch.log(torch.clamp(posterior, min=1.0e-12)), dim=0
    )
    information_gain = torch.clamp(
        (prior_entropy - posterior_entropy) / torch.clamp(prior_entropy, min=1.0e-8),
        min=0.0,
        max=1.0,
    )
    posterior_shift = posterior_log_mean - log_prior_depth
    std_ratio = posterior_std / torch.sqrt(
        torch.clamp(prior_log_variance, min=1.0e-8)
    )
    best_indices = torch.argmax(log_posterior, dim=0)
    hypothesis_indices = torch.arange(hypothesis_count, device=device)[:, None]
    outside_primary_basin = torch.abs(hypothesis_indices - best_indices[None]) > 1
    secondary = torch.where(
        outside_primary_basin,
        log_posterior,
        torch.full_like(log_posterior, -float("inf")),
    ).amax(dim=0)
    best = log_posterior.gather(0, best_indices[None]).squeeze(0)
    mode_nll_margin = best - secondary

    leave_one_out_views = torch.zeros(rows, device=device, dtype=torch.int32)
    leave_one_out_chi2 = torch.full(
        (rows,), float("inf"), device=device, dtype=dtype
    )
    if len(reference_cameras) >= 3:
        loo_means = []
        loo_variances = []
        loo_valid_rows = []
        for omitted in range(len(reference_cameras)):
            retained = torch.ones(
                len(reference_cameras), device=device, dtype=torch.bool
            )
            retained[omitted] = False
            retained_valid = valid[retained]
            retained_costs = costs[retained]
            aggregate = torch.sum(
                torch.where(
                    retained_valid,
                    retained_costs,
                    torch.zeros_like(retained_costs),
                ),
                dim=0,
            ) / torch.clamp(retained_valid.to(dtype).sum(dim=0), min=1.0)
            hypothesis_valid_loo = retained_valid.any(dim=0)
            aggregate = torch.where(
                hypothesis_valid_loo,
                aggregate,
                torch.full_like(aggregate, float("inf")),
            )
            loo_posterior = torch.softmax(
                log_prior - aggregate / temperature, dim=0
            )
            loo_mean = torch.sum(loo_posterior * log_hypotheses, dim=0)
            loo_variance = torch.sum(
                loo_posterior * (log_hypotheses - loo_mean[None]).square(),
                dim=0,
            )
            loo_means.append(loo_mean)
            loo_variances.append(loo_variance)
            loo_valid_rows.append(hypothesis_valid_loo.any(dim=0))
        loo_means = torch.stack(loo_means)
        loo_variances = torch.stack(loo_variances)
        loo_valid = torch.stack(loo_valid_rows)
        step_floor = torch.median(
            torch.abs(log_hypotheses[1:] - log_hypotheses[:-1]), dim=0
        ).values.square()
        loo_statistic = (loo_means - posterior_log_mean[None]).square() / torch.clamp(
            loo_variances + posterior_variance[None] + step_floor[None],
            min=1.0e-8,
        )
        leave_one_out_views = loo_valid.sum(dim=0).to(torch.int32)
        leave_one_out_chi2 = torch.where(
            loo_valid,
            loo_statistic,
            torch.full_like(loo_statistic, -float("inf")),
        ).amax(dim=0)
    leave_one_out_ok = (
        (leave_one_out_views >= 3)
        & (leave_one_out_chi2 <= float(cfg["view_consistency_chi2"]))
        if cfg["leave_one_out_consistency"]
        else torch.ones(rows, device=device, dtype=torch.bool)
    )
    enough_views = (
        consensus_support >= max(2, int(cfg["minimum_valid_views"]))
        if cfg["view_aggregation"] == "consensus"
        else valid_views >= int(cfg["minimum_valid_views"])
    )
    certified = (
        enough_views
        & (information_gain >= float(cfg["information_gain_min"]))
        & (std_ratio <= float(cfg["posterior_std_ratio_max"]))
        & (mode_nll_margin >= float(cfg["mode_nll_margin_min"]))
        & leave_one_out_ok
    )

    prior_index = hypothesis_count // 2
    prior_cost = robust_cost[prior_index]
    best_cost = robust_cost.amin(dim=0)
    conflict = enough_views & torch.isfinite(prior_cost) & torch.isfinite(best_cost) & (
        (prior_cost - best_cost) / temperature >= float(cfg["conflict_nll_margin"])
    )
    return {
        "depths": torch.where(certified, posterior_depth, prior_depths),
        "posterior_log_stds": torch.where(certified, posterior_std, prior_log_stds),
        "certified": certified,
        "conflicted": conflict,
        "information_gain": information_gain,
        "valid_views": valid_views,
        "posterior_shift": posterior_shift,
        "baseline_information": baseline_information,
        "consensus_support": consensus_support,
        "consensus_pairwise_chi2": consensus_pairwise_chi2,
        "mode_nll_margin": mode_nll_margin,
        "leave_one_out_views": leave_one_out_views,
        "leave_one_out_chi2": leave_one_out_chi2,
    }
