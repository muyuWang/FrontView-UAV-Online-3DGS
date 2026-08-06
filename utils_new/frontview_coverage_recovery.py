"""Residual-certified keyframe recovery for sparse front-view trajectories."""

import math
from collections import deque
from copy import deepcopy

import numpy as np
import torch


DEFAULT_FRONT_VIEW_COVERAGE_RECOVERY_CONFIG = {
    "enabled": False,
    "min_frame_gap": 20,
    "min_translation_m": 1.0,
    "min_rotation_deg": 3.0,
    "residual_threshold": 0.08,
    "min_failure_fraction": 0.15,
    "opacity_threshold": 0.50,
    "depth_fallback_enabled": False,
    "depth_prior_window_frames": 240,
    "depth_prior_quantile": 0.90,
    "depth_prior_min_m": 20.0,
    "depth_prior_max_m": 120.0,
    "depth_fallback_min_valid": 128,
    "depth_fallback_confidence": 0.50,
    "depth_fallback_cell_px": 18,
    "depth_fallback_scale_multiplier": 10.0,
    "newborn_optimization_iters": 0,
    "newborn_max_scale_expansion": 1.25,
    "newborn_freeze_positions": True,
    "tracking_update_interval": 0,
    "tracking_optimization_iters": 1,
    "tracking_window_frames": 1,
    "depth_fallback_motion_floor_enabled": False,
    "depth_fallback_max_projected_drift_px": 8.0,
    "depth_fallback_motion_floor_max_m": 500.0,
    "depth_fallback_map_enabled": False,
    "depth_fallback_map_front_ratio": 0.95,
    "depth_fallback_map_min_opacity": 0.50,
    "depth_fallback_map_min_prior_ratio": 0.50,
    "depth_fallback_map_max_prior_ratio": 1.50,
}


def validate_front_view_coverage_recovery_config(config=None):
    result = deepcopy(DEFAULT_FRONT_VIEW_COVERAGE_RECOVERY_CONFIG)
    if config is not None:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(
                "Unknown FrontViewCoverageRecovery options: {}".format(
                    sorted(unknown)
                )
            )
        result.update(config)
    if not isinstance(result["enabled"], bool):
        raise TypeError("FrontViewCoverageRecovery.enabled must be boolean")
    if not isinstance(result["depth_fallback_enabled"], bool):
        raise TypeError(
            "FrontViewCoverageRecovery.depth_fallback_enabled must be boolean"
        )
    if not isinstance(result["depth_fallback_map_enabled"], bool):
        raise TypeError(
            "FrontViewCoverageRecovery.depth_fallback_map_enabled must be boolean"
        )
    if not isinstance(result["depth_fallback_motion_floor_enabled"], bool):
        raise TypeError(
            "FrontViewCoverageRecovery.depth_fallback_motion_floor_enabled must be boolean"
        )
    if not isinstance(result["newborn_freeze_positions"], bool):
        raise TypeError(
            "FrontViewCoverageRecovery.newborn_freeze_positions must be boolean"
        )
    if not isinstance(result["min_frame_gap"], int) or result["min_frame_gap"] < 1:
        raise ValueError("FrontViewCoverageRecovery.min_frame_gap must be positive")
    for key in (
        "depth_prior_window_frames",
        "depth_fallback_min_valid",
        "depth_fallback_cell_px",
        "newborn_optimization_iters",
        "tracking_update_interval",
        "tracking_optimization_iters",
        "tracking_window_frames",
    ):
        minimum = 0 if key in (
            "newborn_optimization_iters",
            "tracking_update_interval",
        ) else 1
        if not isinstance(result[key], int) or result[key] < minimum:
            raise ValueError(
                "FrontViewCoverageRecovery.{} must be an integer >= {}".format(
                    key, minimum
                )
            )
    for key in ("min_translation_m", "min_rotation_deg"):
        if float(result[key]) < 0.0:
            raise ValueError("FrontViewCoverageRecovery.{} must be non-negative".format(key))
    if float(result["min_translation_m"]) == 0.0 and float(
        result["min_rotation_deg"]
    ) == 0.0:
        raise ValueError("FrontViewCoverageRecovery requires pose novelty")
    for key in (
        "residual_threshold",
        "min_failure_fraction",
        "opacity_threshold",
        "depth_prior_quantile",
        "depth_fallback_confidence",
        "depth_fallback_map_min_opacity",
    ):
        value = float(result[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError("FrontViewCoverageRecovery.{} must be in [0, 1]".format(key))
    if float(result["depth_prior_min_m"]) <= 0.0:
        raise ValueError("FrontViewCoverageRecovery.depth_prior_min_m must be positive")
    if float(result["depth_prior_max_m"]) <= float(result["depth_prior_min_m"]):
        raise ValueError(
            "FrontViewCoverageRecovery.depth_prior_max_m must exceed depth_prior_min_m"
        )
    if float(result["depth_fallback_scale_multiplier"]) <= 0.0:
        raise ValueError(
            "FrontViewCoverageRecovery.depth_fallback_scale_multiplier must be positive"
        )
    if float(result["newborn_max_scale_expansion"]) < 1.0:
        raise ValueError(
            "FrontViewCoverageRecovery.newborn_max_scale_expansion must be >= 1"
        )
    if float(result["depth_fallback_max_projected_drift_px"]) <= 0.0:
        raise ValueError(
            "FrontViewCoverageRecovery.depth_fallback_max_projected_drift_px must be positive"
        )
    if float(result["depth_fallback_motion_floor_max_m"]) <= float(
        result["depth_prior_min_m"]
    ):
        raise ValueError(
            "FrontViewCoverageRecovery.depth_fallback_motion_floor_max_m must exceed depth_prior_min_m"
        )
    if not 0.0 < float(result["depth_fallback_map_front_ratio"]) <= 1.0:
        raise ValueError(
            "FrontViewCoverageRecovery.depth_fallback_map_front_ratio must be in (0, 1]"
        )
    if float(result["depth_fallback_map_min_prior_ratio"]) <= 0.0:
        raise ValueError(
            "FrontViewCoverageRecovery.depth_fallback_map_min_prior_ratio must be positive"
        )
    if float(result["depth_fallback_map_max_prior_ratio"]) < float(
        result["depth_fallback_map_min_prior_ratio"]
    ):
        raise ValueError(
            "FrontViewCoverageRecovery map-depth prior-ratio range is invalid"
        )
    return result


def motion_conditioned_depth_floor(
    translation_m,
    focal_px,
    max_projected_drift_px,
    max_depth_m,
):
    """Choose a depth whose translational parallax stays within a pixel budget."""

    translation_m = float(translation_m)
    focal_px = float(focal_px)
    max_projected_drift_px = float(max_projected_drift_px)
    max_depth_m = float(max_depth_m)
    if translation_m < 0.0 or focal_px <= 0.0:
        raise ValueError("Motion-conditioned depth requires valid metric motion and focal length")
    if max_projected_drift_px <= 0.0 or max_depth_m <= 0.0:
        raise ValueError("Motion-conditioned depth bounds must be positive")
    return min(
        max_depth_m,
        translation_m * focal_px / max_projected_drift_px,
    )


class SparseFarDepthPrior:
    """Robust metric far-depth memory built only from persistent sparse tracks."""

    def __init__(self, config):
        self.config = validate_front_view_coverage_recovery_config(config)
        self.samples = deque()
        self.observed_frames = 0
        self.last_frame_id = 0

    def _expire(self, frame_id):
        oldest = int(frame_id) - int(self.config["depth_prior_window_frames"])
        while self.samples and self.samples[0][0] < oldest:
            self.samples.popleft()

    def observe(self, frame_id, sparse_depth):
        self.last_frame_id = int(frame_id)
        self._expire(frame_id)
        depth = torch.as_tensor(sparse_depth).detach().reshape(-1).float()
        valid = (
            torch.isfinite(depth)
            & (depth >= float(self.config["depth_prior_min_m"]))
            & (depth <= float(self.config["depth_prior_max_m"]))
        )
        if bool(valid.any().item()):
            sample = float(
                torch.quantile(
                    depth[valid], float(self.config["depth_prior_quantile"])
                ).item()
            )
            self.samples.append((int(frame_id), sample))
            self.observed_frames += 1
        return self.estimate(frame_id)

    def estimate(self, frame_id):
        self.last_frame_id = max(self.last_frame_id, int(frame_id))
        self._expire(frame_id)
        if not self.samples:
            return None
        values = np.asarray([sample for _, sample in self.samples], dtype=np.float64)
        estimate = float(np.quantile(values, self.config["depth_prior_quantile"]))
        return float(
            np.clip(
                estimate,
                self.config["depth_prior_min_m"],
                self.config["depth_prior_max_m"],
            )
        )

    def summary(self):
        return {
            "depth_prior_observed_frames": int(self.observed_frames),
            "depth_prior_active_samples": int(len(self.samples)),
            "depth_prior_m": self.estimate(self.last_frame_id),
        }


def apply_sparse_depth_prior_fallback(
    estimated_depth,
    valid_mask,
    depth_std,
    *,
    prior_depth_m,
    std_valid_threshold,
    min_valid,
    confidence,
):
    """Fill invalid DepthCov rows only when its sparse calibration collapses."""

    estimated_depth = torch.as_tensor(estimated_depth).clone()
    valid_mask = torch.as_tensor(valid_mask, device=estimated_depth.device).bool().clone()
    depth_std = torch.as_tensor(depth_std, device=estimated_depth.device).clone()
    if prior_depth_m is None or int(valid_mask.sum().item()) >= int(min_valid):
        return estimated_depth, valid_mask, depth_std, 0
    fallback = ~valid_mask
    fallback_rows = int(fallback.sum().item())
    if fallback_rows == 0:
        return estimated_depth, valid_mask, depth_std, 0
    estimated_depth[fallback] = float(prior_depth_m)
    depth_std[fallback] = float(std_valid_threshold) * (1.0 - float(confidence))
    valid_mask[fallback] = True
    return estimated_depth, valid_mask, depth_std, fallback_rows


def apply_visible_surface_depth_fallback(
    estimated_depth,
    valid_mask,
    depth_std,
    *,
    prior_depth_m,
    visible_depth,
    visible_opacity,
    std_valid_threshold,
    min_valid,
    confidence,
    front_ratio,
    min_opacity,
    min_prior_ratio,
    max_prior_ratio,
):
    """Recover invalid rows at reliable visible surfaces, with the far prior as backup."""

    original_valid = torch.as_tensor(valid_mask).bool().clone()
    estimated_depth, valid_mask, depth_std, fallback_rows = (
        apply_sparse_depth_prior_fallback(
            estimated_depth,
            valid_mask,
            depth_std,
            prior_depth_m=prior_depth_m,
            std_valid_threshold=std_valid_threshold,
            min_valid=min_valid,
            confidence=confidence,
        )
    )
    if fallback_rows == 0 or visible_depth is None or visible_opacity is None:
        return estimated_depth, valid_mask, depth_std, fallback_rows, 0

    visible_depth = torch.as_tensor(
        visible_depth, device=estimated_depth.device, dtype=estimated_depth.dtype
    ).reshape(-1)
    visible_opacity = torch.as_tensor(
        visible_opacity, device=estimated_depth.device, dtype=estimated_depth.dtype
    ).reshape(-1)
    if visible_depth.shape != estimated_depth.shape:
        raise ValueError("Visible fallback depth must match DepthCov candidate rows")
    if visible_opacity.shape != estimated_depth.shape:
        raise ValueError("Visible fallback opacity must match DepthCov candidate rows")

    lower = float(prior_depth_m) * float(min_prior_ratio)
    upper = float(prior_depth_m) * float(max_prior_ratio)
    map_rows = (
        ~original_valid.to(device=estimated_depth.device)
        & torch.isfinite(visible_depth)
        & torch.isfinite(visible_opacity)
        & (visible_depth > 0.0)
        & (visible_depth >= lower)
        & (visible_depth <= upper)
        & (visible_opacity >= float(min_opacity))
    )
    estimated_depth[map_rows] = visible_depth[map_rows] * float(front_ratio)
    return (
        estimated_depth,
        valid_mask,
        depth_std,
        fallback_rows,
        int(map_rows.sum().item()),
    )


def residual_grid_indices(valid_mask, residual, cell_px, max_points):
    """Select the strongest failed pixel per projective cell."""

    valid_mask = torch.as_tensor(valid_mask).bool()
    residual = torch.as_tensor(residual, device=valid_mask.device).float()
    if valid_mask.ndim != 2 or residual.shape != valid_mask.shape:
        raise ValueError("Residual-grid inputs must share shape HxW")
    if int(cell_px) < 1 or int(max_points) < 1:
        raise ValueError("Residual-grid cell and budget must be positive")
    flat = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).reshape(-1)
    if len(flat) == 0:
        return flat
    height, width = valid_mask.shape
    columns = (int(width) + int(cell_px) - 1) // int(cell_px)
    rows = torch.div(flat, int(width), rounding_mode="floor")
    image_columns = torch.remainder(flat, int(width))
    cell = (
        torch.div(rows, int(cell_px), rounding_mode="floor") * columns
        + torch.div(image_columns, int(cell_px), rounding_mode="floor")
    )
    flat_cpu = flat.detach().cpu().numpy()
    cell_cpu = cell.detach().cpu().numpy()
    score_cpu = residual.reshape(-1)[flat].detach().cpu().numpy()
    order = np.lexsort((-score_cpu, cell_cpu))
    ordered_cells = cell_cpu[order]
    first = np.empty(len(order), dtype=np.bool_)
    first[0] = True
    first[1:] = ordered_cells[1:] != ordered_cells[:-1]
    selected = order[first]
    if len(selected) > int(max_points):
        strongest = np.argsort(-score_cpu[selected], kind="stable")[: int(max_points)]
        selected = selected[strongest]
    return torch.from_numpy(flat_cpu[selected]).to(device=valid_mask.device)


def pose_novelty(last_world_to_camera, current_world_to_camera):
    """Return camera-center translation and relative rotation in metric units."""

    last = np.asarray(last_world_to_camera, dtype=np.float64)
    current = np.asarray(current_world_to_camera, dtype=np.float64)
    if last.shape != (4, 4) or current.shape != (4, 4):
        raise ValueError("Coverage-recovery poses must be 4x4 matrices")
    last_center = -last[:3, :3].T @ last[:3, 3]
    current_center = -current[:3, :3].T @ current[:3, 3]
    translation = float(np.linalg.norm(current_center - last_center))
    relative = current[:3, :3] @ last[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    rotation = math.degrees(math.acos(cosine))
    return translation, rotation


def coverage_failure_measurements(rendered, target, opacity, config):
    """Measure pixels that are uncovered or remain photometrically unexplained."""

    rendered = torch.as_tensor(rendered).float()
    target = torch.as_tensor(target, device=rendered.device).float()
    opacity = torch.as_tensor(opacity, device=rendered.device).float().squeeze()
    if rendered.shape != target.shape or rendered.ndim != 3 or rendered.shape[-1] != 3:
        raise ValueError("Coverage-recovery RGB tensors must share shape HxWx3")
    if opacity.shape != rendered.shape[:2]:
        raise ValueError("Coverage-recovery opacity must have shape HxW")
    residual = torch.mean(torch.abs(rendered - target), dim=-1)
    high_residual = residual >= float(config["residual_threshold"])
    low_opacity = opacity < float(config["opacity_threshold"])
    failure = high_residual | low_opacity
    return {
        "mean_residual": float(residual.mean().item()),
        "failure_fraction": float(failure.float().mean().item()),
        "high_residual_fraction": float(high_residual.float().mean().item()),
        "low_opacity_fraction": float(low_opacity.float().mean().item()),
    }


def coverage_recovery_certificate(
    *,
    frame_id,
    last_keyframe_id,
    last_world_to_camera,
    current_world_to_camera,
    rendered,
    target,
    opacity,
    config,
):
    """Certify one pose-novel sparse-dropout frame for map recovery."""

    translation, rotation = pose_novelty(
        last_world_to_camera, current_world_to_camera
    )
    frame_gap = int(frame_id) - int(last_keyframe_id)
    pose_novel = (
        translation >= float(config["min_translation_m"])
        or rotation >= float(config["min_rotation_deg"])
    )
    measurements = coverage_failure_measurements(
        rendered, target, opacity, config
    )
    admitted = (
        frame_gap >= int(config["min_frame_gap"])
        and pose_novel
        and measurements["failure_fraction"]
        >= float(config["min_failure_fraction"])
    )
    return {
        **measurements,
        "admitted": bool(admitted),
        "frame_gap": frame_gap,
        "translation_m": translation,
        "rotation_deg": rotation,
        "pose_novel": bool(pose_novel),
    }
