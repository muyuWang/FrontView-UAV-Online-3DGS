"""Certified directional appearance for low-parallax sparse-dropout views."""

from copy import deepcopy
import math
from pathlib import Path

import torch
import torch.nn.functional as F


DIRECTIONAL_LAYER_FILENAME = "frontview_directional_layer.pt"

DEFAULT_FRONT_VIEW_DIRECTIONAL_LAYER_CONFIG = {
    "enabled": False,
    "sparse_point_threshold": 10,
    "anchor_interval_frames": 20,
    "max_anchors": 12,
    "min_anchors": 2,
    "far_depth_m": 80.0,
    "low_opacity_threshold": 0.50,
    "consistency_threshold": 0.12,
    "blend_weight": 0.75,
    "exclude_exact_frame": True,
    "causal_only": True,
    "use_geometry_gate": False,
    # None preserves the legacy use_geometry_gate behavior for old checkpoints.
    "geometry_gate_mode": None,
    "warp_mode": "rotation",
    "source_fusion": "first",
    "boundary_taper": False,
    "uncertainty_cell_px": None,
    "uncertainty_bootstrap_enabled": False,
    "uncertainty_bootstrap_cell_px": 48.0,
    "uncertainty_bootstrap_max_anchors": 48,
    "uncertainty_bootstrap_blend_weight": 0.75,
    "uncertainty_bootstrap_boundary_taper": True,
    "warp_depth_control": "aligned",
    "pose_score_mode": "fixed_depth",
    "anchor_selection_mode": "interval_fifo",
}


def validate_front_view_directional_layer_config(config=None):
    result = deepcopy(DEFAULT_FRONT_VIEW_DIRECTIONAL_LAYER_CONFIG)
    if config is not None:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(
                "Unknown FrontViewDirectionalLayer options: {}".format(
                    sorted(unknown)
                )
            )
        result.update(config)
    for key in (
        "enabled",
        "exclude_exact_frame",
        "causal_only",
        "use_geometry_gate",
        "boundary_taper",
        "uncertainty_bootstrap_enabled",
        "uncertainty_bootstrap_boundary_taper",
    ):
        if not isinstance(result[key], bool):
            raise TypeError("FrontViewDirectionalLayer.{} must be boolean".format(key))
    if result["geometry_gate_mode"] not in (
        None,
        "none",
        "opacity",
        "depth_or_opacity",
        "metric_transmittance",
        "uncertainty_mass",
    ):
        raise ValueError(
            "FrontViewDirectionalLayer.geometry_gate_mode must be none, opacity, "
            "depth_or_opacity, metric_transmittance, or uncertainty_mass"
        )
    for key in ("sparse_point_threshold", "max_anchors", "min_anchors"):
        if not isinstance(result[key], int) or int(result[key]) < 1:
            raise ValueError(
                "FrontViewDirectionalLayer.{} must be a positive integer".format(
                    key
                )
            )
    if int(result["min_anchors"]) > int(result["max_anchors"]):
        raise ValueError(
            "FrontViewDirectionalLayer.min_anchors cannot exceed max_anchors"
        )
    interval = result["anchor_interval_frames"]
    if interval is not None and (
        not isinstance(interval, int) or int(interval) < 1
    ):
        raise ValueError(
            "FrontViewDirectionalLayer.anchor_interval_frames must be null or "
            "a positive integer"
        )
    for key in (
        "low_opacity_threshold",
        "consistency_threshold",
        "blend_weight",
    ):
        value = float(result[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "FrontViewDirectionalLayer.{} must be in [0, 1]".format(key)
            )
    if float(result["consistency_threshold"]) <= 0.0:
        raise ValueError(
            "FrontViewDirectionalLayer.consistency_threshold must be positive"
        )
    uncertainty_cell = result["uncertainty_cell_px"]
    if uncertainty_cell is not None and float(uncertainty_cell) <= 0.0:
        raise ValueError(
            "FrontViewDirectionalLayer.uncertainty_cell_px must be positive"
        )
    if result["geometry_gate_mode"] == "uncertainty_mass" and uncertainty_cell is None:
        raise ValueError(
            "uncertainty_mass ownership requires uncertainty_cell_px"
        )
    if float(result["uncertainty_bootstrap_cell_px"]) <= 0.0:
        raise ValueError(
            "FrontViewDirectionalLayer.uncertainty_bootstrap_cell_px must be positive"
        )
    if (
        not isinstance(result["uncertainty_bootstrap_max_anchors"], int)
        or int(result["uncertainty_bootstrap_max_anchors"]) < 2
    ):
        raise ValueError(
            "FrontViewDirectionalLayer.uncertainty_bootstrap_max_anchors must be at least 2"
        )
    if not 0.0 <= float(result["uncertainty_bootstrap_blend_weight"]) <= 1.0:
        raise ValueError(
            "FrontViewDirectionalLayer.uncertainty_bootstrap_blend_weight must be in [0, 1]"
        )
    if result["warp_mode"] not in (
        "rotation",
        "adaptive_se3",
        "se3_fallback",
    ):
        raise ValueError(
            "FrontViewDirectionalLayer.warp_mode must be rotation, adaptive_se3, "
            "or se3_fallback"
        )
    if result["source_fusion"] not in ("first", "mean", "causal_crossfade"):
        raise ValueError(
            "FrontViewDirectionalLayer.source_fusion must be first, mean, or "
            "causal_crossfade"
        )
    if result["warp_depth_control"] not in ("aligned", "spatial_roll"):
        raise ValueError(
            "FrontViewDirectionalLayer.warp_depth_control must be aligned or "
            "spatial_roll"
        )
    if result["pose_score_mode"] not in (
        "fixed_depth",
        "rendered_inverse_depth",
    ):
        raise ValueError(
            "FrontViewDirectionalLayer.pose_score_mode must be fixed_depth or "
            "rendered_inverse_depth"
        )
    if result["anchor_selection_mode"] not in (
        "interval_fifo",
        "streaming_kcenter",
        "ordered_ward",
        "episode_ordered_ward",
        "episode_bridge_ward",
    ):
        raise ValueError(
            "FrontViewDirectionalLayer.anchor_selection_mode must be "
            "interval_fifo, streaming_kcenter, ordered_ward, or "
            "episode_ordered_ward, or episode_bridge_ward"
        )
    if result["anchor_selection_mode"] == "interval_fifo" and interval is None:
        raise ValueError("interval_fifo requires anchor_interval_frames")
    if (
        result["anchor_selection_mode"] == "episode_bridge_ward"
        and int(result["max_anchors"]) < 3
    ):
        raise ValueError("episode_bridge_ward requires at least three anchors")
    far_depth = result["far_depth_m"]
    gate_mode = result["geometry_gate_mode"]
    if gate_mode is None:
        gate_mode = "depth_or_opacity" if result["use_geometry_gate"] else "none"
    requires_far_depth = result["pose_score_mode"] == "fixed_depth" or (
        gate_mode == "depth_or_opacity"
    )
    if far_depth is not None and float(far_depth) <= 0.0:
        raise ValueError("FrontViewDirectionalLayer.far_depth_m must be positive")
    if requires_far_depth and far_depth is None:
        raise ValueError(
            "Fixed-depth scoring or geometry gating requires far_depth_m"
        )
    return result


def camera_center(world_to_camera):
    pose = torch.as_tensor(world_to_camera, dtype=torch.float32)
    return -pose[:3, :3].T @ pose[:3, 3]


def directional_pose_score(
    anchor_pose, target_pose, far_depth_m=None, inverse_depth_scale=None
):
    """Bound angular mismatch by rotation plus far-depth translation drift."""

    anchor_pose = torch.as_tensor(anchor_pose, dtype=torch.float32)
    target_pose = torch.as_tensor(target_pose, dtype=torch.float32)
    relative = anchor_pose[:3, :3] @ target_pose[:3, :3].T
    cosine = torch.clamp((torch.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    rotation = torch.acos(cosine)
    translation = torch.linalg.norm(
        camera_center(anchor_pose) - camera_center(target_pose)
    )
    if inverse_depth_scale is None:
        if far_depth_m is None or float(far_depth_m) <= 0.0:
            raise ValueError("A positive far depth or inverse-depth scale is required")
        inverse_depth_scale = 1.0 / float(far_depth_m)
    inverse_depth_scale = float(inverse_depth_scale)
    if not math.isfinite(inverse_depth_scale) or inverse_depth_scale < 0.0:
        raise ValueError("Inverse-depth scale must be finite and nonnegative")
    return float((rotation + translation * inverse_depth_scale).item())


def rendered_inverse_depth_scale(depth, opacity=None):
    """Return the opacity-weighted harmonic scene scale from current evidence."""

    depth = torch.as_tensor(depth, dtype=torch.float32).squeeze(-1)
    valid = torch.isfinite(depth) & (depth > 0.0)
    if opacity is None:
        weights = valid.to(depth)
    else:
        weights = torch.clamp(
            torch.as_tensor(opacity, device=depth.device, dtype=depth.dtype).squeeze(-1),
            0.0,
            1.0,
        )
        weights = torch.where(valid, weights, torch.zeros_like(weights))
    denominator = torch.sum(weights)
    if float(denominator.item()) <= torch.finfo(depth.dtype).eps:
        return 0.0
    inverse_depth = torch.where(valid, torch.reciprocal(depth), torch.zeros_like(depth))
    return float((torch.sum(weights * inverse_depth) / denominator).item())


def causal_anchor_crossfade_weight(primary_frame, secondary_frame, target_frame):
    """Return the minimal causal interpolation across two discrete samples."""

    primary_frame = int(primary_frame)
    secondary_frame = int(secondary_frame)
    target_frame = int(target_frame)
    if primary_frame <= secondary_frame:
        return 1.0
    age = target_frame - primary_frame
    if age <= 0:
        return 0.0
    if age == 1:
        return 0.5
    return 1.0


def warp_boundary_support(
    coordinates,
    valid,
    width,
    height,
    minimum_support_px=0.0,
):
    """Certify a warp pair where image support exceeds their disagreement."""

    if len(coordinates) != 2 or len(valid) != 2:
        raise ValueError("Boundary support requires exactly two source warps")
    first, second = coordinates
    support_radius = torch.linalg.norm(first - second, dim=-1).clamp_min(
        max(float(minimum_support_px), 1.0)
    )
    margins = []
    for source in coordinates:
        u, v = source.unbind(dim=-1)
        margins.append(
            torch.minimum(
                torch.minimum(u, float(width - 1) - u),
                torch.minimum(v, float(height - 1) - v),
            ).clamp_min(0.0)
        )
    phase = torch.clamp(
        torch.minimum(margins[0], margins[1]) / support_radius, 0.0, 1.0
    )
    support = phase.square() * (3.0 - 2.0 * phase)
    return support * (valid[0] & valid[1]).to(support.dtype)


def anchor_pair_config(selected, key, default):
    """Use an override only when every anchor in a pair shares the profile."""

    values = []
    for anchor in selected:
        profile = anchor.get("ownership_profile")
        if not isinstance(profile, dict) or key not in profile:
            return default
        values.append(profile[key])
    if not values or any(value != values[0] for value in values[1:]):
        return default
    return values[0]


def pose_distance_matrix(poses, translation_scale):
    """Pairwise SE(3) distance with data-normalized translation units."""

    if isinstance(poses, (list, tuple)):
        poses = torch.stack(
            [torch.as_tensor(pose, dtype=torch.float32) for pose in poses]
        )
    else:
        poses = torch.as_tensor(poses, dtype=torch.float32)
    poses = poses.reshape(-1, 4, 4)
    count = len(poses)
    if count == 0:
        return torch.empty((0, 0), dtype=poses.dtype)
    rotations = poses[:, :3, :3]
    relative = rotations[:, None] @ rotations[None].transpose(-1, -2)
    cosines = torch.clamp(
        (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    rotation = torch.acos(cosines)
    centers = -(rotations.transpose(-1, -2) @ poses[:, :3, 3, None]).squeeze(-1)
    translation = torch.cdist(centers, centers)
    scale = max(float(translation_scale), torch.finfo(poses.dtype).eps)
    distance = rotation + translation / scale
    distance.fill_diagonal_(0.0)
    return distance


def maximin_streaming_subset(poses, budget, translation_scale):
    """Select the budget-sized subset with maximum pairwise separation."""

    if isinstance(poses, (list, tuple)):
        poses = torch.stack(
            [torch.as_tensor(pose, dtype=torch.float32) for pose in poses]
        )
    else:
        poses = torch.as_tensor(poses, dtype=torch.float32)
    poses = poses.reshape(-1, 4, 4)
    budget = int(budget)
    if budget <= 0 or len(poses) != budget + 1:
        raise ValueError("Streaming k-center expects exactly budget + 1 poses")
    distance = pose_distance_matrix(poses, translation_scale)
    best_remove = 0
    best_separation = -math.inf
    for remove in range(len(poses)):
        keep = torch.arange(len(poses)) != remove
        subset = distance[keep][:, keep]
        upper = torch.triu_indices(budget, budget, offset=1)
        separation = float(subset[upper[0], upper[1]].min().item())
        if separation > best_separation:
            best_separation = separation
            best_remove = remove
    return best_remove, best_separation


class FrontViewDirectionalLayer:
    """Small online image bank rendered only where metric geometry is unobservable."""

    def __init__(self, config=None):
        self.config = validate_front_view_directional_layer_config(config)
        self.anchors = []
        self.uncertainty_bootstrap_anchors = []
        self.active = False
        self._pixel_grid_cache = {}
        self._anchor_tensor_cache = {}
        self._pose_center_count = 0
        self._pose_center_mean = torch.zeros(3, dtype=torch.float64)
        self._pose_center_m2 = 0.0
        self._dropout_episode_active = False
        self._uncertainty_bootstrap_started = False
        self._uncertainty_bootstrap_complete = False
        self._uncertainty_bootstrap_recovery_streak = 0
        self._uncertainty_bootstrap_uncertain_streak = 0
        self._uncertainty_bootstrap_max_uncertain_streak = 0
        self._uncertainty_bootstrap_recovery_begin_frame = -1
        self._uncertainty_bootstrap_recovery_snapshot = None
        self._bootstrap_pose_center_count = 0
        self._bootstrap_pose_center_mean = torch.zeros(3, dtype=torch.float64)
        self._bootstrap_pose_center_m2 = 0.0
        self.stats = {
            "sparse_observations": 0,
            "anchors_captured": 0,
            "anchors_evicted": 0,
            "anchors_rejected": 0,
            "render_calls": 0,
            "rendered_pixels": 0,
            "consistency_pixels": 0,
            "far_pixels": 0,
            "certified_pixels": 0,
            "se3_selected_pixels": 0,
            "ownership_weight_sum": 0.0,
            "crossfade_weight_sum": 0.0,
            "crossfade_calls": 0,
            "boundary_support_sum": 0.0,
            "boundary_support_pixels": 0,
            "metric_coverage_sum": 0.0,
            "metric_coverage_pixels": 0,
            "uncertainty_mass_sum": 0.0,
            "uncertainty_mass_pixels": 0,
            "uncertainty_bootstrap_observations": 0,
            "uncertainty_bootstrap_anchors_captured": 0,
            "uncertainty_bootstrap_anchors_merged": 0,
            "uncertainty_bootstrap_begin_frame": -1,
            "uncertainty_bootstrap_end_frame": -1,
            "uncertainty_bootstrap_score_sum": 0.0,
            "uncertainty_bootstrap_recovery_streak": 0,
            "uncertainty_bootstrap_uncertain_streak": 0,
            "uncertainty_bootstrap_max_uncertain_streak": 0,
            "uncertainty_bootstrap_recovery_begin_frame": -1,
            "uncertainty_bootstrap_change_delay": -1,
            "uncertainty_bootstrap_score_timeline": [],
            "last_anchor_frame": -1,
            "last_kcenter_separation": 0.0,
            "last_ward_merge_cost": 0.0,
        }

    @property
    def enabled(self):
        return bool(self.config["enabled"])

    def _combined_anchors(self):
        return self.uncertainty_bootstrap_anchors + self.anchors

    @staticmethod
    def _camera_anchor(camera):
        image = camera.get_gt_image(0).detach().float()
        return {
            "frame_id": int(camera.cam_idx),
            "image": torch.round(torch.clamp(image, 0.0, 1.0).cpu() * 255.0).to(
                torch.uint8
            ),
            "exposure_gain": max(float(camera.exposure_gain), 1.0e-8),
            "pose": camera.get_pose().detach().cpu().float(),
            "intrinsics": camera.get_int_mat(0).detach().cpu().float(),
        }

    def _observe_uncertainty_bootstrap(self, camera, uncertainty_mass):
        if (
            not self.config["uncertainty_bootstrap_enabled"]
            or self._uncertainty_bootstrap_complete
        ):
            return False
        frame_id = int(camera.cam_idx)
        if not self._uncertainty_bootstrap_started:
            self._uncertainty_bootstrap_started = True
            self.stats["uncertainty_bootstrap_begin_frame"] = frame_id

        if uncertainty_mass is not None:
            score = float(
                torch.as_tensor(uncertainty_mass).detach().float().mean().item()
            )
            self.stats["uncertainty_bootstrap_observations"] += 1
            self.stats["uncertainty_bootstrap_score_sum"] += score
            self.stats["uncertainty_bootstrap_score_timeline"].append(
                [frame_id, score]
            )
            # A low-score run is provisional until it outlasts every prior
            # uncertain burst. The snapshot preserves the archive at that boundary.
            if score <= 0.5:
                if self._uncertainty_bootstrap_max_uncertain_streak > 0:
                    if self._uncertainty_bootstrap_recovery_streak == 0:
                        self._uncertainty_bootstrap_recovery_begin_frame = frame_id
                        self._uncertainty_bootstrap_recovery_snapshot = deepcopy(
                            self.uncertainty_bootstrap_anchors
                        )
                    self._uncertainty_bootstrap_recovery_streak += 1
                self._uncertainty_bootstrap_uncertain_streak = 0
            else:
                self._uncertainty_bootstrap_uncertain_streak += 1
                self._uncertainty_bootstrap_max_uncertain_streak = max(
                    self._uncertainty_bootstrap_max_uncertain_streak,
                    self._uncertainty_bootstrap_uncertain_streak,
                )
                self._uncertainty_bootstrap_recovery_streak = 0
                self._uncertainty_bootstrap_recovery_begin_frame = -1
                self._uncertainty_bootstrap_recovery_snapshot = None
            self.stats["uncertainty_bootstrap_recovery_streak"] = int(
                self._uncertainty_bootstrap_recovery_streak
            )
            self.stats["uncertainty_bootstrap_uncertain_streak"] = int(
                self._uncertainty_bootstrap_uncertain_streak
            )
            self.stats["uncertainty_bootstrap_max_uncertain_streak"] = int(
                self._uncertainty_bootstrap_max_uncertain_streak
            )
            self.stats["uncertainty_bootstrap_recovery_begin_frame"] = int(
                self._uncertainty_bootstrap_recovery_begin_frame
            )

        anchor = self._camera_anchor(camera)
        anchor["support_begin_frame"] = int(
            self.stats["uncertainty_bootstrap_begin_frame"]
        )
        anchor["ownership_profile"] = {
            "geometry_gate_mode": "uncertainty_mass",
            "uncertainty_cell_px": float(
                self.config["uncertainty_bootstrap_cell_px"]
            ),
            "blend_weight": float(
                self.config["uncertainty_bootstrap_blend_weight"]
            ),
            "boundary_taper": bool(
                self.config["uncertainty_bootstrap_boundary_taper"]
            ),
            "source_fusion": "first",
        }
        anchor["segment_weight"] = 1
        anchor["segment_begin"] = frame_id
        anchor["segment_end"] = frame_id

        center = camera_center(anchor["pose"]).double()
        self._bootstrap_pose_center_count += 1
        delta = center - self._bootstrap_pose_center_mean
        self._bootstrap_pose_center_mean += delta / float(
            self._bootstrap_pose_center_count
        )
        self._bootstrap_pose_center_m2 += float(
            torch.dot(delta, center - self._bootstrap_pose_center_mean).item()
        )
        translation_scale = math.sqrt(
            max(
                self._bootstrap_pose_center_m2
                / float(self._bootstrap_pose_center_count),
                0.0,
            )
        )
        anchors = self.uncertainty_bootstrap_anchors
        anchors.append(anchor)
        budget = int(self.config["uncertainty_bootstrap_max_anchors"])
        if len(anchors) > budget:
            distance = pose_distance_matrix(
                [item["pose"] for item in anchors], translation_scale
            )
            costs = []
            for index in range(len(anchors) - 1):
                left = int(anchors[index].get("segment_weight", 1))
                right = int(anchors[index + 1].get("segment_weight", 1))
                costs.append(
                    (left * right / float(left + right))
                    * float(distance[index, index + 1].item() ** 2)
                )
            merge = int(torch.argmin(torch.tensor(costs)).item())
            left = anchors[merge]
            right = anchors[merge + 1]
            left_weight = int(left.get("segment_weight", 1))
            right_weight = int(right.get("segment_weight", 1))
            representative = left if left_weight >= right_weight else right
            merged = dict(representative)
            merged["segment_weight"] = left_weight + right_weight
            merged["segment_begin"] = int(left.get("segment_begin", left["frame_id"]))
            merged["segment_end"] = int(right.get("segment_end", right["frame_id"]))
            anchors[merge : merge + 2] = [merged]
            self.stats["uncertainty_bootstrap_anchors_merged"] += 1
        self._anchor_tensor_cache.clear()
        self.stats["uncertainty_bootstrap_anchors_captured"] += 1

        if (
            self._uncertainty_bootstrap_max_uncertain_streak > 0
            and self._uncertainty_bootstrap_recovery_streak
            > self._uncertainty_bootstrap_max_uncertain_streak
        ):
            support_end = self._uncertainty_bootstrap_recovery_begin_frame - 1
            if self._uncertainty_bootstrap_recovery_snapshot is not None:
                self.uncertainty_bootstrap_anchors = (
                    self._uncertainty_bootstrap_recovery_snapshot
                )
            for item in self.uncertainty_bootstrap_anchors:
                item["support_end_frame"] = support_end
            self.stats["uncertainty_bootstrap_end_frame"] = support_end
            self.stats["uncertainty_bootstrap_change_delay"] = (
                frame_id - support_end
            )
            self._uncertainty_bootstrap_complete = True
            self._uncertainty_bootstrap_recovery_snapshot = None
            self._anchor_tensor_cache.clear()
        return True

    def observe(self, camera, uncertainty_mass=None):
        if not self.enabled:
            return False
        bootstrap_captured = self._observe_uncertainty_bootstrap(
            camera, uncertainty_mass
        )
        sparse_count = len(camera.get_pts())
        if sparse_count >= int(self.config["sparse_point_threshold"]):
            if (
                self.config["anchor_selection_mode"]
                in ("episode_ordered_ward", "episode_bridge_ward")
                and self.anchors
                and self._dropout_episode_active
            ):
                if self.config["anchor_selection_mode"] == "episode_bridge_ward":
                    bridge = dict(self.anchors[-1])
                    bridge["episode_boundary"] = True
                    self.anchors[:] = [bridge]
                else:
                    self.anchors.clear()
                self._anchor_tensor_cache.clear()
                self._pose_center_count = 0
                self._pose_center_mean.zero_()
                self._pose_center_m2 = 0.0
                self._dropout_episode_active = False
            return bootstrap_captured
        self.stats["sparse_observations"] += 1
        frame_id = int(camera.cam_idx)
        starts_episode = not self._dropout_episode_active
        self._dropout_episode_active = True
        if self.config["anchor_selection_mode"] == "interval_fifo":
            if self.anchors and frame_id - int(self.anchors[-1]["frame_id"]) < int(
                self.config["anchor_interval_frames"]
            ):
                return False

        anchor = self._camera_anchor(camera)
        if self.config["anchor_selection_mode"] in (
            "streaming_kcenter",
            "ordered_ward",
            "episode_ordered_ward",
            "episode_bridge_ward",
        ):
            center = camera_center(anchor["pose"]).double()
            self._pose_center_count += 1
            delta = center - self._pose_center_mean
            self._pose_center_mean += delta / float(self._pose_center_count)
            self._pose_center_m2 += float(
                torch.dot(delta, center - self._pose_center_mean).item()
            )
            translation_scale = math.sqrt(
                max(self._pose_center_m2 / float(self._pose_center_count), 0.0)
            )
            budget = int(self.config["max_anchors"])
            if self.config["anchor_selection_mode"] in (
                "ordered_ward",
                "episode_ordered_ward",
                "episode_bridge_ward",
            ):
                anchor["segment_weight"] = 1
                anchor["segment_begin"] = frame_id
                anchor["segment_end"] = frame_id
                if (
                    self.config["anchor_selection_mode"]
                    == "episode_bridge_ward"
                    and starts_episode
                ):
                    anchor["episode_boundary"] = True
                self.anchors.append(anchor)
                if len(self.anchors) > budget:
                    poses = [item["pose"] for item in self.anchors]
                    distance = pose_distance_matrix(poses, translation_scale)
                    costs = []
                    for index in range(len(self.anchors) - 1):
                        if bool(
                            self.anchors[index].get("episode_boundary", False)
                        ) or bool(
                            self.anchors[index + 1].get("episode_boundary", False)
                        ):
                            costs.append(math.inf)
                            continue
                        left = int(self.anchors[index].get("segment_weight", 1))
                        right = int(
                            self.anchors[index + 1].get("segment_weight", 1)
                        )
                        ward = (left * right / float(left + right)) * float(
                            distance[index, index + 1].item() ** 2
                        )
                        costs.append(ward)
                    merge = int(torch.argmin(torch.tensor(costs)).item())
                    left = self.anchors[merge]
                    right = self.anchors[merge + 1]
                    left_weight = int(left.get("segment_weight", 1))
                    right_weight = int(right.get("segment_weight", 1))
                    representative = left if left_weight >= right_weight else right
                    merged = dict(representative)
                    merged["segment_weight"] = left_weight + right_weight
                    merged["segment_begin"] = int(
                        left.get("segment_begin", left["frame_id"])
                    )
                    merged["segment_end"] = int(
                        right.get("segment_end", right["frame_id"])
                    )
                    self.anchors[merge : merge + 2] = [merged]
                    self.stats["anchors_evicted"] += 1
                    self.stats["last_ward_merge_cost"] = float(costs[merge])
            elif len(self.anchors) < budget:
                self.anchors.append(anchor)
            else:
                candidates = self.anchors + [anchor]
                remove, separation = maximin_streaming_subset(
                    [item["pose"] for item in candidates],
                    budget,
                    translation_scale,
                )
                self.stats["last_kcenter_separation"] = float(separation)
                if remove == budget:
                    self.stats["anchors_rejected"] += 1
                    return False
                self.anchors = [
                    item for index, item in enumerate(candidates) if index != remove
                ]
                self.stats["anchors_evicted"] += 1
        else:
            self.anchors.append(anchor)
            if len(self.anchors) > int(self.config["max_anchors"]):
                self.anchors.pop(0)
                self.stats["anchors_evicted"] += 1
        self._anchor_tensor_cache.clear()
        self.stats["anchors_captured"] += 1
        self.stats["last_anchor_frame"] = frame_id
        return True

    def activate(self, enabled=True):
        self.active = bool(enabled) and len(self._combined_anchors()) >= int(
            self.config["min_anchors"]
        )
        return self.active

    @staticmethod
    def _pixel_grid(height, width, device, dtype):
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((x, y, torch.ones_like(x)), dim=-1)

    def _anchor_tensors(self, anchor, device, dtype):
        key = (id(anchor), str(device), dtype)
        cached = self._anchor_tensor_cache.get(key)
        if cached is None:
            cached = (
                anchor["pose"].to(device=device, dtype=dtype),
                anchor["intrinsics"].to(device=device, dtype=dtype),
                anchor["image"].to(device=device, dtype=dtype) / 255.0,
            )
            self._anchor_tensor_cache[key] = cached
        return cached

    def _cached_pixel_grid(self, height, width, device, dtype):
        key = (int(height), int(width), str(device), dtype)
        pixels = self._pixel_grid_cache.get(key)
        if pixels is None:
            pixels = self._pixel_grid(height, width, device, dtype)
            self._pixel_grid_cache[key] = pixels
        return pixels

    def _warp_anchor(
        self,
        anchor,
        target_pose,
        target_intrinsics,
        height,
        width,
        exposure,
        pixels=None,
        inverse_target_intrinsics=None,
        target_depth=None,
        return_coordinates=False,
    ):
        device = target_pose.device
        dtype = target_pose.dtype
        source_pose, source_intrinsics, image = self._anchor_tensors(
            anchor, device, dtype
        )
        target_intrinsics = target_intrinsics.to(device=device, dtype=dtype)
        if inverse_target_intrinsics is None:
            inverse_target_intrinsics = torch.linalg.inv(target_intrinsics)
        if pixels is None:
            pixels = self._cached_pixel_grid(height, width, device, dtype)
        if target_depth is None:
            target_to_source = (
                source_intrinsics
                @ source_pose[:3, :3]
                @ target_pose[:3, :3].T
                @ inverse_target_intrinsics
            )
            projected = pixels @ target_to_source.T
            target_valid = torch.ones(
                (height, width), device=device, dtype=torch.bool
            )
        else:
            target_depth = target_depth.to(device=device, dtype=dtype).reshape(
                height, width
            )
            target_camera = (pixels @ inverse_target_intrinsics.T) * target_depth[
                ..., None
            ]
            world = (
                target_camera - target_pose[:3, 3]
            ) @ target_pose[:3, :3]
            source_camera = (
                world @ source_pose[:3, :3].T + source_pose[:3, 3]
            )
            projected = source_camera @ source_intrinsics.T
            target_valid = torch.isfinite(target_depth) & (target_depth > 0.0)
        z = projected[..., 2]
        u = projected[..., 0] / torch.clamp(z, min=1.0e-8)
        v = projected[..., 1] / torch.clamp(z, min=1.0e-8)
        valid = (
            (z > 0.0)
            & (u >= 0.0)
            & (u <= float(width - 1))
            & (v >= 0.0)
            & (v <= float(height - 1))
            & target_valid
        )
        grid = torch.stack(
            (
                2.0 * (u + 0.5) / float(width) - 1.0,
                2.0 * (v + 0.5) / float(height) - 1.0,
            ),
            dim=-1,
        ).unsqueeze(0)
        warped = F.grid_sample(
            image.permute(2, 0, 1).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0].permute(1, 2, 0)
        source_exposure = anchor.get("exposure_gain")
        exposure_scale = (
            float(exposure)
            if source_exposure is None
            else float(exposure) / max(float(source_exposure), 1.0e-8)
        )
        warped = torch.clamp(warped * exposure_scale, 0.0, 1.0)
        if return_coordinates:
            return warped, valid, torch.stack((u, v), dim=-1)
        return warped, valid

    def _select_anchors(self, camera, depth=None, opacity=None, count=2):
        target_pose = camera.get_pose().detach().cpu().float()
        inverse_depth_scale = None
        if self.config["pose_score_mode"] == "rendered_inverse_depth":
            inverse_depth_scale = rendered_inverse_depth_scale(depth, opacity)
        candidates = []
        for anchor in self._combined_anchors():
            support_begin = anchor.get("support_begin_frame")
            support_end = anchor.get("support_end_frame")
            if support_begin is not None and int(camera.cam_idx) < int(support_begin):
                continue
            if support_end is not None and int(camera.cam_idx) > int(support_end):
                continue
            if bool(self.config["causal_only"]) and int(anchor["frame_id"]) >= int(
                camera.cam_idx
            ):
                continue
            if bool(self.config["exclude_exact_frame"]) and int(
                anchor["frame_id"]
            ) == int(camera.cam_idx):
                continue
            score = directional_pose_score(
                anchor["pose"],
                target_pose,
                self.config["far_depth_m"],
                inverse_depth_scale=inverse_depth_scale,
            )
            candidates.append((score, int(anchor["frame_id"]), anchor))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in candidates[: int(count)]]

    def uncertainty_cell_px_for_camera(self, camera):
        if self.config["geometry_gate_mode"] == "uncertainty_mass":
            return self.config["uncertainty_cell_px"]
        cells = []
        for anchor in self._combined_anchors():
            profile = anchor.get("ownership_profile")
            if not isinstance(profile, dict):
                continue
            if profile.get("geometry_gate_mode") != "uncertainty_mass":
                continue
            support_begin = anchor.get("support_begin_frame")
            support_end = anchor.get("support_end_frame")
            if support_begin is not None and int(camera.cam_idx) < int(support_begin):
                continue
            if support_end is not None and int(camera.cam_idx) > int(support_end):
                continue
            if bool(self.config["causal_only"]) and int(anchor["frame_id"]) >= int(
                camera.cam_idx
            ):
                continue
            if bool(self.config["exclude_exact_frame"]) and int(
                anchor["frame_id"]
            ) == int(camera.cam_idx):
                continue
            cell = profile.get("uncertainty_cell_px")
            if cell is not None:
                cells.append(float(cell))
        if len(cells) < 2:
            if (
                self.config["uncertainty_bootstrap_enabled"]
                and not self._uncertainty_bootstrap_complete
            ):
                return float(self.config["uncertainty_bootstrap_cell_px"])
            return None
        return min(cells)

    def _pair_result(
        self,
        camera,
        colors,
        depth,
        opacity,
        metric_opacity,
        uncertainty_opacity,
        selected,
        record_stats,
    ):
        target_pose = camera.get_pose().detach().to(colors)
        intrinsics = camera.get_int_mat(0).detach().to(colors)
        height, width = colors.shape[:2]
        gate_mode = anchor_pair_config(
            selected, "geometry_gate_mode", self.config["geometry_gate_mode"]
        )
        if gate_mode is None:
            gate_mode = (
                "depth_or_opacity" if self.config["use_geometry_gate"] else "none"
            )
        pixels = self._cached_pixel_grid(height, width, colors.device, colors.dtype)
        inverse_intrinsics = torch.linalg.inv(intrinsics)
        rotation_warped = []
        rotation_valid = []
        rotation_coordinates = []
        return_coordinates = bool(
            anchor_pair_config(
                selected, "boundary_taper", self.config["boundary_taper"]
            )
        )
        for anchor in selected:
            result = self._warp_anchor(
                anchor,
                target_pose,
                intrinsics,
                height,
                width,
                camera.exposure_gain,
                pixels=pixels,
                inverse_target_intrinsics=inverse_intrinsics,
                return_coordinates=return_coordinates,
            )
            image, mask = result[:2]
            rotation_warped.append(image)
            rotation_valid.append(mask)
            if return_coordinates:
                rotation_coordinates.append(result[2])

        warped = rotation_warped
        valid = rotation_valid
        coordinates = rotation_coordinates
        se3_selected = torch.zeros(
            (height, width), device=colors.device, dtype=torch.bool
        )
        if self.config["warp_mode"] in ("adaptive_se3", "se3_fallback") and depth is not None:
            warp_depth = depth
            if self.config["warp_depth_control"] == "spatial_roll":
                warp_depth = torch.roll(
                    depth, shifts=(height // 2, width // 2), dims=(0, 1)
                )
            se3_warped = []
            se3_valid = []
            se3_coordinates = []
            for anchor in selected:
                result = self._warp_anchor(
                    anchor,
                    target_pose,
                    intrinsics,
                    height,
                    width,
                    camera.exposure_gain,
                    pixels=pixels,
                    inverse_target_intrinsics=inverse_intrinsics,
                    target_depth=warp_depth,
                    return_coordinates=return_coordinates,
                )
                image, mask = result[:2]
                se3_warped.append(image)
                se3_valid.append(mask)
                if return_coordinates:
                    se3_coordinates.append(result[2])
            rotation_pair_valid = rotation_valid[0] & rotation_valid[1]
            se3_pair_valid = se3_valid[0] & se3_valid[1]
            rotation_error = torch.mean(
                torch.abs(rotation_warped[0] - rotation_warped[1]), dim=-1
            )
            se3_error = torch.mean(
                torch.abs(se3_warped[0] - se3_warped[1]), dim=-1
            )
            if self.config["warp_mode"] == "se3_fallback":
                threshold = float(self.config["consistency_threshold"])
                rotation_certified = rotation_pair_valid & (
                    rotation_error <= threshold
                )
                se3_certified = se3_pair_valid & (se3_error <= threshold)
                se3_selected = ~rotation_certified & se3_certified
            else:
                se3_selected = se3_pair_valid & (
                    ~rotation_pair_valid | (se3_error < rotation_error)
                )
            warped = [
                torch.where(se3_selected[..., None], se3_image, rotation_image)
                for se3_image, rotation_image in zip(se3_warped, rotation_warped)
            ]
            valid = [
                torch.where(se3_selected, se3_mask, rotation_mask)
                for se3_mask, rotation_mask in zip(se3_valid, rotation_valid)
            ]
            if return_coordinates:
                coordinates = [
                    torch.where(
                        se3_selected[..., None], se3_coordinate, rotation_coordinate
                    )
                    for se3_coordinate, rotation_coordinate in zip(
                        se3_coordinates, rotation_coordinates
                    )
                ]

        consistency = torch.mean(torch.abs(warped[0] - warped[1]), dim=-1) <= float(
            self.config["consistency_threshold"]
        )
        alpha = opacity.squeeze(-1)
        geometry_far = alpha <= float(self.config["low_opacity_threshold"])
        if (
            gate_mode == "depth_or_opacity"
            and depth is not None
            and self.config["far_depth_m"] is not None
        ):
            metric_depth = depth.squeeze(-1)
            geometry_far |= torch.isfinite(metric_depth) & (
                metric_depth >= float(self.config["far_depth_m"])
            )
        certified = valid[0] & valid[1] & consistency
        boundary_support = torch.ones_like(alpha)
        if return_coordinates:
            uncertainty_cell_px = anchor_pair_config(
                selected,
                "uncertainty_cell_px",
                self.config["uncertainty_cell_px"],
            )
            minimum_support_px = (
                float(uncertainty_cell_px)
                if gate_mode == "uncertainty_mass"
                else 0.0
            )
            boundary_support = warp_boundary_support(
                coordinates,
                valid,
                width,
                height,
                minimum_support_px=minimum_support_px,
            )
        mask = certified if gate_mode == "none" else certified & geometry_far
        weight = float(
            anchor_pair_config(
                selected, "blend_weight", self.config["blend_weight"]
            )
        )
        source_fusion = anchor_pair_config(
            selected, "source_fusion", self.config["source_fusion"]
        )
        source_color = (
            0.5 * (warped[0] + warped[1])
            if source_fusion == "mean"
            else warped[0]
        )
        if gate_mode == "metric_transmittance":
            if metric_opacity is None:
                raise RuntimeError(
                    "metric_transmittance ownership requires rendered metric opacity"
                )
            metric_coverage = torch.clamp(metric_opacity.squeeze(-1), 0.0, 1.0)
            ownership = (
                certified.to(colors.dtype)
                * (1.0 - metric_coverage)
                * boundary_support
                * weight
            )
            result = torch.lerp(colors, source_color, ownership.unsqueeze(-1))
            mask = ownership > 0.0
        elif gate_mode == "uncertainty_mass":
            if uncertainty_opacity is None:
                raise RuntimeError(
                    "uncertainty_mass ownership requires projected uncertainty mass"
                )
            uncertainty = torch.clamp(
                uncertainty_opacity.squeeze(-1), 0.0, 1.0
            )
            ownership = (
                certified.to(colors.dtype)
                * uncertainty
                * boundary_support
                * weight
            )
            result = torch.lerp(colors, source_color, ownership.unsqueeze(-1))
            mask = ownership > 0.0
        else:
            replacement = torch.lerp(
                colors, source_color, weight * boundary_support.unsqueeze(-1)
            )
            result = torch.where(mask.unsqueeze(-1), replacement, colors)

        if record_stats:
            self.stats["render_calls"] += 1
            self.stats["rendered_pixels"] += int(mask.sum().item())
            self.stats["consistency_pixels"] += int(consistency.sum().item())
            self.stats["far_pixels"] += int(geometry_far.sum().item())
            self.stats["certified_pixels"] += int(certified.sum().item())
            self.stats["se3_selected_pixels"] += int(se3_selected.sum().item())
            if gate_mode == "metric_transmittance":
                self.stats["ownership_weight_sum"] += float(ownership.sum().item())
                self.stats["metric_coverage_sum"] += float(
                    metric_coverage[certified].sum().item()
                )
                self.stats["metric_coverage_pixels"] += int(certified.sum().item())
            elif gate_mode == "uncertainty_mass":
                self.stats["ownership_weight_sum"] += float(ownership.sum().item())
                self.stats["uncertainty_mass_sum"] += float(uncertainty.sum().item())
                self.stats["uncertainty_mass_pixels"] += int(
                    torch.count_nonzero(uncertainty).item()
                )
            if return_coordinates:
                self.stats["boundary_support_sum"] += float(
                    boundary_support[certified].sum().item()
                )
                self.stats["boundary_support_pixels"] += int(certified.sum().item())
        return result

    @torch.no_grad()
    def composite(
        self,
        camera,
        colors,
        depth,
        opacity,
        metric_opacity=None,
        uncertainty_opacity=None,
    ):
        if not self.active:
            return colors
        anchors = self._combined_anchors()
        if not anchors or int(camera.cam_idx) < min(
            int(anchor["frame_id"]) for anchor in anchors
        ):
            return colors
        smooth_pair = self.config["source_fusion"] == "causal_crossfade"
        selected = self._select_anchors(
            camera, depth=depth, opacity=opacity, count=3 if smooth_pair else 2
        )
        if len(selected) < 2:
            return colors
        current_pair = selected[:2]
        smooth_pair = (
            anchor_pair_config(
                current_pair, "source_fusion", self.config["source_fusion"]
            )
            == "causal_crossfade"
        )
        if not smooth_pair or len(selected) < 3:
            return self._pair_result(
                camera,
                colors,
                depth,
                opacity,
                metric_opacity,
                uncertainty_opacity,
                current_pair,
                True,
            )

        incoming = max(current_pair, key=lambda anchor: int(anchor["frame_id"]))
        older = [anchor for anchor in selected if anchor is not incoming]
        previous_frames = [
            int(anchor["frame_id"])
            for anchor in older
            if int(anchor["frame_id"]) < int(incoming["frame_id"])
        ]
        if len(older) < 2 or not previous_frames:
            return self._pair_result(
                camera,
                colors,
                depth,
                opacity,
                metric_opacity,
                uncertainty_opacity,
                current_pair,
                True,
            )
        previous_frame = max(previous_frames)
        crossfade = causal_anchor_crossfade_weight(
            incoming["frame_id"], previous_frame, camera.cam_idx
        )
        current = self._pair_result(
            camera,
            colors,
            depth,
            opacity,
            metric_opacity,
            uncertainty_opacity,
            current_pair,
            True,
        )
        if crossfade >= 1.0:
            return current
        previous = self._pair_result(
            camera,
            colors,
            depth,
            opacity,
            metric_opacity,
            uncertainty_opacity,
            older[:2],
            False,
        )
        self.stats["crossfade_weight_sum"] += float(crossfade)
        self.stats["crossfade_calls"] += 1
        height, width = colors.shape[:2]
        pixels = self._cached_pixel_grid(height, width, colors.device, colors.dtype)
        u = pixels[..., 0] / max(float(width - 1), 1.0)
        v = pixels[..., 1] / max(float(height - 1), 1.0)
        boundary_distance = 2.0 * torch.minimum(
            torch.minimum(u, 1.0 - u), torch.minimum(v, 1.0 - v)
        )
        boundary_distance = torch.clamp(boundary_distance, 0.0, 1.0)
        boundary_support = boundary_distance.square() * (
            3.0 - 2.0 * boundary_distance
        )
        spatial_crossfade = crossfade + (1.0 - crossfade) * boundary_support
        return torch.lerp(previous, current, spatial_crossfade.unsqueeze(-1))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "anchors": self.anchors,
                "uncertainty_bootstrap_anchors": self.uncertainty_bootstrap_anchors,
                "stats": self.stats,
                "stream_state": {
                    "pose_center_count": self._pose_center_count,
                    "pose_center_mean": self._pose_center_mean,
                    "pose_center_m2": self._pose_center_m2,
                    "dropout_episode_active": self._dropout_episode_active,
                    "uncertainty_bootstrap_started": self._uncertainty_bootstrap_started,
                    "uncertainty_bootstrap_complete": self._uncertainty_bootstrap_complete,
                    "uncertainty_bootstrap_recovery_streak": self._uncertainty_bootstrap_recovery_streak,
                    "uncertainty_bootstrap_uncertain_streak": self._uncertainty_bootstrap_uncertain_streak,
                    "uncertainty_bootstrap_max_uncertain_streak": self._uncertainty_bootstrap_max_uncertain_streak,
                    "uncertainty_bootstrap_recovery_begin_frame": self._uncertainty_bootstrap_recovery_begin_frame,
                    "bootstrap_pose_center_count": self._bootstrap_pose_center_count,
                    "bootstrap_pose_center_mean": self._bootstrap_pose_center_mean,
                    "bootstrap_pose_center_m2": self._bootstrap_pose_center_m2,
                },
            },
            path,
        )

    def load(self, path):
        payload = torch.load(Path(path), map_location="cpu")
        self.config = validate_front_view_directional_layer_config(payload["config"])
        self.anchors = payload["anchors"]
        self.uncertainty_bootstrap_anchors = payload.get(
            "uncertainty_bootstrap_anchors", []
        )
        self._pixel_grid_cache.clear()
        self._anchor_tensor_cache.clear()
        self.stats.update(payload.get("stats", {}))
        stream_state = payload.get("stream_state", {})
        self._pose_center_count = int(stream_state.get("pose_center_count", 0))
        self._pose_center_mean = torch.as_tensor(
            stream_state.get("pose_center_mean", torch.zeros(3)),
            dtype=torch.float64,
        )
        self._pose_center_m2 = float(stream_state.get("pose_center_m2", 0.0))
        self._dropout_episode_active = bool(
            stream_state.get("dropout_episode_active", False)
        )
        self._uncertainty_bootstrap_started = bool(
            stream_state.get("uncertainty_bootstrap_started", False)
        )
        self._uncertainty_bootstrap_complete = bool(
            stream_state.get("uncertainty_bootstrap_complete", False)
        )
        self._uncertainty_bootstrap_recovery_streak = int(
            stream_state.get("uncertainty_bootstrap_recovery_streak", 0)
        )
        self._uncertainty_bootstrap_uncertain_streak = int(
            stream_state.get("uncertainty_bootstrap_uncertain_streak", 0)
        )
        self._uncertainty_bootstrap_max_uncertain_streak = int(
            stream_state.get("uncertainty_bootstrap_max_uncertain_streak", 0)
        )
        self._uncertainty_bootstrap_recovery_begin_frame = int(
            stream_state.get("uncertainty_bootstrap_recovery_begin_frame", -1)
        )
        self._uncertainty_bootstrap_recovery_snapshot = None
        self._bootstrap_pose_center_count = int(
            stream_state.get("bootstrap_pose_center_count", 0)
        )
        self._bootstrap_pose_center_mean = torch.as_tensor(
            stream_state.get("bootstrap_pose_center_mean", torch.zeros(3)),
            dtype=torch.float64,
        )
        self._bootstrap_pose_center_m2 = float(
            stream_state.get("bootstrap_pose_center_m2", 0.0)
        )
        self.activate(True)

    def summary(self):
        result = dict(self.stats)
        result.update(
            {
                "enabled": self.enabled,
                "active": bool(self.active),
                "anchor_count": len(self._combined_anchors()),
                "metric_anchor_count": len(self.anchors),
                "uncertainty_bootstrap_anchor_count": len(
                    self.uncertainty_bootstrap_anchors
                ),
            }
        )
        return result
