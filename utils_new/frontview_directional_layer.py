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
    ):
        if not isinstance(result[key], bool):
            raise TypeError("FrontViewDirectionalLayer.{} must be boolean".format(key))
    for key in (
        "sparse_point_threshold",
        "anchor_interval_frames",
        "max_anchors",
        "min_anchors",
    ):
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
    if float(result["far_depth_m"]) <= 0.0:
        raise ValueError("FrontViewDirectionalLayer.far_depth_m must be positive")
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
    return result


def camera_center(world_to_camera):
    pose = torch.as_tensor(world_to_camera, dtype=torch.float32)
    return -pose[:3, :3].T @ pose[:3, 3]


def directional_pose_score(anchor_pose, target_pose, far_depth_m):
    """Bound angular mismatch by rotation plus far-depth translation drift."""

    anchor_pose = torch.as_tensor(anchor_pose, dtype=torch.float32)
    target_pose = torch.as_tensor(target_pose, dtype=torch.float32)
    relative = anchor_pose[:3, :3] @ target_pose[:3, :3].T
    cosine = torch.clamp((torch.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    rotation = torch.acos(cosine)
    translation = torch.linalg.norm(
        camera_center(anchor_pose) - camera_center(target_pose)
    )
    return float((rotation + translation / float(far_depth_m)).item())


class FrontViewDirectionalLayer:
    """Small online image bank rendered only where metric geometry is unobservable."""

    def __init__(self, config=None):
        self.config = validate_front_view_directional_layer_config(config)
        self.anchors = []
        self.active = False
        self._pixel_grid_cache = {}
        self._anchor_tensor_cache = {}
        self.stats = {
            "sparse_observations": 0,
            "anchors_captured": 0,
            "anchors_evicted": 0,
            "render_calls": 0,
            "rendered_pixels": 0,
            "consistency_pixels": 0,
            "far_pixels": 0,
            "certified_pixels": 0,
            "last_anchor_frame": -1,
        }

    @property
    def enabled(self):
        return bool(self.config["enabled"])

    def observe(self, camera):
        if not self.enabled:
            return False
        sparse_count = len(camera.get_pts())
        if sparse_count >= int(self.config["sparse_point_threshold"]):
            return False
        self.stats["sparse_observations"] += 1
        frame_id = int(camera.cam_idx)
        if self.anchors and frame_id - int(self.anchors[-1]["frame_id"]) < int(
            self.config["anchor_interval_frames"]
        ):
            return False

        image = camera.get_gt_image(0).detach().float()
        exposure = max(float(camera.exposure_gain), 1.0e-8)
        anchor = {
            "frame_id": frame_id,
            "image": torch.round(torch.clamp(image, 0.0, 1.0).cpu() * 255.0).to(
                torch.uint8
            ),
            "exposure_gain": exposure,
            "pose": camera.get_pose().detach().cpu().float(),
            "intrinsics": camera.get_int_mat(0).detach().cpu().float(),
        }
        self.anchors.append(anchor)
        if len(self.anchors) > int(self.config["max_anchors"]):
            self.anchors.pop(0)
            self.stats["anchors_evicted"] += 1
        self._anchor_tensor_cache.clear()
        self.stats["anchors_captured"] += 1
        self.stats["last_anchor_frame"] = frame_id
        return True

    def activate(self, enabled=True):
        self.active = bool(enabled) and len(self.anchors) >= int(
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
    ):
        device = target_pose.device
        dtype = target_pose.dtype
        source_pose, source_intrinsics, image = self._anchor_tensors(
            anchor, device, dtype
        )
        target_intrinsics = target_intrinsics.to(device=device, dtype=dtype)
        if inverse_target_intrinsics is None:
            inverse_target_intrinsics = torch.linalg.inv(target_intrinsics)
        target_to_source = (
            source_intrinsics
            @ source_pose[:3, :3]
            @ target_pose[:3, :3].T
            @ inverse_target_intrinsics
        )
        if pixels is None:
            pixels = self._cached_pixel_grid(height, width, device, dtype)
        projected = pixels @ target_to_source.T
        z = projected[..., 2]
        u = projected[..., 0] / torch.clamp(z, min=1.0e-8)
        v = projected[..., 1] / torch.clamp(z, min=1.0e-8)
        valid = (
            (z > 0.0)
            & (u >= 0.0)
            & (u <= float(width - 1))
            & (v >= 0.0)
            & (v <= float(height - 1))
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
        return torch.clamp(warped * exposure_scale, 0.0, 1.0), valid

    def _select_anchors(self, camera):
        target_pose = camera.get_pose().detach().cpu().float()
        candidates = []
        for anchor in self.anchors:
            if bool(self.config["causal_only"]) and int(anchor["frame_id"]) >= int(
                camera.cam_idx
            ):
                continue
            if bool(self.config["exclude_exact_frame"]) and int(
                anchor["frame_id"]
            ) == int(camera.cam_idx):
                continue
            score = directional_pose_score(
                anchor["pose"], target_pose, self.config["far_depth_m"]
            )
            candidates.append((score, int(anchor["frame_id"]), anchor))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in candidates[:2]]

    @torch.no_grad()
    def composite(self, camera, colors, depth, opacity):
        if not self.active:
            return colors
        if int(camera.cam_idx) < int(self.anchors[0]["frame_id"]):
            return colors
        selected = self._select_anchors(camera)
        if len(selected) < 2:
            return colors

        target_pose = camera.get_pose().detach().to(colors)
        intrinsics = camera.get_int_mat(0).detach().to(colors)
        height, width = colors.shape[:2]
        pixels = self._cached_pixel_grid(height, width, colors.device, colors.dtype)
        inverse_intrinsics = torch.linalg.inv(intrinsics)
        warped = []
        valid = []
        for anchor in selected:
            image, mask = self._warp_anchor(
                anchor,
                target_pose,
                intrinsics,
                height,
                width,
                camera.exposure_gain,
                pixels=pixels,
                inverse_target_intrinsics=inverse_intrinsics,
            )
            warped.append(image)
            valid.append(mask)

        consistency = torch.mean(torch.abs(warped[0] - warped[1]), dim=-1) <= float(
            self.config["consistency_threshold"]
        )
        alpha = opacity.squeeze(-1)
        geometry_far = alpha <= float(self.config["low_opacity_threshold"])
        if depth is not None:
            metric_depth = depth.squeeze(-1)
            geometry_far |= torch.isfinite(metric_depth) & (
                metric_depth >= float(self.config["far_depth_m"])
            )
        certified = valid[0] & valid[1] & consistency
        mask = (
            certified & geometry_far
            if bool(self.config["use_geometry_gate"])
            else certified
        )
        weight = float(self.config["blend_weight"])
        replacement = torch.lerp(colors, warped[0], weight)
        result = torch.where(mask.unsqueeze(-1), replacement, colors)

        self.stats["render_calls"] += 1
        self.stats["rendered_pixels"] += int(mask.sum().item())
        self.stats["consistency_pixels"] += int(consistency.sum().item())
        self.stats["far_pixels"] += int(geometry_far.sum().item())
        self.stats["certified_pixels"] += int(certified.sum().item())
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "anchors": self.anchors,
                "stats": self.stats,
            },
            path,
        )

    def load(self, path):
        payload = torch.load(Path(path), map_location="cpu")
        self.config = validate_front_view_directional_layer_config(payload["config"])
        self.anchors = payload["anchors"]
        self._pixel_grid_cache.clear()
        self._anchor_tensor_cache.clear()
        self.stats.update(payload.get("stats", {}))
        self.activate(True)

    def summary(self):
        result = dict(self.stats)
        result.update(
            {
                "enabled": self.enabled,
                "active": bool(self.active),
                "anchor_count": len(self.anchors),
            }
        )
        return result
