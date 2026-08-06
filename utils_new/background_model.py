"""Compact far-field background model for depthless sky observations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


BACKGROUND_MODEL_FILENAME = "background_model.json"


@dataclass(frozen=True)
class SkyBackgroundModel:
    rgb: tuple[float, float, float]
    render_top_fraction: float = 0.35

    def composite(self, render: torch.Tensor, opacity: torch.Tensor, camera=None) -> torch.Tensor:
        if render.ndim != 3 or render.shape[-1] != 3:
            raise ValueError("Sky background expects an HxWx3 render")
        alpha = opacity.squeeze(-1) if opacity.ndim == 3 else opacity
        if alpha.shape != render.shape[:2]:
            raise ValueError("Sky background opacity must match the render")
        height = render.shape[0]
        rows = torch.arange(height, device=render.device).reshape(-1, 1)
        sky_mask = rows < int(round(height * self.render_top_fraction))
        weight = (1.0 - alpha).clamp(0.0, 1.0) * sky_mask.to(render.dtype)
        color = render.new_tensor(self.rgb).reshape(1, 1, 3)
        return render * (1.0 - weight[..., None]) + color * weight[..., None]

    def to_dict(self, **metadata):
        return {
            "type": "robust_constant_sky",
            "rgb": list(self.rgb),
            "render_top_fraction": float(self.render_top_fraction),
            **metadata,
        }

    @classmethod
    def from_dict(cls, payload):
        if payload.get("type") != "robust_constant_sky":
            raise ValueError("Unsupported background model type")
        return cls(
            rgb=tuple(float(value) for value in payload["rgb"]),
            render_top_fraction=float(payload["render_top_fraction"]),
        )


class DirectionalSkyBackgroundModel:
    """An infinite-distance sky whose support is certified in world-ray space."""

    def __init__(self, rgb, grid_shape, valid_indices, min_support_frames):
        self.rgb = tuple(float(value) for value in rgb)
        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.valid_indices = np.asarray(valid_indices, dtype=np.int64)
        self.min_support_frames = int(min_support_frames)
        self._grid_cache = {}
        self._ray_cache = {}

    def _valid_grid(self, device):
        key = str(device)
        if key not in self._grid_cache:
            grid = torch.zeros(
                self.grid_shape[0] * self.grid_shape[1],
                device=device,
                dtype=torch.bool,
            )
            indices = torch.as_tensor(
                self.valid_indices, device=device, dtype=torch.long
            )
            grid[indices] = True
            self._grid_cache[key] = grid.reshape(self.grid_shape)
        return self._grid_cache[key]

    def _camera_rays(self, camera, render):
        intrinsic = camera.get_int_mat(0)
        height, width = render.shape[:2]
        values = tuple(float(intrinsic[row, column].item()) for row, column in (
            (0, 0), (1, 1), (0, 2), (1, 2)
        ))
        key = (str(render.device), height, width) + values
        if key not in self._ray_cache:
            fy, fx = values[1], values[0]
            cx, cy = values[2], values[3]
            rows, columns = torch.meshgrid(
                torch.arange(height, device=render.device, dtype=render.dtype),
                torch.arange(width, device=render.device, dtype=render.dtype),
                indexing="ij",
            )
            rays = torch.stack(
                ((columns - cx) / fx, (rows - cy) / fy, torch.ones_like(rows)),
                dim=-1,
            )
            self._ray_cache[key] = torch.nn.functional.normalize(rays, dim=-1)
        return self._ray_cache[key]

    def composite(self, render, opacity, camera=None):
        if camera is None:
            raise ValueError("Directional sky composition requires a camera")
        alpha = opacity.squeeze(-1) if opacity.ndim == 3 else opacity
        if alpha.shape != render.shape[:2]:
            raise ValueError("Directional sky opacity must match the render")
        rays = self._camera_rays(camera, render)
        camera_to_world = torch.linalg.inv(camera.get_pose())[:3, :3]
        directions = rays @ camera_to_world.T
        grid_height, grid_width = self.grid_shape
        longitude = torch.atan2(directions[..., 0], directions[..., 2])
        latitude = torch.asin(directions[..., 1].clamp(-1.0, 1.0))
        x = torch.floor((longitude / (2.0 * torch.pi) + 0.5) * grid_width)
        y = torch.floor((latitude / torch.pi + 0.5) * grid_height)
        x = x.long().remainder(grid_width)
        y = y.long().clamp(0, grid_height - 1)
        support = self._valid_grid(render.device)[y, x]
        weight = (1.0 - alpha).clamp(0.0, 1.0) * support.to(render.dtype)
        color = render.new_tensor(self.rgb).reshape(1, 1, 3)
        return render * (1.0 - weight[..., None]) + color * weight[..., None]

    def to_dict(self, **metadata):
        return {
            "type": "world_directional_sky",
            "rgb": list(self.rgb),
            "grid_shape": list(self.grid_shape),
            "valid_indices": self.valid_indices.tolist(),
            "min_support_frames": self.min_support_frames,
            **metadata,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            rgb=payload["rgb"],
            grid_shape=payload["grid_shape"],
            valid_indices=payload["valid_indices"],
            min_support_frames=payload["min_support_frames"],
        )


def _selected_camera_infos(dataset_config):
    data_source = dataset_config.get("data_source", "aria")
    trajectory_name = "trajectory_orb.json" if data_source == "orb" else "trajectory.json"
    dataset_path = Path(dataset_config["dataset_path"])
    cameras = json.loads(
        (dataset_path / trajectory_name).read_text(encoding="utf-8")
    )["cameras"]
    begin = int(dataset_config.get("begin_cutoff", -1))
    end = int(dataset_config.get("end_cutoff", -1))
    if begin > 0:
        cameras = cameras[begin:]
    if end > 0:
        cameras = cameras[:-end]
    return cameras[:: int(dataset_config.get("stride", 1))]


def fit_sky_background(config, options=None):
    options = dict(options or {})
    dataset_config = config.get("Testset", config["Dataset"])
    dataset_path = Path(dataset_config["dataset_path"])
    observation_top_fraction = float(options.get("observation_top_fraction", 0.4))
    render_top_fraction = float(options.get("render_top_fraction", 0.35))
    frame_stride = max(1, int(options.get("frame_stride", 1)))
    pixel_stride = max(1, int(options.get("pixel_stride", 4)))
    min_luminance = float(options.get("min_luminance", 0.45))
    max_saturation = float(options.get("max_saturation", 0.22))
    environment_height = max(8, int(options.get("environment_height", 256)))
    environment_width = max(16, int(options.get("environment_width", 512)))
    min_support_frames = max(2, int(options.get("min_support_frames", 20)))

    histograms = np.zeros((3, 256), dtype=np.int64)
    frame_support = np.zeros(
        (environment_height, environment_width), dtype=np.uint16
    )
    sampled_frames = 0
    candidate_pixels = 0
    for info in _selected_camera_infos(dataset_config)[::frame_stride]:
        path = dataset_path / "rectified" / info["image"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError("Could not read background sample: {}".format(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height = max(1, int(round(rgb.shape[0] * observation_top_fraction)))
        pixels = rgb[:height:pixel_stride, ::pixel_stride].reshape(-1, 3)
        normalized = pixels.astype(np.float32) / 255.0
        luminance = normalized.mean(axis=1)
        saturation = normalized.max(axis=1) - normalized.min(axis=1)
        selected = pixels[(luminance >= min_luminance) & (saturation <= max_saturation)]
        selected_mask = (luminance >= min_luminance) & (saturation <= max_saturation)
        calibration = dataset_config["Calibration"]
        rows, columns = np.mgrid[
            0 : rgb.shape[0] : pixel_stride,
            0 : rgb.shape[1] : pixel_stride,
        ]
        rows = rows[: int(np.ceil(height / pixel_stride))].reshape(-1)
        columns = columns[: int(np.ceil(height / pixel_stride))].reshape(-1)
        camera_rays = np.stack(
            (
                (columns - float(calibration["cx"])) / float(calibration["fx"]),
                (rows - float(calibration["cy"])) / float(calibration["fy"]),
                np.ones_like(columns),
            ),
            axis=1,
        ).astype(np.float64)
        camera_rays /= np.linalg.norm(camera_rays, axis=1, keepdims=True)
        camera_to_world = np.linalg.inv(
            np.asarray(info["T_camera_world"], dtype=np.float64)
        )
        directions = camera_rays[selected_mask] @ camera_to_world[:3, :3].T
        x = np.floor(
            (
                np.arctan2(directions[:, 0], directions[:, 2]) / (2.0 * np.pi)
                + 0.5
            )
            * environment_width
        ).astype(np.int64) % environment_width
        y = np.floor(
            (
                np.arcsin(np.clip(directions[:, 1], -1.0, 1.0)) / np.pi
                + 0.5
            )
            * environment_height
        ).astype(np.int64)
        y = np.clip(y, 0, environment_height - 1)
        unique = np.unique(y * environment_width + x)
        frame_support.reshape(-1)[unique] += 1
        for channel in range(3):
            histograms[channel] += np.bincount(
                selected[:, channel], minlength=256
            )
        candidate_pixels += int(len(selected))
        sampled_frames += 1
    if candidate_pixels == 0:
        raise RuntimeError("No valid sky pixels were found for the background model")

    medians = []
    for histogram in histograms:
        target = (int(histogram.sum()) + 1) // 2
        medians.append(int(np.searchsorted(np.cumsum(histogram), target)))
    valid_indices = np.flatnonzero(frame_support.reshape(-1) >= min_support_frames)
    if valid_indices.size == 0:
        raise RuntimeError("No world sky directions passed multi-frame support")
    model = DirectionalSkyBackgroundModel(
        rgb=tuple(value / 255.0 for value in medians),
        grid_shape=frame_support.shape,
        valid_indices=valid_indices,
        min_support_frames=min_support_frames,
    )
    metadata = {
        "observation_top_fraction": observation_top_fraction,
        "frame_stride": frame_stride,
        "pixel_stride": pixel_stride,
        "min_luminance": min_luminance,
        "max_saturation": max_saturation,
        "sampled_frames": sampled_frames,
        "candidate_pixels": candidate_pixels,
        "environment_valid_bins": int(valid_indices.size),
        "environment_valid_percent": float(100.0 * valid_indices.size / frame_support.size),
        "render_top_fraction_legacy_unused": render_top_fraction,
        "causal_input": "observed_training_frames_only",
    }
    return model, metadata


def fit_and_save_sky_background(config, output_dir, options=None):
    model, metadata = fit_sky_background(config, options=options)
    output_path = Path(output_dir) / BACKGROUND_MODEL_FILENAME
    output_path.write_text(
        json.dumps(model.to_dict(**metadata), indent=2), encoding="utf-8"
    )
    return model, output_path


def load_sky_background(path):
    path = Path(path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "world_directional_sky":
        return DirectionalSkyBackgroundModel.from_dict(payload)
    return SkyBackgroundModel.from_dict(payload)
