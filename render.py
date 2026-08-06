import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import cv2
import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

REPO_ROOT = Path(__file__).resolve().parent
CONDA_BIN = Path(sys.executable).resolve().parent
if CONDA_BIN.exists():
    os.environ["PATH"] = f"{CONDA_BIN}:{os.environ.get('PATH', '')}"

from utils_new.camera_utils import Camera
from utils_new.background_model import (
    BACKGROUND_MODEL_FILENAME,
    load_sky_background,
)
from utils_new.aerocommit.view_detail import (
    DETAIL_FILENAME,
    ViewConditionedDetailStore,
)
from utils_new.aerocommit.frequency_cache import (
    FREQUENCY_CACHE_FILENAME,
    FrequencyResidualCache,
)
from utils_new.gaussian_models import GaussianModel
from utils_new.frontview_directional_layer import DIRECTIONAL_LAYER_FILENAME
from utils_new.render_utils import select_gaussian_ply
from utils_new.tool_utils import focal2fov

DEFAULT_RUN_DIR = (
    REPO_ROOT / "Logs" / "Paper-bike-shop" / "2026-07-01-15-51-47_test_bikeshop"
)


def load_run_config(run_dir: Path):
    with open(run_dir / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "scene_exposure_gain" not in config["Mapper"]:
        config["Mapper"]["scene_exposure_gain"] = 20.0
    for key in ("Dataset", "Testset"):
        if key in config:
            config[key]["scene_exposure_gain"] = config["Mapper"]["scene_exposure_gain"]
    return config


def load_camera_infos(dataset_config, render_begin=None, render_end=None):
    data_source = dataset_config.get("data_source", "aria")
    if data_source == "aria":
        trajectory_name = "trajectory.json"
    elif data_source == "orb":
        trajectory_name = "trajectory_orb.json"
    else:
        raise ValueError(f"Unsupported data_source: {data_source}")

    dataset_path = Path(dataset_config["dataset_path"])
    with open(dataset_path / trajectory_name, "r", encoding="utf-8") as f:
        infos = json.load(f)["cameras"]

    if render_begin is not None or render_end is not None:
        begin = 0 if render_begin is None else render_begin
        end = len(infos) - 1 if render_end is None else render_end
        if begin < 0 or end < begin or end >= len(infos):
            raise ValueError(
                f"Invalid render range [{begin}, {end}] for {len(infos)} frames."
            )
        infos = infos[begin : end + 1]
        begin_cutoff = -1
        end_cutoff = -1
    else:
        begin_cutoff = dataset_config.get("begin_cutoff", -1)
        end_cutoff = dataset_config.get("end_cutoff", -1)

    stride = dataset_config.get("stride", 1)
    exclude_interval = dataset_config.get("exclude_interval", -1)

    if begin_cutoff > 0:
        infos = infos[begin_cutoff:]
    if end_cutoff > 0:
        infos = infos[:-end_cutoff]
    infos = infos[::stride]
    if exclude_interval > 0:
        infos = [info for i, info in enumerate(infos) if i % exclude_interval != 0]

    return infos


def load_vignette(run_dir: Path, dataset_config):
    if not dataset_config.get("vignette", False):
        return None

    candidates = [
        run_dir / "vignette.png",
        Path(dataset_config["dataset_path"]) / "vignette.png",
    ]
    for path in candidates:
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return None


def read_gt_bgr(info, dataset_config, vignette):
    image_path = Path(dataset_config["dataset_path"]) / "rectified" / info["image"]
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read GT image: {image_path}")

    if (
        vignette is not None
        and dataset_config.get("use_vignette_type", "post-render") == "pre-render"
    ):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        valid = vignette > 0
        corrected = np.zeros_like(img_rgb)
        corrected[valid] = img_rgb[valid] / vignette[valid]
        img_bgr = cv2.cvtColor(
            np.clip(corrected * 255.0, 0.0, 255.0).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )

    return img_bgr


def ensure_even_frame(frame):
    height, width = frame.shape[:2]
    return frame[: height - height % 2, : width - width % 2]


class RenderMetricEvaluator:
    def __init__(self, device, far_depth_threshold=50.0, opacity_threshold=0.05):
        self.device = torch.device(device)
        self.far_depth_threshold = float(far_depth_threshold)
        self.opacity_threshold = float(opacity_threshold)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)
        self.lpips.eval()

    def _to_rgb_tensor(self, image_bgr):
        image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        return (
            torch.from_numpy(image_rgb)
            .to(device=self.device, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        )

    @torch.inference_mode()
    def _masked_lpips(self, render, gt, mask):
        if not torch.any(mask):
            return None
        masked_render = torch.where(mask.expand_as(render), render, gt)
        value = self.lpips(masked_render, gt)
        self.lpips.reset()
        return float(value.item())

    @staticmethod
    def _masked_psnr(error, mask):
        if not torch.any(mask):
            return None
        mse = error[mask.expand_as(error)].mean()
        if mse.item() == 0.0:
            return float("inf")
        return float((-10.0 * torch.log10(mse)).item())

    @staticmethod
    def _masked_ssim(ssim_image, mask):
        if not torch.any(mask):
            return None
        return float(ssim_image[mask.expand_as(ssim_image)].mean().item())

    @torch.inference_mode()
    def evaluate(self, render_bgr, gt_bgr, depth=None, opacity=None):
        render = self._to_rgb_tensor(render_bgr)
        gt = self._to_rgb_tensor(gt_bgr)

        invalid_gt = gt.sum(dim=1, keepdim=True) <= 1e-7
        gt = torch.where(invalid_gt, render, gt)

        squared_error = (render - gt) ** 2
        mse = torch.mean(squared_error)
        psnr = (
            torch.tensor(float("inf"), device=self.device)
            if mse.item() == 0.0
            else -10.0 * torch.log10(mse)
        )
        ssim, ssim_image = structural_similarity_index_measure(
            render,
            gt,
            data_range=1.0,
            return_full_image=True,
        )
        lpips = self.lpips(render, gt)
        self.lpips.reset()
        metrics = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
        }
        if depth is None:
            return metrics

        depth = torch.as_tensor(
            np.asarray(depth, dtype=np.float32).squeeze(), device=self.device
        ).view(1, 1, render.shape[-2], render.shape[-1])
        valid_depth = torch.isfinite(depth) & (depth > 0.0) & ~invalid_gt
        if opacity is not None:
            opacity = torch.as_tensor(
                np.asarray(opacity, dtype=np.float32).squeeze(), device=self.device
            ).view_as(depth)
            valid_depth &= opacity >= self.opacity_threshold
        far_mask = valid_depth & (depth >= self.far_depth_threshold)
        near_mask = valid_depth & ~far_mask

        gray = gt.mean(dim=1, keepdim=True)
        sobel_x = gt.new_tensor(
            ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        edge_magnitude = torch.sqrt(
            F.conv2d(gray, sobel_x, padding=1).square()
            + F.conv2d(gray, sobel_y, padding=1).square()
        )
        far_edge_mask = far_mask & (edge_magnitude >= 0.08)

        valid_count = max(int(valid_depth.sum().item()), 1)
        metrics.update(
            {
                "near_psnr": self._masked_psnr(squared_error, near_mask),
                "near_ssim": self._masked_ssim(ssim_image, near_mask),
                "near_lpips": self._masked_lpips(render, gt, near_mask),
                "far_psnr": self._masked_psnr(squared_error, far_mask),
                "far_ssim": self._masked_ssim(ssim_image, far_mask),
                "far_lpips": self._masked_lpips(render, gt, far_mask),
                "far_edge_psnr": self._masked_psnr(
                    squared_error, far_edge_mask
                ),
                "far_edge_ssim": self._masked_ssim(ssim_image, far_edge_mask),
                "far_edge_lpips": self._masked_lpips(
                    render, gt, far_edge_mask
                ),
                "near_pixel_count": int(near_mask.sum().item()),
                "far_pixel_count": int(far_mask.sum().item()),
                "far_edge_pixel_count": int(far_edge_mask.sum().item()),
                "far_pixel_fraction": float(far_mask.sum().item() / valid_count),
            }
        )
        return metrics


def write_render_metrics(output_dir, rows):
    excluded = {"frame_index", "frame_name"}
    metric_names = sorted(
        {
            name
            for row in rows
            for name, value in row.items()
            if name not in excluded and isinstance(value, (int, float))
        }
    )
    values = {}
    for name in metric_names:
        finite_values = [
            row[name]
            for row in rows
            if row.get(name) is not None and np.isfinite(row[name])
        ]
        if finite_values:
            values[name] = np.asarray(finite_values, dtype=np.float64)
    payload = {
        "schema_version": 2,
        "frame_count": len(rows),
        "mean": {name: float(metric_values.mean()) for name, metric_values in values.items()},
        "median": {
            name: float(np.median(metric_values))
            for name, metric_values in values.items()
        },
        "definitions": {
            "psnr": "RGB PSNR in dB with data range [0, 1]. Higher is better.",
            "ssim": "RGB structural similarity with data range [0, 1]. Higher is better.",
            "lpips": "AlexNet LPIPS with normalized [0, 1] RGB input. Lower is better.",
            "invalid_gt": "Pixels whose GT RGB sum is zero are excluded by replacing GT with render.",
            "depth_regions": (
                "Near/far masks use rendered expected depth, valid opacity, and the "
                "same far depth threshold as the primitive diagnostic. Region LPIPS "
                "sets pixels outside the mask to GT in both inputs."
            ),
        },
        "frames": rows,
    }
    metrics_path = Path(output_dir) / "render_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    mean = payload["mean"]
    print(
        "Mean render metrics: "
        f"PSNR={mean['psnr']:.6f} dB, "
        f"SSIM={mean['ssim']:.6f}, "
        f"LPIPS={mean['lpips']:.6f}"
    )
    print(f"  {metrics_path}")
    return metrics_path


class H264VideoWriter:
    def __init__(self, path: Path, fps: float, size):
        self.path = Path(path)
        self.width, self.height = size
        self.released = False
        self.proc = None

        with tempfile.NamedTemporaryFile(
            prefix=f".{self.path.stem}_",
            suffix=self.path.suffix,
            dir=str(self.path.parent),
            delete=False,
        ) as tmp:
            self.tmp_path = Path(tmp.name)

        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-preset",
            "veryfast",
            "-threads",
            "2",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.tmp_path),
        ]
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if self.proc.stdin is None:
            raise RuntimeError(f"Could not open ffmpeg stdin for video writer: {path}")

    def write(self, frame):
        if self.released:
            raise RuntimeError(f"Video writer already released: {self.path}")
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise ValueError(
                f"Frame size {frame.shape[1]}x{frame.shape[0]} does not match "
                f"writer size {self.width}x{self.height}: {self.path}"
            )
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self):
        if self.released:
            return
        self.released = True
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        stdout = (
            self.proc.stdout.read().decode("utf-8", errors="replace")
            if self.proc.stdout
            else ""
        )
        stderr = (
            self.proc.stderr.read().decode("utf-8", errors="replace")
            if self.proc.stderr
            else ""
        )
        ret = self.proc.wait()
        if ret != 0:
            self.tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"ffmpeg failed for {self.path} with code {ret}\n"
                f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        self.tmp_path.replace(self.path)


def make_writer(path: Path, fps: float, size):
    return H264VideoWriter(path, fps, size)


def release_writers(writers):
    first_error = None
    for writer in writers.values():
        try:
            writer.release()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def draw_label(frame, label, x, y):
    cv2.putText(
        frame,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


class DepthColorizer:
    def __init__(self, depth_min=None, depth_max=None):
        if depth_min is not None and depth_max is not None and depth_max <= depth_min:
            raise ValueError("depth_max must be greater than depth_min.")
        self.depth_min = depth_min
        self.depth_max = depth_max

    def _ensure_bounds(self, values):
        values = np.asarray(values, dtype=np.float32)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size == 0:
            if self.depth_min is None:
                self.depth_min = 0.0
            if self.depth_max is None:
                self.depth_max = self.depth_min + 1.0
            return

        if self.depth_min is None:
            self.depth_min = float(np.percentile(values, 2.0))
        if self.depth_max is None:
            self.depth_max = float(np.percentile(values, 98.0))
        if self.depth_max <= self.depth_min:
            self.depth_max = self.depth_min + max(abs(self.depth_min) * 0.01, 1e-3)

    def colors_for_values(self, values):
        values = np.asarray(values, dtype=np.float32)
        self._ensure_bounds(values)
        scale = max(self.depth_max - self.depth_min, 1e-8)
        normalized = np.clip((values - self.depth_min) / scale, 0.0, 1.0)
        # Near geometry is warm and far geometry is cool.
        color_indices = np.rint((1.0 - normalized) * 255.0).astype(np.uint8)
        return cv2.applyColorMap(
            color_indices.reshape(-1, 1), cv2.COLORMAP_TURBO
        ).reshape(-1, 3)

    def colorize(self, depth, opacity=None):
        depth = np.asarray(depth, dtype=np.float32).squeeze()
        valid = np.isfinite(depth) & (depth > 0)
        if opacity is not None:
            valid &= np.asarray(opacity).squeeze() > 1e-4

        self._ensure_bounds(depth[valid])
        scale = max(self.depth_max - self.depth_min, 1e-8)
        safe_depth = np.where(valid, depth, self.depth_min)
        normalized = np.clip((safe_depth - self.depth_min) / scale, 0.0, 1.0)
        color_indices = np.rint((1.0 - normalized) * 255.0).astype(np.uint8)
        frame = cv2.applyColorMap(color_indices, cv2.COLORMAP_TURBO)
        frame[~valid] = (12, 12, 12)
        draw_label(
            frame,
            f"Depth {self.depth_min:.3f} - {self.depth_max:.3f}",
            20,
            40,
        )
        return frame

    def metadata(self):
        return {
            "depth_min": float(self.depth_min),
            "depth_max": float(self.depth_max),
            "colormap": "turbo_inverted",
        }


def write_far_gs_statistics(output_dir, stem, rows, depth_threshold):
    visible_counts = np.array([row["visible_gs"] for row in rows], dtype=np.float64)
    far_counts = np.array([row["far_gs"] for row in rows], dtype=np.float64)
    percentages = np.array([row["far_percentage"] for row in rows], dtype=np.float64)

    total_visible = int(visible_counts.sum())
    total_far = int(far_counts.sum())
    min_row = min(rows, key=lambda row: row["far_percentage"])
    max_row = max(rows, key=lambda row: row["far_percentage"])
    summary = {
        "frame_count": len(rows),
        "depth_threshold": float(depth_threshold),
        "average_visible_gs_per_frame": float(visible_counts.mean()),
        "average_far_gs_per_frame": float(far_counts.mean()),
        "mean_frame_far_percentage": float(percentages.mean()),
        "median_frame_far_percentage": float(np.median(percentages)),
        "min_frame_far_percentage": float(percentages.min()),
        "min_frame_index": int(min_row["frame_index"]),
        "max_frame_far_percentage": float(percentages.max()),
        "max_frame_index": int(max_row["frame_index"]),
        "weighted_far_percentage": (
            100.0 * total_far / total_visible if total_visible > 0 else 0.0
        ),
        "total_visible_gs": total_visible,
        "total_far_gs": total_far,
    }
    for key in (
        "near_radius_mean_px",
        "near_radius_median_px",
        "near_radius_p90_px",
        "far_radius_mean_px",
        "far_radius_median_px",
        "far_radius_p90_px",
    ):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary["mean_frame_" + key] = float(values.mean())
    payload = {
        "definition": (
            "Projected Gaussian primitives with camera-space z depth greater than "
            "or equal to depth_threshold."
        ),
        "summary": summary,
        "frames": rows,
    }

    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Far GS (depth >= {depth_threshold:g}): "
        f"{summary['average_far_gs_per_frame']:.2f}/frame, "
        f"mean {summary['mean_frame_far_percentage']:.4f}%, "
        f"weighted {summary['weighted_far_percentage']:.4f}%"
    )
    print(f"  {json_path}")
    print(f"  {csv_path}")
    return summary


class GaussianPrimitiveRenderer:
    def __init__(
        self,
        gaussians,
        scale,
        outline_scale,
        opacity_floor,
        far_depth_threshold,
        depth_min=None,
        depth_max=None,
    ):
        self.means = gaussians.get_xyz().detach().contiguous()
        self.scales = gaussians.get_scaling().detach().contiguous()
        self.quats = gaussians.get_rotation().detach().contiguous()
        self.opacities = gaussians.get_opacity().detach().reshape(-1).contiguous()
        self.features = gaussians.get_features().detach().contiguous()
        self.sh_degree = gaussians.active_sh_degree
        self.scale = scale
        self.outline_scale = outline_scale
        self.opacity_floor = opacity_floor
        self.far_depth_threshold = far_depth_threshold
        self.depth_min = depth_min
        self.depth_max = depth_max

        if self.sh_degree == 0:
            self.base_colors = torch.clamp(
                self.features[:, 0] * 0.28209479177387814 + 0.5, 0.0, 1.0
            ).contiguous()
        else:
            self.base_colors = None

        color_indices = np.arange(256, dtype=np.uint8).reshape(-1, 1)
        turbo_bgr = cv2.applyColorMap(color_indices, cv2.COLORMAP_TURBO).reshape(-1, 3)
        turbo_rgb = np.ascontiguousarray(turbo_bgr[:, ::-1], dtype=np.float32) / 255.0
        self.turbo_lut = torch.from_numpy(turbo_rgb).to(self.means.device)

    def _get_rgb_colors(self, viewmat):
        if self.base_colors is not None:
            return self.base_colors

        from gsplat.cuda._wrapper import spherical_harmonics

        camera_position = torch.linalg.inv(viewmat)[:3, 3]
        view_directions = self.means - camera_position
        return torch.clamp(
            spherical_harmonics(
                self.sh_degree,
                view_directions,
                self.features,
            )
            + 0.5,
            0.0,
            1.0,
        ).contiguous()

    def _get_depth_colors(self, viewmat, near, far, depth_bounds):
        camera_depths = self.means @ viewmat[2, :3] + viewmat[2, 3]
        valid = (camera_depths > near) & (camera_depths < far)

        if depth_bounds is not None:
            depth_min, depth_max = depth_bounds
        else:
            depth_min, depth_max = self.depth_min, self.depth_max
            valid_depths = camera_depths[valid]
            if valid_depths.numel() > 0:
                if depth_min is None:
                    depth_min = float(torch.quantile(valid_depths, 0.02).item())
                if depth_max is None:
                    depth_max = float(torch.quantile(valid_depths, 0.98).item())
            if depth_min is None:
                depth_min = near
            if depth_max is None:
                depth_max = far

        if depth_max <= depth_min:
            depth_max = depth_min + max(abs(depth_min) * 0.01, 1e-3)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)

        normalized = torch.clamp(
            (camera_depths - self.depth_min) / (self.depth_max - self.depth_min),
            0.0,
            1.0,
        )
        indices = torch.round((1.0 - normalized) * 255.0).long()
        return self.turbo_lut[indices]

    def render(self, cam, depth_bounds=None):
        from gsplat.rendering import rasterization

        viewmat = cam.get_pose()
        viewmats = viewmat[None, :, :]
        intrinsics = cam.get_int_mat(0)[None, ...]
        width = cam.get_width(0)
        height = cam.get_height(0)

        with torch.no_grad():
            rgb_colors = self._get_rgb_colors(viewmat)
            depth_colors = self._get_depth_colors(
                viewmat,
                cam.near,
                cam.far,
                depth_bounds,
            )
            outline_opacities = torch.clamp(
                self.opacities, min=self.opacity_floor, max=0.99
            )
            render_opacities = torch.clamp(self.opacities, max=0.99)

            outer_rgb, outer_alpha, outer_info = rasterization(
                means=self.means,
                quats=self.quats,
                scales=self.scales * self.scale * self.outline_scale,
                opacities=outline_opacities,
                colors=depth_colors,
                viewmats=viewmats,
                Ks=intrinsics,
                width=width,
                height=height,
                near_plane=cam.near,
                far_plane=cam.far,
                radius_clip=0.0,
                render_mode="RGB",
                packed=True,
                rasterize_mode="classic",
            )
            del outer_info

            inner_rgb, inner_alpha, inner_info = rasterization(
                means=self.means,
                quats=self.quats,
                scales=self.scales * self.scale,
                opacities=render_opacities,
                colors=rgb_colors,
                viewmats=viewmats,
                Ks=intrinsics,
                width=width,
                height=height,
                near_plane=cam.near,
                far_plane=cam.far,
                radius_clip=0.0,
                render_mode="RGB",
                packed=True,
                rasterize_mode="classic",
            )
            projected_depths = inner_info.get("depths")
            if projected_depths is None:
                raise RuntimeError("gsplat did not return projected primitive depths.")
            projected_radii = inner_info.get("radii")
            if projected_radii is None:
                raise RuntimeError("gsplat did not return projected primitive radii.")
            if projected_radii.ndim > 1:
                projected_radii = projected_radii.amax(dim=-1)
            visible_count = int(projected_depths.numel())
            far_mask = projected_depths >= self.far_depth_threshold
            far_count = int(torch.count_nonzero(far_mask).item())
            far_percentage = (
                100.0 * far_count / visible_count if visible_count > 0 else 0.0
            )

            def radius_stats(mask):
                values = projected_radii[mask].to(torch.float32)
                if values.numel() == 0:
                    return 0.0, 0.0, 0.0
                return (
                    float(values.mean().item()),
                    float(torch.quantile(values, 0.5).item()),
                    float(torch.quantile(values, 0.9).item()),
                )

            near_radius = radius_stats(~far_mask)
            far_radius = radius_stats(far_mask)
            del inner_info

            primitive_rgb = inner_rgb + (1.0 - inner_alpha) * outer_rgb
            primitive_alpha = inner_alpha + (1.0 - inner_alpha) * outer_alpha
            primitive_rgb = primitive_rgb + (1.0 - primitive_alpha) * (12.0 / 255.0)
            primitive_rgb = torch.clamp(primitive_rgb[0], 0.0, 1.0)

        frame_rgb = (primitive_rgb.cpu().numpy() * 255.0).astype(np.uint8)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        draw_label(
            frame,
            (
                f"Gaussian primitives: {visible_count} | "
                f">={self.far_depth_threshold:g}: {far_count} "
                f"({far_percentage:.2f}%)"
            ),
            20,
            40,
        )
        return frame, {
            "visible_gs": visible_count,
            "far_gs": far_count,
            "far_percentage": far_percentage,
            "near_radius_mean_px": near_radius[0],
            "near_radius_median_px": near_radius[1],
            "near_radius_p90_px": near_radius[2],
            "far_radius_mean_px": far_radius[0],
            "far_radius_median_px": far_radius[1],
            "far_radius_p90_px": far_radius[2],
        }

    def metadata(self):
        return {
            "renderer": "gpu_splat_outline",
            "all_visible": True,
            "scale": float(self.scale),
            "outline_scale": float(self.outline_scale),
            "opacity_floor": float(self.opacity_floor),
            "far_depth_threshold": float(self.far_depth_threshold),
            "depth_min": float(self.depth_min),
            "depth_max": float(self.depth_max),
            "colormap": "turbo_inverted",
        }


def load_tracked_camera_map(run_dir: Path):
    tracked_path = run_dir / "tracked_info.json"
    if not tracked_path.exists():
        return {}

    with open(tracked_path, "r", encoding="utf-8") as f:
        tracked = json.load(f)

    camera_map = {}
    for cam in tracked.get("cameras", []):
        name = cam.get("name")
        pose = cam.get("pose") or cam.get("raw_pose")
        if name is not None and pose is not None:
            camera_map[name] = {
                "pose": np.array(pose, dtype=np.float32),
                "exposure_gain": cam.get("exposure_gain"),
            }
    return camera_map


def build_render_gt_video(
    run_dir,
    output_dir,
    config,
    fps,
    stride,
    max_frames,
    device,
    render_begin,
    render_end,
    generate_metrics,
    generate_depth,
    generate_opacity,
    generate_primitives,
    depth_min,
    depth_max,
    primitive_scale,
    primitive_outline_scale,
    primitive_opacity_floor,
    far_gs_depth_threshold,
    use_view_detail=True,
    view_detail_path=None,
    view_detail_exclude_exact=None,
    view_detail_sources=1,
    use_frequency_cache=True,
    frequency_cache_path=None,
    frequency_cache_exclude_exact=None,
    frequency_cache_strength=None,
    frequency_cache_warp=False,
    use_background=True,
    use_cached_renders=True,
):
    dataset_config = config["Testset"] if "Testset" in config else config["Dataset"]
    infos = load_camera_infos(dataset_config, render_begin, render_end)
    vignette = load_vignette(run_dir, dataset_config)
    render_dir = run_dir / "eval" / "renders"

    indices = list(range(0, len(infos), stride))
    if max_frames > 0:
        indices = indices[:max_frames]
    if not indices:
        raise RuntimeError("No frames selected for render/GT video.")

    gaussians = None
    tracked_camera_map = None
    view_detail_store = None
    frequency_cache = None
    background_model = None
    primitive_renderer = None
    depth_colorizer = DepthColorizer(depth_min, depth_max)
    far_gs_rows = []
    metric_rows = []
    metric_evaluator = (
        RenderMetricEvaluator(
            device,
            far_depth_threshold=far_gs_depth_threshold,
            opacity_threshold=primitive_opacity_floor,
        )
        if generate_metrics
        else None
    )

    resolved_view_detail_path = (
        Path(view_detail_path) if view_detail_path is not None else run_dir / DETAIL_FILENAME
    )
    view_detail_requested = bool(use_view_detail and resolved_view_detail_path.exists())
    resolved_frequency_cache_path = (
        Path(frequency_cache_path)
        if frequency_cache_path is not None
        else run_dir / FREQUENCY_CACHE_FILENAME
    )
    frequency_cache_requested = bool(
        use_frequency_cache and resolved_frequency_cache_path.exists()
    )
    background_requested = bool(
        use_background and (run_dir / BACKGROUND_MODEL_FILENAME).exists()
    )

    def ensure_model():
        nonlocal gaussians, tracked_camera_map, primitive_renderer, view_detail_store
        nonlocal frequency_cache, background_model
        if gaussians is not None:
            return
        gaussians = load_gaussians(run_dir, config, device, vignette)
        if use_background:
            background_model = load_sky_background(
                run_dir / BACKGROUND_MODEL_FILENAME
            )
            if background_model is not None:
                print("Loaded far-field background: {}".format(background_model.rgb))
        tracked_camera_map = load_tracked_camera_map(run_dir)
        if view_detail_requested:
            view_detail_store = ViewConditionedDetailStore.load(
                resolved_view_detail_path,
                device=device,
                sh_degree=int(config["Model"].get("sh_degree", 0)),
            )
            print(
                "Loaded view-conditioned detail: {} sources, {} Gaussians".format(
                    view_detail_store.source_count,
                    view_detail_store.gaussian_count,
                )
            )
        if frequency_cache_requested:
            frequency_cache = FrequencyResidualCache.load(
                resolved_frequency_cache_path
            )
            print(
                "Loaded frequency residual cache: {} sources".format(
                    frequency_cache.source_count
                )
            )
        if generate_primitives:
            primitive_renderer = GaussianPrimitiveRenderer(
                gaussians=gaussians,
                scale=primitive_scale,
                outline_scale=primitive_outline_scale,
                opacity_floor=primitive_opacity_floor,
                far_depth_threshold=far_gs_depth_threshold,
                depth_min=depth_min,
                depth_max=depth_max,
            )

    def get_modalities(idx):
        render_path = render_dir / f"{idx:05d}.png"
        render_bgr = None
        if use_cached_renders and render_path.exists():
            render_bgr = cv2.imread(str(render_path), cv2.IMREAD_COLOR)

        needs_model = (
            generate_depth
            or generate_opacity
            or generate_primitives
            or generate_metrics
            or render_bgr is None
            or view_detail_requested
            or frequency_cache_requested
            or background_requested
        )
        if not needs_model:
            return {"rgb": render_bgr}

        ensure_model()
        info = infos[idx]
        tracked_camera = tracked_camera_map.get(info["image"], {})
        pose = tracked_camera.get(
            "pose", np.array(info["T_camera_world"], dtype=np.float32)
        )
        modalities = render_pose_modalities(
            gaussians,
            pose,
            dataset_config,
            device,
            f"traj_{idx:05d}",
            idx,
            exposure_gain=tracked_camera.get("exposure_gain"),
            view_detail_store=view_detail_store,
            view_detail_exclude_exact=view_detail_exclude_exact,
            view_detail_sources=view_detail_sources,
            frequency_cache=frequency_cache,
            frequency_cache_exclude_exact=frequency_cache_exclude_exact,
            frequency_cache_strength=frequency_cache_strength,
            frequency_cache_warp=frequency_cache_warp,
            background_model=background_model,
        )
        if (
            render_bgr is not None
            and not view_detail_requested
            and not frequency_cache_requested
            and not background_requested
        ):
            modalities["rgb"] = render_bgr
        return modalities

    def frames_for_index(idx):
        modalities = get_modalities(idx)
        gt_bgr = read_gt_bgr(infos[idx], dataset_config, vignette)
        gt_bgr = cv2.resize(
            gt_bgr, (modalities["rgb"].shape[1], modalities["rgb"].shape[0])
        )

        if metric_evaluator is not None:
            metric_rows.append(
                {
                    "frame_index": int(idx),
                    "frame_name": infos[idx].get("image", f"frame_{idx:05d}"),
                    **metric_evaluator.evaluate(
                        modalities["rgb"],
                        gt_bgr,
                        modalities.get("depth"),
                        modalities.get("opacity"),
                    ),
                }
            )

        rgb_gt = np.concatenate([modalities["rgb"], gt_bgr], axis=1)
        draw_label(rgb_gt, "Render", 20, 40)
        draw_label(rgb_gt, "GT", modalities["rgb"].shape[1] + 20, 40)
        frames = {"rgb": ensure_even_frame(rgb_gt)}
        if generate_depth:
            depth_frame = depth_colorizer.colorize(
                modalities["depth"], modalities["opacity"]
            )
            frames["depth"] = ensure_even_frame(depth_frame)
        if generate_opacity:
            opacity = np.asarray(modalities["opacity"], dtype=np.float32).squeeze()
            opacity_u8 = np.rint(np.clip(opacity, 0.0, 1.0) * 255.0).astype(np.uint8)
            opacity_frame = cv2.cvtColor(opacity_u8, cv2.COLOR_GRAY2BGR)
            draw_label(opacity_frame, "Permanent-map opacity", 20, 40)
            frames["opacity"] = ensure_even_frame(opacity_frame)
        if generate_primitives:
            depth_bounds = None
            if depth_colorizer.depth_min is not None:
                depth_bounds = (
                    depth_colorizer.depth_min,
                    depth_colorizer.depth_max,
                )
            primitive_frame, primitive_stats = primitive_renderer.render(
                modalities["camera"], depth_bounds=depth_bounds
            )
            far_gs_rows.append(
                {
                    "frame_index": int(idx),
                    "frame_name": infos[idx].get("image", f"frame_{idx:05d}"),
                    **primitive_stats,
                }
            )
            frames["primitives"] = ensure_even_frame(primitive_frame)
        return frames

    output_paths = {"rgb": output_dir / "render_vs_gt.mp4"}
    if generate_depth:
        output_paths["depth"] = output_dir / "render_depth.mp4"
    if generate_opacity:
        output_paths["opacity"] = output_dir / "render_opacity.mp4"
    if generate_primitives:
        output_paths["primitives"] = output_dir / "render_primitives.mp4"

    first_frames = frames_for_index(indices[0])
    writers = {
        key: make_writer(output_paths[key], fps, (frame.shape[1], frame.shape[0]))
        for key, frame in first_frames.items()
    }
    try:
        for key, frame in first_frames.items():
            writers[key].write(frame)
        for idx in tqdm(indices[1:], desc="render_trajectory"):
            frames = frames_for_index(idx)
            for key, frame in frames.items():
                writers[key].write(frame)
    finally:
        release_writers(writers)

    metadata = {}
    if metric_rows:
        write_render_metrics(output_dir, metric_rows)
    if generate_depth:
        metadata["depth"] = depth_colorizer.metadata()
    if generate_primitives:
        metadata["primitives"] = primitive_renderer.metadata()
        metadata["far_gs_summary"] = write_far_gs_statistics(
            output_dir,
            "render_far_gs_stats",
            far_gs_rows,
            far_gs_depth_threshold,
        )
    if metadata:
        with open(output_dir / "render_modalities.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return list(output_paths.values())


def prepare_render_config(config, device):
    render_config = copy.deepcopy(config)
    render_config["Model"]["device"] = device
    render_config["Model"]["render_mode"] = "RGB+ED"
    render_config["Mapper"]["device"] = device
    if "Loss" in render_config:
        render_config["Loss"]["device"] = device
    if "CameraOptimizer" in render_config["Mapper"]:
        render_config["Mapper"]["CameraOptimizer"]["device"] = device
    render_config["Model"].pop("DepthCovEstimator", None)
    return render_config


def load_gaussians(run_dir, config, device, vignette):
    ply_path = select_gaussian_ply(run_dir, config)
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing Gaussian PLY: {ply_path}")

    print(f"Loading Gaussian PLY: {ply_path}")
    gaussians = GaussianModel(prepare_render_config(config, device))
    gaussians.load_from_ply(str(ply_path))
    directional_layer_path = run_dir / DIRECTIONAL_LAYER_FILENAME
    if directional_layer_path.exists():
        gaussians.load_frontview_directional_layer(directional_layer_path)
    gaussians.set_vignette_img(vignette)
    return gaussians


def camera_centers_and_down_axes(infos):
    w2cs = np.array(
        [np.array(info["T_camera_world"], dtype=np.float32) for info in infos]
    )
    c2ws = np.linalg.inv(w2cs)
    centers = c2ws[:, :3, 3]
    down_axes = c2ws[:, :3, 1]
    forward_axes = c2ws[:, :3, 2]
    right_axes = c2ws[:, :3, 0]
    return centers, right_axes, down_axes, forward_axes


def normalize(vec, eps=1e-8):
    norm = np.linalg.norm(vec)
    if norm < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return vec / norm


def look_at_w2c(center, target, preferred_down):
    forward = normalize(target - center)
    down = preferred_down - np.dot(preferred_down, forward) * forward
    if np.linalg.norm(down) < 1e-6:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        down = fallback - np.dot(fallback, forward) * forward
    down = normalize(down)
    right = normalize(np.cross(down, forward))
    down = normalize(np.cross(forward, right))

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = center
    return np.linalg.inv(c2w).astype(np.float32)


def generate_orbit_poses(
    infos,
    num_frames,
    radius_scale,
    height_offset,
    start_angle,
    revolutions,
):
    centers, right_axes, down_axes, forward_axes = camera_centers_and_down_axes(infos)
    target = np.median(centers, axis=0)

    ref_idx = len(infos) // 2
    ref_down = normalize(np.median(down_axes, axis=0))
    ref_right = right_axes[ref_idx]
    ref_forward = forward_axes[ref_idx]

    up = -ref_down
    plane_x = ref_right - np.dot(ref_right, up) * up
    if np.linalg.norm(plane_x) < 1e-6:
        plane_x = np.cross(up, ref_forward)
    plane_x = normalize(plane_x)
    plane_y = normalize(np.cross(up, plane_x))

    offsets = centers - target
    horizontal_offsets = offsets - np.outer(offsets @ up, up)
    radius = np.median(np.linalg.norm(horizontal_offsets, axis=1)) * radius_scale
    if not np.isfinite(radius) or radius < 1e-4:
        radius = np.median(np.linalg.norm(offsets, axis=1)) * radius_scale
    if not np.isfinite(radius) or radius < 1e-4:
        radius = 1.0

    poses = []
    angles = np.linspace(
        start_angle,
        start_angle + 2.0 * np.pi * revolutions,
        num_frames,
        endpoint=False,
    )
    for angle in angles:
        center = (
            target
            + radius * np.cos(angle) * plane_x
            + radius * np.sin(angle) * plane_y
            + height_offset * up
        )
        poses.append(
            look_at_w2c(center.astype(np.float32), target.astype(np.float32), ref_down)
        )

    meta = {
        "target": target.tolist(),
        "radius": float(radius),
        "height_offset": float(height_offset),
        "radius_scale": float(radius_scale),
        "revolutions": float(revolutions),
        "num_frames": int(num_frames),
        "poses_w2c": [pose.tolist() for pose in poses],
    }
    return poses, meta


def build_camera(pose, dataset_config, device, name, uid, exposure_gain=None):
    cal = dataset_config["Calibration"]
    width = int(cal["width"])
    height = int(cal["height"])
    fx = float(cal["fx"])
    fy = float(cal["fy"])
    cx = float(cal["cx"])
    cy = float(cal["cy"])
    pose_tensor = torch.from_numpy(np.asarray(pose, dtype=np.float32))

    cam = Camera.init_from_dataset(
        uid=uid,
        color=None,
        color_for_pts=None,
        pose=pose_tensor,
        pts=None,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        fovx=focal2fov(fx, width),
        fovy=focal2fov(fy, height),
        image_height=height,
        image_width=width,
        near=float(cal["near"]),
        far=float(cal["far"]),
        exposure_gain=(
            dataset_config["scene_exposure_gain"]
            if exposure_gain is None
            else float(exposure_gain)
        ),
        rot_speed=0,
        depth=None,
        name=name,
    )
    cam.to_device(device)
    return cam


def render_pose_modalities(
    gaussians,
    pose,
    dataset_config,
    device,
    name,
    uid,
    exposure_gain=None,
    view_detail_store=None,
    view_detail_exclude_exact=None,
    view_detail_sources=1,
    frequency_cache=None,
    frequency_cache_exclude_exact=None,
    frequency_cache_strength=None,
    frequency_cache_warp=False,
    background_model=None,
):
    cam = build_camera(
        pose, dataset_config, device, name, uid, exposure_gain=exposure_gain
    )

    with torch.no_grad():
        external_splats = (
            view_detail_store.external_splats_for_pose(
                pose,
                target_frame_index=uid,
                exclude_exact=view_detail_exclude_exact,
                source_count=view_detail_sources,
            )
            if view_detail_store is not None
            else None
        )
        render_pkg = gaussians.render(cam, external_splats=external_splats)
        if frequency_cache is not None:
            residual = frequency_cache.residual_for_pose(
                pose,
                target_frame_index=uid,
                exclude_exact=frequency_cache_exclude_exact,
                device=render_pkg["render"].device,
                dtype=render_pkg["render"].dtype,
                target_depth=render_pkg["depth"],
                target_intrinsics=cam.get_int_mat(0),
                warp_to_target=frequency_cache_warp,
            )
            if residual is not None:
                strength = (
                    float(frequency_cache.metadata.get("strength", 1.0))
                    if frequency_cache_strength is None
                    else float(frequency_cache_strength)
                )
                render_pkg["render"] = torch.clamp(
                    render_pkg["render"] + strength * residual, 0.0, 1.0
                )
        if background_model is not None:
            render_pkg["render"] = background_model.composite(
                render_pkg["render"], render_pkg["opacity"], cam
            )
    if "depth" not in render_pkg:
        raise RuntimeError(
            "Gaussian renderer did not return depth. The render configuration must "
            "use render_mode RGB+ED."
        )

    rgb = torch.clamp(render_pkg["render"], 0.0, 1.0).detach().cpu().numpy()
    rgb_bgr = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return {
        "rgb": rgb_bgr,
        "depth": render_pkg["depth"].detach().cpu().numpy(),
        "opacity": render_pkg["opacity"].detach().cpu().numpy(),
        "camera": cam,
    }


def render_pose(gaussians, pose, dataset_config, device, name, uid):
    return render_pose_modalities(gaussians, pose, dataset_config, device, name, uid)[
        "rgb"
    ]


def build_novel_orbit_video(
    run_dir,
    output_dir,
    config,
    fps,
    num_frames,
    radius_scale,
    height_offset,
    start_angle,
    revolutions,
    device,
    render_begin,
    render_end,
    generate_depth,
    generate_primitives,
    depth_min,
    depth_max,
    primitive_scale,
    primitive_outline_scale,
    primitive_opacity_floor,
    far_gs_depth_threshold,
    use_view_detail=True,
    view_detail_path=None,
    view_detail_sources=1,
    use_background=True,
):
    dataset_config = config["Testset"] if "Testset" in config else config["Dataset"]
    infos = load_camera_infos(dataset_config, render_begin, render_end)
    vignette = load_vignette(run_dir, dataset_config)
    gaussians = load_gaussians(run_dir, config, device, vignette)
    background_model = (
        load_sky_background(run_dir / BACKGROUND_MODEL_FILENAME)
        if use_background
        else None
    )
    resolved_view_detail_path = (
        Path(view_detail_path) if view_detail_path is not None else run_dir / DETAIL_FILENAME
    )
    view_detail_store = (
        ViewConditionedDetailStore.load(
            resolved_view_detail_path,
            device=device,
            sh_degree=int(config["Model"].get("sh_degree", 0)),
        )
        if use_view_detail and resolved_view_detail_path.exists()
        else None
    )
    depth_colorizer = DepthColorizer(depth_min, depth_max)
    far_gs_rows = []
    primitive_renderer = None
    if generate_primitives:
        primitive_renderer = GaussianPrimitiveRenderer(
            gaussians=gaussians,
            scale=primitive_scale,
            outline_scale=primitive_outline_scale,
            opacity_floor=primitive_opacity_floor,
            far_depth_threshold=far_gs_depth_threshold,
            depth_min=depth_min,
            depth_max=depth_max,
        )

    poses, meta = generate_orbit_poses(
        infos=infos,
        num_frames=num_frames,
        radius_scale=radius_scale,
        height_offset=height_offset,
        start_angle=start_angle,
        revolutions=revolutions,
    )

    def frames_for_pose(pose, idx):
        modalities = render_pose_modalities(
            gaussians,
            pose,
            dataset_config,
            device,
            f"novel_{idx:05d}",
            idx,
            view_detail_store=view_detail_store,
            view_detail_exclude_exact=False,
            view_detail_sources=view_detail_sources,
            background_model=background_model,
        )
        frames = {"rgb": ensure_even_frame(modalities["rgb"])}
        if generate_depth:
            depth_frame = depth_colorizer.colorize(
                modalities["depth"], modalities["opacity"]
            )
            frames["depth"] = ensure_even_frame(depth_frame)
        if generate_primitives:
            depth_bounds = None
            if depth_colorizer.depth_min is not None:
                depth_bounds = (
                    depth_colorizer.depth_min,
                    depth_colorizer.depth_max,
                )
            primitive_frame, primitive_stats = primitive_renderer.render(
                modalities["camera"], depth_bounds=depth_bounds
            )
            far_gs_rows.append(
                {
                    "frame_index": int(idx),
                    "frame_name": f"novel_{idx:05d}",
                    **primitive_stats,
                }
            )
            frames["primitives"] = ensure_even_frame(primitive_frame)
        return frames

    output_paths = {"rgb": output_dir / "novel_orbit.mp4"}
    if generate_depth:
        output_paths["depth"] = output_dir / "novel_orbit_depth.mp4"
    if generate_primitives:
        output_paths["primitives"] = output_dir / "novel_orbit_primitives.mp4"

    first_frames = frames_for_pose(poses[0], 0)
    writers = {
        key: make_writer(output_paths[key], fps, (frame.shape[1], frame.shape[0]))
        for key, frame in first_frames.items()
    }
    try:
        for key, frame in first_frames.items():
            writers[key].write(frame)
        for idx, pose in enumerate(tqdm(poses[1:], desc="novel_orbit"), start=1):
            frames = frames_for_pose(pose, idx)
            for key, frame in frames.items():
                writers[key].write(frame)
    finally:
        release_writers(writers)

    if generate_depth:
        meta["depth"] = depth_colorizer.metadata()
    if generate_primitives:
        meta["primitives"] = primitive_renderer.metadata()
        meta["far_gs_summary"] = write_far_gs_statistics(
            output_dir,
            "novel_orbit_far_gs_stats",
            far_gs_rows,
            far_gs_depth_threshold,
        )
    with open(output_dir / "novel_orbit_poses.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return list(output_paths.values())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render videos from an Online-3DGS-Monocular run directory."
    )
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument(
        "--render_begin",
        type=int,
        default=None,
        help="Absolute source frame index to start rendering, e.g. 173.",
    )
    parser.add_argument(
        "--render_end",
        type=int,
        default=None,
        help="Absolute source frame index to stop rendering inclusively, e.g. 373.",
    )
    parser.add_argument("--novel_frames", type=int, default=240)
    parser.add_argument("--radius_scale", type=float, default=0.7)
    parser.add_argument("--height_offset", type=float, default=0.0)
    parser.add_argument("--start_angle", type=float, default=0.0)
    parser.add_argument("--revolutions", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Do not calculate PSNR, SSIM, and LPIPS for the render/GT video.",
    )
    parser.add_argument(
        "--skip_depth",
        action="store_true",
        help="Do not generate colorized expected-depth videos.",
    )
    parser.add_argument(
        "--save_opacity",
        action="store_true",
        help="Generate a permanent-map opacity diagnostic video.",
    )
    parser.add_argument(
        "--skip_primitives",
        action="store_true",
        help="Do not generate projected Gaussian ellipsoid videos.",
    )
    parser.add_argument(
        "--depth_min",
        type=float,
        default=None,
        help="Fixed near bound for depth colors. Defaults to the first frame's 2nd percentile.",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=None,
        help="Fixed far bound for depth colors. Defaults to the first frame's 98th percentile.",
    )
    parser.add_argument("--primitive_scale", type=float, default=1.0)
    parser.add_argument("--primitive_outline_scale", type=float, default=1.25)
    parser.add_argument("--primitive_opacity_floor", type=float, default=0.15)
    parser.add_argument(
        "--far_gs_depth_threshold",
        type=float,
        default=50.0,
        help="Camera-space depth threshold used to count very distant visible Gaussians.",
    )
    parser.add_argument(
        "--skip_render_gt",
        action="store_true",
        help="Only render the novel-view orbit video.",
    )
    parser.add_argument(
        "--skip_novel",
        action="store_true",
        help="Only build the render-vs-GT comparison video.",
    )
    parser.add_argument(
        "--skip_view_detail",
        action="store_true",
        help="Ignore a view-conditioned detail sidecar even if one exists.",
    )
    parser.add_argument(
        "--view_detail_path",
        type=Path,
        default=None,
        help=f"Optional appearance sidecar. Defaults to RUN_DIR/{DETAIL_FILENAME}.",
    )
    parser.add_argument(
        "--view_detail_exclude_exact",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Exclude carriers captured from the exact target frame.",
    )
    parser.add_argument("--view_detail_sources", type=int, default=1)
    parser.add_argument(
        "--skip_frequency_cache",
        action="store_true",
        help="Ignore a frequency residual cache even if one exists.",
    )
    parser.add_argument(
        "--skip_background",
        action="store_true",
        help="Ignore a learned far-field background sidecar even if one exists.",
    )
    parser.add_argument(
        "--ignore_cached_renders",
        action="store_true",
        help="Render RGB directly from the final PLY even when eval/renders PNGs exist.",
    )
    parser.add_argument("--frequency_cache_path", type=Path, default=None)
    parser.add_argument(
        "--frequency_cache_exclude_exact",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--frequency_cache_strength", type=float, default=None)
    parser.add_argument(
        "--frequency_cache_warp",
        action="store_true",
        help="Reproject the selected residual through the stable GS depth.",
    )
    return parser.parse_args()


def configure_cuda_arch(device):
    if not device.startswith("cuda") or "TORCH_CUDA_ARCH_LIST" in os.environ:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu or run on a GPU.")
    torch_device = torch.device(device)
    device_index = torch_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


def main():
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("stride must be greater than zero.")
    if args.novel_frames <= 0:
        raise ValueError("novel_frames must be greater than zero.")
    if args.primitive_scale <= 0:
        raise ValueError("primitive_scale must be greater than zero.")
    if args.primitive_outline_scale <= 1.0:
        raise ValueError("primitive_outline_scale must be greater than one.")
    if not 0.0 <= args.primitive_opacity_floor <= 1.0:
        raise ValueError("primitive_opacity_floor must be between zero and one.")
    if args.far_gs_depth_threshold <= 0:
        raise ValueError("far_gs_depth_threshold must be greater than zero.")
    if args.view_detail_sources <= 0:
        raise ValueError("view_detail_sources must be greater than zero.")

    configure_cuda_arch(args.device)
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "videos")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_run_config(run_dir)
    generated = []

    if not args.skip_render_gt:
        generated.extend(
            build_render_gt_video(
                run_dir=run_dir,
                output_dir=output_dir,
                config=config,
                fps=args.fps,
                stride=args.stride,
                max_frames=args.max_frames,
                device=args.device,
                render_begin=args.render_begin,
                render_end=args.render_end,
                generate_metrics=not args.skip_metrics,
                generate_depth=not args.skip_depth,
                generate_opacity=args.save_opacity,
                generate_primitives=not args.skip_primitives,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                primitive_scale=args.primitive_scale,
                primitive_outline_scale=args.primitive_outline_scale,
                primitive_opacity_floor=args.primitive_opacity_floor,
                far_gs_depth_threshold=args.far_gs_depth_threshold,
                use_view_detail=not args.skip_view_detail,
                view_detail_path=args.view_detail_path,
                view_detail_exclude_exact=args.view_detail_exclude_exact,
                view_detail_sources=args.view_detail_sources,
                use_frequency_cache=not args.skip_frequency_cache,
                frequency_cache_path=args.frequency_cache_path,
                frequency_cache_exclude_exact=args.frequency_cache_exclude_exact,
                frequency_cache_strength=args.frequency_cache_strength,
                frequency_cache_warp=args.frequency_cache_warp,
                use_background=not args.skip_background,
                use_cached_renders=not args.ignore_cached_renders,
            )
        )

    if not args.skip_novel:
        generated.extend(
            build_novel_orbit_video(
                run_dir=run_dir,
                output_dir=output_dir,
                config=config,
                fps=args.fps,
                num_frames=args.novel_frames,
                radius_scale=args.radius_scale,
                height_offset=args.height_offset,
                start_angle=args.start_angle,
                revolutions=args.revolutions,
                device=args.device,
                render_begin=args.render_begin,
                render_end=args.render_end,
                generate_depth=not args.skip_depth,
                generate_primitives=not args.skip_primitives,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                primitive_scale=args.primitive_scale,
                primitive_outline_scale=args.primitive_outline_scale,
                primitive_opacity_floor=args.primitive_opacity_floor,
                far_gs_depth_threshold=args.far_gs_depth_threshold,
                use_view_detail=not args.skip_view_detail,
                view_detail_path=args.view_detail_path,
                view_detail_sources=args.view_detail_sources,
                use_background=not args.skip_background,
            )
        )

    print("Generated videos:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
