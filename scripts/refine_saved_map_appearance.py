#!/usr/bin/env python3
"""Continue full-trajectory appearance optimization with geometry held fixed."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils_new.render_utils import select_gaussian_ply
from utils_new.scene_mapper import SceneMapper


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sh-lr", type=float, default=0.005)
    parser.add_argument("--opacity-lr", type=float, default=0.05)
    parser.add_argument("--sh-degree", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--hard-fraction", type=float, default=0.75)
    parser.add_argument("--color-loss-type", choices=("l1", "l2"), default="l1")
    parser.add_argument(
        "--ssim-weight",
        type=float,
        default=None,
        help="Override Loss.lambda_ssim during appearance-only replay.",
    )
    parser.add_argument("--calibrate-exposure", action="store_true")
    parser.add_argument("--exposure-min-scale", type=float, default=0.8)
    parser.add_argument("--exposure-max-scale", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--appearance-lod",
        choices=("off", "evidence", "shuffled"),
        default="off",
    )
    parser.add_argument("--lod-observation-stride", type=int, default=2)
    parser.add_argument("--lod-max-observation-frames", type=int, default=80)
    parser.add_argument("--lod-max-sh1-fraction", type=float, default=0.65)
    parser.add_argument("--lod-max-sh2-fraction", type=float, default=0.30)
    parser.add_argument("--lod-min-views-sh1", type=int, default=3)
    parser.add_argument("--lod-min-views-sh2", type=int, default=6)
    parser.add_argument("--lod-min-radius-sh1", type=float, default=1.0)
    parser.add_argument("--lod-min-radius-sh2", type=float, default=2.0)
    parser.add_argument("--lod-min-dispersion-sh1", type=float, default=1.0e-4)
    parser.add_argument("--lod-min-dispersion-sh2", type=float, default=5.0e-4)
    parser.add_argument("--detail-l1-weight", type=float, default=0.0)
    parser.add_argument("--detail-gradient-weight", type=float, default=0.0)
    parser.add_argument("--detail-laplacian-fine-weight", type=float, default=0.0)
    parser.add_argument("--detail-laplacian-coarse-weight", type=float, default=0.0)
    parser.add_argument("--detail-gradient-threshold", type=float, default=0.04)
    parser.add_argument("--detail-side-boost", type=float, default=1.0)
    parser.add_argument("--detail-side-start", type=float, default=1.0)
    parser.add_argument("--far-structure-tail-steps", type=int, default=0)
    parser.add_argument("--far-structure-weight", type=float, default=0.0)
    parser.add_argument("--far-structure-depth", type=float, default=25.0)
    parser.add_argument("--far-structure-opacity", type=float, default=0.1)
    parser.add_argument("--anchor-sh0-weight", type=float, default=0.0)
    parser.add_argument("--anchor-shn-weight", type=float, default=0.0)
    parser.add_argument("--anchor-opacity-weight", type=float, default=0.0)
    return parser.parse_args()


def load_tracked_cameras(path):
    if not path.is_file():
        return {"cameras": []}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    camera_map = {
        camera["name"]: camera
        for camera in payload.get("cameras", [])
        if camera.get("name") is not None and camera.get("pose") is not None
    }
    return payload, camera_map


def calibrate_exposures(mapper, tracked, minimum_scale, maximum_scale):
    """Fit one bounded photometric gain per observed frame."""

    start = time.perf_counter()
    scales = []
    for camera in mapper.post_refinement_frames:
        camera.to_device(mapper.device)
        with torch.no_grad():
            rendered = torch.clamp(mapper.gaussians.render(camera)["render"], 0.0, 1.0)
            target = camera.get_gt_image()
            valid = (target.sum(dim=-1, keepdim=True) > 1.0e-7).to(rendered.dtype)
            numerator = (rendered * target * valid).sum()
            denominator = (rendered.square() * valid).sum().clamp_min(1.0e-12)
            scale = float((numerator / denominator).item())
        scale = min(max(scale, minimum_scale), maximum_scale)
        camera.exposure_gain = float(camera.exposure_gain) * scale
        record = tracked.get(camera.name)
        if record is not None:
            record["exposure_gain"] = float(camera.exposure_gain)
        scales.append(scale)
        camera.to_device("cpu")
    values = np.asarray(scales, dtype=np.float64)
    return {
        "enabled": True,
        "frames": int(len(values)),
        "min_scale": float(values.min()) if len(values) else 1.0,
        "median_scale": float(np.median(values)) if len(values) else 1.0,
        "mean_scale": float(values.mean()) if len(values) else 1.0,
        "max_scale": float(values.max()) if len(values) else 1.0,
        "lower_bound": float(minimum_scale),
        "upper_bound": float(maximum_scale),
        "seconds": time.perf_counter() - start,
    }


def snapshot_geometry(model):
    names = ("means", "scales", "quats")
    return {
        name: torch.cat(
            [
                model.gaussian_groups[group_id].splats[name].detach().cpu()
                for group_id in model.valid_groups
                if model.gaussian_groups[group_id].splats is not None
                and model.gaussian_groups[group_id].get_num > 0
            ],
            dim=0,
        ).clone()
        for name in names
    }


def geometry_delta(model, before):
    after = snapshot_geometry(model)
    result = {}
    for name, reference in before.items():
        current = after[name]
        if current.shape != reference.shape:
            raise RuntimeError(
                "Geometry shape changed for {}: {} != {}".format(
                    name, tuple(current.shape), tuple(reference.shape)
                )
            )
        difference = torch.abs(current - reference)
        result[name + "_max_abs_change"] = (
            float(difference.max().item()) if difference.numel() else 0.0
        )
        result[name + "_exactly_equal"] = bool(torch.equal(current, reference))
    return result


def main():
    args = parse_args()
    if args.steps < 0 or (args.steps == 0 and not args.calibrate_exposure):
        raise ValueError(
            "--steps must be positive unless exposure calibration is enabled"
        )
    if args.sh_lr <= 0.0 or args.opacity_lr <= 0.0:
        raise ValueError("Appearance learning rates must be positive")
    if not 0.0 <= args.hard_fraction <= 1.0:
        raise ValueError("--hard-fraction must be in [0, 1]")
    if not 0.0 <= args.lod_max_sh1_fraction <= 1.0:
        raise ValueError("--lod-max-sh1-fraction must be in [0, 1]")
    if not 0.0 <= args.lod_max_sh2_fraction <= 1.0:
        raise ValueError("--lod-max-sh2-fraction must be in [0, 1]")
    detail_weights = (
        args.detail_l1_weight,
        args.detail_gradient_weight,
        args.detail_laplacian_fine_weight,
        args.detail_laplacian_coarse_weight,
    )
    if any(weight < 0.0 for weight in detail_weights):
        raise ValueError("detail loss weights must be non-negative")
    if args.ssim_weight is not None and not 0.0 <= args.ssim_weight <= 1.0:
        raise ValueError("--ssim-weight must be in [0, 1]")
    if not 0.0 < args.exposure_min_scale <= args.exposure_max_scale:
        raise ValueError("Exposure scale bounds are invalid")
    if not 0 <= args.far_structure_tail_steps <= args.steps:
        raise ValueError("--far-structure-tail-steps must be in [0, steps]")
    if args.far_structure_weight < 0.0:
        raise ValueError("--far-structure-weight must be non-negative")
    if args.far_structure_depth <= 0.0:
        raise ValueError("--far-structure-depth must be positive")
    if not 0.0 <= args.far_structure_opacity <= 1.0:
        raise ValueError("--far-structure-opacity must be in [0, 1]")
    if min(
        args.anchor_sh0_weight,
        args.anchor_shn_weight,
        args.anchor_opacity_weight,
    ) < 0.0:
        raise ValueError("Appearance anchor weights must be non-negative")

    source_run = args.source_run.resolve()
    output_run = args.output_run.resolve()
    if output_run.exists() and any(output_run.iterdir()):
        raise FileExistsError("Output run is not empty: {}".format(output_run))
    output_run.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = yaml.safe_load((source_run / "config.yaml").read_text(encoding="utf-8"))
    source_ply = select_gaussian_ply(source_run, config).resolve()
    config["Results"]["save_dir"] = str(output_run)
    config["Results"]["skip_eval"] = True
    config["Results"]["save_gt"] = False
    config["Mapper"].setdefault("scene_exposure_gain", 20.0)
    config["Dataset"]["max_pts_num"] = 1
    config["Model"].pop("DepthCovEstimator", None)
    config["Loss"]["color_loss_type"] = args.color_loss_type
    if args.ssim_weight is not None:
        config["Loss"]["lambda_ssim"] = float(args.ssim_weight)
    config["Mapper"]["pin_kf_gpu"] = False
    config["Mapper"]["CameraOptimizer"]["use_camera_opt"] = False
    if args.sh_degree is not None:
        config["Model"]["sh_degree"] = int(args.sh_degree)
    if args.appearance_lod != "off" and config["Model"]["sh_degree"] < 1:
        raise ValueError("--appearance-lod requires --sh-degree 1 or 2")
    post = config["Mapper"]["post_refinement"]
    post.update(
        {
            "max_steps": int(args.steps),
            "opt_cam": False,
            "freeze_geometry": True,
            "restore_aerocommit_archives": False,
            "use_all_frames": True,
            "all_frame_hard_fraction": float(args.hard_fraction),
            "appearance_only_start_step": 0,
            "appearance_freeze_scales": True,
            "means_lr_init": 0.0,
            "means_lr_final": 0.0,
            "scales_lr_init": 0.0,
            "scales_lr_final": 0.0,
            "quats_lr": 0.0,
            "sh_lr": float(args.sh_lr),
            "opacities_lr": float(args.opacity_lr),
            "detail_l1_weight": float(args.detail_l1_weight),
            "detail_gradient_weight": float(args.detail_gradient_weight),
            "detail_laplacian_fine_weight": float(args.detail_laplacian_fine_weight),
            "detail_laplacian_coarse_weight": float(
                args.detail_laplacian_coarse_weight
            ),
            "detail_gradient_threshold": float(args.detail_gradient_threshold),
            "detail_side_boost": float(args.detail_side_boost),
            "detail_side_start": float(args.detail_side_start),
            "far_structure_tail_steps": int(args.far_structure_tail_steps),
            "far_structure_weight": float(args.far_structure_weight),
            "far_structure_depth_m": float(args.far_structure_depth),
            "far_structure_opacity": float(args.far_structure_opacity),
        }
    )
    post["appearance_lod"] = {
        "enabled": args.appearance_lod != "off",
        "selection_mode": args.appearance_lod,
        "observation_stride": int(args.lod_observation_stride),
        "max_observation_frames": int(args.lod_max_observation_frames),
        "max_sh1_fraction": float(args.lod_max_sh1_fraction),
        "max_sh2_fraction": float(args.lod_max_sh2_fraction),
        "min_views_sh1": int(args.lod_min_views_sh1),
        "min_views_sh2": int(args.lod_min_views_sh2),
        "min_mean_radius_sh1": float(args.lod_min_radius_sh1),
        "min_mean_radius_sh2": float(args.lod_min_radius_sh2),
        "min_angular_dispersion_sh1": float(args.lod_min_dispersion_sh1),
        "min_angular_dispersion_sh2": float(args.lod_min_dispersion_sh2),
        "shuffle_seed": int(args.seed),
        "zero_inactive": True,
    }
    post["appearance_anchor"] = {
        "sh0_weight": float(args.anchor_sh0_weight),
        "shN_weight": float(args.anchor_shn_weight),
        "opacity_weight": float(args.anchor_opacity_weight),
    }
    if "stable_detail_split" in post:
        post["stable_detail_split"]["enabled"] = False

    mapper = SceneMapper(config)
    mapper.gaussians.load_from_ply(str(source_ply))
    tracked_payload, tracked = load_tracked_cameras(source_run / "tracked_info.json")
    while True:
        camera = mapper.get_frame()
        if camera is None:
            break
        record = tracked.get(camera.name)
        if record is not None:
            camera.update_pose_numpy(
                np.asarray(record["pose"], dtype=np.float32),
                exposure_gain=record.get("exposure_gain", camera.exposure_gain),
            )
        mapper.post_refinement_frames.append(camera)
        mapper.opt_log["poses_pair"][camera.cam_idx] = (
            camera.get_raw_pose().detach().cpu().numpy(),
            camera.get_pose().detach().cpu().numpy(),
            bool(record.get("is_key_frame", False)) if record else False,
        )
        mapper.cur_view = camera

    before = snapshot_geometry(mapper.gaussians)
    start = time.perf_counter()
    if args.steps > 0:
        mapper.post_refinement()
    exposure = (
        calibrate_exposures(
            mapper,
            tracked,
            args.exposure_min_scale,
            args.exposure_max_scale,
        )
        if args.calibrate_exposure
        else {"enabled": False}
    )
    elapsed = time.perf_counter() - start
    delta = geometry_delta(mapper.gaussians, before)
    if not all(value for key, value in delta.items() if key.endswith("_exactly_equal")):
        raise RuntimeError("Geometry changed during appearance-only refinement")

    output_ply = output_run / "point_cloud.ply"
    mapper.gaussians.save_as_ply(str(output_ply))
    config["OfflineAppearanceRefinement"] = {
        "source_run": str(source_run),
        "source_ply": str(source_ply),
        "steps": int(args.steps),
        "sh_lr": float(args.sh_lr),
        "opacity_lr": float(args.opacity_lr),
        "hard_fraction": float(args.hard_fraction),
        "color_loss_type": args.color_loss_type,
        "ssim_weight": float(config["Loss"]["lambda_ssim"]),
        "exposure_calibration": exposure,
        "seconds": elapsed,
        "geometry_delta": delta,
        "appearance_lod": mapper.appearance_lod_stats,
        "appearance_anchor": mapper.appearance_anchor_stats,
        "far_structure_tail": {
            "steps": int(args.far_structure_tail_steps),
            "weight": float(args.far_structure_weight),
            "depth_m": float(args.far_structure_depth),
            "opacity": float(args.far_structure_opacity),
        },
        "detail_loss": {
            "l1_weight": float(args.detail_l1_weight),
            "gradient_weight": float(args.detail_gradient_weight),
            "laplacian_fine_weight": float(args.detail_laplacian_fine_weight),
            "laplacian_coarse_weight": float(args.detail_laplacian_coarse_weight),
            "gradient_threshold": float(args.detail_gradient_threshold),
            "side_boost": float(args.detail_side_boost),
            "side_start": float(args.detail_side_start),
        },
    }
    (output_run / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    if tracked_payload.get("cameras"):
        tracked_payload.setdefault("scenes", {})["sh_degree"] = int(
            config["Model"]["sh_degree"]
        )
        (output_run / "tracked_info.json").write_text(
            json.dumps(tracked_payload, indent=2) + "\n", encoding="utf-8"
        )
    for name in ("background_model.json", "vignette.png"):
        source = source_run / name
        if source.is_file():
            shutil.copy2(source, output_run / name)
    source_results = source_run / "results.json"
    results = (
        json.loads(source_results.read_text(encoding="utf-8"))
        if source_results.is_file()
        else {}
    )
    results["online_recon_time"] = (
        float(results.get("online_recon_time", 0.0)) + elapsed
    )
    results["num_gaussians"] = mapper.gaussians.get_num_gaussians
    results["offline_appearance_refinement"] = config["OfflineAppearanceRefinement"]
    (output_run / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config["OfflineAppearanceRefinement"], indent=2))
    print(output_run)


if __name__ == "__main__":
    main()
