#!/usr/bin/env python3
"""Build view-conditioned high-frequency surface carriers from an existing run."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from render import (
    build_camera,
    load_camera_infos,
    load_gaussians,
    load_run_config,
    load_tracked_pose_map,
    load_vignette,
    read_gt_bgr,
)
from utils_new.camera_utils import unproject_pts_tensor
from utils_new.aerocommit.view_detail import DETAIL_FILENAME


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--depth_directory",
        type=Path,
        default=None,
        help="Optional ORB-aligned monocular depth used for carrier geometry.",
    )
    parser.add_argument("--stable_depth_consistency_ratio", type=float, default=0.50)
    parser.add_argument("--max_points_per_source", type=int, default=4096)
    parser.add_argument("--grid_px", type=int, default=3)
    parser.add_argument("--gradient_threshold", type=float, default=0.025)
    parser.add_argument("--frequency_gap_threshold", type=float, default=0.008)
    parser.add_argument("--residual_threshold", type=float, default=0.04)
    parser.add_argument("--opacity_threshold", type=float, default=0.65)
    parser.add_argument("--side_start", type=float, default=0.30)
    parser.add_argument("--vertical_start", type=float, default=0.18)
    parser.add_argument("--near_depth_m", type=float, default=60.0)
    parser.add_argument("--projected_scale_px", type=float, default=0.55)
    parser.add_argument("--surface_offset_m", type=float, default=0.03)
    parser.add_argument("--opacity", type=float, default=0.82)
    parser.add_argument(
        "--include_non_keyframes", action="store_true", help="Use every frame as a source."
    )
    parser.add_argument(
        "--full_surface",
        action="store_true",
        help="Sample the complete stable side surface instead of only residual edges.",
    )
    parser.add_argument(
        "--regular_grid",
        action="store_true",
        help="Use a deterministic image grid before the per-source TopK budget.",
    )
    return parser.parse_args()


def image_gradient(image):
    gray = image.mean(dim=-1)
    gradient = torch.zeros_like(gray)
    horizontal = torch.abs(gray[:, 1:] - gray[:, :-1])
    vertical = torch.abs(gray[1:, :] - gray[:-1, :])
    gradient[:, 1:] = torch.maximum(gradient[:, 1:], horizontal)
    gradient[:, :-1] = torch.maximum(gradient[:, :-1], horizontal)
    gradient[1:, :] = torch.maximum(gradient[1:, :], vertical)
    gradient[:-1, :] = torch.maximum(gradient[:-1, :], vertical)
    return gradient


def spatial_topk(flat_indices, scores, width, grid_px, budget):
    if flat_indices.numel() == 0:
        return flat_indices
    prelimit = min(int(flat_indices.numel()), max(int(budget) * 12, int(budget)))
    top = torch.topk(scores[flat_indices], prelimit, sorted=True).indices
    ranked = flat_indices[top].detach().cpu().numpy()
    x = ranked % int(width)
    y = ranked // int(width)
    cell = (y // int(grid_px)) * ((int(width) + int(grid_px) - 1) // int(grid_px))
    cell += x // int(grid_px)
    _, first = np.unique(cell, return_index=True)
    chosen = ranked[np.sort(first)[: int(budget)]]
    return torch.as_tensor(chosen, device=flat_indices.device, dtype=torch.long)


def main():
    args = parse_args()
    if args.max_points_per_source <= 0 or args.grid_px <= 0:
        raise ValueError("point budget and grid size must be positive")
    run_dir = args.run_dir.resolve()
    output = (args.output or (run_dir / DETAIL_FILENAME)).resolve()
    config = load_run_config(run_dir)
    dataset_config = config.get("Testset", config["Dataset"])
    infos = load_camera_infos(dataset_config)
    tracked_pose_map = load_tracked_pose_map(run_dir)
    with open(run_dir / "tracked_info.json", "r", encoding="utf-8") as handle:
        tracked = json.load(handle)
    tracked_by_name = {cam["name"]: cam for cam in tracked["cameras"]}
    vignette = load_vignette(run_dir, dataset_config)
    gaussians = load_gaussians(run_dir, config, args.device, vignette)

    all_means = []
    all_scales = []
    all_colors = []
    all_opacities = []
    offsets = [0]
    source_indices = []
    source_poses = []
    source_counts = []
    source_times = []
    camera_centers = []
    start_time = time.perf_counter()

    for local_index, info in enumerate(infos):
        tracked_info = tracked_by_name.get(info["image"], {})
        if not args.include_non_keyframes and not tracked_info.get("is_key_frame", False):
            continue
        frame_start = time.perf_counter()
        pose = tracked_pose_map.get(
            info["image"], np.asarray(info["T_camera_world"], dtype=np.float32)
        )
        cam = build_camera(
            pose, dataset_config, args.device, info["image"], local_index
        )
        gt_bgr = read_gt_bgr(info, dataset_config, vignette)
        if gt_bgr.shape[:2] != (cam.get_height(0), cam.get_width(0)):
            gt_bgr = cv2.resize(
                gt_bgr,
                (cam.get_width(0), cam.get_height(0)),
                interpolation=cv2.INTER_AREA,
            )
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(gt_rgb).to(args.device)
        with torch.no_grad():
            render_pkg = gaussians.render(cam)
        rendered = render_pkg["render"]
        stable_depth = render_pkg["depth"].squeeze(-1)
        depth = stable_depth
        if args.depth_directory is not None:
            depth_path = args.depth_directory / "depth_{:04d}.npz".format(local_index)
            if not depth_path.exists():
                raise FileNotFoundError(depth_path)
            with np.load(depth_path) as depth_payload:
                dense_depth = depth_payload["depth"].astype(np.float32)
            if dense_depth.shape != stable_depth.shape:
                dense_depth = cv2.resize(
                    dense_depth,
                    (stable_depth.shape[1], stable_depth.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            depth = torch.from_numpy(dense_depth).to(args.device)
        opacity = render_pkg["opacity"].squeeze(-1)
        gt_gradient = image_gradient(image)
        render_gradient = image_gradient(rendered)
        frequency_gap = torch.clamp(gt_gradient - render_gradient, min=0.0)
        residual = torch.abs(image - rendered).mean(dim=-1)
        height, width = depth.shape
        side = torch.linspace(-1.0, 1.0, width, device=args.device).abs()[None, :]
        rows = torch.linspace(0.0, 1.0, height, device=args.device)[:, None]
        surface_candidate = (
            (opacity >= args.opacity_threshold)
            & torch.isfinite(depth)
            & (depth > args.surface_offset_m)
            & (depth <= args.near_depth_m)
            & (
                ~torch.isfinite(stable_depth)
                | (stable_depth <= 0.0)
                | (
                    torch.abs(depth - stable_depth) / torch.clamp(depth, min=1.0e-6)
                    <= args.stable_depth_consistency_ratio
                )
            )
            & (side >= args.side_start)
            & (rows >= args.vertical_start)
        )
        candidate = surface_candidate
        if not args.full_surface:
            candidate &= (
                (gt_gradient >= args.gradient_threshold)
                & (frequency_gap >= args.frequency_gap_threshold)
                & (residual >= args.residual_threshold)
            )
        importance = (
            (gt_gradient + 2.0 * frequency_gap)
            * torch.clamp(residual, min=0.01)
            * (1.0 + side)
        ).reshape(-1)
        if args.regular_grid:
            grid_mask = torch.zeros_like(candidate)
            grid_mask[:: args.grid_px, :: args.grid_px] = True
            flat_candidates = torch.nonzero(
                (candidate & grid_mask).reshape(-1), as_tuple=False
            ).reshape(-1)
            if flat_candidates.numel() > args.max_points_per_source:
                top = torch.topk(
                    importance[flat_candidates],
                    args.max_points_per_source,
                    sorted=False,
                ).indices
                selected = flat_candidates[top]
            else:
                selected = flat_candidates
        else:
            selected = spatial_topk(
                torch.nonzero(candidate.reshape(-1), as_tuple=False).reshape(-1),
                importance,
                width,
                args.grid_px,
                args.max_points_per_source,
            )
        if selected.numel() == 0:
            continue
        x = selected % width
        y = selected // width
        uv = torch.stack((x + 0.5, y + 0.5), dim=-1).float()
        selected_depth = torch.clamp(
            depth.reshape(-1)[selected] - args.surface_offset_m,
            min=cam.near * 2.0,
        )
        means = unproject_pts_tensor(
            uv, selected_depth, cam.get_int_mat(0), cam.get_pose()
        )
        focal = 0.5 * (float(cam.get_fx(0)) + float(cam.get_fy(0)))
        scale = args.projected_scale_px * selected_depth / focal
        scales = scale[:, None].expand(-1, 3)
        colors = image.reshape(-1, 3)[selected]
        colors = colors * float(config["Mapper"]["scene_exposure_gain"]) / max(
            float(cam.exposure_gain), 1.0e-8
        )
        count = int(selected.numel())
        all_means.append(means.detach().cpu().numpy())
        all_scales.append(scales.detach().cpu().numpy())
        all_colors.append(colors.detach().cpu().numpy())
        all_opacities.append(np.full((count,), args.opacity, dtype=np.float32))
        offsets.append(offsets[-1] + count)
        source_indices.append(local_index)
        source_poses.append(np.asarray(pose, dtype=np.float32))
        source_counts.append(count)
        camera_centers.append(np.linalg.inv(pose)[:3, 3])
        source_times.append(time.perf_counter() - frame_start)

    if not all_means:
        raise RuntimeError("No view-conditioned appearance carriers were selected")
    camera_centers = np.asarray(camera_centers, dtype=np.float32)
    translation_scale = (
        float(np.median(np.linalg.norm(np.diff(camera_centers, axis=0), axis=1)))
        if len(camera_centers) > 1
        else 1.0
    )
    metadata = {
        "version": 1,
        "method": "frequency-faithful view-conditioned surface appearance",
        "exclude_exact_frame": True,
        "translation_scale": max(translation_scale, 1.0e-6),
        "orientation_weight": 2.0,
        "build_seconds": time.perf_counter() - start_time,
        "mean_source_seconds": float(np.mean(source_times)),
        "source_count": len(source_indices),
        "gaussian_count": int(offsets[-1]),
        "max_points_per_source": args.max_points_per_source,
        "grid_px": args.grid_px,
        "gradient_threshold": args.gradient_threshold,
        "frequency_gap_threshold": args.frequency_gap_threshold,
        "residual_threshold": args.residual_threshold,
        "opacity_threshold": args.opacity_threshold,
        "side_start": args.side_start,
        "vertical_start": args.vertical_start,
        "near_depth_m": args.near_depth_m,
        "projected_scale_px": args.projected_scale_px,
        "surface_offset_m": args.surface_offset_m,
        "opacity": args.opacity,
        "source_counts": source_counts,
        "depth_source": (
            "orb_aligned_monocular"
            if args.depth_directory is not None
            else "stable_render"
        ),
        "stable_depth_consistency_ratio": args.stable_depth_consistency_ratio,
        "full_surface": bool(args.full_surface),
        "regular_grid": bool(args.regular_grid),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        means=np.concatenate(all_means).astype(np.float32),
        scales=np.concatenate(all_scales).astype(np.float32),
        colors=np.concatenate(all_colors).astype(np.float32),
        opacities=np.concatenate(all_opacities).astype(np.float32),
        source_offsets=np.asarray(offsets, dtype=np.int64),
        source_frame_indices=np.asarray(source_indices, dtype=np.int64),
        source_poses=np.asarray(source_poses, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with open(output.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))
    print(output)


if __name__ == "__main__":
    main()
