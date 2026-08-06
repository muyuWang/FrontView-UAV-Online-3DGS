#!/usr/bin/env python3
"""Build RGB-only multiview detail geometry around the stable map depth."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from render import (
    build_camera,
    load_camera_infos,
    load_gaussians,
    load_run_config,
    load_tracked_pose_map,
    load_vignette,
    read_gt_bgr,
)
from scripts.build_view_conditioned_detail import image_gradient, spatial_topk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--depth_directory",
        type=Path,
        default=None,
        help="Optional ORB-aligned monocular depth used only as the sweep center.",
    )
    parser.add_argument("--history_views", type=int, default=2)
    parser.add_argument("--depth_hypotheses", type=int, default=24)
    parser.add_argument("--min_depth_ratio", type=float, default=0.55)
    parser.add_argument("--max_depth_ratio", type=float, default=1.20)
    parser.add_argument("--max_candidates_per_source", type=int, default=2048)
    parser.add_argument("--max_commits_per_source", type=int, default=512)
    parser.add_argument("--max_total_gaussians", type=int, default=16000)
    parser.add_argument("--candidate_grid_px", type=int, default=3)
    parser.add_argument("--patch_radius", type=int, default=1)
    parser.add_argument("--gradient_threshold", type=float, default=0.025)
    parser.add_argument("--residual_threshold", type=float, default=0.04)
    parser.add_argument("--opacity_threshold", type=float, default=0.55)
    parser.add_argument("--side_start", type=float, default=0.30)
    parser.add_argument("--vertical_start", type=float, default=0.18)
    parser.add_argument("--near_depth_m", type=float, default=65.0)
    parser.add_argument("--max_photometric_error", type=float, default=0.13)
    parser.add_argument("--min_baseline_improvement", type=float, default=0.006)
    parser.add_argument("--occlusion_front_ratio", type=float, default=0.50)
    parser.add_argument("--occlusion_back_ratio", type=float, default=1.15)
    parser.add_argument("--projected_scale_px", type=float, default=0.65)
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument("--voxel_size", type=float, default=0.035)
    parser.add_argument(
        "--view_conditioned",
        action="store_true",
        help="Keep accepted geometry separated by its reference view cone.",
    )
    return parser.parse_args()


def pixel_grid(uv, width, height):
    grid = uv.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / float(width) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / float(height) - 1.0
    return grid


def sample_patches(image, uv, radius):
    height, width = image.shape[:2]
    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(-radius, radius + 1, device=image.device),
            torch.arange(-radius, radius + 1, device=image.device),
            indexing="ij",
        ),
        dim=-1,
    )[..., [1, 0]].reshape(-1, 2)
    patch_uv = uv[..., None, :] + offsets.to(uv.dtype)
    leading_shape = patch_uv.shape[:-2]
    patch_size = patch_uv.shape[-2]
    grid = pixel_grid(patch_uv, width, height).reshape(
        1, -1, patch_size, 2
    )
    values = F.grid_sample(
        image.permute(2, 0, 1)[None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return values[0].permute(1, 2, 0).reshape(*leading_shape, patch_size, 3)


def sample_scalar(image, uv):
    height, width = image.shape
    grid = pixel_grid(uv, width, height).reshape(1, -1, 1, 2)
    values = F.grid_sample(
        image[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return values.reshape(uv.shape[:-1])


@torch.no_grad()
def planesweep_depths(
    reference_image,
    reference_depth,
    reference_pose,
    neighbor_images,
    neighbor_depths,
    neighbor_poses,
    intrinsics,
    candidate_uv,
    depth_ratios,
    patch_radius,
    max_photometric_error,
    min_baseline_improvement,
    occlusion_front_ratio,
    occlusion_back_ratio,
):
    height, width = reference_depth.shape
    count = candidate_uv.shape[0]
    hypothesis_count = depth_ratios.numel()
    candidate_x = torch.clamp(candidate_uv[:, 0].long(), 0, width - 1)
    candidate_y = torch.clamp(candidate_uv[:, 1].long(), 0, height - 1)
    initial_depth = reference_depth[candidate_y, candidate_x]
    depths = initial_depth[:, None] * depth_ratios[None, :]
    homogeneous_uv = torch.cat(
        (candidate_uv, torch.ones_like(candidate_uv[:, :1])), dim=-1
    )
    rays = homogeneous_uv @ torch.linalg.inv(intrinsics.T)
    local = rays[:, None, :] * depths[..., None]
    world_h = torch.cat(
        (local, torch.ones((count, hypothesis_count, 1), device=local.device)),
        dim=-1,
    ) @ torch.linalg.inv(reference_pose.T)
    reference_patch = sample_patches(
        reference_image, candidate_uv, patch_radius
    )[:, None]

    error_sum = torch.zeros_like(depths)
    support = torch.zeros_like(depths, dtype=torch.int32)
    for neighbor_image, neighbor_depth, neighbor_pose in zip(
        neighbor_images, neighbor_depths, neighbor_poses
    ):
        camera = world_h @ neighbor_pose.T
        camera_z = camera[..., 2]
        projected = camera[..., :3] @ intrinsics.T
        projected_uv = projected[..., :2] / torch.clamp(
            projected[..., 2:], min=1.0e-6
        )
        margin = float(patch_radius) + 0.5
        in_bounds = (
            (camera_z > 0.0)
            & (projected_uv[..., 0] >= margin)
            & (projected_uv[..., 0] < width - margin)
            & (projected_uv[..., 1] >= margin)
            & (projected_uv[..., 1] < height - margin)
        )
        stable_neighbor_depth = sample_scalar(neighbor_depth, projected_uv)
        visible = (
            in_bounds
            & torch.isfinite(stable_neighbor_depth)
            & (stable_neighbor_depth > 0.0)
            & (camera_z >= occlusion_front_ratio * stable_neighbor_depth)
            & (camera_z <= occlusion_back_ratio * stable_neighbor_depth)
        )
        neighbor_patch = sample_patches(
            neighbor_image, projected_uv, patch_radius
        )
        raw_error = torch.abs(neighbor_patch - reference_patch).mean(dim=(-1, -2))
        error_sum += torch.where(visible, raw_error, torch.zeros_like(raw_error))
        support += visible.to(torch.int32)

    required_support = len(neighbor_images)
    mean_error = error_sum / torch.clamp(support, min=1)
    mean_error = torch.where(
        support >= required_support,
        mean_error,
        torch.full_like(mean_error, float("inf")),
    )
    best_error, best_index = torch.min(mean_error, dim=1)
    baseline_index = int(torch.argmin(torch.abs(depth_ratios - 1.0)).item())
    baseline_error = mean_error[:, baseline_index]
    improvement = baseline_error - best_error
    best_ratio = depth_ratios[best_index]
    valid = (
        torch.isfinite(best_error)
        & torch.isfinite(baseline_error)
        & (best_error <= max_photometric_error)
        & (improvement >= min_baseline_improvement)
        & (torch.abs(best_ratio - 1.0) >= 0.025)
    )
    return depths[torch.arange(count, device=depths.device), best_index], best_error, improvement, valid


def main():
    args = parse_args()
    if args.history_views < 2:
        raise ValueError("history_views must be at least two")
    run_dir = args.run_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else run_dir / "planesweep_detail.npz"
    )
    config = load_run_config(run_dir)
    dataset_config = config.get("Testset", config["Dataset"])
    infos = load_camera_infos(dataset_config)
    tracked_pose_map = load_tracked_pose_map(run_dir)
    with open(run_dir / "tracked_info.json", "r", encoding="utf-8") as handle:
        tracked = json.load(handle)
    tracked_by_name = {cam["name"]: cam for cam in tracked["cameras"]}
    vignette = load_vignette(run_dir, dataset_config)
    gaussians = load_gaussians(run_dir, config, args.device, vignette)

    views = []
    build_start = time.perf_counter()
    for local_index, info in enumerate(infos):
        tracked_info = tracked_by_name.get(info["image"], {})
        if not tracked_info.get("is_key_frame", False):
            continue
        pose = tracked_pose_map.get(
            info["image"], np.asarray(info["T_camera_world"], dtype=np.float32)
        )
        cam = build_camera(pose, dataset_config, args.device, info["image"], local_index)
        gt_bgr = read_gt_bgr(info, dataset_config, vignette)
        if gt_bgr.shape[:2] != (cam.get_height(0), cam.get_width(0)):
            gt_bgr = cv2.resize(
                gt_bgr,
                (cam.get_width(0), cam.get_height(0)),
                interpolation=cv2.INTER_AREA,
            )
        image = torch.from_numpy(
            cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ).to(args.device)
        with torch.no_grad():
            render_pkg = gaussians.render(cam)
        sweep_depth = render_pkg["depth"].squeeze(-1)
        if args.depth_directory is not None:
            depth_path = args.depth_directory / "depth_{:04d}.npz".format(local_index)
            if not depth_path.exists():
                raise FileNotFoundError(depth_path)
            with np.load(depth_path) as depth_payload:
                dense_depth = depth_payload["depth"].astype(np.float32)
            if dense_depth.shape != (cam.get_height(0), cam.get_width(0)):
                dense_depth = cv2.resize(
                    dense_depth,
                    (cam.get_width(0), cam.get_height(0)),
                    interpolation=cv2.INTER_LINEAR,
                )
            sweep_depth = torch.from_numpy(dense_depth).to(args.device)
        views.append(
            {
                "index": local_index,
                "pose": torch.as_tensor(pose, device=args.device),
                "image": image,
                "render": render_pkg["render"],
                "depth": sweep_depth,
                "stable_depth": render_pkg["depth"].squeeze(-1),
                "opacity": render_pkg["opacity"].squeeze(-1),
                "intrinsics": cam.get_int_mat(0),
                "focal": 0.5 * (float(cam.get_fx(0)) + float(cam.get_fy(0))),
            }
        )

    all_means = []
    all_scales = []
    all_colors = []
    all_scores = []
    committed_by_source = []
    accepted_source_indices = []
    accepted_source_poses = []
    accepted_source_offsets = [0]
    ratios = torch.linspace(
        args.min_depth_ratio,
        args.max_depth_ratio,
        args.depth_hypotheses,
        device=args.device,
    )
    if torch.min(torch.abs(ratios - 1.0)) > 1.0e-5:
        ratios = torch.sort(torch.cat((ratios, torch.ones(1, device=args.device)))).values

    for view_index in range(args.history_views, len(views)):
        reference = views[view_index]
        history = views[view_index - args.history_views : view_index]
        image = reference["image"]
        gradient = image_gradient(image)
        render_gradient = image_gradient(reference["render"])
        residual = torch.abs(image - reference["render"]).mean(dim=-1)
        height, width = reference["depth"].shape
        side = torch.linspace(-1.0, 1.0, width, device=args.device).abs()[None, :]
        rows = torch.linspace(0.0, 1.0, height, device=args.device)[:, None]
        candidate = (
            (reference["opacity"] >= args.opacity_threshold)
            & torch.isfinite(reference["depth"])
            & (reference["depth"] > 0.0)
            & (reference["depth"] <= args.near_depth_m)
            & (
                ~torch.isfinite(reference["stable_depth"])
                | (reference["stable_depth"] <= 0.0)
                | (
                    torch.abs(reference["depth"] - reference["stable_depth"])
                    / torch.clamp(reference["depth"], min=1.0e-6)
                    <= 0.50
                )
            )
            & (gradient >= args.gradient_threshold)
            & (residual >= args.residual_threshold)
            & (side >= args.side_start)
            & (rows >= args.vertical_start)
        )
        importance = (
            gradient
            * (1.0 + torch.clamp(gradient - render_gradient, min=0.0))
            * torch.clamp(residual, min=0.01)
            * (1.0 + side)
        ).reshape(-1)
        selected = spatial_topk(
            torch.nonzero(candidate.reshape(-1), as_tuple=False).reshape(-1),
            importance,
            width,
            args.candidate_grid_px,
            args.max_candidates_per_source,
        )
        if selected.numel() == 0:
            committed_by_source.append(0)
            continue
        uv = torch.stack(
            (selected % width + 0.5, selected // width + 0.5), dim=-1
        ).float()
        depth, error, improvement, valid = planesweep_depths(
            reference["image"],
            reference["depth"],
            reference["pose"],
            [item["image"] for item in history],
            [item["depth"] for item in history],
            [item["pose"] for item in history],
            reference["intrinsics"],
            uv,
            ratios,
            args.patch_radius,
            args.max_photometric_error,
            args.min_baseline_improvement,
            args.occlusion_front_ratio,
            args.occlusion_back_ratio,
        )
        valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if valid_indices.numel() > args.max_commits_per_source:
            score = improvement * torch.clamp(gradient.reshape(-1)[selected], min=0.01)
            top = torch.topk(
                score[valid_indices], args.max_commits_per_source, sorted=False
            ).indices
            valid_indices = valid_indices[top]
        uv = uv[valid_indices]
        depth = depth[valid_indices]
        if depth.numel() == 0:
            committed_by_source.append(0)
            continue
        local = torch.cat((uv, torch.ones_like(uv[:, :1])), dim=-1)
        local = local @ torch.linalg.inv(reference["intrinsics"].T) * depth[:, None]
        world = torch.cat((local, torch.ones_like(local[:, :1])), dim=-1)
        world = world @ torch.linalg.inv(reference["pose"].T)
        scale = args.projected_scale_px * depth / reference["focal"]
        all_means.append(world[:, :3].detach().cpu().numpy())
        all_scales.append(scale[:, None].expand(-1, 3).detach().cpu().numpy())
        colors = image.reshape(-1, 3)[selected[valid_indices]]
        all_colors.append(colors.detach().cpu().numpy())
        all_scores.append(improvement[valid_indices].detach().cpu().numpy())
        committed_by_source.append(int(depth.numel()))
        accepted_source_indices.append(int(reference["index"]))
        accepted_source_poses.append(reference["pose"].detach().cpu().numpy())
        accepted_source_offsets.append(accepted_source_offsets[-1] + int(depth.numel()))

    if not all_means:
        raise RuntimeError("Plane sweep did not accept any geometry candidates")
    means = np.concatenate(all_means)
    scales = np.concatenate(all_scales)
    colors = np.concatenate(all_colors)
    scores = np.concatenate(all_scores)
    if args.view_conditioned:
        count = len(means)
        source_offsets = np.asarray(accepted_source_offsets, dtype=np.int64)
        source_frame_indices = np.asarray(accepted_source_indices, dtype=np.int64)
        source_poses = np.asarray(accepted_source_poses, dtype=np.float32)
    else:
        order = np.argsort(-scores, kind="stable")
        voxels = np.floor(means / args.voxel_size).astype(np.int64)
        selected_rows = []
        occupied = set()
        for row in order:
            key = tuple(voxels[row])
            if key in occupied:
                continue
            occupied.add(key)
            selected_rows.append(int(row))
            if len(selected_rows) >= args.max_total_gaussians:
                break
        selected_rows = np.asarray(selected_rows, dtype=np.int64)
        means = means[selected_rows]
        scales = scales[selected_rows]
        colors = colors[selected_rows]
        count = len(selected_rows)
        source_offsets = np.asarray([0, count], dtype=np.int64)
        source_frame_indices = np.asarray([-1], dtype=np.int64)
        source_poses = np.eye(4, dtype=np.float32)[None]
    metadata = {
        "version": 1,
        "method": "RGB-only causal multiview plane-sweep geometry responsibility",
        "exclude_exact_frame": bool(args.view_conditioned),
        "translation_scale": 1.0,
        "orientation_weight": 0.0,
        "build_seconds": time.perf_counter() - build_start,
        "source_count": int(len(source_frame_indices)),
        "gaussian_count": count,
        "history_views": args.history_views,
        "depth_hypotheses": int(ratios.numel()),
        "depth_ratio_range": [args.min_depth_ratio, args.max_depth_ratio],
        "committed_by_source_before_voxel_filter": committed_by_source,
        "max_photometric_error": args.max_photometric_error,
        "min_baseline_improvement": args.min_baseline_improvement,
        "voxel_size": args.voxel_size,
        "view_conditioned": bool(args.view_conditioned),
        "depth_source": (
            "orb_aligned_monocular_sweep_center"
            if args.depth_directory is not None
            else "stable_render_sweep_center"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        means=means.astype(np.float32),
        scales=scales.astype(np.float32),
        colors=colors.astype(np.float32),
        opacities=np.full((count,), args.opacity, dtype=np.float32),
        source_offsets=source_offsets,
        source_frame_indices=source_frame_indices,
        source_poses=source_poses,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with open(output.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))
    print(output)


if __name__ == "__main__":
    main()
