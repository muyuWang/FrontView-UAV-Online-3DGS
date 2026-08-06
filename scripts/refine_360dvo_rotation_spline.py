#!/usr/bin/env python3
"""Refine a low-frequency 360DVO rotation correction with held-out evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.merge_360dvo_orbslam3_pose_windows import c2w, summarize
    from scripts.refine_360dvo_rotation_prior import (
        camera_model,
        load_samples,
        read_json,
        select_pair_paths,
        write_pose_dataset,
    )
except ModuleNotFoundError:
    from merge_360dvo_orbslam3_pose_windows import c2w, summarize
    from refine_360dvo_rotation_prior import (
        camera_model,
        load_samples,
        read_json,
        select_pair_paths,
        write_pose_dataset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--initial-pose", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-pairs", type=int, default=768)
    parser.add_argument("--max-matches-per-pair", type=int, default=256)
    parser.add_argument("--min-match-score", type=float, default=0.15)
    parser.add_argument("--epipolar-threshold-px", type=float, default=1.5)
    parser.add_argument("--knot-spacing", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=0.001)
    parser.add_argument("--min-validation-gain", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def initial_correction(statistics: dict) -> np.ndarray:
    epipolar = statistics.get("epipolar_refinement_certificate")
    if epipolar is not None:
        return np.asarray(epipolar["refined_correction"], dtype=np.float64)
    continuity = statistics.get("continuity_certificate")
    if continuity is not None and "camera_axis_correction" in continuity:
        return np.asarray(continuity["camera_axis_correction"], dtype=np.float64)
    raise RuntimeError("Initial pose does not contain a camera-axis correction")


def matrix_to_6d(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def rotation_6d_to_matrix(values: torch.Tensor) -> torch.Tensor:
    first = torch.nn.functional.normalize(values[..., :3], dim=-1)
    second_raw = values[..., 3:]
    second = torch.nn.functional.normalize(
        second_raw - torch.sum(first * second_raw, dim=-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def interpolate_knots(
    knot_values: torch.Tensor, frame_count: int, spacing: int
) -> torch.Tensor:
    frame = torch.arange(frame_count, device=knot_values.device)
    left = torch.div(frame, spacing, rounding_mode="floor").clamp_max(
        len(knot_values) - 1
    )
    right = (left + 1).clamp_max(len(knot_values) - 1)
    fraction = ((frame % spacing).to(knot_values.dtype) / float(spacing)).unsqueeze(1)
    values = (1.0 - fraction) * knot_values[left] + fraction * knot_values[right]
    return rotation_6d_to_matrix(values)


def pack_samples(samples: list, device: torch.device) -> dict[str, torch.Tensor]:
    frame_i = []
    frame_j = []
    pair_ids = []
    points_i = []
    points_j = []
    for pair_id, (left, right, left_points, right_points) in enumerate(samples):
        frame_i.append(left)
        frame_j.append(right)
        pair_ids.extend([pair_id] * len(left_points))
        points_i.append(left_points)
        points_j.append(right_points)
    return {
        "frame_i": torch.as_tensor(frame_i, dtype=torch.long, device=device),
        "frame_j": torch.as_tensor(frame_j, dtype=torch.long, device=device),
        "pair_ids": torch.as_tensor(pair_ids, dtype=torch.long, device=device),
        "points_i": torch.as_tensor(
            np.concatenate(points_i), dtype=torch.float32, device=device
        ),
        "points_j": torch.as_tensor(
            np.concatenate(points_j), dtype=torch.float32, device=device
        ),
    }


def skew_batch(vectors: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(vectors[:, 0])
    x, y, z = vectors.unbind(dim=1)
    return torch.stack(
        (zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=1
    ).reshape(-1, 3, 3)


def epipolar_errors(
    corrections: torch.Tensor,
    source_rotations: torch.Tensor,
    centers: torch.Tensor,
    k_inverse: torch.Tensor,
    samples: dict[str, torch.Tensor],
) -> torch.Tensor:
    rotations = source_rotations @ corrections
    frame_i = samples["frame_i"]
    frame_j = samples["frame_j"]
    rotation_i = rotations[frame_i]
    rotation_j = rotations[frame_j]
    relative_rotation = rotation_j.transpose(1, 2) @ rotation_i
    relative_translation = (
        rotation_j.transpose(1, 2)
        @ (centers[frame_i] - centers[frame_j]).unsqueeze(2)
    ).squeeze(2)
    essential = skew_batch(relative_translation) @ relative_rotation
    fundamental = k_inverse.T.unsqueeze(0) @ essential @ k_inverse.unsqueeze(0)
    matrix = fundamental[samples["pair_ids"]]

    ones = torch.ones(
        (len(samples["points_i"]), 1),
        dtype=samples["points_i"].dtype,
        device=samples["points_i"].device,
    )
    left = torch.cat((samples["points_i"], ones), dim=1)
    right = torch.cat((samples["points_j"], ones), dim=1)
    f_left = torch.bmm(matrix, left.unsqueeze(2)).squeeze(2)
    ft_right = torch.bmm(matrix.transpose(1, 2), right.unsqueeze(2)).squeeze(2)
    numerator = torch.sum(right * f_left, dim=1).square()
    denominator = (
        f_left[:, 0].square()
        + f_left[:, 1].square()
        + ft_right[:, 0].square()
        + ft_right[:, 1].square()
        + 1.0e-12
    )
    return torch.sqrt(numerator / denominator + 1.0e-12)


def metrics(errors: torch.Tensor, threshold: float) -> dict:
    values = errors.detach().cpu().numpy().astype(np.float64)
    return {
        "match_count": len(values),
        "inlier_count": int(np.sum(values <= threshold)),
        "inlier_fraction": float(np.mean(values <= threshold)),
        "error_px": summarize(values),
    }


def main() -> int:
    args = parse_args()
    if args.knot_spacing <= 0 or args.steps <= 0:
        raise ValueError("knot-spacing and steps must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    source = args.source.resolve()
    initial_pose = args.initial_pose.resolve()
    cache = args.feature_cache.resolve()
    output = args.output.resolve()
    source_cameras = read_json(source / "trajectory_orb.json")["cameras"]
    initial_stats = read_json(initial_pose / "conversion_stats.json")
    correction = initial_correction(initial_stats)
    source_c2w = np.asarray([c2w(camera) for camera in source_cameras])
    source_rotations = torch.as_tensor(
        source_c2w[:, :3, :3], dtype=torch.float32, device=device
    )
    centers = torch.as_tensor(
        source_c2w[:, :3, 3], dtype=torch.float32, device=device
    )
    k_inverse = torch.as_tensor(
        np.linalg.inv(camera_model(source_cameras)), dtype=torch.float32, device=device
    )
    pair_paths = select_pair_paths(cache / "matches", args.max_pairs)
    fit_samples, validation_samples = load_samples(
        cache,
        pair_paths,
        args.min_match_score,
        args.max_matches_per_pair,
    )
    fit = pack_samples(fit_samples, device)
    validation = pack_samples(validation_samples, device)

    knot_count = (len(source_cameras) - 1) // args.knot_spacing + 2
    initial_matrix = torch.as_tensor(correction, dtype=torch.float32, device=device)
    initial_values = matrix_to_6d(initial_matrix).repeat(knot_count, 1)
    knot_values = torch.nn.Parameter(initial_values.clone())
    optimizer = torch.optim.Adam([knot_values], lr=args.learning_rate)

    with torch.no_grad():
        initial_frames = interpolate_knots(
            initial_values, len(source_cameras), args.knot_spacing
        )
        initial_fit = metrics(
            epipolar_errors(
                initial_frames, source_rotations, centers, k_inverse, fit
            ),
            args.epipolar_threshold_px,
        )
        initial_validation = metrics(
            epipolar_errors(
                initial_frames, source_rotations, centers, k_inverse, validation
            ),
            args.epipolar_threshold_px,
        )

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        frame_corrections = interpolate_knots(
            knot_values, len(source_cameras), args.knot_spacing
        )
        errors = epipolar_errors(
            frame_corrections, source_rotations, centers, k_inverse, fit
        )
        scaled = errors / args.epipolar_threshold_px
        data_loss = torch.mean(scaled.square() / (1.0 + scaled.square()))
        knot_rotations = rotation_6d_to_matrix(knot_values)
        relative = knot_rotations[:-1].transpose(1, 2) @ knot_rotations[1:]
        cosine = ((torch.diagonal(relative, dim1=1, dim2=2).sum(1) - 1.0) / 2.0).clamp(
            -1.0, 1.0
        )
        smoothness = torch.mean(1.0 - cosine)
        anchor = torch.mean((knot_values - initial_values).square())
        loss = (
            data_loss
            + args.smoothness_weight * smoothness
            + args.anchor_weight * anchor
        )
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            print(
                f"step={step + 1}/{args.steps} loss={loss.item():.6f} "
                f"data={data_loss.item():.6f} smooth={smoothness.item():.6f}",
                flush=True,
            )

    with torch.no_grad():
        refined_frames = interpolate_knots(
            knot_values, len(source_cameras), args.knot_spacing
        )
        refined_fit = metrics(
            epipolar_errors(
                refined_frames, source_rotations, centers, k_inverse, fit
            ),
            args.epipolar_threshold_px,
        )
        refined_validation = metrics(
            epipolar_errors(
                refined_frames, source_rotations, centers, k_inverse, validation
            ),
            args.epipolar_threshold_px,
        )
    validation_gain = (
        refined_validation["inlier_fraction"]
        - initial_validation["inlier_fraction"]
    )
    certificate = {
        "initial_pose": str(initial_pose),
        "feature_cache": str(cache),
        "frame_count": len(source_cameras),
        "knot_count": knot_count,
        "knot_spacing": args.knot_spacing,
        "steps": args.steps,
        "smoothness_weight": args.smoothness_weight,
        "anchor_weight": args.anchor_weight,
        "sampled_pair_count": len(pair_paths),
        "fit_pair_count": len(fit_samples),
        "validation_pair_count": len(validation_samples),
        "initial_fit": initial_fit,
        "refined_fit": refined_fit,
        "initial_validation": initial_validation,
        "refined_validation": refined_validation,
        "validation_inlier_fraction_gain": validation_gain,
        "required_validation_gain": args.min_validation_gain,
        "knot_rotations": rotation_6d_to_matrix(knot_values).detach().cpu().tolist(),
    }
    print(json.dumps(certificate, indent=2), flush=True)
    if validation_gain < args.min_validation_gain:
        raise RuntimeError(
            f"Held-out spline gain {validation_gain:.6f} is below "
            f"{args.min_validation_gain:.6f}"
        )

    statistics = {
        "schema_version": 1,
        "method": "held-out epipolar low-frequency SO(3) rotation correction",
        "pose_source": "gt_centers_epipolar_rotation_spline",
        "sparse_world_geometry": "not_exported_pose_stage",
        "source": str(source),
        "frame_count": len(source_cameras),
        "pose_mode": "epipolar_rotation_spline",
        "epipolar_spline_certificate": certificate,
    }
    rotations = source_c2w[:, :3, :3] @ refined_frames.cpu().numpy()
    write_pose_dataset(source, output, source_cameras, rotations, statistics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
