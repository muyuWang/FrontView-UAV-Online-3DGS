"""Cross-frame projective anchor tracking and inverse-depth filtering."""

import math
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

from .geometry import (
    fronto_parallel_quaternion,
    parallax_angle,
    project_world,
    project_world_batch,
    unproject_pixel,
)
from .observation_extractor import ObservationExtractor
from .types import Observation, ProjectiveAnchor


def promotion_thresholds_for_anchor(
    anchor: ProjectiveAnchor, config: Dict[str, object]
) -> Dict[str, float]:
    """Use relaxed evidence thresholds only for P backed by valid near depth."""
    thresholds = {
        "min_observations": int(config["promotion_min_observations"]),
        "min_best_weight": float(config["promotion_min_best_weight"]),
        "max_normalized_entropy": float(config["promotion_max_normalized_entropy"]),
        "max_relative_std": float(config["promotion_max_relative_std"]),
        "min_parallax_deg": float(config["promotion_min_parallax_deg"]),
        "max_match_error": float(config["promotion_max_match_error"]),
    }
    is_reliable_near = (
        anchor.reference_depth_valid
        and anchor.reference_depth_prior > 0.0
        and anchor.reference_depth_prior <= float(config["near_promotion_max_depth_m"])
    )
    if is_reliable_near:
        thresholds.update(
            {
                "min_observations": min(
                    thresholds["min_observations"],
                    int(config["near_promotion_min_observations"]),
                ),
                "min_best_weight": min(
                    thresholds["min_best_weight"],
                    float(config["near_promotion_min_best_weight"]),
                ),
                "max_normalized_entropy": max(
                    thresholds["max_normalized_entropy"],
                    float(config["near_promotion_max_normalized_entropy"]),
                ),
                "max_relative_std": max(
                    thresholds["max_relative_std"],
                    float(config["near_promotion_max_relative_std"]),
                ),
                "min_parallax_deg": min(
                    thresholds["min_parallax_deg"],
                    float(config["near_promotion_min_parallax_deg"]),
                ),
                "max_match_error": max(
                    thresholds["max_match_error"],
                    float(config["near_promotion_max_match_error"]),
                ),
            }
        )
    return thresholds


class ProjectiveAnchorBank:
    """Maintain projective scene tracks that do not own optimizers or hash entries."""

    def __init__(self, config: Dict[str, object]):
        self.config = config
        self.anchors: Dict[int, ProjectiveAnchor] = {}
        self._next_anchor_id = 0

    @staticmethod
    def posterior_statistics(anchor: ProjectiveAnchor) -> Tuple[float, float, float]:
        weights = torch.softmax(anchor.mode_log_weights, dim=0)
        mean = torch.sum(weights * anchor.inverse_depth_modes)
        variance = torch.sum(weights * (anchor.inverse_depth_modes - mean).square())
        entropy = -(
            weights * torch.log(torch.clamp(weights, min=1.0e-12))
        ).sum() / math.log(max(2, weights.numel()))
        values = torch.stack((mean, variance, entropy)).detach().cpu().tolist()
        return float(values[0]), float(values[1]), float(values[2])

    def initialize_inverse_depth_modes(
        self,
        depth: float,
        near: float,
        far: float,
        device: torch.device,
        offsets: Optional[Sequence[float]] = None,
    ) -> torch.Tensor:
        if depth <= 0.0 or near <= 0.0 or far <= near:
            raise ValueError("depth and camera near/far must define a positive range")
        offsets = torch.tensor(
            self.config["inverse_depth_log_offsets"] if offsets is None else offsets,
            device=device,
            dtype=torch.float32,
        )
        center = 1.0 / max(depth, 1.0e-8)
        modes = center * torch.exp(offsets)
        return torch.clamp(modes, min=1.0 / far, max=1.0 / near)

    def create_anchor(
        self,
        observation: Observation,
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        near: float,
        far: float,
        sparse_depth: Optional[torch.Tensor] = None,
    ) -> ProjectiveAnchor:
        use_near_prior = (
            observation.depth_valid
            and observation.depth_prior <= float(self.config["near_promotion_max_depth_m"])
        )
        modes = self.initialize_inverse_depth_modes(
            observation.depth_prior,
            near,
            far,
            observation.uv.device,
            self.config["near_inverse_depth_log_offsets"] if use_near_prior else None,
        ).to(dtype=observation.uv.dtype)
        log_weights = torch.full_like(modes, -math.log(modes.numel()))
        surface_normal, surface_confidence, surface_support = self.fit_sparse_plane(
            observation, sparse_depth, intrinsics, world_to_camera
        )
        anchor = ProjectiveAnchor(
            anchor_id=self._next_anchor_id,
            reference_frame_id=observation.frame_id,
            reference_pose=world_to_camera.detach().clone(),
            reference_intrinsics=intrinsics.detach().clone(),
            uv=observation.uv.detach().clone(),
            patch_size_px=torch.tensor(
                [self.config["patch_size"], self.config["patch_size"]],
                device=observation.uv.device,
                dtype=observation.uv.dtype,
            ),
            descriptor=observation.descriptor.detach().clone(),
            mean_color=observation.mean_color.detach().clone(),
            appearance_grid=observation.appearance_grid.detach().clone(),
            inverse_depth_modes=modes.detach(),
            mode_log_weights=log_weights.detach(),
            reference_depth_prior=float(observation.depth_prior),
            reference_depth_valid=bool(observation.depth_valid),
            reference_depth_uncertainty=float(observation.depth_uncertainty),
            reference_surface_normal=surface_normal,
            reference_surface_confidence=surface_confidence,
            reference_surface_support=surface_support,
            last_seen_frame=observation.frame_id,
            static_confidence=0.0,
            best_error_ema=1.0,
        )
        anchor.posterior_mean, anchor.posterior_variance, anchor.posterior_entropy = self.posterior_statistics(anchor)
        self.anchors[anchor.anchor_id] = anchor
        self._next_anchor_id += 1
        return anchor

    @torch.no_grad()
    def fit_sparse_plane(
        self,
        observation: Observation,
        sparse_depth: Optional[torch.Tensor],
        intrinsics: torch.Tensor,
        world_to_camera: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], float, int]:
        """Fit a guarded local plane to sparse depth inside one observation patch."""
        if (
            not bool(self.config["enable_sparse_plane_initialization"])
            or sparse_depth is None
            or not observation.depth_valid
        ):
            return None, 0.0, 0
        x0, y0, x1, y1 = [
            int(value) for value in observation.patch_bbox.detach().cpu().tolist()
        ]
        patch = sparse_depth[y0:y1, x0:x1]
        local_y, local_x = torch.nonzero(
            torch.isfinite(patch) & (patch > 0), as_tuple=True
        )
        min_points = int(self.config["sparse_plane_min_points"])
        if local_x.numel() < min_points:
            return None, 0.0, int(local_x.numel())

        depth = patch[local_y, local_x]
        median = depth.median()
        mad = torch.median(torch.abs(depth - median))
        depth_band = torch.maximum(
            float(self.config["sparse_plane_depth_mad_scale"]) * mad,
            float(self.config["sparse_plane_min_relative_depth_band"]) * median,
        )
        keep = torch.abs(depth - median) <= depth_band
        local_x = local_x[keep]
        local_y = local_y[keep]
        depth = depth[keep]
        support = int(depth.numel())
        if support < min_points:
            return None, 0.0, support

        pixel_x = local_x.to(depth.dtype) + float(x0) + 0.5
        pixel_y = local_y.to(depth.dtype) + float(y0) + 0.5
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        camera_points = torch.stack(
            (
                (pixel_x - cx) * depth / torch.clamp(fx, min=1.0e-8),
                (pixel_y - cy) * depth / torch.clamp(fy, min=1.0e-8),
                depth,
            ),
            dim=1,
        )
        centered = camera_points - camera_points.mean(dim=0)
        covariance = centered.T @ centered / max(1, support - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        normal_camera = eigenvectors[:, 0]
        normal_camera = torch.where(
            normal_camera[2] < 0, -normal_camera, normal_camera
        )
        plane_rmse = torch.sqrt(torch.mean((centered @ normal_camera).square()))
        confidence = 1.0 - eigenvalues[0] / torch.clamp(
            eigenvalues[1], min=1.0e-12
        )

        pixel_points = torch.stack((pixel_x, pixel_y), dim=1)
        centered_pixels = pixel_points - pixel_points.mean(dim=0)
        pixel_covariance = centered_pixels.T @ centered_pixels / max(1, support - 1)
        min_uv_variance = torch.linalg.eigvalsh(pixel_covariance)[0]
        relative_rmse = plane_rmse / torch.clamp(median, min=1.0e-8)
        tilt_deg = torch.rad2deg(
            torch.acos(torch.clamp(normal_camera[2], -1.0, 1.0))
        )
        quality = torch.stack(
            (relative_rmse, confidence, min_uv_variance, tilt_deg)
        ).detach().cpu().tolist()
        if (
            quality[0] > float(self.config["sparse_plane_max_relative_rmse"])
            or quality[1] < float(self.config["sparse_plane_min_confidence"])
            or quality[2] < float(self.config["sparse_plane_min_uv_variance"])
            or quality[3] > float(self.config["sparse_plane_max_tilt_deg"])
        ):
            return None, float(quality[1]), support

        camera_to_world = torch.linalg.inv(world_to_camera)
        normal_world = camera_to_world[:3, :3] @ normal_camera
        normal_world = normal_world / torch.clamp(
            torch.linalg.norm(normal_world), min=1.0e-8
        )
        return normal_world.detach(), float(quality[1]), support

    @staticmethod
    def _world_points(anchor: ProjectiveAnchor) -> List[torch.Tensor]:
        return [
            unproject_pixel(
                anchor.uv,
                1.0 / torch.clamp(rho, min=1.0e-8),
                anchor.reference_intrinsics,
                anchor.reference_pose,
            )
            for rho in anchor.inverse_depth_modes
        ]

    def _project_modes(
        self,
        anchor: ProjectiveAnchor,
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: Tuple[int, int],
        near: float,
        far: float,
    ) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
        projected = []
        for mode_index, point in enumerate(self._world_points(anchor)):
            uv, _, valid = project_world(point, world_to_camera, intrinsics, image_size, near, far)
            if valid:
                projected.append((mode_index, uv, point))
        return projected

    def _project_all_modes(
        self,
        anchors: Sequence[ProjectiveAnchor],
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: Tuple[int, int],
        near: float,
        far: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unproject and project all P depth modes as two batched transforms."""
        if not anchors:
            mode_count = int(self.config["num_inverse_depth_modes"])
            return (
                world_to_camera.new_empty((0, mode_count, 3)),
                world_to_camera.new_empty((0, mode_count, 2)),
                torch.empty(
                    (0, mode_count), dtype=torch.bool, device=world_to_camera.device
                ),
            )
        device = world_to_camera.device
        dtype = world_to_camera.dtype
        reference_uv = torch.stack(
            [anchor.uv.to(device=device, dtype=dtype) for anchor in anchors]
        )
        reference_intrinsics = torch.stack(
            [
                anchor.reference_intrinsics.to(device=device, dtype=dtype)
                for anchor in anchors
            ]
        )
        reference_poses = torch.stack(
            [anchor.reference_pose.to(device=device, dtype=dtype) for anchor in anchors]
        )
        inverse_depth_modes = torch.stack(
            [
                anchor.inverse_depth_modes.to(device=device, dtype=dtype)
                for anchor in anchors
            ]
        )
        uv1 = torch.cat((reference_uv, torch.ones_like(reference_uv[:, :1])), dim=1)
        rays = torch.linalg.solve(reference_intrinsics, uv1.unsqueeze(-1)).squeeze(-1)
        depths = 1.0 / torch.clamp(inverse_depth_modes, min=1.0e-8)
        reference_points = rays[:, None, :] * depths[:, :, None]
        reference_homogeneous = torch.cat(
            (reference_points, torch.ones_like(reference_points[:, :, :1])), dim=2
        )
        camera_to_world = torch.linalg.inv(reference_poses)
        world_points = torch.einsum(
            "aij,amj->ami", camera_to_world, reference_homogeneous
        )[:, :, :3]
        projected_uv, _, valid = project_world_batch(
            world_points.reshape(-1, 3),
            world_to_camera,
            intrinsics,
            image_size,
            near,
            far,
        )
        mode_count = inverse_depth_modes.shape[1]
        return (
            world_points,
            projected_uv.reshape(len(anchors), mode_count, 2),
            valid.reshape(len(anchors), mode_count),
        )

    def associate(
        self,
        observations: Sequence[Observation],
        image: torch.Tensor,
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        near: float,
        far: float,
        extractor: ObservationExtractor,
    ) -> Tuple[Dict[int, int], Set[int]]:
        """Greedily match observations through a projected 2D grid, then update anchors."""
        if not observations or not self.anchors:
            return {}, set()
        height, width = image.shape[:2]
        radius = float(self.config["association_radius_px"])
        cell_size = max(1, int(math.ceil(radius)))
        bins: DefaultDict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        anchors = list(self.anchors.values())
        anchor_indices_by_id = {
            anchor.anchor_id: index for index, anchor in enumerate(anchors)
        }
        world_points, projected_uv, valid_modes = self._project_all_modes(
            anchors, world_to_camera, intrinsics, (height, width), near, far
        )
        valid_mode_indices = torch.nonzero(valid_modes, as_tuple=False)
        projected_uv_cpu = projected_uv[
            valid_mode_indices[:, 0], valid_mode_indices[:, 1]
        ].detach().cpu().tolist()
        for (anchor_index, mode_index), uv in zip(
            valid_mode_indices.cpu().tolist(), projected_uv_cpu
        ):
            bins[(int(uv[0]) // cell_size, int(uv[1]) // cell_size)].append(
                (anchor_index, mode_index)
            )

        feature_threshold = float(self.config["association_feature_threshold"])
        pixel_weight = float(self.config["association_pixel_weight"])
        observation_uv = torch.stack([observation.uv for observation in observations])
        pair_observations = []
        pair_anchors = []
        pair_modes = []
        for observation_index, uv in enumerate(observation_uv.detach().cpu().tolist()):
            bx = int(uv[0]) // cell_size
            by = int(uv[1]) // cell_size
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for anchor_index, mode_index in bins.get((bx + dx, by + dy), []):
                        pair_observations.append(observation_index)
                        pair_anchors.append(anchor_index)
                        pair_modes.append(mode_index)

        candidates = []
        if pair_observations:
            device = observation_uv.device
            observation_indices = torch.tensor(
                pair_observations, device=device, dtype=torch.long
            )
            anchor_indices = torch.tensor(pair_anchors, device=device, dtype=torch.long)
            mode_indices = torch.tensor(pair_modes, device=device, dtype=torch.long)
            pixel_distances = torch.linalg.norm(
                observation_uv[observation_indices]
                - projected_uv[anchor_indices, mode_indices],
                dim=1,
            )
            observation_descriptors = torch.stack(
                [observation.descriptor for observation in observations]
            )
            anchor_descriptors = torch.stack(
                [
                    anchor.descriptor.to(
                        device=observation_descriptors.device,
                        dtype=observation_descriptors.dtype,
                    )
                    for anchor in anchors
                ]
            )
            feature_errors = 1.0 - F.cosine_similarity(
                observation_descriptors[observation_indices],
                anchor_descriptors[anchor_indices],
                dim=1,
            )
            pair_valid = (pixel_distances < radius) & (
                feature_errors < feature_threshold
            )
            pair_scores = feature_errors + (
                pixel_weight * pixel_distances / max(radius, 1.0)
            )
            valid_pair_indices = torch.nonzero(pair_valid, as_tuple=False).flatten()
            per_observation_anchor: Dict[Tuple[int, int], Tuple[float, int]] = {}
            for score, observation_index, anchor_index, mode_index in zip(
                pair_scores[valid_pair_indices].detach().cpu().tolist(),
                observation_indices[valid_pair_indices].cpu().tolist(),
                anchor_indices[valid_pair_indices].cpu().tolist(),
                mode_indices[valid_pair_indices].cpu().tolist(),
            ):
                key = (observation_index, anchor_index)
                if key not in per_observation_anchor or score < per_observation_anchor[key][0]:
                    per_observation_anchor[key] = (score, mode_index)
            candidates = [
                (score, observation_index, anchors[anchor_index].anchor_id, mode_index)
                for (observation_index, anchor_index), (
                    score,
                    mode_index,
                ) in per_observation_anchor.items()
            ]

        matches: Dict[int, int] = {}
        used_observations: Set[int] = set()
        used_anchors: Set[int] = set()
        for _, observation_index, anchor_id, _ in sorted(candidates):
            if observation_index in used_observations or anchor_id in used_anchors:
                continue
            matches[observation_index] = anchor_id
            used_observations.add(observation_index)
            used_anchors.add(anchor_id)
            anchor_index = anchor_indices_by_id[anchor_id]
            projected_modes = [
                (
                    mode_index,
                    projected_uv[anchor_index, mode_index],
                    world_points[anchor_index, mode_index],
                )
                for mode_index, is_valid in enumerate(
                    valid_modes[anchor_index].detach().cpu().tolist()
                )
                if is_valid
            ]
            self.update_anchor(
                self.anchors[anchor_id], observations[observation_index], image, world_to_camera,
                intrinsics, near, far, extractor, projected_modes
            )
        return matches, used_observations

    def update_anchor(
        self,
        anchor: ProjectiveAnchor,
        observation: Observation,
        image: torch.Tensor,
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        near: float,
        far: float,
        extractor: ObservationExtractor,
        projected_modes: Optional[List[Tuple[int, torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        """Apply a causal appearance likelihood update to all inverse-depth modes."""
        if projected_modes is None:
            projected_modes = self._project_modes(
                anchor, world_to_camera, intrinsics, image.shape[:2], near, far
            )
        mode_errors = torch.full_like(anchor.mode_log_weights, 2.0)
        points_by_mode: Dict[int, torch.Tensor] = {}
        photo_weight = float(self.config["association_photo_weight"])
        if projected_modes:
            projected_uv = torch.stack([item[1] for item in projected_modes])
            descriptors, mean_colors = extractor.describe_patches(image, projected_uv)
            feature_errors = 1.0 - F.cosine_similarity(
                anchor.descriptor.reshape(1, -1).expand_as(descriptors),
                descriptors,
                dim=1,
            )
            photo_errors = torch.mean(
                torch.abs(anchor.mean_color.reshape(1, 3) - mean_colors), dim=1
            )
            projected_errors = feature_errors + photo_weight * photo_errors
            for projected_index, (mode_index, _, world_point) in enumerate(
                projected_modes
            ):
                mode_errors[mode_index] = projected_errors[projected_index]
                points_by_mode[mode_index] = world_point

        temperature = max(1.0e-6, float(self.config["association_temperature"]))
        probabilities = torch.softmax(anchor.mode_log_weights - mode_errors / temperature, dim=0)
        floor = float(self.config["minimum_probability_floor"])
        if floor * probabilities.numel() >= 1.0:
            raise ValueError("minimum_probability_floor is too large for the mode count")
        probabilities = (1.0 - floor * probabilities.numel()) * probabilities + floor
        anchor.mode_log_weights = torch.log(probabilities).detach()
        best_index = int(probabilities.argmax().item())
        best_error = float(mode_errors[best_index].item())
        ema = float(self.config["residual_ema"])
        anchor.best_error_ema = best_error if anchor.valid_update_count == 0 else (
            (1.0 - ema) * anchor.best_error_ema + ema * best_error
        )
        if best_error < float(self.config["association_feature_threshold"]):
            descriptor_ema = float(self.config["descriptor_ema"])
            updated_descriptor = (1.0 - descriptor_ema) * anchor.descriptor + descriptor_ema * observation.descriptor
            anchor.descriptor = updated_descriptor / torch.clamp(
                torch.linalg.norm(updated_descriptor), min=1.0e-8
            )
        color_ema = float(self.config["color_ema"])
        anchor.mean_color = (1.0 - color_ema) * anchor.mean_color + color_ema * observation.mean_color
        anchor.appearance_grid = (
            (1.0 - color_ema) * anchor.appearance_grid
            + color_ema * observation.appearance_grid
        )
        anchor.observation_count += 1
        anchor.valid_update_count += int(best_index in points_by_mode)
        anchor.last_seen_frame = observation.frame_id
        anchor.static_confidence = max(0.0, min(1.0, 1.0 - anchor.best_error_ema))
        if best_index in points_by_mode:
            anchor.max_parallax_rad = max(
                anchor.max_parallax_rad,
                parallax_angle(points_by_mode[best_index], anchor.reference_pose, world_to_camera),
            )
        anchor.posterior_mean, anchor.posterior_variance, anchor.posterior_entropy = self.posterior_statistics(anchor)

    def is_promotion_candidate(
        self, anchor: ProjectiveAnchor, best_weight: Optional[float] = None
    ) -> bool:
        if best_weight is None:
            weights = torch.softmax(anchor.mode_log_weights, dim=0)
            best_weight = float(weights.max().item())
        relative_std = math.sqrt(max(0.0, anchor.posterior_variance)) / max(anchor.posterior_mean, 1.0e-8)
        thresholds = promotion_thresholds_for_anchor(anchor, self.config)
        threshold_gate = bool(
            anchor.observation_count >= thresholds["min_observations"]
            and best_weight >= thresholds["min_best_weight"]
            and anchor.posterior_entropy <= thresholds["max_normalized_entropy"]
            and relative_std <= thresholds["max_relative_std"]
            and anchor.max_parallax_rad >= math.radians(thresholds["min_parallax_deg"])
            and anchor.best_error_ema <= thresholds["max_match_error"]
        )
        if not threshold_gate or not bool(self.config["commitment_score_enabled"]):
            return threshold_gate
        return bool(
            anchor.valid_update_count
            >= int(self.config["commitment_min_valid_updates"])
            and self.commitment_score(anchor, best_weight, relative_std)
            >= float(self.config["commitment_score_threshold"])
        )

    def commitment_score(
        self,
        anchor: ProjectiveAnchor,
        best_weight: Optional[float] = None,
        relative_std: Optional[float] = None,
    ) -> float:
        """Return a cheap calibrated observability proxy for admission.

        The score uses evidence already maintained by the P track. It does not
        profile pose covariance or claim to bound local recovery error.
        """
        if best_weight is None:
            best_weight = float(
                torch.softmax(anchor.mode_log_weights, dim=0).max().item()
            )
        if relative_std is None:
            relative_std = math.sqrt(
                max(0.0, anchor.posterior_variance)
            ) / max(anchor.posterior_mean, 1.0e-8)
        parallax_reference = math.radians(
            float(self.config["commitment_parallax_reference_deg"])
        )
        parallax_ratio = math.sin(anchor.max_parallax_rad) / max(
            math.sin(parallax_reference), 1.0e-8
        )
        relative_std_ratio = relative_std / float(
            self.config["commitment_relative_std_reference"]
        )
        match_error_ratio = max(0.0, anchor.best_error_ema) / float(
            self.config["commitment_match_error_reference"]
        )
        nuisance = (
            1.0
            + relative_std_ratio * relative_std_ratio
            + match_error_ratio * match_error_ratio
            + anchor.posterior_entropy * anchor.posterior_entropy
        )
        anchor.commitment_score = float(
            max(0, anchor.valid_update_count)
            * max(0.0, best_weight)
            * parallax_ratio
            * parallax_ratio
            / max(nuisance, 1.0e-8)
        )
        return anchor.commitment_score

    def promotion_candidates(self) -> List[ProjectiveAnchor]:
        anchors = list(self.anchors.values())
        if not anchors:
            return []
        log_weights = torch.stack([anchor.mode_log_weights for anchor in anchors])
        best_weights = torch.softmax(log_weights, dim=1).amax(dim=1).detach().cpu().tolist()
        return [
            anchor
            for anchor, best_weight in zip(anchors, best_weights)
            if self.is_promotion_candidate(anchor, best_weight)
        ]

    def remove(self, anchor_id: int) -> None:
        self.anchors.pop(anchor_id, None)

    def collapse_to_best_mode(self, anchor_ids: Optional[Iterable[int]] = None) -> None:
        """Retain one effective hypothesis while preserving the fixed four-mode tensor shape."""
        ids = self.anchors.keys() if anchor_ids is None else anchor_ids
        for anchor_id in list(ids):
            anchor = self.anchors.get(anchor_id)
            if anchor is None:
                continue
            best = int(anchor.mode_log_weights.argmax().item())
            log_weights = torch.full_like(anchor.mode_log_weights, -30.0)
            log_weights[best] = 0.0
            anchor.mode_log_weights = torch.log_softmax(log_weights, dim=0)
            anchor.posterior_mean, anchor.posterior_variance, anchor.posterior_entropy = self.posterior_statistics(anchor)

    def build_proxy_splats(
        self, device: torch.device, dtype: torch.dtype, max_sh_degree: int = 0
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Generate detached renderer-ready 2DGS proxies for the most likely modes."""
        means: List[torch.Tensor] = []
        scales: List[torch.Tensor] = []
        quats: List[torch.Tensor] = []
        opacities: List[torch.Tensor] = []
        colors: List[torch.Tensor] = []
        top_k = int(self.config["projective_top_k_render_modes"])
        base_opacity = float(self.config["projective_base_opacity"])
        sh_count = (max_sh_degree + 1) ** 2
        for anchor in self.anchors.values():
            weights = torch.softmax(anchor.mode_log_weights, dim=0)
            for mode_index in torch.topk(weights, k=min(top_k, weights.numel())).indices.tolist():
                rho = anchor.inverse_depth_modes[mode_index]
                depth = 1.0 / torch.clamp(rho, min=1.0e-8)
                point = unproject_pixel(anchor.uv, depth, anchor.reference_intrinsics, anchor.reference_pose)
                fx = anchor.reference_intrinsics[0, 0]
                fy = anchor.reference_intrinsics[1, 1]
                sx = depth * anchor.patch_size_px[0] / torch.clamp(fx, min=1.0e-8)
                sy = depth * anchor.patch_size_px[1] / torch.clamp(fy, min=1.0e-8)
                sz = torch.clamp(torch.minimum(sx, sy) * 0.01, min=1.0e-6)
                means.append(point.to(device=device, dtype=dtype))
                scales.append(torch.stack((sx, sy, sz)).to(device=device, dtype=dtype))
                quats.append(fronto_parallel_quaternion(anchor.reference_pose).to(device=device, dtype=dtype))
                opacities.append((base_opacity * weights[mode_index]).to(device=device, dtype=dtype))
                sh = torch.zeros((sh_count, 3), device=device, dtype=dtype)
                sh[0] = (anchor.mean_color.to(device=device, dtype=dtype) - 0.5) / 0.28209479177387814
                colors.append(sh)
        if not means:
            return None
        return {
            "means": torch.stack(means).detach(),
            "scales": torch.stack(scales).detach(),
            "quats": torch.stack(quats).detach(),
            "opacities": torch.stack(opacities).detach(),
            "shs": torch.stack(colors).detach(),
        }
