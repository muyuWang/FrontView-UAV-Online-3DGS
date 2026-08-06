"""Causal per-frame state machine for P -> M -> S -> A mapping."""

import math
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

from .archive_store import ArchiveStore
from .budget_manager import BudgetManager
from .config import validate_progressive_config
from .debug_utils import ProgressiveDebugWriter
from .gaussian_tree_registry import GaussianTreeRegistry
from .geometry import (
    fronto_parallel_quaternion,
    project_world,
    project_world_batch,
    quaternion_from_normal,
    quaternion_to_matrix,
    unproject_pixel,
)
from .observation_extractor import ObservationExtractor
from .progressive_gaussian_store import ProgressiveGaussianStore, concatenate_raw_params
from .projective_anchor_bank import ProjectiveAnchorBank
from .types import GaussianTreeNode, NodeState, Observation, ProgressiveFrameStats, ProjectiveAnchor


def _inverse_sigmoid(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, 1.0e-6, 1.0 - 1.0e-6)
    return torch.log(value / (1.0 - value))


def _merge_external_splats(
    first: Optional[Dict[str, torch.Tensor]], second: Optional[Dict[str, torch.Tensor]]
) -> Optional[Dict[str, torch.Tensor]]:
    if first is None:
        return second
    if second is None:
        return first
    return {key: torch.cat((first[key], second[key]), dim=0).detach() for key in first}


class ProgressiveManager:
    """Run strictly causal observation association and hierarchy transitions."""

    def __init__(self, config: Dict[str, object], gaussian_model=None, output_dir: Optional[str] = None):
        self.config = validate_progressive_config(config)
        if (
            self.config["enabled"]
            and gaussian_model is not None
            and gaussian_model.gaussian_type not in ("2dgs", "3dgs")
        ):
            raise ValueError(
                "ProgressiveMapping requires Model.gaussian_type=2dgs or 3dgs"
            )
        self.gaussian_model = gaussian_model
        self.max_sh_degree = 0 if gaussian_model is None else int(gaussian_model.max_sh_degree)
        self.extractor = ObservationExtractor(self.config)
        self.anchor_bank = ProjectiveAnchorBank(self.config)
        self.registry = GaussianTreeRegistry()
        self.store = ProgressiveGaussianStore(gaussian_model)
        archive_dir = None if output_dir is None else os.path.join(output_dir, "progressive_archive")
        self.archive_store = ArchiveStore(archive_dir)
        self.budget = BudgetManager(self.config)
        self.debug_writer = ProgressiveDebugWriter(
            output_dir, bool(self.config["debug"]), int(self.config["debug_save_interval"])
        )
        self.last_candidate_mask: Optional[torch.Tensor] = None
        self.last_stable_render: Optional[Dict[str, torch.Tensor]] = None
        self.last_scale_clamp_count = 0
        self.last_opacity_clamp_count = 0
        self.last_optimization_visibility = {"visible": 0, "enabled": 0, "frozen": 0}
        self.optimization_enabled_root_ids: Set[int] = set()
        self.last_metric_depth_corrections = 0
        self.last_metric_depth_correction_rejections = 0
        self._eligible_keyframe_count = 0
        self._last_process_decision_frame: Optional[int] = None
        self._last_process_decision = False

    @property
    def enabled(self) -> bool:
        return bool(self.config["enabled"])

    def should_use_baseline_densification(self, processed_frames: int) -> bool:
        if not self.enabled:
            return True
        if not bool(self.config["replace_original_densification_after_bootstrap"]):
            return True
        return processed_frames <= int(self.config["bootstrap_frames"])

    def should_process_progressive_frame(
        self, processed_frames: int, is_keyframe: bool = False
    ) -> bool:
        """Start P/M/S updates after bootstrap, including in hybrid baseline mode."""
        if not self.enabled or processed_frames <= int(self.config["bootstrap_frames"]):
            return False

        interval = int(self.config["process_frame_interval"])
        if not bool(self.config["process_keyframes_only"]):
            first_eligible = int(self.config["bootstrap_frames"]) + 1
            return (processed_frames - first_eligible) % interval == 0
        if not is_keyframe:
            return False

        # Count eligible keyframes, rather than using the absolute frame id. This
        # avoids aliasing when a tracker emits keyframes with a periodic cadence.
        if self._last_process_decision_frame != processed_frames:
            self._eligible_keyframe_count += 1
            self._last_process_decision_frame = processed_frames
            self._last_process_decision = (
                (self._eligible_keyframe_count - 1) % interval == 0
            )
        return self._last_process_decision

    @torch.no_grad()
    def configure_post_refinement_optimization(self) -> int:
        """Prepare the incremental layer for final joint or baseline refinement."""
        if self.gaussian_model is None:
            return 0
        if bool(self.config["post_refinement_merge_into_baseline"]):
            merged_groups, merged_rows = self.store.merge_active_into_baseline()
            self.optimization_enabled_root_ids.clear()
            if merged_groups:
                print(
                    "ProgressiveMapping: merged {} groups / {} rows into the baseline map".format(
                        merged_groups, merged_rows
                    )
                )
            return merged_groups
        if bool(self.config["post_refinement_optimize_progressive"]):
            return 0
        frozen = 0
        for root_id in list(self.store.group_ids):
            frozen += int(
                self.gaussian_model.remove_optimization(
                    self.store.group_ids[root_id]
                )
            )
        self.optimization_enabled_root_ids.clear()
        return frozen

    def _raw_params(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        count = means.shape[0]
        sh0 = ((colors - 0.5) / 0.28209479177387814).reshape(count, 1, 3)
        shn_count = (self.max_sh_degree + 1) ** 2 - 1
        return {
            "means": means,
            "scales": torch.log(torch.clamp(scales, min=1.0e-8)),
            "quats": quats,
            "opacities": _inverse_sigmoid(opacities),
            "sh0": sh0,
            "shN": torch.zeros((count, shn_count, 3), device=means.device, dtype=means.dtype),
        }

    def _anchor_geometry(self, anchor: ProjectiveAnchor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rho = torch.tensor(anchor.posterior_mean, device=anchor.uv.device, dtype=anchor.uv.dtype)
        depth = 1.0 / torch.clamp(rho, min=1.0e-8)
        center = unproject_pixel(anchor.uv, depth, anchor.reference_intrinsics, anchor.reference_pose)
        fx = anchor.reference_intrinsics[0, 0]
        fy = anchor.reference_intrinsics[1, 1]
        scale_factor = float(self.config["metric_scale_factor"])
        sx = depth * anchor.patch_size_px[0] / torch.clamp(fx, min=1.0e-8) * scale_factor
        sy = depth * anchor.patch_size_px[1] / torch.clamp(fy, min=1.0e-8) * scale_factor
        fronto_sx = sx
        fronto_sy = sy
        if anchor.reference_surface_normal is not None:
            camera_to_world = torch.linalg.inv(anchor.reference_pose)
            normal = anchor.reference_surface_normal.to(
                device=center.device, dtype=center.dtype
            )
            rotation = quaternion_from_normal(normal, camera_to_world[:3, 0])
            rotation_matrix = quaternion_to_matrix(rotation)
            normal_camera = anchor.reference_pose[:3, :3] @ normal
            center_camera = (
                anchor.reference_pose
                @ torch.cat(
                    (center, torch.ones(1, device=center.device, dtype=center.dtype))
                )
            )[:3]
            half_patch = 0.5 * anchor.patch_size_px
            corner_offsets = torch.tensor(
                [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
                device=center.device,
                dtype=center.dtype,
            ) * half_patch.reshape(1, 2)
            corner_uv = anchor.uv.reshape(1, 2) + corner_offsets
            corner_uv1 = torch.cat(
                (corner_uv, torch.ones_like(corner_uv[:, :1])), dim=1
            )
            corner_rays = torch.linalg.solve(
                anchor.reference_intrinsics, corner_uv1.T
            ).T
            numerator = torch.dot(normal_camera, center_camera)
            corner_depths = numerator / torch.clamp(
                corner_rays @ normal_camera, min=1.0e-4
            )
            corner_camera = corner_rays * corner_depths[:, None]
            corner_world = (
                camera_to_world
                @ torch.cat(
                    (corner_camera, torch.ones_like(corner_camera[:, :1])), dim=1
                ).T
            ).T[:, :3]
            local_corners = (corner_world - center.reshape(1, 3)) @ rotation_matrix
            plane_sx = (
                local_corners[:, 0].amax() - local_corners[:, 0].amin()
            ) * scale_factor
            plane_sy = (
                local_corners[:, 1].amax() - local_corners[:, 1].amin()
            ) * scale_factor
            max_scale_multiplier = float(
                self.config["sparse_plane_max_scale_multiplier"]
            )
            sx = torch.minimum(plane_sx, fronto_sx * max_scale_multiplier)
            sy = torch.minimum(plane_sy, fronto_sy * max_scale_multiplier)
        else:
            rotation = fronto_parallel_quaternion(anchor.reference_pose)
        sz = torch.clamp(torch.minimum(sx, sy) * 0.01, min=1.0e-6)
        scale = torch.stack((sx, sy, sz))
        return center, scale, rotation

    def promote_anchor(self, anchor_id: int, frame_id: int) -> Tuple[int, bool]:
        """Promote one P anchor, performing metric-space and appearance deduplication first."""
        anchor = self.anchor_bank.anchors[anchor_id]
        center, scale, rotation = self._anchor_geometry(anchor)
        existing = self.registry.find_merge_candidate(
            center,
            scale,
            anchor.descriptor,
            float(self.config["metric_merge_radius_factor"]),
            float(self.config["metric_merge_feature_threshold"]),
        )
        if existing is not None:
            self.registry.merge_anchor_support(existing, anchor, center, scale)
            updated = self.store._sync(
                existing.node_id, self.store.active_metric[existing.node_id]
            )
            updated["means"] = existing.world_center.to(center.device).reshape(1, 3)
            updated["scales"] = torch.log(
                torch.clamp(
                    existing.root_scale.to(center.device).reshape(1, 3),
                    min=1.0e-8,
                )
            )
            updated["sh0"] = (
                (existing.root_color.to(center.device) - 0.5)
                / 0.28209479177387814
            ).reshape(1, 1, 3)
            self.store.update_metric(existing.node_id, updated)
            self.anchor_bank.remove(anchor_id)
            return existing.node_id, True

        node = self.registry.create_metric_root(anchor, center, rotation, scale, frame_id)
        opacity = torch.full(
            (1,), float(self.config["metric_initial_opacity"]), device=center.device, dtype=center.dtype
        )
        params = self._raw_params(
            center.reshape(1, 3),
            scale.reshape(1, 3),
            rotation.reshape(1, 4),
            opacity,
            anchor.mean_color.reshape(1, 3),
        )
        node.metric_gaussian_id = self.store.add_metric(node.node_id, params)
        self.anchor_bank.remove(anchor_id)
        return node.node_id, False

    def surface_child_layout(
        self, root: GaussianTreeNode, current_depth: Optional[float]
    ) -> Tuple[int, float]:
        """Choose a finer square child grid for close or large projected roots."""
        child_count = int(self.config["children_per_root"])
        scale_ratio = float(self.config["child_scale_ratio"])
        is_near = (
            current_depth is not None
            and current_depth <= float(self.config["near_surface_depth_m"])
        ) or root.projected_radius_ema >= float(
            self.config["near_surface_projected_radius_px"]
        )
        if is_near:
            child_count = int(self.config["near_surface_children"])
            scale_ratio = float(self.config["near_child_scale_ratio"])

        is_very_near = (
            current_depth is not None
            and current_depth <= float(self.config["very_near_surface_depth_m"])
            and root.residual_ema >= float(
                self.config["very_near_surface_min_residual"]
            )
        ) or root.projected_radius_ema >= float(
            self.config["very_near_surface_projected_radius_px"]
        )
        if is_very_near:
            child_count = int(self.config["very_near_surface_children"])
            scale_ratio = float(self.config["very_near_child_scale_ratio"])
        return child_count, scale_ratio

    def refine_root(
        self, root_id: int, current_depth: Optional[float] = None
    ) -> List[int]:
        root = self.registry.nodes[root_id]
        if root.state != NodeState.METRIC:
            raise ValueError("Only a METRIC root can be refined")
        params = self.store._sync(root_id, self.store.active_metric[root_id])
        center = params["means"][0]
        scale = torch.exp(params["scales"][0])
        quat = params["quats"][0]
        rotation = quaternion_to_matrix(quat)
        tangent_x = rotation[:, 0]
        tangent_y = rotation[:, 1]
        child_count, scale_ratio = self.surface_child_layout(root, current_depth)
        grid_size = math.isqrt(child_count)
        coordinates = (
            (torch.arange(grid_size, device=center.device, dtype=center.dtype) + 0.5)
            / grid_size
            - 0.5
        )
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        offsets = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
        child_centers = (
            center.reshape(1, 3)
            + offsets[:, :1] * scale[0] * tangent_x
            + offsets[:, 1:] * scale[1] * tangent_y
        )
        child_scale = scale * scale_ratio
        if root.appearance_grid is None:
            raise RuntimeError("Metric root is missing its appearance grid")
        appearance = root.appearance_grid.to(
            device=center.device, dtype=center.dtype
        ).permute(2, 0, 1).unsqueeze(0)
        child_colors = F.interpolate(
            appearance,
            size=(grid_size, grid_size),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0).reshape(child_count, 3)
        child_ids = self.registry.refine(root_id, child_centers, child_scale)
        parent_opacity = params["opacities"][0].reshape(1)
        opacity_floor = float(self.config["surface_initial_opacity_floor"])
        if opacity_floor > 0.0:
            child_opacity = _inverse_sigmoid(
                torch.clamp(torch.sigmoid(parent_opacity), min=opacity_floor)
            )
        else:
            child_opacity = parent_opacity
        child_params = {
            "means": child_centers,
            "scales": torch.log(torch.clamp(child_scale, min=1.0e-8))
            .reshape(1, 3)
            .repeat(child_count, 1),
            "quats": quat.reshape(1, 4).repeat(child_count, 1),
            "opacities": child_opacity.repeat(child_count),
            "sh0": ((child_colors - 0.5) / 0.28209479177387814).reshape(
                child_count, 1, 3
            ),
            "shN": params["shN"][0]
            .reshape(1, params["shN"].shape[1], 3)
            .repeat(child_count, 1, 1),
        }
        self.store.refine(root_id, child_ids, child_params)
        root.world_center = center.detach().clone()
        root.root_scale = scale.detach().clone()
        root.root_rotation = quat.detach().clone()
        return child_ids

    def archive_root(self, root_id: int, frame_id: int) -> str:
        root = self.registry.nodes[root_id]
        params = self.store.remove_surface(root_id)
        handle = self.archive_store.archive(
            root_id,
            root.children_ids,
            params,
            root.world_bbox_min,
            root.world_bbox_max,
            metadata={"archived_frame": frame_id},
        )
        proxy = self.store.root_snapshots.get(root_id)
        if proxy is None:
            weights = torch.sigmoid(params["opacities"])
            center = torch.sum(params["means"] * weights[:, None], dim=0) / torch.clamp(weights.sum(), min=1.0e-8)
            proxy = {key: value[:1].detach().cpu() for key, value in params.items()}
            proxy["means"] = center.reshape(1, 3).detach().cpu()
        self.store.set_archive_proxy(root_id, proxy)
        self.registry.mark_archived(root_id, handle)
        for node_id in [root_id] + list(root.children_ids):
            node = self.registry.nodes[node_id]
            for field_name in (
                "world_center",
                "world_bbox_min",
                "world_bbox_max",
                "root_rotation",
                "root_scale",
                "root_color",
                "appearance_grid",
                "descriptor",
            ):
                value = getattr(node, field_name)
                if value is not None:
                    setattr(node, field_name, value.detach().cpu())
        return handle

    def reactivate_root(self, root_id: int) -> List[int]:
        root = self.registry.nodes[root_id]
        params = self.archive_store.restore(root_id, self.store.device, torch.float32)
        self.store.pop_archive_proxy(root_id)
        self.store.add_surface(root_id, root.children_ids, params)
        self.registry.mark_reactivated(root_id)
        return list(root.children_ids)

    def _associate_persistent_nodes(
        self,
        observations: Sequence[Observation],
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: Tuple[int, int],
        near: float,
        far: float,
    ) -> Set[int]:
        claimed: Set[int] = set()
        used_roots: Set[int] = set()
        self.last_metric_depth_corrections = 0
        self.last_metric_depth_correction_rejections = 0
        radius = float(self.config["association_radius_px"])
        threshold = float(self.config["association_feature_threshold"])
        cell_size = max(1, int(math.ceil(radius)))
        bins = defaultdict(list)
        roots = self.registry.root_nodes(
            (NodeState.METRIC, NodeState.SURFACE, NodeState.ARCHIVED)
        )
        root_indices_by_id = {
            root.node_id: root_index for root_index, root in enumerate(roots)
        }
        if not observations or not roots:
            return claimed

        centers = torch.stack(
            [
                root.world_center.to(
                    device=world_to_camera.device, dtype=world_to_camera.dtype
                )
                for root in roots
            ]
        )
        projected_uv, projected_depth, valid = project_world_batch(
            centers, world_to_camera, intrinsics, image_size, near, far
        )
        valid_root_indices = torch.nonzero(valid, as_tuple=False).flatten()
        valid_uv_cpu = projected_uv[valid_root_indices].detach().cpu().tolist()
        for root_index, uv in zip(valid_root_indices.cpu().tolist(), valid_uv_cpu):
            bins[(int(uv[0]) // cell_size, int(uv[1]) // cell_size)].append(
                (root_index, uv)
            )

        observation_uv = torch.stack([observation.uv for observation in observations])
        observation_uv_cpu = observation_uv.detach().cpu().tolist()
        pair_observations = []
        pair_roots = []
        for index, uv in enumerate(observation_uv_cpu):
            bx = int(uv[0]) // cell_size
            by = int(uv[1]) // cell_size
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for root_index, _ in bins.get((bx + dx, by + dy), []):
                        pair_observations.append(index)
                        pair_roots.append(root_index)
        if not pair_observations:
            return claimed

        pair_observation_tensor = torch.tensor(
            pair_observations, device=observation_uv.device, dtype=torch.long
        )
        pair_root_tensor = torch.tensor(
            pair_roots, device=projected_uv.device, dtype=torch.long
        )
        pixel_distances = torch.linalg.norm(
            observation_uv[pair_observation_tensor] - projected_uv[pair_root_tensor],
            dim=1,
        )
        observation_descriptors = torch.stack(
            [observation.descriptor for observation in observations]
        )
        root_descriptors = torch.stack(
            [
                root.descriptor.to(
                    device=observation_descriptors.device,
                    dtype=observation_descriptors.dtype,
                )
                for root in roots
            ]
        )
        feature_errors = 1.0 - F.cosine_similarity(
            observation_descriptors[pair_observation_tensor],
            root_descriptors[pair_root_tensor],
            dim=1,
        )
        valid_pairs = (pixel_distances < radius) & (feature_errors < threshold)
        valid_pair_indices = torch.nonzero(valid_pairs, as_tuple=False).flatten()
        scores = feature_errors + pixel_distances / radius
        candidates = [
            (score, observation_index, roots[root_index].node_id)
            for score, observation_index, root_index in zip(
                scores[valid_pair_indices].detach().cpu().tolist(),
                pair_observation_tensor[valid_pair_indices].cpu().tolist(),
                pair_root_tensor[valid_pair_indices].cpu().tolist(),
            )
        ]
        accepted_corrections = 0
        rejected_corrections = 0
        for _, index, root_id in sorted(candidates):
            if index in claimed or root_id in used_roots:
                continue
            root = self.registry.nodes[root_id]
            observation = observations[index]
            root.observation_count += 1
            root.last_seen_frame = observation.frame_id
            root.residual_ema = (
                (1.0 - float(self.config["residual_ema"])) * root.residual_ema
                + float(self.config["residual_ema"]) * observation.residual_score
            )
            if observation.frame_id not in root.support_keyframes:
                root.support_keyframes.append(observation.frame_id)
            if bool(self.config["enable_metric_depth_correction"]) and root.state == NodeState.METRIC:
                root_index = root_indices_by_id[root_id]
                corrected = self._correct_metric_root_depth(
                    root,
                    observation,
                    pose=world_to_camera,
                    intrinsics=intrinsics,
                    projected_uv=projected_uv[root_index],
                    projected_depth=projected_depth[root_index],
                )
                accepted_corrections += int(corrected)
                rejected_corrections += int(not corrected)
            claimed.add(index)
            used_roots.add(root_id)
        self.last_metric_depth_corrections = accepted_corrections
        self.last_metric_depth_correction_rejections = rejected_corrections
        return claimed

    @torch.no_grad()
    def _correct_metric_root_depth(
        self,
        root: GaussianTreeNode,
        observation: Observation,
        pose: torch.Tensor,
        intrinsics: torch.Tensor,
        projected_uv: torch.Tensor,
        projected_depth: torch.Tensor,
    ) -> bool:
        """Fuse a guarded current-view sparse-depth measurement into one M root."""
        pixel_error = torch.linalg.norm(observation.uv - projected_uv)
        feature_error = 1.0 - F.cosine_similarity(
            observation.descriptor.reshape(1, -1),
            root.descriptor.to(
                device=observation.descriptor.device,
                dtype=observation.descriptor.dtype,
            ).reshape(1, -1),
            dim=1,
        )[0]
        measured_depth = float(observation.depth_prior)
        predicted_depth = float(projected_depth.item())
        relative_depth_error = abs(measured_depth - predicted_depth) / max(
            predicted_depth, 1.0e-6
        )
        relative_uncertainty = float(observation.depth_uncertainty) / max(
            measured_depth, 1.0e-6
        )
        valid = (
            observation.depth_valid
            and observation.depth_support
            >= int(self.config["metric_correction_min_depth_support"])
            and measured_depth > 0.0
            and predicted_depth > 0.0
            and torch.isfinite(pixel_error)
            and float(pixel_error.item())
            <= float(self.config["metric_correction_max_pixel_error_px"])
            and torch.isfinite(feature_error)
            and float(feature_error.item())
            <= float(self.config["metric_correction_max_feature_error"])
            and relative_depth_error
            <= float(self.config["metric_correction_max_relative_depth_error"])
            and relative_uncertainty
            <= float(self.config["metric_correction_max_relative_uncertainty"])
        )
        if not valid:
            root.metric_depth_reject_count += 1
            return False

        params = self.store._sync(root.node_id, self.store.active_metric[root.node_id])
        old_center = params["means"][0]
        measured_center = unproject_pixel(
            observation.uv.to(device=old_center.device, dtype=old_center.dtype),
            torch.tensor(measured_depth, device=old_center.device, dtype=old_center.dtype),
            intrinsics.to(device=old_center.device, dtype=old_center.dtype),
            pose.to(device=old_center.device, dtype=old_center.dtype),
        )
        base_alpha = float(self.config["metric_correction_ema"])
        uncertainty_weight = 1.0 / (1.0 + 10.0 * relative_uncertainty)
        alpha = min(1.0, max(0.0, base_alpha * uncertainty_weight))
        corrected_center = torch.lerp(old_center, measured_center, alpha)

        reference_pose = pose.to(device=old_center.device, dtype=old_center.dtype)
        old_h = torch.cat((old_center, torch.ones_like(old_center[:1])))
        corrected_h = torch.cat(
            (corrected_center, torch.ones_like(corrected_center[:1]))
        )
        old_reference_depth = (reference_pose @ old_h)[2]
        corrected_reference_depth = (reference_pose @ corrected_h)[2]
        scale_ratio = torch.clamp(
            corrected_reference_depth / torch.clamp(old_reference_depth, min=1.0e-6),
            min=0.75,
            max=1.25,
        )
        corrected_scale = torch.exp(params["scales"][0]) * scale_ratio
        params["means"][0] = corrected_center
        params["scales"][0] = torch.log(torch.clamp(corrected_scale, min=1.0e-8))
        self.store.update_metric(root.node_id, params)

        innovation = float(
            (
                torch.linalg.norm(measured_center - old_center)
                / max(measured_depth, 1.0e-6)
            ).item()
        )
        ema = float(self.config["residual_ema"])
        root.metric_center_innovation_ema = (
            innovation
            if root.metric_depth_update_count == 0
            else (1.0 - ema) * root.metric_center_innovation_ema + ema * innovation
        )
        root.metric_depth_update_count += 1
        root.world_center = corrected_center.detach().clone()
        root.root_scale = corrected_scale.detach().clone()
        radius = torch.max(corrected_scale[:2])
        root.world_bbox_min = (corrected_center - radius).detach().clone()
        root.world_bbox_max = (corrected_center + radius).detach().clone()
        return True

    def _metric_depth_ready_for_refine(
        self, root: GaussianTreeNode, frame_id: int
    ) -> bool:
        if not bool(self.config["enable_metric_depth_correction"]):
            return True
        required = int(self.config["metric_correction_min_updates_for_refine"])
        if root.metric_depth_update_count >= required:
            return True
        promoted_frame = min(root.support_keyframes) if root.support_keyframes else frame_id
        return frame_id - promoted_frame >= int(
            self.config["metric_correction_max_wait_frames"]
        )

    def _select_spawn_observations(
        self,
        observations: Sequence[Observation],
        claimed_indices: Set[int],
    ) -> List[Observation]:
        """Reserve most, but not all, new-P capacity for reliable near depth."""
        available = [
            observation for index, observation in enumerate(observations)
            if index not in claimed_indices
            and (
                not bool(self.config["spawn_requires_valid_depth"])
                or observation.depth_valid
            )
        ]
        max_new = int(self.config["max_new_anchors_per_keyframe"])
        near_depth = float(self.config["near_observation_depth_m"])
        near = [
            observation for observation in available
            if observation.depth_valid
            and 0.0 < observation.depth_prior <= near_depth
        ]
        other = [
            observation for observation in available
            if not (
                observation.depth_valid
                and 0.0 < observation.depth_prior <= near_depth
            )
        ]
        near_quota = min(
            len(near), int(round(max_new * float(self.config["near_spawn_fraction"])))
        )
        selected = near[:near_quota]
        selected.extend(other[: max_new - len(selected)])
        if len(selected) < max_new:
            selected.extend(near[near_quota : near_quota + max_new - len(selected)])
        return selected

    def _sync_registry_geometry(self) -> None:
        """Reflect optimizer-updated group geometry in root metadata before association."""
        for root_id, fallback in self.store.active_metric.items():
            if root_id not in self.optimization_enabled_root_ids:
                continue
            params = (
                self.gaussian_model.gaussian_groups[self.store.group_ids[root_id]].splats
                if self.gaussian_model is not None
                else fallback
            )
            root = self.registry.nodes[root_id]
            root.world_center = params["means"][0].detach().clone()
            root.root_scale = torch.exp(params["scales"][0]).detach().clone()
            root.root_rotation = params["quats"][0].detach().clone()
        for root_id, fallback in self.store.active_surface.items():
            if root_id not in self.optimization_enabled_root_ids:
                continue
            params = (
                self.gaussian_model.gaussian_groups[self.store.group_ids[root_id]].splats
                if self.gaussian_model is not None
                else fallback
            )
            root = self.registry.nodes[root_id]
            root.world_center = params["means"].mean(dim=0).detach().clone()
            root.world_bbox_min = params["means"].amin(dim=0).detach().clone()
            root.world_bbox_max = params["means"].amax(dim=0).detach().clone()

    @torch.no_grad()
    def configure_optimization_visibility(self, cameras) -> Dict[str, int]:
        """Optimize only active roots visible in the selected causal views."""
        roots = list(self.registry.root_nodes((NodeState.METRIC, NodeState.SURFACE)))
        if self.gaussian_model is None or not roots:
            result = {"visible": len(roots), "enabled": len(roots), "frozen": 0}
            self.optimization_enabled_root_ids = {root.node_id for root in roots}
            self.last_optimization_visibility = result
            return result

        camera_list = cameras if isinstance(cameras, (list, tuple)) else [cameras]
        visible_ids: Set[int] = set()
        gate_enabled = bool(self.config["optimize_visible_roots_only"])
        if not gate_enabled:
            result = {"visible": len(roots), "enabled": len(roots), "frozen": 0}
            self.optimization_enabled_root_ids = {root.node_id for root in roots}
            self.last_optimization_visibility = result
            return result
        if gate_enabled and camera_list:
            device = camera_list[0].get_pose().device
            dtype = camera_list[0].get_pose().dtype
            centers = torch.stack(
                [root.world_center.to(device=device, dtype=dtype) for root in roots], dim=0
            )
            homogeneous = torch.cat((centers, torch.ones_like(centers[:, :1])), dim=1)
            visible = torch.zeros(len(roots), dtype=torch.bool, device=device)
            margin = float(self.config["optimization_visibility_margin_px"])
            for camera in camera_list:
                pose = camera.get_pose().detach().to(device=device, dtype=dtype)
                intrinsics = camera.get_int_mat(0).detach().to(device=device, dtype=dtype)
                camera_points = (pose @ homogeneous.T).T[:, :3]
                depth = camera_points[:, 2]
                safe_depth = torch.clamp(depth, min=1.0e-6)
                pixels = (intrinsics @ camera_points.T).T
                uv = pixels[:, :2] / safe_depth[:, None]
                visible |= (
                    (depth > float(camera.near))
                    & (depth < float(camera.far))
                    & torch.isfinite(uv).all(dim=1)
                    & (uv[:, 0] >= -margin)
                    & (uv[:, 0] < camera.get_width(0) + margin)
                    & (uv[:, 1] >= -margin)
                    & (uv[:, 1] < camera.get_height(0) + margin)
                )
            visible_ids = {
                root.node_id for root, is_visible in zip(roots, visible.tolist()) if is_visible
            }
        visible_count = len(visible_ids)
        max_optimized = int(self.config["max_optimized_roots_per_step"])
        if max_optimized > 0 and len(visible_ids) > max_optimized:
            ranked_visible = sorted(
                (root for root in roots if root.node_id in visible_ids),
                key=lambda root: (
                    root.last_seen_frame,
                    root.projected_radius_ema * max(root.residual_ema, 1.0e-4),
                    root.observation_count,
                    root.node_id,
                ),
                reverse=True,
            )
            visible_ids = {
                root.node_id for root in ranked_visible[:max_optimized]
            }

        enabled = 0
        for root in roots:
            group_id = self.store.group_ids[root.node_id]
            if root.node_id in visible_ids:
                self.gaussian_model.add_optimization(group_id)
                enabled += 1
            else:
                self.gaussian_model.remove_optimization(group_id)
        result = {
            "visible": visible_count,
            "enabled": enabled,
            "frozen": len(roots) - enabled,
        }
        self.optimization_enabled_root_ids = visible_ids
        self.last_optimization_visibility = result
        return result

    @torch.no_grad()
    def constrain_active_surface_scales(self, camera=None) -> int:
        """Keep optimized S children fine enough to preserve visible street detail."""
        clamped_rows = 0
        clamped_opacities = 0
        pose = intrinsics = None
        image_size = None
        if camera is not None:
            pose = camera.get_pose().detach()
            intrinsics = camera.get_int_mat(0)
            image_size = (camera.get_height(0), camera.get_width(0))

        min_factor = float(self.config["surface_scale_min_factor"])
        max_factor = float(self.config["surface_scale_max_factor"])
        max_sigma_px = float(self.config["surface_max_projected_sigma_px"])
        for root_id, fallback in list(self.store.active_surface.items()):
            params = self.store._sync(root_id, fallback)
            root = self.registry.nodes[root_id]
            initial_scales = self.registry.nodes[root.children_ids[0]].root_scale.to(
                params["scales"].device, dtype=params["scales"].dtype
            ).reshape(1, 3).expand(params["scales"].shape[0], 3)
            scales = torch.exp(params["scales"])
            lower = initial_scales * min_factor
            upper = initial_scales * max_factor

            if pose is not None and intrinsics is not None:
                means = params["means"]
                pose_device = pose.to(means.device, dtype=means.dtype)
                intrinsics_device = intrinsics.to(means.device, dtype=means.dtype)
                homogeneous = torch.cat(
                    (means, torch.ones_like(means[:, :1])), dim=1
                )
                camera_points = (pose_device @ homogeneous.T).T[:, :3]
                depth = camera_points[:, 2]
                safe_depth = torch.clamp(depth, min=1.0e-6)
                pixels = (intrinsics_device @ camera_points.T).T
                uv = pixels[:, :2] / safe_depth[:, None]
                visible = (
                    (depth > float(camera.near))
                    & (depth < float(camera.far))
                    & torch.isfinite(uv).all(dim=1)
                    & (uv[:, 0] >= 0)
                    & (uv[:, 0] < image_size[1])
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < image_size[0])
                )
                projected_upper_x = max_sigma_px * safe_depth / torch.clamp(
                    intrinsics_device[0, 0], min=1.0e-6
                )
                projected_upper_y = max_sigma_px * safe_depth / torch.clamp(
                    intrinsics_device[1, 1], min=1.0e-6
                )
                upper[visible, 0] = torch.minimum(
                    upper[visible, 0], projected_upper_x[visible]
                )
                upper[visible, 1] = torch.minimum(
                    upper[visible, 1], projected_upper_y[visible]
                )
                lower = torch.minimum(lower, upper)

            constrained = torch.minimum(torch.maximum(scales, lower), upper)
            changed = torch.any(
                torch.abs(constrained - scales) > 1.0e-7, dim=1
            )
            opacity = torch.sigmoid(params["opacities"])
            constrained_opacity = torch.clamp(
                opacity, min=float(self.config["surface_opacity_min"])
            )
            opacity_changed = torch.abs(constrained_opacity - opacity) > 1.0e-7
            if not bool(changed.any()) and not bool(opacity_changed.any()):
                continue
            params["scales"] = torch.log(torch.clamp(constrained, min=1.0e-8))
            params["opacities"] = _inverse_sigmoid(constrained_opacity)
            self.store.update_surface(root_id, params)
            clamped_rows += int(changed.sum().item())
            clamped_opacities += int(opacity_changed.sum().item())
        self.last_scale_clamp_count = clamped_rows
        self.last_opacity_clamp_count = clamped_opacities
        return clamped_rows

    def _update_visibility(
        self,
        frame_id: int,
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        residual: torch.Tensor,
        near: float,
        far: float,
    ) -> Tuple[Set[int], List[int]]:
        height, width = residual.shape
        visible: Set[int] = set()
        reactivate = []
        roots = self.registry.root_nodes()
        if not roots:
            return visible, reactivate
        centers = torch.stack(
            [
                root.world_center.to(
                    device=world_to_camera.device, dtype=world_to_camera.dtype
                )
                for root in roots
            ]
        )
        scales = torch.stack(
            [
                root.root_scale[:2].to(
                    device=intrinsics.device, dtype=intrinsics.dtype
                )
                for root in roots
            ]
        )
        uv, depth, valid = project_world_batch(
            centers, world_to_camera, intrinsics, (height, width), near, far
        )
        projected_radii = intrinsics[0, 0] * scales.amax(dim=1) / torch.clamp(
            depth, min=1.0e-6
        )
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return visible, reactivate

        visible_uv = uv[valid_indices]
        visible_radii = projected_radii[valid_indices]
        xs = visible_uv[:, 0].to(torch.long).clamp(0, width - 1)
        ys = visible_uv[:, 1].to(torch.long).clamp(0, height - 1)
        radius_pixels = visible_radii.to(torch.long).clamp(min=1)
        x0 = (xs - radius_pixels).clamp(min=0)
        y0 = (ys - radius_pixels).clamp(min=0)
        x1 = (xs + radius_pixels + 1).clamp(max=width)
        y1 = (ys + radius_pixels + 1).clamp(max=height)
        integral = F.pad(residual.cumsum(dim=0).cumsum(dim=1), (1, 0, 1, 0))
        patch_sums = (
            integral[y1, x1]
            - integral[y0, x1]
            - integral[y1, x0]
            + integral[y0, x0]
        )
        patch_means = patch_sums / ((x1 - x0) * (y1 - y0)).clamp(min=1)
        visible_values = zip(
            valid_indices.cpu().tolist(),
            visible_radii.detach().cpu().tolist(),
            patch_means.detach().cpu().tolist(),
        )
        for root_index, projected_radius, patch_residual in visible_values:
            root = roots[root_index]
            visible.add(root.node_id)
            if root.state == NodeState.ARCHIVED:
                if bool(self.config["enable_reactivation"]) and projected_radius >= float(
                    self.config["reactivate_min_projected_radius_px"]
                ):
                    reactivate.append(root.node_id)
                continue
            root.last_seen_frame = frame_id
            root.observation_count += 1
            ema = float(self.config["residual_ema"])
            root.projected_radius_ema = projected_radius if root.projected_radius_ema == 0.0 else (
                (1.0 - ema) * root.projected_radius_ema + ema * projected_radius
            )
            root.residual_ema = patch_residual if root.residual_ema <= 0.0 else (
                (1.0 - ema) * root.residual_ema + ema * patch_residual
            )
        return visible, reactivate

    def process_frame(
        self, camera, stable_render: Dict[str, torch.Tensor], is_keyframe: bool
    ) -> ProgressiveFrameStats:
        """Process exactly one current frame; no future frame or offline state is accessed."""
        process_start = time.perf_counter()
        frame_id = int(camera.cam_idx)
        image = camera.get_gt_image(0)
        sparse_depth = camera.get_sparse_depth(0)
        opacity = stable_render["opacity"]
        residual = stable_render.get("diff")
        if residual is None:
            residual = torch.mean(torch.abs(image - stable_render["render"]), dim=-1)
        residual = residual.squeeze()
        observations, candidate_mask = self.extractor.extract(
            frame_id, image, sparse_depth, opacity, residual
        )
        near_observation_count = sum(
            observation.depth_valid
            and 0.0 < observation.depth_prior <= float(
                self.config["near_observation_depth_m"]
            )
            for observation in observations
        )
        self.last_candidate_mask = candidate_mask
        self.last_stable_render = stable_render
        pose = camera.get_pose().detach()
        intrinsics = camera.get_int_mat(0)
        near, far = float(camera.near), float(camera.far)
        self._sync_registry_geometry()
        persistent_claimed = self._associate_persistent_nodes(
            observations, pose, intrinsics, image.shape[:2], near, far
        )
        remaining_indices = [index for index in range(len(observations)) if index not in persistent_claimed]
        remaining = [observations[index] for index in remaining_indices]
        p_matches, p_claimed_local = self.anchor_bank.associate(
            remaining, image, pose, intrinsics, near, far, self.extractor
        )
        spawn_allowed = is_keyframe if bool(self.config["use_keyframes_for_spawn"]) else (
            frame_id % int(self.config["spawn_interval"]) == 0
        )
        new_count = 0
        near_new_count = 0
        new_plane_count = 0
        if spawn_allowed:
            for observation in self._select_spawn_observations(
                remaining, p_claimed_local
            ):
                anchor = self.anchor_bank.create_anchor(
                    observation,
                    pose,
                    intrinsics,
                    near,
                    far,
                    sparse_depth=sparse_depth,
                )
                new_count += 1
                new_plane_count += int(anchor.reference_surface_normal is not None)
                near_new_count += int(
                    observation.depth_valid
                    and 0.0 < observation.depth_prior <= float(
                        self.config["near_observation_depth_m"]
                    )
                )

        promoted = 0
        promoted_plane = 0
        merged = 0
        promotion_candidates = list(self.anchor_bank.promotion_candidates())
        promotion_candidates.sort(
            key=lambda anchor: (
                int(
                    anchor.reference_depth_valid
                    and 0.0 < anchor.reference_depth_prior
                    <= float(self.config["near_promotion_max_depth_m"])
                ),
                anchor.observation_count,
                anchor.static_confidence,
                -anchor.posterior_entropy,
                anchor.max_parallax_rad,
                -anchor.anchor_id,
            ),
            reverse=True,
        )
        promotion_limit = int(self.config["max_promotions_per_frame"])
        if promotion_limit > 0:
            promotion_candidates = promotion_candidates[:promotion_limit]
        for anchor in promotion_candidates:
            if not self.budget.can_promote(self.store.num_metric):
                break
            used_plane = anchor.reference_surface_normal is not None
            _, was_merged = self.promote_anchor(anchor.anchor_id, frame_id)
            promoted += 1
            promoted_plane += int(used_plane)
            merged += int(was_merged)

        visible, reactivate_ids = self._update_visibility(
            frame_id, pose, intrinsics, residual, near, far
        )
        reactivated = 0
        for root_id in reactivate_ids:
            visible_archive_count = sum(
                1 for root in self.registry.root_nodes((NodeState.ARCHIVED,))
                if root.node_id in visible
            )
            total_before = (
                self._persistent_active_count()
                + visible_archive_count
                + self.projective_proxy_count()
            )
            required = len(self.registry.nodes[root_id].children_ids)
            if not self.budget.can_refine(self.store.num_surface, required):
                continue
            if total_before + max(0, required - 1) > int(
                self.config["max_active_gaussians"]
            ):
                continue
            self.reactivate_root(root_id)
            reactivated += 1

        refine_candidates = [] if bool(self.config["admission_only"]) else [
            root for root in self.registry.root_nodes((NodeState.METRIC,))
            if root.observation_count >= int(self.config["refine_min_observations"])
            and root.projected_radius_ema >= float(self.config["refine_min_projected_radius_px"])
            and root.residual_ema >= float(self.config["refine_min_residual"])
            and root.confidence >= float(self.config["refine_min_confidence"])
            and self._metric_depth_ready_for_refine(root, frame_id)
        ]
        waiting_for_depth = sum(
            not self._metric_depth_ready_for_refine(root, frame_id)
            for root in self.registry.root_nodes((NodeState.METRIC,))
        )
        refine_candidates.sort(key=self.budget.refinement_priority, reverse=True)
        refinement_limit = int(self.config["max_refinements_per_frame"])
        if refinement_limit > 0:
            refine_candidates = refine_candidates[:refinement_limit]
        refined = 0
        new_surface_gaussians = 0
        for root in refine_candidates:
            _, current_depth, valid = project_world(
                root.world_center.to(pose.device),
                pose,
                intrinsics,
                image.shape[:2],
                near,
                far,
            )
            depth_value = float(current_depth.item()) if valid else None
            required, _ = self.surface_child_layout(root, depth_value)
            if not self.budget.can_refine(self.store.num_surface, required):
                continue
            child_ids = self.refine_root(root.node_id, depth_value)
            refined += 1
            new_surface_gaussians += len(child_ids)

        prune_ids = self.budget.anchor_prune_candidates(
            self.anchor_bank.anchors.values(), frame_id
        )
        for anchor_id in prune_ids:
            self.anchor_bank.remove(anchor_id)
        visible_archive_count = sum(
            1 for root in self.registry.root_nodes((NodeState.ARCHIVED,))
            if root.node_id in visible
        )
        total_active = (
            self._persistent_active_count()
            + visible_archive_count
            + self.projective_proxy_count()
        )
        archive_ids = self.budget.surface_archive_candidates(
            self.registry.root_nodes((NodeState.SURFACE,)), frame_id, visible,
            self.store.num_surface, total_active
        )
        archived = 0
        for root_id in archive_ids:
            self.archive_root(root_id, frame_id)
            archived += 1

        visible_archive_count = sum(
            1 for root in self.registry.root_nodes((NodeState.ARCHIVED,))
            if root.node_id in visible
        )
        total_active = (
            self._persistent_active_count()
            + visible_archive_count
            + self.projective_proxy_count()
        )
        if total_active > int(self.config["max_active_gaussians"]):
            self.anchor_bank.collapse_to_best_mode()
            warnings.warn(
                "Progressive active Gaussian budget remains exceeded: {} > {}".format(
                    total_active, self.config["max_active_gaussians"]
                ), RuntimeWarning
            )
        optimization_visibility = self.configure_optimization_visibility(camera)
        stats = ProgressiveFrameStats(
            frame_id=frame_id,
            num_observations=len(observations),
            num_near_observations=near_observation_count,
            num_observations_matched_to_P=len(p_matches),
            num_new_P=new_count,
            num_near_new_P=near_new_count,
            num_new_P_with_plane=new_plane_count,
            num_pruned_P=len(prune_ids),
            num_active_P=len(self.anchor_bank.anchors),
            num_promoted_P_to_M=promoted,
            num_promoted_plane_P_to_M=promoted_plane,
            num_merged_into_existing_M=merged,
            num_metric_depth_corrections=self.last_metric_depth_corrections,
            num_metric_depth_correction_rejections=self.last_metric_depth_correction_rejections,
            num_metric_waiting_for_depth=waiting_for_depth,
            num_active_M=self.store.num_metric,
            num_refined_M_to_S=refined,
            num_new_surface_gaussians=new_surface_gaussians,
            num_active_S=self.store.num_surface,
            num_archived_S_to_A=archived,
            num_active_A=len(self.store.archive_proxies),
            num_reactivated_A=reactivated,
            num_proxy_splats=self.projective_proxy_count(),
            num_total_active_gaussians=total_active,
            num_clamped_S_scales=self.last_scale_clamp_count,
            num_clamped_S_opacities=self.last_opacity_clamp_count,
            num_visible_roots=optimization_visibility["visible"],
            num_optimized_roots=optimization_visibility["enabled"],
            num_frozen_roots=optimization_visibility["frozen"],
            progressive_process_seconds=time.perf_counter() - process_start,
            cpu_archive_bytes=self.archive_store.cpu_bytes,
        )
        self.last_scale_clamp_count = 0
        self.last_opacity_clamp_count = 0
        if torch.cuda.is_available():
            stats.gpu_memory_allocated = torch.cuda.memory_allocated(self.store.device)
            stats.gpu_memory_reserved = torch.cuda.memory_reserved(self.store.device)
        self.debug_writer.write_stats(stats)
        state_depth_bands = None
        if self.debug_writer.should_write_histograms(frame_id):
            state_depth_bands = self._state_depth_bands(
                pose, intrinsics, image.shape[:2], near, far
            )
        self.debug_writer.write_p_histograms(
            frame_id,
            self.anchor_bank.anchors.values(),
            self.config,
            num_promoted=promoted,
            num_pruned=len(prune_ids),
            state_depth_bands=state_depth_bands,
        )
        return stats

    def _persistent_active_count(self) -> int:
        if self.gaussian_model is not None:
            return int(self.gaussian_model.get_num_gaussians)
        return self.store.num_metric + self.store.num_surface

    def _state_depth_bands(
        self,
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: Tuple[int, int],
        near: float,
        far: float,
    ) -> Dict[str, object]:
        """Count current-view P/M/S/A support in configurable metric depth bands."""
        low, high = [float(value) for value in self.config["depth_histogram_edges_m"]]
        counts = {
            state: {"near": 0, "mid": 0, "far": 0, "not_visible": 0, "total": 0}
            for state in ("P", "M", "S", "A")
        }

        def add_depth(state: str, depth: float, weight: int = 1) -> None:
            counts[state]["total"] += weight
            if not math.isfinite(depth) or depth <= 0.0:
                counts[state]["not_visible"] += weight
            elif depth <= low:
                counts[state]["near"] += weight
            elif depth <= high:
                counts[state]["mid"] += weight
            else:
                counts[state]["far"] += weight

        def add_point(state: str, point: torch.Tensor, weight: int = 1) -> None:
            _, depth, valid = project_world(
                point.to(world_to_camera.device),
                world_to_camera,
                intrinsics,
                image_size,
                near,
                far,
            )
            if not valid:
                counts[state]["total"] += weight
                counts[state]["not_visible"] += weight
            else:
                add_depth(state, float(depth.item()), weight)

        for anchor in self.anchor_bank.anchors.values():
            add_depth("P", 1.0 / max(anchor.posterior_mean, 1.0e-8))
        for root in self.registry.root_nodes():
            if root.state == NodeState.METRIC:
                add_point("M", root.world_center)
            elif root.state == NodeState.SURFACE:
                add_point("S", root.world_center, len(root.children_ids))
            elif root.state == NodeState.ARCHIVED:
                add_point("A", root.world_center, len(root.children_ids))
        return {
            "edges_m": [low, high],
            "p_depth_basis": "reference_posterior",
            "persistent_depth_basis": "current_camera_visible",
            "counts": counts,
        }

    def surface_regularization_loss(self) -> torch.Tensor:
        """Keep optimized children within their parent's coarse support region."""
        if self.gaussian_model is None or not bool(
            self.config["enable_center_regularization"]
        ):
            return torch.tensor(0.0, device=self.store.device)
        losses = []
        allowed_factor = float(self.config["center_regularization_allowed_factor"])
        for root_id in self.store.active_surface:
            root = self.registry.nodes[root_id]
            group = self.gaussian_model.gaussian_groups[self.store.group_ids[root_id]]
            if not group.is_optimize:
                continue
            centers = group.splats["means"]
            scales = torch.exp(group.splats["scales"])
            parent_center = root.world_center.to(centers.device)
            parent_radius = torch.max(root.root_scale[:2].to(centers.device))
            initial_scales = self.registry.nodes[root.children_ids[0]].root_scale.to(
                centers.device, dtype=centers.dtype
            ).reshape(1, 3)
            center_excess = torch.relu(
                torch.linalg.norm(centers - parent_center, dim=1)
                - parent_radius * allowed_factor
            )
            scale_excess = torch.relu(
                scales[:, :2]
                - initial_scales[:, :2]
                * float(self.config["surface_scale_max_factor"])
            )
            losses.append(center_excess.mean() + scale_excess.mean())
        if not losses:
            return torch.tensor(0.0, device=self.store.device)
        return float(self.config["center_regularization_weight"]) * torch.stack(losses).mean()

    def record_bootstrap_frame(self, frame_id: int) -> ProgressiveFrameStats:
        """Emit a complete stats row while legacy bootstrap densification is active."""
        stats = ProgressiveFrameStats(
            frame_id=frame_id,
            num_active_P=len(self.anchor_bank.anchors),
            num_active_M=self.store.num_metric,
            num_active_S=self.store.num_surface,
            num_active_A=len(self.store.archive_proxies),
            num_proxy_splats=self.projective_proxy_count(),
            num_total_active_gaussians=(
                self._persistent_active_count()
                + self.store.num_archive_proxies
                + self.projective_proxy_count()
            ),
            cpu_archive_bytes=self.archive_store.cpu_bytes,
        )
        if torch.cuda.is_available():
            stats.gpu_memory_allocated = torch.cuda.memory_allocated(self.store.device)
            stats.gpu_memory_reserved = torch.cuda.memory_reserved(self.store.device)
        self.debug_writer.write_stats(stats)
        return stats

    def projective_proxy_count(self) -> int:
        top_k = int(self.config["projective_top_k_render_modes"])
        return sum(min(top_k, anchor.mode_log_weights.numel()) for anchor in self.anchor_bank.anchors.values())

    def stable_external_splats(
        self, device: torch.device, dtype: torch.dtype, cameras=None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Return A proxies only; P proxies never count as reliable observation coverage."""
        visible_ids = None
        if cameras is not None:
            camera_list = cameras if isinstance(cameras, (list, tuple)) else [cameras]
            visible_ids = set()
            for camera in camera_list:
                pose = camera.get_pose().detach()
                intrinsics = camera.get_int_mat(0)
                image_size = (camera.get_height(0), camera.get_width(0))
                for root in self.registry.root_nodes((NodeState.ARCHIVED,)):
                    _, _, valid = project_world(
                        root.world_center.to(pose.device), pose, intrinsics, image_size,
                        camera.near, camera.far
                    )
                    if valid:
                        visible_ids.add(root.node_id)
        return self.store.archive_external_splats(device, dtype, root_ids=visible_ids)

    def visualization_external_splats(
        self, device: torch.device, dtype: torch.dtype, cameras=None
    ) -> Optional[Dict[str, torch.Tensor]]:
        archive = self.stable_external_splats(device, dtype, cameras=cameras)
        projective = self.anchor_bank.build_proxy_splats(device, dtype, self.max_sh_degree)
        return _merge_external_splats(archive, projective)

    def debug_projection_points(self, camera) -> Tuple[List[Tuple], List[Tuple], List[Tuple], List[Tuple]]:
        pose = camera.get_pose().detach()
        intrinsics = camera.get_int_mat(0)
        image_size = (camera.get_height(0), camera.get_width(0))
        p_points = []
        for anchor in self.anchor_bank.anchors.values():
            best = int(anchor.mode_log_weights.argmax().item())
            point = self.anchor_bank._world_points(anchor)[best]
            uv, _, valid = project_world(point, pose, intrinsics, image_size, camera.near, camera.far)
            if valid:
                p_points.append((float(uv[0]), float(uv[1]), anchor.posterior_entropy, float(torch.softmax(anchor.mode_log_weights, 0).max())))
        by_state = {NodeState.METRIC: [], NodeState.SURFACE: [], NodeState.ARCHIVED: []}
        for root in self.registry.root_nodes():
            uv, _, valid = project_world(root.world_center.to(pose.device), pose, intrinsics, image_size, camera.near, camera.far)
            if valid:
                by_state[root.state].append((float(uv[0]), float(uv[1])))
        return p_points, by_state[NodeState.METRIC], by_state[NodeState.SURFACE], by_state[NodeState.ARCHIVED]

    def save_debug_frame(self, camera, proxy_render: Optional[torch.Tensor] = None) -> None:
        if (
            self.last_candidate_mask is None
            or self.last_stable_render is None
            or not self.debug_writer.enabled
            or int(camera.cam_idx) % self.debug_writer.save_interval != 0
        ):
            return
        points = self.debug_projection_points(camera)
        self.debug_writer.save_frame(
            int(camera.cam_idx), camera.get_gt_image(0), self.last_stable_render["render"],
            self.last_stable_render["opacity"], self.last_candidate_mask,
            *points,
            proxy_render=proxy_render,
            state_counts={
                "P": len(self.anchor_bank.anchors),
                "M": self.store.num_metric,
                "S": self.store.num_surface,
                "A": len(self.store.archive_proxies),
            },
        )

    def export_full_progressive_map(self, path: str) -> None:
        """Export legacy seed, active detail, and archived fine children without GPU restoration."""
        param_sets = []
        if self.gaussian_model is not None:
            baseline = self.gaussian_model.export_raw_splats(exclude_group_ids=self.store.backend_group_ids())
            if baseline:
                param_sets.append(baseline)
        param_sets.extend(self.store.active_raw_params())
        for root in self.registry.root_nodes((NodeState.ARCHIVED,)):
            try:
                param_sets.append(self.archive_store.get(root.node_id).tensor_dict())
            except KeyError:
                proxy = self.store.archive_proxies.get(root.node_id)
                if proxy is not None:
                    param_sets.append(proxy)
        combined = concatenate_raw_params(param_sets)
        if not combined:
            raise RuntimeError("Cannot export an empty progressive map")
        if self.gaussian_model is not None:
            self.gaussian_model.save_raw_splats_as_ply(combined, path)
        else:
            torch.save(combined, path)
