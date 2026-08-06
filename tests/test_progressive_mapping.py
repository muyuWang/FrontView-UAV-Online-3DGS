"""CPU coverage for the progressive UAV mapping state machine."""

import json
import math

import pytest
import torch

from utils_new.progressive_mapping.budget_manager import BudgetManager
from utils_new.progressive_mapping.config import validate_progressive_config
from utils_new.progressive_mapping.geometry import (
    fronto_parallel_quaternion,
    parallax_angle,
    project_world,
    project_world_batch,
    quaternion_to_matrix,
    unproject_pixel,
)
from utils_new.progressive_mapping.observation_extractor import ObservationExtractor
from utils_new.progressive_mapping.progressive_manager import ProgressiveManager
from utils_new.progressive_mapping.projective_anchor_bank import ProjectiveAnchorBank
from utils_new.progressive_mapping.types import NodeState, Observation


def config(**overrides):
    values = {"enabled": True, "debug": False}
    values.update(overrides)
    return validate_progressive_config(values)


def image_and_observation(frame_id=0, uv=(32.5, 32.5), depth=4.0):
    y, x = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64), indexing="ij")
    image = torch.stack((x, y, 0.5 + 0.25 * torch.sin(10 * x) * torch.cos(8 * y)), dim=-1)
    extractor = ObservationExtractor(config())
    uv_tensor = torch.tensor(uv)
    descriptor, color = extractor.describe_patch(image, uv_tensor)
    appearance_grid = extractor.appearance_grid(image, uv_tensor)
    observation = Observation(
        frame_id=frame_id,
        uv=uv_tensor,
        patch_bbox=torch.tensor([24, 24, 40, 40]),
        descriptor=descriptor,
        mean_color=color,
        appearance_grid=appearance_grid,
        depth_prior=depth,
        depth_valid=True,
        depth_uncertainty=0.1,
        gradient_score=0.1,
        residual_score=0.2,
    )
    return image, extractor, observation


def camera_matrices(tx=0.0):
    pose = torch.eye(4)
    pose[0, 3] = tx
    intrinsics = torch.tensor([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]])
    return pose, intrinsics


def make_anchor(bank=None, frame_id=0, uv=(32.5, 32.5)):
    cfg = config() if bank is None else bank.config
    bank = ProjectiveAnchorBank(cfg) if bank is None else bank
    image, extractor, observation = image_and_observation(frame_id, uv)
    pose, intrinsics = camera_matrices()
    return bank, bank.create_anchor(observation, pose, intrinsics, 0.1, 100.0), image, extractor


def promotable(anchor):
    anchor.mode_log_weights = torch.log_softmax(torch.tensor([8.0, 0.0, -2.0, -4.0]), dim=0)
    anchor.observation_count = 5
    anchor.valid_update_count = 4
    anchor.max_parallax_rad = math.radians(3.0)
    anchor.best_error_ema = 0.05
    anchor.posterior_mean = float(anchor.inverse_depth_modes[0])
    anchor.posterior_variance = 0.0
    anchor.posterior_entropy = 0.01


def test_inverse_depth_mode_initialization():
    bank = ProjectiveAnchorBank(config())
    modes = bank.initialize_inverse_depth_modes(4.0, 0.1, 100.0, torch.device("cpu"))
    expected = 0.25 * torch.exp(torch.tensor([-0.7, -0.25, 0.25, 0.7]))
    assert modes.shape == (4,)
    assert torch.allclose(modes, expected)


def test_probability_update_is_normalized():
    bank, anchor, image, extractor = make_anchor()
    _, _, observation = image_and_observation(frame_id=1)
    pose, intrinsics = camera_matrices()
    bank.update_anchor(anchor, observation, image, pose, intrinsics, 0.1, 100.0, extractor)
    assert torch.allclose(torch.softmax(anchor.mode_log_weights, 0).sum(), torch.tensor(1.0))
    assert torch.isfinite(anchor.mode_log_weights).all()


def test_posterior_mean_variance_entropy():
    bank, anchor, _, _ = make_anchor()
    anchor.mode_log_weights = torch.log(torch.tensor([0.7, 0.2, 0.08, 0.02]))
    mean, variance, entropy = bank.posterior_statistics(anchor)
    expected_mean = float(torch.sum(torch.tensor([0.7, 0.2, 0.08, 0.02]) * anchor.inverse_depth_modes))
    assert math.isclose(mean, expected_mean, rel_tol=1e-6)
    assert variance > 0.0
    assert 0.0 < entropy < 1.0


def test_projection_unprojection_and_parallax():
    pose, intrinsics = camera_matrices()
    uv = torch.tensor([30.0, 35.0])
    point = unproject_pixel(uv, torch.tensor(5.0), intrinsics, pose)
    projected, depth, valid = project_world(point, pose, intrinsics, (64, 64), 0.1, 100.0)
    moved_pose, _ = camera_matrices(tx=-1.0)
    assert valid and torch.allclose(projected, uv, atol=1e-5)
    assert math.isclose(float(depth), 5.0)
    assert parallax_angle(point, pose, moved_pose) > 0.0


def test_batched_projection_matches_scalar_projection():
    pose, intrinsics = camera_matrices()
    points = torch.tensor(
        [[0.0, 0.0, 4.0], [1.0, -0.5, 5.0], [0.0, 0.0, -1.0]]
    )

    uv, depth, valid = project_world_batch(
        points, pose, intrinsics, (64, 64), 0.1, 100.0
    )

    scalar = [
        project_world(point, pose, intrinsics, (64, 64), 0.1, 100.0)
        for point in points
    ]
    assert torch.allclose(uv[:2], torch.stack([item[0] for item in scalar[:2]]))
    assert torch.allclose(depth, torch.stack([item[1] for item in scalar]))
    assert valid.tolist() == [item[2] for item in scalar]


def test_fronto_parallel_rotation_aligns_normal_to_camera_view_axis():
    angle = math.radians(35.0)
    camera_to_world = torch.eye(4)
    camera_to_world[:3, :3] = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    world_to_camera = torch.linalg.inv(camera_to_world)

    rotation = quaternion_to_matrix(
        fronto_parallel_quaternion(world_to_camera)
    )

    assert torch.allclose(rotation[:, 2], camera_to_world[:3, 2], atol=1.0e-5)


def test_sparse_plane_initialization_recovers_guarded_local_normal():
    cfg = config(
        enable_sparse_plane_initialization=True,
        sparse_plane_min_points=5,
        sparse_plane_max_relative_rmse=0.01,
        sparse_plane_min_confidence=0.7,
        sparse_plane_min_uv_variance=4.0,
        sparse_plane_max_tilt_deg=80.0,
    )
    bank = ProjectiveAnchorBank(cfg)
    image, extractor, observation = image_and_observation()
    pose, intrinsics = camera_matrices()
    expected_normal = torch.tensor([0.35, -0.15, 0.924])
    expected_normal = expected_normal / torch.linalg.norm(expected_normal)
    plane_point = torch.tensor([0.0, 0.0, 4.0])
    sparse_depth = torch.zeros((64, 64))
    for y in range(24, 41, 4):
        for x in range(24, 41, 4):
            ray = torch.linalg.solve(
                intrinsics, torch.tensor([x + 0.5, y + 0.5, 1.0])
            )
            depth = torch.dot(expected_normal, plane_point) / torch.dot(
                expected_normal, ray
            )
            sparse_depth[y, x] = depth
    observation.patch_bbox = torch.tensor([20, 20, 44, 44])
    observation.depth_prior = float(
        sparse_depth[sparse_depth > 0].median().item()
    )

    anchor = bank.create_anchor(
        observation,
        pose,
        intrinsics,
        0.1,
        100.0,
        sparse_depth=sparse_depth,
    )

    assert anchor.reference_surface_normal is not None
    assert anchor.reference_surface_support >= 5
    assert torch.dot(anchor.reference_surface_normal, expected_normal) > 0.99

    manager = ProgressiveManager(cfg, gaussian_model=None)
    manager.anchor_bank = bank
    promotable(anchor)
    root_id, _ = manager.promote_anchor(anchor.anchor_id, 2)
    root_normal = quaternion_to_matrix(
        manager.registry.nodes[root_id].root_rotation
    )[:, 2]
    assert torch.dot(root_normal, expected_normal) > 0.99
    root_scale = manager.registry.nodes[root_id].root_scale
    root_depth = 1.0 / anchor.posterior_mean
    fronto_scale = root_depth * cfg["patch_size"] / intrinsics[0, 0]
    assert torch.all(
        root_scale[:2]
        <= fronto_scale * cfg["sparse_plane_max_scale_multiplier"] + 1.0e-6
    )


def test_same_patch_reuses_projective_anchor():
    bank, anchor, image, extractor = make_anchor()
    _, _, observation = image_and_observation(frame_id=1)
    pose, intrinsics = camera_matrices()
    matches, claimed = bank.associate(
        [observation], image, pose, intrinsics, 0.1, 100.0, extractor
    )
    assert matches == {0: anchor.anchor_id}
    assert claimed == {0}
    assert len(bank.anchors) == 1
    assert anchor.observation_count == 2


def test_batched_patch_descriptors_match_scalar_calls():
    image, extractor, _ = image_and_observation()
    uvs = torch.tensor([[16.5, 20.5], [32.5, 32.5], [55.5, 50.5]])

    descriptors, colors = extractor.describe_patches(image, uvs)
    scalar = [extractor.describe_patch(image, uv) for uv in uvs]

    assert torch.allclose(descriptors, torch.stack([item[0] for item in scalar]))
    assert torch.allclose(colors, torch.stack([item[1] for item in scalar]))


def test_anchor_becomes_promotion_candidate():
    bank, anchor, _, _ = make_anchor()
    promotable(anchor)
    assert bank.is_promotion_candidate(anchor)


def test_empirical_commitment_score_rejects_weak_parallax():
    bank = ProjectiveAnchorBank(
        config(
            commitment_score_enabled=True,
            commitment_score_threshold=0.08,
            commitment_min_valid_updates=2,
        )
    )
    _, anchor, _, _ = make_anchor(bank=bank)
    promotable(anchor)
    anchor.max_parallax_rad = math.radians(0.1)

    assert not bank.is_promotion_candidate(anchor)
    assert anchor.commitment_score < 0.08


def test_empirical_commitment_score_accepts_supported_parallax():
    bank = ProjectiveAnchorBank(
        config(
            commitment_score_enabled=True,
            commitment_score_threshold=0.08,
            commitment_min_valid_updates=2,
        )
    )
    _, anchor, _, _ = make_anchor(bank=bank)
    promotable(anchor)

    assert bank.is_promotion_candidate(anchor)
    assert anchor.commitment_score >= 0.08


def test_near_depth_gets_relaxed_promotion_without_relaxing_far_anchor():
    cfg = config(
        promotion_min_observations=3,
        promotion_min_best_weight=0.55,
        promotion_max_match_error=0.40,
        near_promotion_max_depth_m=30.0,
        near_promotion_min_observations=2,
        near_promotion_min_best_weight=0.50,
        near_promotion_max_match_error=0.45,
    )
    near_bank = ProjectiveAnchorBank(cfg)
    _, near, _, _ = make_anchor(bank=near_bank)
    far_bank = ProjectiveAnchorBank(cfg)
    image, extractor, far_observation = image_and_observation(depth=80.0)
    pose, intrinsics = camera_matrices()
    far = far_bank.create_anchor(far_observation, pose, intrinsics, 0.1, 100.0)
    for anchor in (near, far):
        anchor.mode_log_weights = torch.log(
            torch.tensor([0.52, 0.20, 0.15, 0.13])
        )
        anchor.observation_count = 2
        anchor.max_parallax_rad = math.radians(3.0)
        anchor.best_error_ema = 0.44
        anchor.posterior_mean = float(anchor.inverse_depth_modes[0])
        anchor.posterior_variance = 0.0
        anchor.posterior_entropy = 0.1

    assert near_bank.is_promotion_candidate(near)
    assert not far_bank.is_promotion_candidate(far)


def test_observation_extractor_reserves_near_depth_quota():
    cfg = config(
        patch_stride=16,
        max_observations_per_frame=4,
        near_observation_depth_m=30.0,
        near_observation_fraction=0.5,
    )
    extractor = ObservationExtractor(cfg)
    image, _, _ = image_and_observation()
    depth = torch.full((64, 64), 80.0)
    depth[32:] = 10.0
    observations, _ = extractor.extract(
        0,
        image,
        depth,
        torch.zeros((64, 64)),
        torch.zeros((64, 64)),
    )

    assert len(observations) == 4
    assert all(observation.depth_prior <= 30.0 for observation in observations[:2])
    assert sum(observation.depth_prior <= 30.0 for observation in observations) == 2


def test_spawn_quota_preserves_non_near_projective_anchors():
    manager = ProgressiveManager(
        config(max_new_anchors_per_keyframe=4, near_spawn_fraction=0.5),
        gaussian_model=None,
    )
    observations = []
    for index, depth in enumerate((10.0, 12.0, 14.0, 16.0, 60.0, 70.0)):
        _, _, observation = image_and_observation(
            uv=(8.5 + index * 8.0, 32.5), depth=depth
        )
        observations.append(observation)

    selected = manager._select_spawn_observations(observations, set())

    assert len(selected) == 4
    assert sum(observation.depth_prior <= 30.0 for observation in selected) == 2
    assert sum(observation.depth_prior > 30.0 for observation in selected) == 2


def test_spawn_can_require_valid_depth():
    manager = ProgressiveManager(
        config(max_new_anchors_per_keyframe=4, spawn_requires_valid_depth=True),
        gaussian_model=None,
    )
    _, _, valid = image_and_observation(depth=10.0)
    _, _, invalid = image_and_observation(depth=10.0, uv=(48.5, 48.5))
    invalid.depth_valid = False

    selected = manager._select_spawn_observations([invalid, valid], set())

    assert selected == [valid]


def test_nearby_promotions_merge_into_one_metric_root():
    manager = ProgressiveManager(config(), gaussian_model=None)
    bank = manager.anchor_bank
    _, anchor1, _, _ = make_anchor(bank=bank, uv=(32.5, 32.5))
    _, anchor2, _, _ = make_anchor(bank=bank, uv=(32.6, 32.5))
    promotable(anchor1)
    promotable(anchor2)
    first_id, first_merged = manager.promote_anchor(anchor1.anchor_id, 5)
    second_id, second_merged = manager.promote_anchor(anchor2.anchor_id, 5)
    assert not first_merged and second_merged
    assert first_id == second_id
    assert manager.store.num_metric == 1


def test_metric_root_depth_correction_moves_center_and_updates_scale():
    manager = ProgressiveManager(
        config(
            enable_metric_depth_correction=True,
            metric_correction_min_depth_support=2,
            metric_correction_ema=0.5,
        ),
        gaussian_model=None,
    )
    _, anchor, _, _ = make_anchor(bank=manager.anchor_bank)
    promotable(anchor)
    root_id, _ = manager.promote_anchor(anchor.anchor_id, 5)
    root = manager.registry.nodes[root_id]
    _, _, observation = image_and_observation(frame_id=6, depth=5.0)
    observation.depth_support = 4
    pose, intrinsics = camera_matrices()
    projected_uv, projected_depth, valid = project_world(
        root.world_center, pose, intrinsics, (64, 64), 0.1, 100.0
    )
    old_center = root.world_center.clone()
    old_scale = root.root_scale.clone()

    corrected = manager._correct_metric_root_depth(
        root,
        observation,
        pose,
        intrinsics,
        projected_uv,
        projected_depth,
    )

    assert valid and corrected
    assert root.metric_depth_update_count == 1
    assert root.world_center[2] > old_center[2]
    assert root.root_scale[0] > old_scale[0]
    assert torch.allclose(
        manager.store.active_metric[root_id]["means"][0], root.world_center
    )


def test_metric_root_rejects_unsupported_depth_and_refine_wait_has_fallback():
    manager = ProgressiveManager(
        config(
            enable_metric_depth_correction=True,
            metric_correction_min_depth_support=3,
            metric_correction_min_updates_for_refine=1,
            metric_correction_max_wait_frames=2,
        ),
        gaussian_model=None,
    )
    _, anchor, _, _ = make_anchor(bank=manager.anchor_bank)
    promotable(anchor)
    root_id, _ = manager.promote_anchor(anchor.anchor_id, 5)
    root = manager.registry.nodes[root_id]
    _, _, observation = image_and_observation(frame_id=6, depth=5.0)
    observation.depth_support = 1
    pose, intrinsics = camera_matrices()
    projected_uv, projected_depth, _ = project_world(
        root.world_center, pose, intrinsics, (64, 64), 0.1, 100.0
    )

    corrected = manager._correct_metric_root_depth(
        root,
        observation,
        pose,
        intrinsics,
        projected_uv,
        projected_depth,
    )

    assert not corrected
    assert root.metric_depth_reject_count == 1
    assert not manager._metric_depth_ready_for_refine(root, 6)
    assert manager._metric_depth_ready_for_refine(root, 7)


def make_surface_manager():
    manager = ProgressiveManager(config(), gaussian_model=None)
    _, anchor, _, _ = make_anchor(bank=manager.anchor_bank)
    promotable(anchor)
    root_id, _ = manager.promote_anchor(anchor.anchor_id, 5)
    child_ids = manager.refine_root(root_id)
    return manager, root_id, child_ids


def test_metric_root_splits_into_four_surface_children():
    manager, root_id, child_ids = make_surface_manager()
    assert len(child_ids) == 4
    assert manager.store.num_surface == 4
    assert manager.registry.nodes[root_id].state == NodeState.SURFACE


def test_near_metric_root_uses_nine_surface_children():
    manager = ProgressiveManager(config(), gaussian_model=None)
    _, anchor, _, _ = make_anchor(bank=manager.anchor_bank)
    promotable(anchor)
    root_id, _ = manager.promote_anchor(anchor.anchor_id, 5)
    root = manager.registry.nodes[root_id]
    root.residual_ema = 0.2
    child_ids = manager.refine_root(root_id, current_depth=40.0)

    assert len(child_ids) == 9
    assert manager.store.num_surface == 9
    child_sh0 = manager.store.active_surface[root_id]["sh0"].reshape(9, 3)
    assert torch.unique(child_sh0, dim=0).shape[0] > 1


def test_very_near_high_residual_root_uses_sixteen_surface_children():
    manager = ProgressiveManager(config(), gaussian_model=None)
    _, anchor, _, _ = make_anchor(bank=manager.anchor_bank)
    promotable(anchor)
    root_id, _ = manager.promote_anchor(anchor.anchor_id, 5)
    root = manager.registry.nodes[root_id]
    root.residual_ema = 0.2
    child_ids = manager.refine_root(root_id, current_depth=10.0)

    assert len(child_ids) == 16
    assert manager.store.num_surface == 16


def test_surface_parent_is_not_in_active_metric_render_set():
    manager, root_id, _ = make_surface_manager()
    assert root_id not in manager.store.active_metric
    assert root_id in manager.store.active_surface
    assert manager.registry.nodes[root_id].metric_gaussian_id is None


def test_root_iteration_does_not_scan_surface_children():
    manager, root_id, child_ids = make_surface_manager()

    roots = manager.registry.root_nodes()

    assert [root.node_id for root in roots] == [root_id]
    assert all(child_id not in manager.registry._root_ids for child_id in child_ids)


def test_surface_creation_applies_configured_opacity_floor():
    manager = ProgressiveManager(
        config(surface_initial_opacity_floor=0.12), gaussian_model=None
    )
    _, anchor, _, _ = make_anchor(bank=manager.anchor_bank)
    promotable(anchor)
    root_id, _ = manager.promote_anchor(anchor.anchor_id, 5)
    params = manager.store.active_metric[root_id]
    params["opacities"].fill_(-10.0)
    manager.store.update_metric(root_id, params)

    manager.refine_root(root_id)
    child_opacity = torch.sigmoid(manager.store.active_surface[root_id]["opacities"])

    assert torch.all(child_opacity >= 0.12 - 1.0e-6)


class _OptimizationCamera:
    near = 0.1
    far = 100.0

    def __init__(self, x_translation=0.0):
        self.pose = torch.eye(4)
        self.pose[0, 3] = x_translation

    def get_pose(self):
        return self.pose

    def get_int_mat(self, level=0):
        del level
        return camera_matrices()[1]

    def get_width(self, level=0):
        del level
        return 64

    def get_height(self, level=0):
        del level
        return 64


class _OptimizationGroup:
    def __init__(self, splats=None):
        self.is_optimize = True
        self.splats = splats


class _OptimizationModel:
    def __init__(self, group_id, splats):
        self.gaussian_groups = [_OptimizationGroup() for _ in range(group_id + 1)]
        self.gaussian_groups[group_id] = _OptimizationGroup(splats)

    def remove_optimization(self, group_id):
        changed = self.gaussian_groups[group_id].is_optimize
        self.gaussian_groups[group_id].is_optimize = False
        return changed

    def add_optimization(self, group_id):
        changed = not self.gaussian_groups[group_id].is_optimize
        self.gaussian_groups[group_id].is_optimize = True
        return changed

    def merge_progressive_groups_into_baseline(self, level=0):
        del level
        return 1, sum(
            params["means"].shape[0]
            for params in (
                getattr(group, "splats", None) for group in self.gaussian_groups
            )
            if params is not None
        )


def test_visibility_gate_freezes_offscreen_root_without_removing_it():
    manager = ProgressiveManager(
        config(
            optimize_visible_roots_only=True,
            optimization_visibility_margin_px=0.0,
        ),
        gaussian_model=None,
    )
    _, anchor, _, _ = make_anchor(bank=manager.anchor_bank)
    promotable(anchor)
    root_id, _ = manager.promote_anchor(anchor.anchor_id, 5)
    manager.refine_root(root_id)
    group_id = manager.store.group_ids[root_id]
    model = _OptimizationModel(group_id, manager.store.active_surface[root_id])
    manager.gaussian_model = model

    visible = manager.configure_optimization_visibility(_OptimizationCamera())
    frozen = manager.configure_optimization_visibility(
        _OptimizationCamera(x_translation=100.0)
    )

    assert visible == {"visible": 1, "enabled": 1, "frozen": 0}
    assert frozen == {"visible": 0, "enabled": 0, "frozen": 1}
    assert root_id in manager.store.active_surface
    assert manager.store.group_ids[root_id] == group_id
    assert model.gaussian_groups[group_id].splats is not None


def test_surface_regularization_skips_frozen_roots():
    manager = ProgressiveManager(
        config(enable_center_regularization=True), gaussian_model=None
    )
    _, anchor, _, _ = make_anchor(bank=manager.anchor_bank)
    promotable(anchor)
    root_id, _ = manager.promote_anchor(anchor.anchor_id, 5)
    manager.refine_root(root_id)
    group_id = manager.store.group_ids[root_id]
    model = _OptimizationModel(group_id, manager.store.active_surface[root_id])
    model.gaussian_groups[group_id].is_optimize = False
    manager.gaussian_model = model

    loss = manager.surface_regularization_loss()

    assert loss.item() == 0.0


def test_archive_moves_surface_parameters_to_cpu_fp16():
    manager, root_id, _ = make_surface_manager()
    manager.archive_root(root_id, 20)
    detail = manager.archive_store.get(root_id)
    assert detail.means_fp16.device.type == "cpu"
    assert detail.means_fp16.dtype == torch.float16
    assert manager.store.num_surface == 0
    assert manager.registry.nodes[root_id].state == NodeState.ARCHIVED


def test_archive_reactivation_restores_surface_parameters():
    manager, root_id, child_ids = make_surface_manager()
    manager.archive_root(root_id, 20)
    restored = manager.reactivate_root(root_id)
    assert restored == child_ids
    assert manager.store.num_surface == 4
    assert root_id not in manager.store.archive_proxies
    assert manager.registry.nodes[root_id].state == NodeState.SURFACE


def test_budget_prunes_low_confidence_p_before_archiving_old_surface():
    cfg = config(max_projective_anchors=1, max_surface_gaussians=1, max_active_gaussians=1)
    manager = ProgressiveManager(cfg, gaussian_model=None)
    _, old_anchor, _, _ = make_anchor(bank=manager.anchor_bank)
    _, good_anchor, _, _ = make_anchor(bank=manager.anchor_bank, uv=(20.5, 20.5))
    old_anchor.static_confidence = 0.1
    good_anchor.static_confidence = 0.9
    prune = BudgetManager(cfg).anchor_prune_candidates(manager.anchor_bank.anchors.values(), 10)
    assert prune == [old_anchor.anchor_id]

    promotable(good_anchor)
    root_id, _ = manager.promote_anchor(good_anchor.anchor_id, 1)
    manager.refine_root(root_id)
    root = manager.registry.nodes[root_id]
    root.last_seen_frame = 0
    archive = BudgetManager(cfg).surface_archive_candidates(
        [root], 50, set(), manager.store.num_surface, manager.store.num_surface
    )
    assert archive == [root_id]


def test_budget_protects_young_p_and_prunes_mature_low_quality_track():
    cfg = config(
        max_projective_anchors=2,
        projective_prune_grace_frames=5,
        projective_prune_stale_frames=10,
    )
    bank = ProjectiveAnchorBank(cfg)
    _, mature_weak, _, _ = make_anchor(bank=bank, frame_id=0, uv=(18.5, 18.5))
    _, mature_supported, _, _ = make_anchor(
        bank=bank, frame_id=0, uv=(32.5, 32.5)
    )
    _, young, _, _ = make_anchor(bank=bank, frame_id=8, uv=(46.5, 46.5))
    mature_weak.observation_count = 1
    mature_weak.static_confidence = 0.1
    mature_supported.observation_count = 5
    mature_supported.static_confidence = 0.7
    young.observation_count = 1
    young.static_confidence = 0.0

    prune = BudgetManager(cfg).anchor_prune_candidates(bank.anchors.values(), 10)
    assert prune == [mature_weak.anchor_id]


def test_budget_stale_track_overrides_pruning_grace_period():
    cfg = config(
        max_projective_anchors=1,
        projective_prune_grace_frames=20,
        projective_prune_stale_frames=20,
    )
    bank = ProjectiveAnchorBank(cfg)
    _, stale, _, _ = make_anchor(bank=bank, frame_id=0, uv=(20.5, 20.5))
    _, young, _, _ = make_anchor(bank=bank, frame_id=24, uv=(40.5, 40.5))
    stale.last_seen_frame = 0
    young.last_seen_frame = 24

    prune = BudgetManager(cfg).anchor_prune_candidates(bank.anchors.values(), 25)
    assert prune == [stale.anchor_id]


def test_p_histogram_records_distributions_and_promotion_failures(tmp_path):
    cfg = config(debug=True, debug_save_interval=1)
    manager = ProgressiveManager(cfg, gaussian_model=None, output_dir=str(tmp_path))
    _, first, _, _ = make_anchor(bank=manager.anchor_bank, frame_id=0)
    _, second, _, _ = make_anchor(
        bank=manager.anchor_bank, frame_id=1, uv=(20.5, 20.5)
    )
    promotable(first)
    second.static_confidence = 0.0

    manager.debug_writer.write_p_histograms(
        5,
        manager.anchor_bank.anchors.values(),
        cfg,
        num_promoted=1,
        num_pruned=2,
        state_depth_bands={
            "edges_m": [20.0, 50.0],
            "counts": {"P": {"near": 1, "mid": 1, "far": 0}},
        },
    )
    record = json.loads((tmp_path / "progressive_p_histograms.jsonl").read_text())
    assert record["num_anchors"] == 2
    assert record["num_promoted_this_frame"] == 1
    assert record["num_pruned_this_frame"] == 2
    assert set(record["promotion_failures"]) == {
        "observations",
        "best_weight",
        "entropy",
        "relative_std",
        "parallax",
        "match_error",
    }
    assert record["near_anchor_count"] == 2
    assert record["state_depth_bands"]["counts"]["P"]["near"] == 1
    for histogram in record["histograms"].values():
        assert sum(histogram["counts"]) == 2


def test_pruning_stale_window_must_not_be_shorter_than_grace():
    with pytest.raises(ValueError, match="projective_prune_stale_frames"):
        config(
            projective_prune_grace_frames=20,
            projective_prune_stale_frames=10,
        )


def test_surface_child_counts_must_be_square_grids():
    with pytest.raises(ValueError, match="square grid"):
        config(near_surface_children=10)


def test_progressive_mapping_is_disabled_by_default():
    disabled = validate_progressive_config(None)
    manager = ProgressiveManager(disabled, gaussian_model=None)
    assert not manager.enabled
    assert manager.should_use_baseline_densification(processed_frames=1000)


def test_progressive_manager_accepts_native_3dgs_backend():
    class Model:
        gaussian_type = "3dgs"
        max_sh_degree = 0
        device = "cpu"

    manager = ProgressiveManager(config(), gaussian_model=Model())

    assert manager.enabled
    assert manager.gaussian_model.gaussian_type == "3dgs"


def test_quality_post_refinement_freezes_progressive_groups():
    manager, root_id, _ = make_surface_manager()
    group_id = manager.store.group_ids[root_id]
    model = _OptimizationModel(group_id, manager.store.active_surface[root_id])
    manager.gaussian_model = model
    manager.config["post_refinement_optimize_progressive"] = False

    frozen = manager.configure_post_refinement_optimization()

    assert frozen == 1
    assert not model.gaussian_groups[group_id].is_optimize


def test_quality_post_refinement_can_merge_progressive_groups():
    manager, root_id, child_ids = make_surface_manager()
    group_id = manager.store.group_ids[root_id]
    model = _OptimizationModel(group_id, manager.store.active_surface[root_id])
    manager.gaussian_model = model
    manager.store.gaussian_model = model
    manager.config["post_refinement_merge_into_baseline"] = True

    merged = manager.configure_post_refinement_optimization()

    assert merged == 1
    assert not manager.store.group_ids
    assert not manager.store.active_surface
    assert len(child_ids) == 4


def test_hybrid_mode_processes_progressive_frames_after_bootstrap():
    manager = ProgressiveManager(
        config(
            bootstrap_frames=5,
            replace_original_densification_after_bootstrap=False,
        ),
        gaussian_model=None,
    )
    assert manager.should_use_baseline_densification(processed_frames=6)
    assert manager.should_process_progressive_frame(processed_frames=6)
    assert not manager.should_process_progressive_frame(processed_frames=5)


def test_keyframe_only_processing_skips_non_keyframes():
    manager = ProgressiveManager(
        config(process_keyframes_only=True, bootstrap_frames=0),
        gaussian_model=None,
    )

    assert not manager.should_process_progressive_frame(1, is_keyframe=False)
    assert manager.should_process_progressive_frame(1, is_keyframe=True)


def test_progressive_processing_honors_causal_frame_interval():
    manager = ProgressiveManager(
        config(
            bootstrap_frames=0,
            process_keyframes_only=True,
            process_frame_interval=2,
        ),
        gaussian_model=None,
    )

    assert manager.should_process_progressive_frame(2, is_keyframe=True)
    assert manager.should_process_progressive_frame(2, is_keyframe=True)
    assert not manager.should_process_progressive_frame(4, is_keyframe=False)
    assert not manager.should_process_progressive_frame(4, is_keyframe=True)
    assert manager.should_process_progressive_frame(7, is_keyframe=True)


def test_surface_scale_limit_config_requires_ordered_factors():
    with pytest.raises(ValueError, match="surface_scale_min_factor"):
        config(surface_scale_min_factor=2.0, surface_scale_max_factor=1.0)


def test_surface_scale_constraint_clamps_to_initial_child_scale():
    manager, root_id, child_ids = make_surface_manager()
    params = manager.store.active_surface[root_id]
    params["scales"] = params["scales"] + math.log(10.0)
    manager.store.update_surface(root_id, params)

    clamped = manager.constrain_active_surface_scales()
    constrained = torch.exp(manager.store.active_surface[root_id]["scales"])
    initial = torch.stack(
        [manager.registry.nodes[child_id].root_scale for child_id in child_ids]
    )

    assert clamped == len(child_ids)
    assert torch.allclose(
        constrained,
        initial * manager.config["surface_scale_max_factor"],
    )


def test_surface_constraint_enforces_opacity_minimum():
    manager = ProgressiveManager(
        config(surface_opacity_min=0.08), gaussian_model=None
    )
    _, anchor, _, _ = make_anchor(bank=manager.anchor_bank)
    promotable(anchor)
    root_id, _ = manager.promote_anchor(anchor.anchor_id, 5)
    manager.refine_root(root_id)
    params = manager.store.active_surface[root_id]
    params["opacities"].fill_(-10.0)
    manager.store.update_surface(root_id, params)

    manager.constrain_active_surface_scales()
    constrained = torch.sigmoid(manager.store.active_surface[root_id]["opacities"])

    assert torch.all(constrained >= 0.08 - 1.0e-6)


def test_full_export_prefers_archived_children_over_coarse_proxy(tmp_path):
    manager, root_id, _ = make_surface_manager()
    manager.archive_root(root_id, 20)
    output = tmp_path / "full_map.pt"
    manager.export_full_progressive_map(str(output))
    exported = torch.load(output)
    assert exported["means"].shape[0] == 4
