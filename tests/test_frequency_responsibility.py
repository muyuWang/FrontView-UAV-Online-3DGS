import numpy as np
import pytest
import torch

from utils_new.aerocommit.frequency_responsibility import (
    PLAYERS,
    all_coalitions,
    checkerboard_masks,
    clip_whitened_update,
    evaluate_shadow_coalitions,
    exact_shapley_values,
    frequency_weighted_pose_loss,
    geometry_responsibility_decision,
    injected_defect_game,
    laplacian_pyramid_reconstruction_loss,
    two_level_laplacian_residual,
)
from utils_new.aerocommit.frequency_sampling import (
    frequency_evidence_map,
    frequency_footprint_log_offset,
    sample_frequency_balanced_indices,
)
from utils_new.aerocommit.sparse_track_geometry import (
    conditional_scale_expansion_limits,
    zbuffer_sparse_tracks,
)
from utils_new.aerocommit.sparse_flow_detail import triangulate_sparse_flow_detail
from utils_new.aerocommit.stable_detail_split import (
    split_gaussian_parameters,
    stable_detail_split_scores,
)
from utils_new.aerocommit.track_detail import (
    SparseTrackDetailAccumulator,
    StableSurfaceDetailSampler,
)
from utils_new.aerocommit.view_detail import ViewConditionedDetailStore
from utils_new.aerocommit.frequency_cache import FrequencyResidualCache
from utils_new.gaussian_models import Gaussians
from utils_new.loss_utils import RGBLoss, masked_ssim_loss


def test_exact_shapley_recovers_additive_loss_reductions():
    gains = {"P": 0.1, "G": 0.4, "A": 0.05}
    losses = {
        coalition: np.asarray([1.0 - sum(gains[player] for player in coalition)])
        for coalition in all_coalitions()
    }
    assert exact_shapley_values(losses) == pytest.approx(gains)


def test_exact_shapley_efficiency_with_interaction():
    losses = {}
    for coalition in all_coalitions():
        gain = 0.1 * len(coalition)
        if {"G", "A"}.issubset(coalition):
            gain += 0.3
        losses[coalition] = np.asarray([1.0 - gain])
    values = exact_shapley_values(losses)
    assert sum(values.values()) == pytest.approx(
        losses[frozenset()][0] - losses[frozenset(PLAYERS)][0]
    )
    assert values["G"] == pytest.approx(0.25)
    assert values["A"] == pytest.approx(0.25)


@pytest.mark.parametrize("defect", PLAYERS)
def test_known_source_defect_is_assigned_to_the_injected_block(defect):
    values = exact_shapley_values(injected_defect_game(defect))
    assert max(values, key=values.get) == defect


def test_geometry_gate_requires_support_dominance_and_view_stability():
    decision = geometry_responsibility_decision(
        injected_defect_game("G"), support_frames=3
    )
    assert decision.eligible
    assert decision.reason == "eligible"
    assert min(decision.leave_one_view_out_geometry) > 0.0

    assert not geometry_responsibility_decision(
        injected_defect_game("G"), support_frames=2
    ).eligible
    assert not geometry_responsibility_decision(
        injected_defect_game("P"), support_frames=3
    ).eligible


def test_geometry_gate_rejects_exact_responsibility_ties():
    losses = {
        coalition: np.asarray([1.0 - 0.2 * len(coalition)] * 3)
        for coalition in all_coalitions()
    }
    decision = geometry_responsibility_decision(losses, support_frames=3)
    assert not decision.eligible
    assert decision.reason == "geometry_not_dominant"


def test_missing_coalition_is_rejected():
    losses = injected_defect_game("G")
    losses.pop(frozenset(("P", "G", "A")))
    with pytest.raises(ValueError, match="every coalition"):
        exact_shapley_values(losses)


def test_trust_radius_is_measured_from_the_frozen_base():
    base = np.zeros(2, dtype=np.float32)
    clipped = clip_whitened_update(
        base, np.asarray([3.0, 4.0]), np.ones(2), radius=2.0
    )
    assert clipped == pytest.approx(np.asarray([1.2, 1.6]))
    second = clip_whitened_update(
        base, clipped + np.asarray([3.0, 4.0]), np.ones(2), radius=2.0
    )
    assert np.linalg.norm(second - base) == pytest.approx(2.0)


def test_checkerboard_fit_and_score_pixels_are_disjoint_and_complete():
    fit, score = checkerboard_masks(16, 16, phase=1)
    assert not np.any(fit & score)
    assert np.all(fit | score)
    assert int(fit.sum()) == int(score.sum()) == 128


def test_shadow_coalitions_receive_fresh_copies_only():
    live = {"means": np.zeros((2, 3), dtype=np.float32)}

    def evaluator(coalition, sandbox):
        assert np.all(sandbox["means"] == 0.0)
        sandbox["means"][:] = len(coalition) + 1
        return [1.0 - 0.1 * len(coalition)]

    results = evaluate_shadow_coalitions(evaluator, live)
    assert len(results) == 8
    assert np.all(live["means"] == 0.0)


def test_two_level_laplacian_residual_is_zero_only_for_matching_images():
    target = torch.zeros((3, 16, 16))
    prediction = target.clone()
    fine, coarse = two_level_laplacian_residual(prediction, target)
    assert torch.count_nonzero(fine) == 0
    assert torch.count_nonzero(coarse) == 0

    prediction[:, :, 8:] = 1.0
    fine, coarse = two_level_laplacian_residual(prediction, target)
    assert fine.sum() > 0
    assert coarse.sum() > 0


def test_zero_weight_rgb_loss_skips_ssim(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("SSIM must be skipped when its weight is zero")

    monkeypatch.setattr("utils_new.loss_utils.fused_ssim", fail_if_called)
    loss_fn = RGBLoss(
        {
            "lambda_ssim": 0.0,
            "mu": 10.0,
            "use_tonemap": False,
            "color_loss_type": "l2",
        }
    )
    prediction = torch.zeros((1, 4, 5, 3), requires_grad=True)
    target = torch.ones_like(prediction)
    loss, _ = loss_fn(prediction, target)
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert prediction.grad is not None


def test_masked_ssim_blocks_gradients_outside_mask(monkeypatch):
    def fake_ssim(prediction, target, padding):
        assert padding == "valid"
        return 1.0 - (prediction - target).square().mean()

    monkeypatch.setattr("utils_new.loss_utils.fused_ssim", fake_ssim)
    prediction = torch.zeros((1, 4, 6, 3), requires_grad=True)
    target = torch.ones_like(prediction)
    mask = torch.zeros((1, 4, 6), dtype=torch.bool)
    mask[:, :, 3:] = True
    loss = masked_ssim_loss(prediction, target, mask)
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:, :, :3]) == 0
    assert torch.count_nonzero(prediction.grad[:, :, 3:]) > 0


def test_laplacian_pyramid_loss_matches_rgb_bands_and_backpropagates():
    target = torch.zeros((1, 12, 16, 3))
    target[:, :, 8:] = torch.tensor((1.0, 0.5, 0.25))
    prediction = torch.zeros_like(target, requires_grad=True)
    weight = torch.ones((1, 12, 16))
    loss = laplacian_pyramid_reconstruction_loss(
        prediction, target, weight, fine_weight=1.0, coarse_weight=0.25
    )
    assert loss > 0
    loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) > 0


def test_laplacian_pyramid_loss_respects_stable_pixel_mask():
    target = torch.zeros((3, 12, 16))
    prediction = target.clone()
    prediction[:, :, 8:] = 1.0
    zero_weight = torch.zeros((12, 16))
    assert laplacian_pyramid_reconstruction_loss(
        prediction, target, zero_weight
    ).item() == pytest.approx(0.0)

    edge_weight = torch.zeros((12, 16))
    edge_weight[:, 7:9] = 1.0
    assert laplacian_pyramid_reconstruction_loss(
        prediction, target, edge_weight
    ).item() > 0.0


def test_frequency_pose_loss_emphasizes_side_edges_and_backpropagates():
    target = torch.zeros((1, 12, 20, 3))
    target[:, :, 16:] = 1.0
    prediction = target.clone()
    prediction[:, :, 15:] = 0.0
    prediction.requires_grad_(True)
    valid = torch.ones((1, 12, 20, 1))

    uniform = frequency_weighted_pose_loss(
        prediction, target, valid, edge_weight=0.0, side_boost=1.0
    )
    emphasized = frequency_weighted_pose_loss(
        prediction, target, valid, edge_weight=2.0, side_boost=3.0
    )

    assert emphasized > uniform
    emphasized.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) > 0


def test_frequency_pose_loss_ignores_invalid_pixels():
    target = torch.zeros((1, 8, 8, 3))
    prediction = target.clone()
    prediction[:, :, 4:] = 1.0
    valid = torch.zeros((1, 8, 8, 1))
    assert frequency_weighted_pose_loss(prediction, target, valid).item() == 0.0


def test_frequency_sampling_is_unique_and_prefers_stable_side_detail():
    image = torch.zeros((16, 24, 3))
    image[:, 18:] = torch.arange(6).view(1, 6, 1) % 2
    residual = torch.full((16, 24), 0.10)
    opacity = torch.zeros((16, 24, 1))
    opacity[:, 18:] = 0.8
    importance, footprint, admission = frequency_evidence_map(
        image,
        residual,
        opacity,
        {
            "frequency_fraction": 1.0,
            "side_start": 0.4,
            "side_boost": 2.0,
            "stable_opacity_boost": 2.0,
        },
    )
    assert importance[:, 18:].mean() > importance[:, :12].mean()
    assert footprint[:, 18:].mean() > 0.0
    assert admission[:, 18:].mean() > admission[:, :12].mean()
    torch.manual_seed(7)
    selected = sample_frequency_balanced_indices(
        torch.ones((16, 24), dtype=torch.bool), importance, 80
    )
    assert selected.unique().numel() == 80
    assert torch.count_nonzero((selected % 24) >= 18) > 20


def test_frequency_footprint_offset_only_shrinks_supported_detail():
    offset = frequency_footprint_log_offset(torch.tensor([0.0, 0.5, 1.0]), 0.30)
    assert offset[0].item() == pytest.approx(0.0)
    assert offset[1] < 0.0
    assert offset[2].item() == pytest.approx(np.log(0.7), rel=1.0e-6)


def test_sparse_track_zbuffer_preserves_world_geometry_and_nearest_surface():
    points = np.asarray(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 4.0],
            [1.0, 0.0, 2.0],
            [-10.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)
    intrinsics = np.asarray(
        [[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    observations = zbuffer_sparse_tracks(points, pose, intrinsics, 5, 5)
    assert observations.world_points.shape == (2, 3)
    assert observations.world_points[0] == pytest.approx(points[0])
    assert observations.depths.tolist() == pytest.approx([2.0, 2.0])
    assert len(np.unique(observations.pixel_indices)) == 2


def test_track_detail_requires_repeated_side_frequency_evidence():
    accumulator = SparseTrackDetailAccumulator(
        {
            "track_quantization": 1.0e-4,
            "min_support_views": 3,
            "max_observations_per_track": 8,
            "gradient_threshold": 0.1,
            "side_start": 0.5,
            "near_depth_m": 10.0,
            "color_mad_threshold": 0.1,
            "projected_scale_px": 0.5,
            "max_commits_per_frame": 4,
            "max_total_gaussians": 4,
            "side_score_boost": 1.0,
            "near_score_boost": 1.0,
        }
    )
    point = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    color = np.asarray([[0.2, 0.4, 0.6]], dtype=np.float32)
    for frame_id in range(2):
        batch = accumulator.observe(
            frame_id,
            point,
            color,
            np.asarray([4.0]),
            np.asarray([0.2]),
            np.asarray([0.8]),
            focal=100.0,
        )
        assert len(batch) == 0
    batch = accumulator.observe(
        2,
        point + 2.0e-5,
        color,
        np.asarray([4.0]),
        np.asarray([0.2]),
        np.asarray([0.8]),
        focal=100.0,
    )
    assert len(batch) == 1
    assert batch.world_points[0] == pytest.approx(point[0])
    assert np.exp(batch.log_scales[0, 0]) == pytest.approx(0.02)


def test_track_detail_rejects_occlusion_color_inconsistency():
    accumulator = SparseTrackDetailAccumulator(
        {
            "track_quantization": 1.0e-4,
            "min_support_views": 3,
            "max_observations_per_track": 8,
            "gradient_threshold": 0.1,
            "side_start": 0.5,
            "near_depth_m": 10.0,
            "color_mad_threshold": 0.05,
            "projected_scale_px": 0.5,
            "max_commits_per_frame": 4,
            "max_total_gaussians": 4,
            "side_score_boost": 1.0,
            "near_score_boost": 1.0,
        }
    )
    point = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    for frame_id, value in enumerate((0.0, 0.5, 1.0)):
        batch = accumulator.observe(
            frame_id,
            point,
            np.full((1, 3), value, dtype=np.float32),
            np.asarray([4.0]),
            np.asarray([0.2]),
            np.asarray([0.8]),
            focal=100.0,
        )
    assert len(batch) == 0


def test_stable_surface_detail_deduplicates_voxels_and_respects_budget():
    sampler = StableSurfaceDetailSampler(
        {
            "voxel_size": 0.1,
            "max_commits_per_keyframe": 2,
            "max_total_gaussians": 3,
        }
    )
    points = np.asarray(
        [[0.01, 0.01, 1.0], [0.02, 0.02, 1.0], [0.2, 0.0, 1.0]],
        dtype=np.float32,
    )
    colors = np.eye(3, dtype=np.float32)
    scales = np.zeros((3, 1), dtype=np.float32)
    batch = sampler.select(points, colors, scales, np.asarray([0.1, 0.9, 0.5]))
    assert len(batch) == 2
    assert batch.colors[0] == pytest.approx(colors[1])
    repeated = sampler.select(points, colors, scales, np.ones(3))
    assert len(repeated) == 0


def test_sparse_flow_detail_triangulates_consistent_side_corners():
    import cv2

    height, width = 96, 128
    image0 = np.zeros((height, width, 3), dtype=np.float32)
    image1 = np.zeros_like(image0)
    for x in range(12, 116, 16):
        for y in range(20, 84, 16):
            cv2.rectangle(image0, (x - 2, y - 2), (x + 2, y + 2), (1, 1, 1), -1)
            cv2.rectangle(image1, (x - 6, y - 2), (x - 2, y + 2), (1, 1, 1), -1)
    intrinsics = np.asarray(
        [[100.0, 0.0, 64.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]]
    )
    pose0 = np.eye(4)
    pose1 = np.eye(4)
    pose1[0, 3] = -0.2
    result = triangulate_sparse_flow_detail(
        image0,
        image1,
        pose0,
        pose1,
        intrinsics,
        {
            "side_start": 0.0,
            "vertical_start": 0.0,
            "max_corners": 200,
            "quality_level": 0.001,
            "min_corner_distance_px": 3.0,
            "corner_block_size": 3,
            "lk_window_size": 21,
            "lk_max_level": 3,
            "lk_iterations": 30,
            "lk_epsilon": 0.01,
            "forward_backward_threshold_px": 1.0,
            "near_depth_m": 20.0,
            "min_parallax_deg": 0.1,
            "max_parallax_deg": 10.0,
            "color_consistency_threshold": 0.1,
        },
    )
    assert len(result.world_points) > 8
    assert np.median(result.depths) == pytest.approx(5.0, abs=0.5)


def test_gaussian_scale_trust_region_clamps_expansion_from_initial_size():
    config = {
        "means_lr_init": 1.0e-4,
        "means_lr_final": 1.0e-4,
        "scales_lr_init": 1.0e-3,
        "scales_lr_final": 1.0e-3,
        "quats_lr_init": 1.0e-3,
        "quats_lr_final": 1.0e-3,
        "opacities_lr": 1.0e-2,
        "sh_lr": 1.0e-3,
        "lr_final_step": 1,
        "max_scale_expansion": 2.0,
    }
    gaussians = Gaussians(BS=1, init_config=config, max_sh_degree=1)
    gaussians.extend_gaussians_from_color_points(
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
        np.asarray([[np.log(0.1)]], dtype=np.float32),
    )
    with torch.no_grad():
        gaussians.splats["scales"][-1] = np.log(1.0)
    changed = gaussians.constrain_scale_expansion()
    assert changed == 3
    assert gaussians.get_scaling[-1].tolist() == pytest.approx([0.2, 0.2, 0.2])


def test_conditional_scale_limits_leave_low_frequency_rows_unbounded():
    limits = conditional_scale_expansion_limits(
        np.asarray([0.2, 0.65, 0.9], dtype=np.float32),
        {
            "conditional_scale_control": {
                "enabled": True,
                "frequency_score_threshold": 0.65,
                "max_scale_expansion": 4.0,
            }
        },
    )
    assert np.isinf(limits[0])
    assert limits[1:].tolist() == pytest.approx([4.0, 4.0])


def test_per_gaussian_scale_limits_only_clamp_attributed_rows():
    config = {
        "means_lr_init": 1.0e-4,
        "means_lr_final": 1.0e-4,
        "scales_lr_init": 1.0e-3,
        "scales_lr_final": 1.0e-3,
        "quats_lr_init": 1.0e-3,
        "quats_lr_final": 1.0e-3,
        "opacities_lr": 1.0e-2,
        "sh_lr": 1.0e-3,
        "lr_final_step": 1,
    }
    gaussians = Gaussians(BS=1, init_config=config, max_sh_degree=1)
    gaussians.extend_gaussians_from_color_points(
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        np.full((2, 3), 0.5, dtype=np.float32),
        np.full((2, 1), np.log(0.1), dtype=np.float32),
        max_scale_expansion=np.asarray([4.0, np.inf], dtype=np.float32),
    )
    with torch.no_grad():
        gaussians.splats["scales"][-2:] = np.log(1.0)
    changed = gaussians.constrain_scale_expansion()
    assert changed == 3
    assert gaussians.get_scaling[-2].tolist() == pytest.approx([0.4, 0.4, 0.4])
    assert gaussians.get_scaling[-1].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_stable_detail_scores_require_large_near_side_frequency_support():
    image = torch.zeros((10, 20, 3))
    image[:, 16:] = 1.0
    info = {
        "gaussian_ids": torch.tensor([3, 7]),
        "means2d": torch.tensor([[16.0, 5.0], [10.0, 5.0]]),
        "radii": torch.tensor([[8, 7], [8, 7]]),
        "depths": torch.tensor([10.0, 10.0]),
    }
    ids, scores = stable_detail_split_scores(
        image,
        info,
        {
            "near_depth_m": 60.0,
            "min_projected_radius_px": 6.0,
            "gradient_threshold": 0.04,
            "side_start": 0.45,
        },
    )
    assert ids.tolist() == [3, 7]
    assert torch.isfinite(scores[0])
    assert not torch.isfinite(scores[1])


def test_stable_detail_scores_require_render_residual_when_configured():
    image = torch.zeros((10, 20, 3))
    image[:, 16:] = 1.0
    rendered = image.clone()
    rendered[:, 16:] = 0.8
    info = {
        "gaussian_ids": torch.tensor([3, 7]),
        "means2d": torch.tensor([[16.0, 5.0], [17.0, 5.0]]),
        "radii": torch.tensor([[8, 7], [8, 7]]),
        "depths": torch.tensor([10.0, 10.0]),
    }
    rendered[5, 17] = image[5, 17]
    ids, scores = stable_detail_split_scores(
        image,
        info,
        {
            "near_depth_m": 60.0,
            "min_projected_radius_px": 6.0,
            "gradient_threshold": 0.04,
            "residual_threshold": 0.1,
            "side_start": 0.45,
        },
        rendered=rendered,
    )
    assert ids.tolist() == [3, 7]
    assert torch.isfinite(scores[0])
    assert not torch.isfinite(scores[1])


def test_stable_detail_split_preserves_parent_opacity_and_replaces_one_by_four():
    params = {
        "means": torch.tensor([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
        "scales": torch.full((2, 3), np.log(0.5)),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
        "opacities": torch.logit(torch.tensor([0.8, 0.4])),
        "sh0": torch.zeros((2, 1, 3)),
        "shN": torch.zeros((2, 3, 3)),
    }
    split = split_gaussian_parameters(
        params,
        torch.tensor([0]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        scale_ratio=0.5,
    )
    assert split["means"].shape[0] == 5
    child_alpha = torch.sigmoid(split["opacities"][-4:])
    combined_alpha = 1.0 - torch.prod(1.0 - child_alpha)
    assert combined_alpha.item() == pytest.approx(0.8, rel=1.0e-5)
    assert torch.exp(split["scales"][-4:]).max().item() == pytest.approx(0.25)
    assert torch.unique(split["means"][-4:], dim=0).shape[0] == 4


def test_view_detail_selects_nearest_non_exact_source_and_builds_sh_features():
    source_poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    source_poses[1, 0, 3] = -1.0
    store = ViewConditionedDetailStore(
        means=np.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]], dtype=np.float32),
        scales=np.full((2, 3), 0.01, dtype=np.float32),
        colors=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        opacities=np.asarray([0.8, 0.7], dtype=np.float32),
        source_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        source_frame_indices=np.asarray([0, 4], dtype=np.int64),
        source_poses=source_poses,
        metadata={"translation_scale": 1.0, "orientation_weight": 2.0},
        device="cpu",
        sh_degree=1,
    )
    assert store.nearest_sources(np.eye(4), 0, exclude_exact=False).tolist() == [0]
    assert store.nearest_sources(np.eye(4), 0, exclude_exact=True).tolist() == [1]
    splats = store.external_splats_for_pose(
        np.eye(4), target_frame_index=0, exclude_exact=True
    )
    assert splats["means"].tolist() == [[1.0, 0.0, 2.0]]
    assert splats["shs"].shape == (1, 4, 3)
    assert all(not value.requires_grad for value in splats.values())


def test_frequency_cache_places_side_bands_and_can_exclude_exact_source():
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    poses[1, 0, 3] = -1.0
    bands = np.zeros((2, 2, 4, 3), dtype=np.int8)
    bands[0, :, :2] = 10
    bands[0, :, 2:] = -10
    bands[1] = 20
    cache = FrequencyResidualCache(
        residual_bands=bands,
        source_frame_indices=np.asarray([0, 4]),
        source_poses=poses,
        image_height=4,
        image_width=8,
        side_band_width=2,
        vertical_start=2,
        quantization_scale=100.0,
        metadata={"translation_scale": 1.0, "orientation_weight": 2.0},
    )
    exact = cache.residual_for_pose(
        np.eye(4), target_frame_index=0, exclude_exact=False, device="cpu"
    )
    heldout = cache.residual_for_pose(
        np.eye(4), target_frame_index=0, exclude_exact=True, device="cpu"
    )
    assert exact[:2].count_nonzero().item() == 0
    assert exact[2:, :2].mean().item() == pytest.approx(0.1)
    assert exact[2:, -2:].mean().item() == pytest.approx(-0.1)
    assert exact[2:, 2:-2].count_nonzero().item() == 0
    assert heldout[2:, :2].mean().item() == pytest.approx(0.2)


def test_frequency_cache_identity_warp_preserves_residual_and_masks_invalid_depth():
    bands = np.zeros((1, 2, 4, 3), dtype=np.int8)
    bands[0, :, :2] = 10
    bands[0, :, 2:] = -10
    cache = FrequencyResidualCache(
        residual_bands=bands,
        source_frame_indices=np.asarray([0]),
        source_poses=np.eye(4, dtype=np.float32)[None],
        image_height=4,
        image_width=8,
        side_band_width=2,
        vertical_start=2,
        quantization_scale=100.0,
    )
    depth = torch.ones((4, 8), dtype=torch.float32)
    depth[3, 0] = 0.0
    intrinsics = torch.tensor(
        [[2.0, 0.0, 4.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]
    )
    warped = cache.residual_for_pose(
        np.eye(4),
        target_frame_index=0,
        exclude_exact=False,
        device="cpu",
        target_depth=depth,
        target_intrinsics=intrinsics,
        warp_to_target=True,
    )
    assert warped[2, :2].mean().item() == pytest.approx(0.1)
    assert warped[2, -2:].mean().item() == pytest.approx(-0.1)
    assert warped[2, 2:-2].count_nonzero().item() == 0
    assert warped[3, 0].count_nonzero().item() == 0
