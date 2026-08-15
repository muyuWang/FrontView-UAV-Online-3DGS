import numpy as np
import pytest
import torch

from utils_new.frontview_coverage_recovery import (
    SparseFarDepthPrior,
    apply_sparse_depth_prior_fallback,
    apply_visible_surface_depth_fallback,
    coverage_recovery_certificate,
    inverse_depth_hypotheses,
    motion_conditioned_depth_floor,
    posterior_inverse_depth_fusion,
    pose_novelty,
    projective_coverage_debt,
    projective_exposure_budget,
    residual_grid_indices,
    validate_front_view_coverage_recovery_config,
)


def _pose(center, yaw_deg=0.0):
    angle = np.deg2rad(yaw_deg)
    rotation = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = -rotation @ np.asarray(center, dtype=np.float64)
    return result


def test_coverage_recovery_config_rejects_zero_pose_novelty():
    with pytest.raises(ValueError, match="pose novelty"):
        validate_front_view_coverage_recovery_config(
            {"min_translation_m": 0.0, "min_rotation_deg": 0.0}
        )


def test_coverage_recovery_config_accepts_disabled_newborn_refinement():
    config = validate_front_view_coverage_recovery_config(
        {"newborn_optimization_iters": 0, "newborn_max_scale_expansion": 1.0}
    )
    assert config["newborn_optimization_iters"] == 0
    assert config["newborn_max_scale_expansion"] == pytest.approx(1.0)
    assert config["tracking_update_interval"] == 0
    assert config["tracking_window_frames"] == 1
    assert config["trigger_mode"] == "fixed_gap"
    assert config["multiview_depth_enabled"] is False


def test_projective_debt_uses_birth_budget_cell_spacing():
    result = projective_coverage_debt(
        translation_m=1.0,
        rotation_deg=0.0,
        focal_px=640.0,
        reference_depth_m=40.0,
        image_size=(1280, 720),
        birth_budget=3600,
    )
    assert result["cell_px"] == pytest.approx(16.0)
    assert result["translation_px"] == pytest.approx(16.0)
    assert result["debt"] == pytest.approx(1.0)


def test_projective_exposure_budget_counts_new_boundary_cells():
    assert projective_exposure_budget((1280, 720), 3200, 0.0) == 1
    assert projective_exposure_budget((1280, 720), 3200, 16.0) == 112
    assert projective_exposure_budget((1280, 720), 3200, 1280.0) == 3200


def test_inverse_depth_hypotheses_include_motion_bound_and_are_ordered():
    hypotheses = inverse_depth_hypotheses(
        [8.0, 10.0, 12.0, 20.0, 40.0], 5, far_depth_m=200.0
    )
    assert len(hypotheses) == 5
    assert np.all(np.diff(hypotheses) > 0.0)
    assert hypotheses[-1] == pytest.approx(200.0)
    inverse_steps = np.diff(1.0 / hypotheses)
    assert np.max(inverse_steps) - np.min(inverse_steps) < 1.0e-6


def test_posterior_inverse_depth_keeps_fallback_for_flat_evidence():
    fused, confidence, supported = posterior_inverse_depth_fusion(
        torch.ones((3, 2)),
        torch.ones((3, 2), dtype=torch.bool),
        torch.tensor([10.0, 20.0, 100.0]),
        torch.tensor([80.0, 120.0]),
    )
    assert fused.tolist() == pytest.approx([80.0, 120.0])
    assert confidence.tolist() == pytest.approx([0.0, 0.0], abs=1.0e-6)
    assert supported.tolist() == [True, True]


def test_posterior_inverse_depth_uses_concentrated_supported_evidence():
    fused, confidence, supported = posterior_inverse_depth_fusion(
        torch.tensor([[0.0], [1.0], [0.0]]),
        torch.ones((3, 1), dtype=torch.bool),
        torch.tensor([10.0, 20.0, 100.0]),
        torch.tensor([80.0]),
    )
    assert fused.item() == pytest.approx(20.0)
    assert confidence.item() == pytest.approx(1.0)
    assert supported.item() is True


def test_pose_novelty_uses_camera_centers_and_rotation():
    translation, rotation = pose_novelty(
        _pose([0.0, 0.0, 0.0]), _pose([2.0, 0.0, 0.0], yaw_deg=5.0)
    )
    assert translation == pytest.approx(2.0)
    assert rotation == pytest.approx(5.0)


def test_motion_conditioned_depth_floor_limits_translational_parallax():
    assert motion_conditioned_depth_floor(4.0, 640.0, 8.0, 500.0) == pytest.approx(
        320.0
    )
    assert motion_conditioned_depth_floor(10.0, 640.0, 8.0, 500.0) == pytest.approx(
        500.0
    )


def test_recovery_requires_interval_pose_novelty_and_failed_coverage():
    config = validate_front_view_coverage_recovery_config({"enabled": True})
    target = torch.ones((10, 10, 3))
    rendered = target.clone()
    rendered[:2] = 0.0
    opacity = torch.ones((10, 10, 1))
    certificate = coverage_recovery_certificate(
        frame_id=40,
        last_keyframe_id=20,
        last_world_to_camera=_pose([0.0, 0.0, 0.0]),
        current_world_to_camera=_pose([2.0, 0.0, 0.0]),
        rendered=rendered,
        target=target,
        opacity=opacity,
        config=config,
    )
    assert certificate["failure_fraction"] == pytest.approx(0.2)
    assert certificate["admitted"] is True

    certificate["frame_gap"] = 19
    rejected = coverage_recovery_certificate(
        frame_id=39,
        last_keyframe_id=20,
        last_world_to_camera=_pose([0.0, 0.0, 0.0]),
        current_world_to_camera=_pose([2.0, 0.0, 0.0]),
        rendered=rendered,
        target=target,
        opacity=opacity,
        config=config,
    )
    assert rejected["admitted"] is False


def test_projective_debt_replaces_fixed_frame_interval():
    config = validate_front_view_coverage_recovery_config(
        {
            "enabled": True,
            "trigger_mode": "projective_debt",
            "min_translation_m": 0.0,
            "min_rotation_deg": 0.0,
        }
    )
    target = torch.ones((10, 10, 3))
    rendered = torch.zeros_like(target)
    opacity = torch.ones((10, 10))
    rejected = coverage_recovery_certificate(
        frame_id=2,
        last_keyframe_id=1,
        last_world_to_camera=_pose([0.0, 0.0, 0.0]),
        current_world_to_camera=_pose([0.1, 0.0, 0.0]),
        rendered=rendered,
        target=target,
        opacity=opacity,
        config=config,
        projective_debt=0.9,
    )
    admitted = coverage_recovery_certificate(
        frame_id=2,
        last_keyframe_id=1,
        last_world_to_camera=_pose([0.0, 0.0, 0.0]),
        current_world_to_camera=_pose([0.1, 0.0, 0.0]),
        rendered=rendered,
        target=target,
        opacity=opacity,
        config=config,
        projective_debt=1.0,
    )
    assert rejected["admitted"] is False
    assert admitted["admitted"] is True


def test_low_opacity_can_recover_an_uncovered_view():
    config = validate_front_view_coverage_recovery_config({"enabled": True})
    target = torch.zeros((10, 10, 3))
    certificate = coverage_recovery_certificate(
        frame_id=20,
        last_keyframe_id=0,
        last_world_to_camera=_pose([0.0, 0.0, 0.0]),
        current_world_to_camera=_pose([0.0, 0.0, 0.0], yaw_deg=4.0),
        rendered=target,
        target=target,
        opacity=torch.zeros((10, 10)),
        config=config,
    )
    assert certificate["low_opacity_fraction"] == pytest.approx(1.0)
    assert certificate["admitted"] is True


def test_sparse_far_depth_prior_is_robust_and_expires():
    prior = SparseFarDepthPrior(
        {
            "depth_prior_window_frames": 10,
            "depth_prior_quantile": 0.5,
            "depth_prior_min_m": 20.0,
            "depth_prior_max_m": 120.0,
        }
    )
    assert prior.observe(0, torch.tensor([5.0, 30.0, 50.0])) == pytest.approx(40.0)
    assert prior.observe(5, torch.tensor([80.0])) == pytest.approx(60.0)
    assert prior.estimate(11) == pytest.approx(80.0)
    assert prior.estimate(16) is None


def test_depth_prior_fills_only_invalid_rows_when_depthcov_collapses():
    estimated, valid, depth_std, rows = apply_sparse_depth_prior_fallback(
        torch.tensor([10.0, 0.0, 0.0]),
        torch.tensor([True, False, False]),
        torch.tensor([0.01, 1.0, 1.0]),
        prior_depth_m=48.0,
        std_valid_threshold=0.06,
        min_valid=2,
        confidence=0.5,
    )
    assert rows == 2
    assert estimated.tolist() == pytest.approx([10.0, 48.0, 48.0])
    assert valid.tolist() == [True, True, True]
    assert depth_std.tolist() == pytest.approx([0.01, 0.03, 0.03])


def test_depth_prior_does_not_modify_sufficient_depthcov_support():
    estimated, valid, depth_std, rows = apply_sparse_depth_prior_fallback(
        torch.tensor([10.0, 20.0]),
        torch.tensor([True, True]),
        torch.tensor([0.01, 0.02]),
        prior_depth_m=48.0,
        std_valid_threshold=0.06,
        min_valid=2,
        confidence=0.5,
    )
    assert rows == 0
    assert estimated.tolist() == pytest.approx([10.0, 20.0])
    assert valid.tolist() == [True, True]
    assert depth_std.tolist() == pytest.approx([0.01, 0.02])


def test_visible_surface_depth_overrides_only_reliable_fallback_rows():
    estimated, valid, depth_std, rows, map_rows = (
        apply_visible_surface_depth_fallback(
            torch.tensor([12.0, 0.0, 0.0, 0.0]),
            torch.tensor([True, False, False, False]),
            torch.tensor([0.01, 1.0, 1.0, 1.0]),
            prior_depth_m=48.0,
            visible_depth=torch.tensor([30.0, 40.0, 20.0, 60.0]),
            visible_opacity=torch.tensor([1.0, 0.8, 0.9, 0.2]),
            std_valid_threshold=0.06,
            min_valid=2,
            confidence=0.5,
            front_ratio=0.95,
            min_opacity=0.5,
            min_prior_ratio=0.5,
            max_prior_ratio=1.5,
        )
    )
    assert rows == 3
    assert map_rows == 1
    assert estimated.tolist() == pytest.approx([12.0, 38.0, 48.0, 48.0])
    assert valid.tolist() == [True, True, True, True]
    assert depth_std.tolist() == pytest.approx([0.01, 0.03, 0.03, 0.03])


def test_visible_surface_depth_is_noop_with_sufficient_depthcov_support():
    estimated, valid, _, rows, map_rows = apply_visible_surface_depth_fallback(
        torch.tensor([12.0, 20.0]),
        torch.tensor([True, True]),
        torch.tensor([0.01, 0.01]),
        prior_depth_m=48.0,
        visible_depth=torch.tensor([30.0, 30.0]),
        visible_opacity=torch.tensor([1.0, 1.0]),
        std_valid_threshold=0.06,
        min_valid=2,
        confidence=0.5,
        front_ratio=0.95,
        min_opacity=0.5,
        min_prior_ratio=0.5,
        max_prior_ratio=1.5,
    )
    assert rows == 0
    assert map_rows == 0
    assert estimated.tolist() == pytest.approx([12.0, 20.0])


def test_residual_grid_selects_strongest_pixel_per_cell():
    valid = torch.ones((4, 4), dtype=torch.bool)
    residual = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    selected = residual_grid_indices(valid, residual, cell_px=2, max_points=4)
    assert set(selected.tolist()) == {5, 7, 13, 15}
