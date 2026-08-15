import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils_new.frontview_causal_metric_birth import (
    CausalMetricBirth,
    bind_tracks_to_responsibility_cells,
    bind_tracks_to_proxy_slots,
    causal_birth_replaces_depth_fallback,
    CausalMetricBirthBatch,
    certify_track_observations,
    fuse_candidate_log_depth_posteriors,
    fundamental_from_poses,
    far_inverse_depth_fisher,
    select_inverse_depth_fisher_references,
    sampson_errors,
    shuffle_metric_depth_binding,
    propagate_local_affine_inverse_depth,
    validate_causal_metric_birth_config,
)


def test_certificate_does_not_advance_mapper_torch_rng(monkeypatch):
    certificate = CausalMetricBirth({"enabled": True}, "cpu")

    def consume_rng(*args, **kwargs):
        del args, kwargs
        torch.rand(32)
        return "certificate"

    monkeypatch.setattr(certificate, "_certify_impl", consume_rng)
    camera = type("Camera", (), {"cam_idx": 17})()
    torch.manual_seed(1234)
    state = torch.random.get_rng_state().clone()

    result = certificate.certify(
        camera,
        [],
        0,
        None,
        None,
        0.1,
        1,
    )

    assert result == "certificate"
    assert torch.equal(torch.random.get_rng_state(), state)


def test_depth_gauge_statistics_keep_per_frame_decisions():
    certificate = CausalMetricBirth(
        {"enabled": True, "birth_mode": "cross_fitted_gauge"}, "cpu"
    )
    gauge = SimpleNamespace(
        sample_count=12,
        fold_nll_gains=(1.0, 2.0),
        selected_model="calibrated_field",
        log_scale=math.log(1.5),
        selected_fallback_depth=float("nan"),
    )
    certificate.record_depth_gauge(gauge, applied_rows=900, frame_id=588)
    summary = certificate.summary()
    assert summary["gauge_field_calls"] == 1
    assert summary["gauge_applied_rows"] == 900
    assert summary["gauge_decisions"][0]["frame_id"] == 588
from utils_new.frontview_parallax_depth_certificate import project_world_point


def _pose(center):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = -np.asarray(center, dtype=np.float64)
    return pose


def test_config_rejects_underconstrained_history():
    with pytest.raises(ValueError):
        validate_causal_metric_birth_config(
            {"history_frames": 1, "minimum_references": 2}
        )


def test_far_inverse_depth_fisher_prefers_lateral_baseline():
    intrinsics = np.array([[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0, 0, 1]])
    current = _pose([0.0, 0.0, 0.0])
    forward = _pose([0.0, 0.0, 10.0])
    lateral = _pose([10.0, 0.0, 0.0])

    forward_information = far_inverse_depth_fisher(
        current, forward, intrinsics, (1280, 720), grid_size=3
    )
    lateral_information = far_inverse_depth_fisher(
        current, lateral, intrinsics, (1280, 720), grid_size=3
    )

    assert np.count_nonzero(lateral_information) > 0
    assert lateral_information.sum() > forward_information.sum()


def test_fisher_reference_selection_uses_geometry_not_frame_recency():
    intrinsics = np.array([[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0, 0, 1]])
    current = _pose([0.0, 0.0, 0.0])
    references = [
        _pose([8.0, 0.0, 0.0]),
        _pose([0.05, 0.0, 0.0]),
        _pose([0.1, 0.0, 0.0]),
    ]

    selected = select_inverse_depth_fisher_references(
        current, references, intrinsics, (1280, 720), 1, grid_size=3
    )

    assert selected.indices == (0,)
    assert selected.objective_gain > 0.0
    assert not selected.used_recent_fallback


def test_config_validates_support_mode():
    assert validate_causal_metric_birth_config(
        {"support_mode": "budget_structure"}
    )["support_mode"] == "budget_structure"
    with pytest.raises(ValueError):
        validate_causal_metric_birth_config({"support_mode": "metric_threshold"})
    assert validate_causal_metric_birth_config(
        {"birth_mode": "depthcov_recondition"}
    )["birth_mode"] == "depthcov_recondition"
    assert validate_causal_metric_birth_config(
        {"birth_mode": "footprint_reanchor"}
    )["birth_mode"] == "footprint_reanchor"
    assert validate_causal_metric_birth_config(
        {"birth_mode": "posterior_proxy"}
    )["birth_mode"] == "posterior_proxy"
    assert validate_causal_metric_birth_config(
        {"birth_mode": "cross_fitted_gauge"}
    )["birth_mode"] == "cross_fitted_gauge"


@pytest.mark.parametrize(
    ("birth_mode", "replaces"),
    (
        ("tracked_features", True),
        ("local_affine_field", True),
        ("depthcov_recondition", True),
        ("footprint_reanchor", False),
        ("posterior_proxy", False),
        ("cross_fitted_gauge", False),
    ),
)
def test_causal_birth_fallback_ownership_is_explicit(birth_mode, replaces):
    config = {"enabled": True, "birth_mode": birth_mode}
    assert causal_birth_replaces_depth_fallback(config) is replaces
    assert not causal_birth_replaces_depth_fallback(
        config, dual_responsibility_enabled=True
    )


def test_disabled_causal_birth_never_replaces_fallback():
    assert not causal_birth_replaces_depth_fallback(
        {"enabled": False, "birth_mode": "tracked_features"}
    )
    assert not causal_birth_replaces_depth_fallback(
        {
            "enabled": True,
            "birth_mode": "tracked_features",
            "posterior_action": "observe_only",
        }
    )
    assert causal_birth_replaces_depth_fallback(
        {
            "enabled": True,
            "birth_mode": "tracked_features",
            "posterior_action": "abstain",
        }
    )


def test_metric_depth_shuffle_preserves_pixels_count_and_marginals():
    batch = CausalMetricBirthBatch(
        uv=np.asarray(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32
        ),
        depths=np.asarray([20.0, 50.0, 90.0], dtype=np.float32),
        world_points=np.zeros((3, 3), dtype=np.float32),
        log_depth_stds=np.asarray([0.01, 0.02, 0.03], dtype=np.float32),
        information_gains=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        supports=np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
    )
    intrinsic = np.asarray(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    )
    shuffled = shuffle_metric_depth_binding(batch, np.eye(4), intrinsic, seed=43)
    assert np.array_equal(shuffled.uv, batch.uv)
    assert sorted(shuffled.depths.tolist()) == sorted(batch.depths.tolist())
    assert sorted(shuffled.log_depth_stds.tolist()) == sorted(
        batch.log_depth_stds.tolist()
    )
    projected = shuffled.world_points @ intrinsic.T
    projected = projected[:, :2] / projected[:, 2:3]
    assert np.allclose(projected, batch.uv, atol=1.0e-5)


def test_proxy_binding_is_unique_and_budget_local():
    positions, tracks, distances = bind_tracks_to_proxy_slots(
        torch.tensor(
            [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]
        ),
        torch.tensor([1, 2, 3]),
        torch.tensor([[19.0, 20.0], [11.0, 10.0], [80.0, 80.0]]),
        support_radius_px=4.0,
    )
    assert positions.tolist() == [2, 1]
    assert tracks.tolist() == [0, 1]
    assert distances.tolist() == pytest.approx([1.0, 1.0])
    assert len(torch.unique(positions)) == len(positions)


def test_proxy_binding_rejects_tracks_outside_budget_cell():
    positions, tracks, distances = bind_tracks_to_proxy_slots(
        torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
        torch.tensor([0, 1]),
        torch.tensor([[20.0, 20.0]]),
        support_radius_px=2.0,
    )
    assert positions.numel() == tracks.numel() == distances.numel() == 0


def test_responsibility_cell_binding_does_not_cross_cell_boundary():
    positions, tracks, distances = bind_tracks_to_responsibility_cells(
        torch.tensor([[9.0, 5.0], [11.0, 5.0], [25.0, 5.0]]),
        torch.tensor([[10.1, 5.0], [24.0, 5.0]]),
        image_size=(100, 100),
        birth_budget=100,
    )
    assert positions.tolist() == [1, 2]
    assert tracks.tolist() == [0, 1]
    assert distances.tolist() == pytest.approx([0.9, 1.0])


def _posterior_tracks(uv, depths, stds, gains=None):
    row_count = len(depths)
    return CausalMetricBirthBatch(
        uv=np.asarray(uv, dtype=np.float32),
        depths=np.asarray(depths, dtype=np.float32),
        world_points=np.zeros((row_count, 3), dtype=np.float32),
        log_depth_stds=np.asarray(stds, dtype=np.float32),
        information_gains=np.asarray(
            np.ones(row_count) if gains is None else gains, dtype=np.float32
        ),
        supports=np.full((row_count,), 2.0, dtype=np.float32),
    )


def test_candidate_posterior_precision_fusion_matches_closed_form():
    prior_depth = torch.tensor([40.0])
    prior_std = torch.tensor([0.08])
    track_depth = 42.0
    track_std = 0.02
    result = fuse_candidate_log_depth_posteriors(
        torch.tensor([[15.0, 15.0]]),
        prior_depth,
        prior_std,
        torch.tensor([True]),
        _posterior_tracks([[16.0, 15.0]], [track_depth], [track_std]),
        image_size=(100, 100),
        birth_budget=100,
        config={
            "enabled": True,
            "birth_mode": "posterior_proxy",
            "minimum_information_gain": 0.1,
        },
    )
    prior_precision = 1.0 / 0.08**2
    track_precision = 1.0 / track_std**2
    expected_log = (
        prior_precision * math.log(40.0) + track_precision * math.log(track_depth)
    ) / (prior_precision + track_precision)
    assert result.certified.tolist() == [True]
    assert result.conflicted.tolist() == [False]
    assert result.depths.item() == pytest.approx(math.exp(expected_log), rel=1.0e-6)
    assert result.log_depth_stds.item() == pytest.approx(
        1.0 / math.sqrt(prior_precision + track_precision), rel=1.0e-6
    )


def test_candidate_posterior_abstains_on_innovation_conflict():
    result = fuse_candidate_log_depth_posteriors(
        torch.tensor([[15.0, 15.0]]),
        torch.tensor([40.0]),
        torch.tensor([0.03]),
        torch.tensor([True]),
        _posterior_tracks([[16.0, 15.0]], [100.0], [0.01]),
        image_size=(100, 100),
        birth_budget=100,
        config={"enabled": True, "birth_mode": "posterior_proxy"},
    )
    assert result.bound.tolist() == [True]
    assert result.certified.tolist() == [False]
    assert result.conflicted.tolist() == [True]
    assert result.depths.item() == pytest.approx(40.0)


def test_candidate_posterior_track_only_repairs_invalid_prior_without_new_row():
    result = fuse_candidate_log_depth_posteriors(
        torch.tensor([[15.0, 15.0], [75.0, 75.0]]),
        torch.tensor([300.0, 20.0]),
        torch.tensor([0.06, 0.06]),
        torch.tensor([False, False]),
        _posterior_tracks([[16.0, 15.0]], [55.0], [0.015], [1.2]),
        image_size=(100, 100),
        birth_budget=100,
        config={"enabled": True, "birth_mode": "posterior_proxy"},
    )
    assert len(result.depths) == 2
    assert result.track_only.tolist() == [True, False]
    assert result.valid.tolist() == [True, False]
    assert result.depths.tolist() == pytest.approx([55.0, 20.0])


def test_candidate_posterior_observe_only_is_matched_noop():
    result = fuse_candidate_log_depth_posteriors(
        torch.tensor([[15.0, 15.0]]),
        torch.tensor([300.0]),
        torch.tensor([0.06]),
        torch.tensor([False]),
        _posterior_tracks([[16.0, 15.0]], [55.0], [0.015], [1.2]),
        image_size=(100, 100),
        birth_budget=100,
        config={
            "enabled": True,
            "birth_mode": "posterior_proxy",
            "posterior_action": "observe_only",
        },
    )
    assert result.certified.tolist() == [True]
    assert result.depths.item() == pytest.approx(300.0)
    assert result.valid.tolist() == [False]


def test_sampson_error_uses_pose_geometry():
    intrinsic = np.asarray(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    )
    first, second = _pose([0.0, 0.0, 0.0]), _pose([2.0, 0.0, 0.0])
    point = np.asarray([1.0, 0.5, 50.0])
    uv0 = project_world_point(first, intrinsic, point)[0]
    uv1 = project_world_point(second, intrinsic, point)[0]
    fundamental = fundamental_from_poses(first, second, intrinsic)
    errors = sampson_errors(
        fundamental,
        np.stack((uv0, uv0)),
        np.stack((uv1, uv1 + np.asarray([0.0, 20.0]))),
    )
    assert errors[0] < 1.0e-8
    assert errors[1] > 10.0


def test_certified_track_birth_is_metric_and_fail_closed():
    intrinsic = np.asarray(
        [[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0.0, 0.0, 1.0]]
    )
    current = _pose([0.0, 0.0, 0.0])
    references = [_pose([2.0, 0.0, 0.0]), _pose([4.0, 0.0, 0.0])]
    point = np.asarray([1.0, -0.5, 55.0])
    current_uv = project_world_point(current, intrinsic, point)[0]
    reference_uv = [
        project_world_point(pose, intrinsic, point)[0] for pose in references
    ]
    config = validate_causal_metric_birth_config({"enabled": True})
    result = certify_track_observations(
        current,
        np.asarray([current_uv, current_uv + 40.0]),
        references,
        {0: [(0, reference_uv[0]), (1, reference_uv[1])], 1: [(0, reference_uv[0])]},
        intrinsic,
        prior_log_depth_std=0.06,
        config=config,
    )
    assert len(result) == 1
    assert np.allclose(result.world_points[0], point, atol=1.0e-5)
    assert result.depths[0] == pytest.approx(55.0, abs=1.0e-5)
    assert result.information_gains[0] >= math.log(math.sqrt(2.0))
    assert result.supports[0] == 2


def test_inconsistent_track_fails_closed():
    intrinsic = np.asarray(
        [[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0.0, 0.0, 1.0]]
    )
    current = _pose([0.0, 0.0, 0.0])
    references = [_pose([2.0, 0.0, 0.0]), _pose([4.0, 0.0, 0.0])]
    point = np.asarray([0.0, 0.0, 60.0])
    current_uv = project_world_point(current, intrinsic, point)[0]
    reference_uv = [
        project_world_point(pose, intrinsic, point)[0] for pose in references
    ]
    reference_uv[1] = reference_uv[1] + np.asarray([30.0, 0.0])
    result = certify_track_observations(
        current,
        np.asarray([current_uv]),
        references,
        {0: [(0, reference_uv[0]), (1, reference_uv[1])]},
        intrinsic,
        prior_log_depth_std=0.06,
        config={"enabled": True},
    )
    assert len(result) == 0


def test_local_affine_inverse_depth_recovers_planar_queries():
    uv = np.asarray(
        [[100.0, 100.0], [200.0, 100.0], [100.0, 200.0], [200.0, 200.0], [150.0, 125.0], [125.0, 150.0]]
    )
    inverse = 0.02 + 2.0e-5 * uv[:, 0] - 1.0e-5 * uv[:, 1]
    anchors = CausalMetricBirthBatch(
        uv=uv.astype(np.float32),
        depths=(1.0 / inverse).astype(np.float32),
        world_points=np.zeros((len(uv), 3), dtype=np.float32),
        log_depth_stds=np.full((len(uv),), 0.005, dtype=np.float32),
        information_gains=np.ones((len(uv),), dtype=np.float32),
        supports=np.full((len(uv),), 2.0, dtype=np.float32),
    )
    query = np.asarray([[150.0, 150.0]])
    result = propagate_local_affine_inverse_depth(
        anchors,
        query,
        np.eye(4),
        np.asarray([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
        0.06,
        {"enabled": True, "birth_mode": "local_affine_field"},
    )
    expected = 1.0 / (0.02 + 2.0e-5 * 150.0 - 1.0e-5 * 150.0)
    assert len(result) == 1
    assert result.depths[0] == pytest.approx(expected, rel=1.0e-4)


def test_local_affine_field_rejects_extrapolation():
    uv = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.25], [0.25, 0.5]])
    anchors = CausalMetricBirthBatch(
        uv=uv.astype(np.float32),
        depths=np.full((len(uv),), 50.0, dtype=np.float32),
        world_points=np.zeros((len(uv), 3), dtype=np.float32),
        log_depth_stds=np.full((len(uv),), 0.005, dtype=np.float32),
        information_gains=np.ones((len(uv),), dtype=np.float32),
        supports=np.full((len(uv),), 2.0, dtype=np.float32),
    )
    result = propagate_local_affine_inverse_depth(
        anchors,
        np.asarray([[20.0, 20.0]]),
        np.eye(4),
        np.eye(3),
        0.06,
        {"enabled": True, "birth_mode": "local_affine_field"},
    )
    assert len(result) == 0
