import numpy as np

from utils_new.frontview_residual_cover import (
    FrontViewResidualCover,
    validate_front_view_residual_cover_config,
)


def candidate_inputs(uv, *, depth=10.0, track_ids=None, sparse_valid=None):
    uv = np.asarray(uv, dtype=np.float32)
    count = len(uv)
    target = np.ones((19, 19, 3), dtype=np.float32)
    rendered = np.zeros_like(target)
    return {
        "frame_id": 3,
        "uv": uv,
        "depths": np.full((count,), depth, dtype=np.float32),
        "world_points": np.concatenate(
            (
                uv / 100.0,
                np.full((count, 1), depth, dtype=np.float32),
            ),
            axis=1,
        ),
        "log_scales": np.full((count, 1), np.log(depth / 100.0), dtype=np.float32),
        "colors": np.ones((count, 3), dtype=np.float32),
        "residual_scores": np.ones((count,), dtype=np.float32),
        "depth_confidences": np.ones((count,), dtype=np.float32),
        "sparse_valid": (
            np.zeros((count,), dtype=np.bool_)
            if sparse_valid is None
            else np.asarray(sparse_valid, dtype=np.bool_)
        ),
        "track_ids": (
            np.full((count,), -1, dtype=np.int64)
            if track_ids is None
            else np.asarray(track_ids, dtype=np.int64)
        ),
        "rendered": rendered,
        "target": target,
        "focal_px": 100.0,
        "depthcov_budget": 2,
    }


def test_config_normalizes_depth_fractions_and_rejects_unknown_options():
    config = validate_front_view_residual_cover_config(
        {"depth_fractions": [1.0, 2.0, 1.0]}
    )
    assert config["depth_fractions"] == [0.25, 0.5, 0.25]

    try:
        validate_front_view_residual_cover_config({"unknown": True})
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("Unknown options must fail closed")


def test_coverage_avoids_spending_both_births_on_the_same_residual_patch():
    uv = [[4.5, 4.5], [4.5, 4.5], [13.5, 4.5], [4.5, 13.5]]
    utility = FrontViewResidualCover(
        {
            "enabled": True,
            "selection_mode": "utility",
            "budget_scale": 1.0,
            "depth_fractions": [1.0, 0.0, 0.0],
        }
    )
    coverage = FrontViewResidualCover(
        {
            "enabled": True,
            "selection_mode": "utility_coverage",
            "budget_scale": 1.0,
            "depth_fractions": [1.0, 0.0, 0.0],
        }
    )

    utility_rows = utility.filter_candidates(**candidate_inputs(uv))
    coverage_rows = coverage.filter_candidates(**candidate_inputs(uv))

    assert set(utility_rows) == {0, 1}
    assert len(set(coverage_rows) & {0, 1}) == 1
    assert len(coverage_rows) == 2


def test_sparse_world_tracks_are_persistent_but_released_after_pruning():
    manager = FrontViewResidualCover({"enabled": True, "budget_scale": 1.0})
    inputs = candidate_inputs(
        [[2.5, 2.5], [3.5, 3.5]],
        track_ids=[17, 17],
        sparse_valid=[True, True],
    )
    inputs["depthcov_budget"] = 0
    first = manager.filter_candidates(**inputs)
    assert first.tolist() == [0]

    class Proposals:
        track_ids = np.asarray([17], dtype=np.int64)

        def __len__(self):
            return 1

    manager.mark_committed(Proposals())
    assert manager.filter_candidates(**inputs).tolist() == []
    manager.release([17])
    assert manager.filter_candidates(**inputs).tolist() == [0]


def test_shuffled_evidence_is_deterministic_and_keeps_the_fixed_budget():
    manager = FrontViewResidualCover(
        {
            "enabled": True,
            "selection_mode": "utility_coverage",
            "budget_scale": 1.0,
            "shuffle_evidence": True,
            "shuffle_seed": 9,
            "depth_fractions": [1.0, 0.0, 0.0],
        }
    )
    inputs = candidate_inputs(
        [[1.5, 1.5], [3.5, 3.5], [5.5, 5.5], [8.5, 8.5]]
    )
    first = manager.filter_candidates(**inputs)
    second_manager = FrontViewResidualCover(manager.config)
    second = second_manager.filter_candidates(**inputs)

    assert np.array_equal(first, second)
    assert len(first) == 2
    assert manager.summary()["hash_calls_zero"] is True


def test_occlusion_aware_counterfactual_gain_attenuates_births_behind_the_map():
    manager = FrontViewResidualCover(
        {
            "enabled": True,
            "use_depth_visibility": True,
            "behind_visibility_floor": 0.0,
        }
    )
    inputs = candidate_inputs([[9.5, 9.5], [9.5, 9.5]])
    inputs["depths"] = np.asarray([5.0, 15.0], dtype=np.float32)
    inputs["log_scales"] = np.log(inputs["depths"] / 100.0).reshape(-1, 1)
    map_depth = np.full((19, 19), 10.0, dtype=np.float32)
    map_opacity = np.ones((19, 19), dtype=np.float32)

    _, gains, _ = manager._counterfactual_support(
        inputs["uv"],
        inputs["depths"],
        inputs["log_scales"],
        inputs["colors"],
        inputs["depth_confidences"],
        inputs["rendered"],
        inputs["target"],
        inputs["focal_px"],
        map_depth,
        map_opacity,
    )

    assert gains[0].sum() > 0.0
    assert gains[1].sum() == 0.0


def test_covariance_lod_separates_duplicates_refinements_and_new_surfaces():
    manager = FrontViewResidualCover(
        {
            "enabled": True,
            "use_covariance_lod": True,
            "covariance_duplicate_floor": 0.1,
            "covariance_refinement_ratio": 2.0,
        }
    )
    weights, overlap, refinement = manager._covariance_lod_weights(
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        np.log(np.asarray([[0.1], [0.025], [0.1]], dtype=np.float32)),
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.1, 0.1, 0.1]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
    )

    assert overlap.tolist() == [True, True, False]
    assert refinement.tolist() == [False, True, False]
    assert np.isclose(weights[0], 0.1)
    assert np.isclose(weights[1], 1.0)
    assert np.isclose(weights[2], 1.0)


def test_projected_covariance_responsibility_unlocks_refinement_when_approaching():
    manager = FrontViewResidualCover(
        {
            "enabled": True,
            "covariance_competition_enabled": True,
            "covariance_refinement_min_radius_px": 1.5,
        }
    )
    admissible, resolvable = manager._covariance_competition_mask(
        depths=np.asarray([100.0, 10.0, 100.0], dtype=np.float32),
        focal_px=100.0,
        overlap=np.asarray([True, True, False]),
        refinement=np.asarray([True, True, False]),
        equivalent_scale=np.asarray([0.2, 0.2, 0.2], dtype=np.float32),
        bands=np.asarray([2, 0, 2], dtype=np.int8),
        frame_id=5,
    )

    assert admissible.tolist() == [False, True, True]
    assert resolvable.tolist() == [False, True, False]


def test_shuffled_covariance_responsibility_preserves_each_depth_band_budget():
    manager = FrontViewResidualCover(
        {
            "enabled": True,
            "covariance_competition_enabled": True,
            "shuffle_covariance_competition": True,
            "shuffle_seed": 13,
        }
    )
    kwargs = {
        "depths": np.asarray([10.0, 10.0, 30.0, 30.0, 80.0, 80.0]),
        "focal_px": 100.0,
        "overlap": np.asarray([True, False, True, False, True, False]),
        "refinement": np.zeros((6,), dtype=np.bool_),
        "equivalent_scale": np.full((6,), 0.1, dtype=np.float32),
        "bands": np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int8),
        "frame_id": 9,
    }
    first, _ = manager._covariance_competition_mask(**kwargs)
    second, _ = manager._covariance_competition_mask(**kwargs)

    assert np.array_equal(first, second)
    assert np.bincount(kwargs["bands"][first], minlength=3).tolist() == [1, 1, 1]


def test_covariance_competition_replaces_duplicate_birth_with_novel_candidate():
    manager = FrontViewResidualCover(
        {
            "enabled": True,
            "selection_mode": "utility",
            "budget_scale": 1.0,
            "depth_fractions": [1.0, 0.0, 0.0],
            "covariance_competition_enabled": True,
            "covariance_overlap_sigma": 3.0,
            "covariance_refinement_ratio": 2.0,
        }
    )
    inputs = candidate_inputs([[1.5, 1.5], [8.5, 8.5], [15.5, 15.5]])
    inputs["log_scales"] = np.full((3, 1), np.log(0.01), dtype=np.float32)
    inputs.update(
        {
            "global_means": inputs["world_points"][:1].copy(),
            "global_scales": np.full((1, 3), 0.01, dtype=np.float32),
            "global_quaternions": np.asarray(
                [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32
            ),
            "global_opacities": np.asarray([1.0], dtype=np.float32),
        }
    )

    selected = manager.filter_candidates(**inputs)
    summary = manager.summary()

    assert selected.tolist() == [1, 2]
    assert summary["covariance_competition_rejected_rows"] == 1
    assert summary["covariance_competition_selected_duplicate_rows"] == 0


def test_mixed_selection_preserves_band_quota_and_adds_pool_order_exploration():
    scores = np.asarray([0.1, 0.9, 0.8, 0.2, 0.7, 0.3], dtype=np.float32)
    bands = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    selected = FrontViewResidualCover._mixed_indices(
        scores, bands, np.asarray([2, 2, 0]), 0.5
    )

    assert set(selected.tolist()) == {0, 1, 3, 4}
    assert np.bincount(bands[selected], minlength=3).tolist() == [2, 2, 0]
