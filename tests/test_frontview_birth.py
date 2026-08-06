import numpy as np
import pytest
import torch

from utils_new.frontview_birth import (
    TrackResponsibilityLedger,
    layered_projective_birth_indices,
    layered_scale_expansion_limits,
    multi_layer_projective_occupancy,
    responsibility_initial_opacities,
    temporal_responsibility_rejections,
    validate_front_view_birth_config,
)


def _config(**updates):
    return validate_front_view_birth_config({"enabled": True, **updates})


def test_frontview_birth_rejects_invalid_projective_bins():
    with pytest.raises(ValueError):
        _config(depth_bin_ratio=1.0)


def test_track_ledger_binds_known_ids_once_and_keeps_depthcov_rows():
    ledger = TrackResponsibilityLedger()
    first = ledger.new_indices([7, 7, -1, 9])
    assert first.tolist() == [0, 2, 3]
    ledger.mark_committed([7, -1])
    second = ledger.new_indices([7, 9, -1])
    assert second.tolist() == [1, 2]
    ledger.mark_committed([9])
    summary = ledger.summary()
    assert summary["committed_tracks"] == 2
    assert summary["proposal_track_rejections"] == 2


def test_track_ledger_allows_only_finer_depth_octave_refinements():
    ledger = TrackResponsibilityLedger(refinement_ratio=1.5)
    assert ledger.new_indices([7], [30.0]).tolist() == [0]
    ledger.mark_committed([7], [30.0])
    assert ledger.new_indices([7], [24.0]).tolist() == []
    assert ledger.new_indices([7], [19.0]).tolist() == [0]
    ledger.mark_committed([7], [19.0])
    assert ledger.new_indices([7], [13.0]).tolist() == []
    assert ledger.new_indices([7], [12.0]).tolist() == [0]
    ledger.mark_committed([7], [12.0])
    summary = ledger.summary()
    assert summary["committed_tracks"] == 1
    assert summary["track_refinement_births"] == 2


def test_track_ledger_releases_pruned_identity_and_counts_rebirth():
    ledger = TrackResponsibilityLedger()
    ledger.mark_committed([7], [20.0])
    assert ledger.new_indices([7], [20.0]).tolist() == []
    assert ledger.release([7, -1]) == 1
    assert ledger.new_indices([7], [20.0]).tolist() == [0]
    ledger.mark_committed([7], [20.0])
    summary = ledger.summary()
    assert summary["committed_tracks"] == 1
    assert summary["active_tracks"] == 1
    assert summary["track_release_events"] == 1
    assert summary["track_rebirths"] == 1


def test_layered_birth_rejects_explained_map_but_keeps_detail_and_new_depth_layer():
    selected, stats = layered_projective_birth_indices(
        uv=torch.tensor([[6.0, 6.0], [30.0, 6.0], [54.0, 6.0]]),
        depths=torch.tensor([10.0, 10.0, 14.0]),
        depth_confidences=torch.ones(3),
        residual_scores=torch.tensor([0.10, 0.20, 0.10]),
        map_depths=torch.tensor([10.0, 10.0, 10.0]),
        map_opacities=torch.tensor([0.9, 0.9, 0.9]),
        budget=3,
        config=_config(),
    )
    assert selected.tolist() == [1, 2]
    assert stats["map_rejected"] == 1


def test_layered_birth_keeps_highest_priority_row_in_a_projective_cell():
    selected, stats = layered_projective_birth_indices(
        uv=torch.tensor([[2.0, 2.0], [4.0, 4.0]]),
        depths=torch.tensor([20.0, 20.0]),
        depth_confidences=torch.ones(2),
        residual_scores=torch.tensor([0.10, 0.30]),
        map_depths=None,
        map_opacities=None,
        budget=1,
        config=_config(map_competition=False, projective_cell_px=12),
    )
    assert selected.tolist() == [1]
    assert stats["selected"] == 1


def test_adaptive_layer_balance_uses_rank_layers_without_metric_thresholds():
    depths = torch.arange(1, 11, dtype=torch.float32)
    selected, stats = layered_projective_birth_indices(
        uv=torch.stack((torch.arange(10) * 20.0, torch.zeros(10)), dim=1),
        depths=depths,
        depth_confidences=torch.ones(10),
        residual_scores=torch.arange(10, dtype=torch.float32) / 20.0,
        map_depths=None,
        map_opacities=None,
        budget=6,
        config=_config(
            map_competition=False,
            adaptive_layer_balance=True,
            layer_quantiles=[0.30, 0.70],
            layer_fractions=[0.33, 0.34, 0.33],
        ),
    )
    assert len(selected) == 6
    assert stats["pool_layer_counts"] == [3, 4, 3]
    assert stats["selected_layer_counts"] == [2, 2, 2]
    assert stats["layer_edges"][0] < stats["layer_edges"][1]


def test_projective_birth_coverage_reserve_is_deterministic_and_not_top_only():
    kwargs = {
        "uv": torch.stack((torch.arange(10) * 20.0, torch.zeros(10)), dim=1),
        "depths": torch.arange(1, 11, dtype=torch.float32),
        "depth_confidences": torch.ones(10),
        "residual_scores": torch.arange(10, dtype=torch.float32),
        "map_depths": None,
        "map_opacities": None,
        "budget": 4,
        "config": _config(map_competition=False, priority_fraction=0.0),
        "seed": 9,
    }
    first, stats = layered_projective_birth_indices(**kwargs)
    second, _ = layered_projective_birth_indices(**kwargs)
    assert torch.equal(first, second)
    assert first.tolist() != [6, 7, 8, 9]
    assert stats["priority_selected"] == 0
    assert stats["coverage_selected"] == 4


def test_strict_responsibility_budget_does_not_refill_rejected_cells():
    kwargs = {
        "uv": torch.tensor([[2.0, 2.0], [4.0, 4.0]]),
        "depths": torch.tensor([20.0, 20.0]),
        "depth_confidences": torch.ones(2),
        "residual_scores": torch.tensor([0.10, 0.30]),
        "map_depths": None,
        "map_opacities": None,
        "budget": 2,
    }
    strict, strict_stats = layered_projective_birth_indices(
        **kwargs,
        config=_config(
            map_competition=False,
            projective_cell_px=12,
            strict_responsibility_budget=True,
        ),
    )
    refilled, refilled_stats = layered_projective_birth_indices(
        **kwargs,
        config=_config(
            map_competition=False,
            projective_cell_px=12,
            strict_responsibility_budget=False,
        ),
    )
    assert strict.tolist() == [1]
    assert strict_stats["fallback_selected"] == 0
    assert refilled.tolist() == [0, 1]
    assert refilled_stats["fallback_selected"] == 1


def test_layer_preserving_overflow_fills_each_depth_budget_before_global_fallback():
    uv = torch.tensor(
        [[2.0, 2.0], [4.0, 4.0], [2.0, 2.0], [4.0, 4.0], [2.0, 2.0], [4.0, 4.0]]
    )
    selected, stats = layered_projective_birth_indices(
        uv=uv,
        depths=torch.tensor([1.0, 1.0, 10.0, 10.0, 100.0, 100.0]),
        depth_confidences=torch.ones(6),
        residual_scores=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.2, 0.1]),
        map_depths=None,
        map_opacities=None,
        budget=6,
        config=_config(
            map_competition=False,
            adaptive_layer_balance=True,
            preserve_layer_budget=True,
            layer_quantiles=[0.34, 0.67],
            layer_fractions=[1.0 / 3.0] * 3,
            projective_cell_px=12,
        ),
    )
    assert selected.tolist() == list(range(6))
    assert stats["selected_layer_counts"] == [2, 2, 2]
    assert stats["fallback_selected"] == 3


def test_shuffled_layer_control_is_deterministic_and_changes_physical_allocation():
    depths = torch.cat(
        (
            torch.linspace(1.0, 2.0, 10),
            torch.linspace(10.0, 20.0, 10),
            torch.linspace(100.0, 200.0, 10),
        )
    )
    kwargs = {
        "uv": torch.stack((torch.arange(30) * 20.0, torch.zeros(30)), dim=1),
        "depths": depths,
        "depth_confidences": torch.ones(30),
        "residual_scores": torch.linspace(1.0, 0.0, 30),
        "map_depths": None,
        "map_opacities": None,
        "budget": 9,
        "seed": 17,
    }
    structured, _ = layered_projective_birth_indices(
        **kwargs,
        config=_config(
            map_competition=False,
            adaptive_layer_balance=True,
            preserve_layer_budget=True,
            layer_quantiles=[1.0 / 3.0, 2.0 / 3.0],
            layer_fractions=[1.0 / 3.0] * 3,
        ),
    )
    shuffled, _ = layered_projective_birth_indices(
        **kwargs,
        config=_config(
            map_competition=False,
            adaptive_layer_balance=True,
            preserve_layer_budget=True,
            shuffle_layer_assignments=True,
            layer_quantiles=[1.0 / 3.0, 2.0 / 3.0],
            layer_fractions=[1.0 / 3.0] * 3,
        ),
    )
    repeated, _ = layered_projective_birth_indices(
        **kwargs,
        config=_config(
            map_competition=False,
            adaptive_layer_balance=True,
            preserve_layer_budget=True,
            shuffle_layer_assignments=True,
            layer_quantiles=[1.0 / 3.0, 2.0 / 3.0],
            layer_fractions=[1.0 / 3.0] * 3,
        ),
    )
    assert not torch.equal(structured, shuffled)
    assert torch.equal(shuffled, repeated)


def test_bounded_layer_overflow_never_exceeds_ray_depth_cell_capacity():
    depths = torch.tensor([1.0] * 3 + [10.0] * 3 + [100.0] * 3)
    selected, stats = layered_projective_birth_indices(
        uv=torch.tensor([[2.0, 2.0]] * 9),
        depths=depths,
        depth_confidences=torch.ones(9),
        residual_scores=torch.linspace(1.0, 0.1, 9),
        map_depths=None,
        map_opacities=None,
        budget=9,
        config=_config(
            map_competition=False,
            adaptive_layer_balance=True,
            preserve_layer_budget=True,
            layer_quantiles=[0.34, 0.67],
            layer_fractions=[1.0 / 3.0] * 3,
            overflow_max_per_cell=2,
        ),
    )
    assert len(selected) == 6
    assert stats["selected_layer_counts"] == [2, 2, 2]
    assert stats["fallback_selected"] == 3


def test_multi_layer_atlas_finds_occluded_depth_layers_not_just_front_surface():
    occupied = multi_layer_projective_occupancy(
        candidate_uv=torch.tensor([[3.0, 3.0], [3.0, 3.0], [30.0, 3.0]]),
        candidate_depths=torch.tensor([10.2, 30.5, 30.5]),
        map_uv=torch.tensor([[4.0, 4.0], [4.0, 4.0]]),
        map_depths=torch.tensor([10.0, 30.0]),
        config=_config(projective_cell_px=12, depth_bin_ratio=1.10),
    )
    assert occupied.tolist() == [True, True, False]


def test_multi_layer_atlas_rejection_allows_only_residual_override():
    kwargs = {
        "uv": torch.tensor([[2.0, 2.0], [20.0, 2.0]]),
        "depths": torch.tensor([20.0, 20.0]),
        "depth_confidences": torch.ones(2),
        "residual_scores": torch.tensor([0.10, 0.20]),
        "map_depths": None,
        "map_opacities": None,
        "budget": 2,
        "map_occupied": torch.tensor([True, True]),
        "config": _config(
            map_competition=False,
            multi_layer_map_competition=True,
            residual_override=0.16,
        ),
    }
    selected, stats = layered_projective_birth_indices(**kwargs)
    assert selected.tolist() == [1]
    assert stats["atlas_rejected"] == 1


def test_sparse_anchor_owns_its_cell_even_under_high_residual():
    selected, stats = layered_projective_birth_indices(
        uv=torch.tensor([[2.0, 2.0], [20.0, 2.0]]),
        depths=torch.tensor([20.0, 20.0]),
        depth_confidences=torch.ones(2),
        residual_scores=torch.tensor([1.0, 1.0]),
        map_depths=None,
        map_opacities=None,
        budget=2,
        anchor_occupied=torch.tensor([True, False]),
        config=_config(
            map_competition=False,
            sparse_anchor_competition=True,
        ),
    )
    assert selected.tolist() == [1]
    assert stats["anchor_rejected"] == 1


def test_responsibility_opacity_only_softens_untracked_depthcov_rows():
    opacities = responsibility_initial_opacities(
        source_kinds=np.asarray(["sparse", "depthcov", "depthcov"]),
        residual_scores=np.asarray([0.0, 0.0, 0.25]),
        coverage_scores=np.asarray([0.0, 0.0, 1.0]),
        depth_confidences=np.asarray([0.0, 0.0, 1.0]),
        base_opacity=0.5,
        config=_config(
            responsibility_opacity=True,
            depthcov_opacity_min=0.2,
            depthcov_opacity_max=0.5,
            opacity_residual_saturation=0.25,
        ),
    )
    assert opacities[0] == pytest.approx(0.5)
    assert opacities[1] == pytest.approx(0.2)
    assert opacities[2] == pytest.approx(0.5)


def test_temporal_responsibility_rejects_prior_free_space_by_vote():
    reject, stats = temporal_responsibility_rejections(
        candidate_depths_by_view=np.asarray(
            [[8.0, 10.0, 12.0], [8.5, 10.2, 14.0]], dtype=np.float32
        ),
        map_depths_by_view=np.asarray(
            [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]], dtype=np.float32
        ),
        map_opacities_by_view=np.ones((2, 3), dtype=np.float32),
        valid_by_view=np.ones((2, 3), dtype=np.bool_),
        config=_config(
            temporal_map_competition=True,
            temporal_reference_frames=2,
            temporal_reject_views=2,
            temporal_free_space_ratio=0.08,
            temporal_reject_duplicates=False,
        ),
    )
    assert reject.tolist() == [True, False, False]
    assert stats["free_space_rows"] == 1
    assert stats["rejected_rows"] == 1


def test_layered_scale_control_only_caps_far_depthcov_quantile():
    limits = layered_scale_expansion_limits(
        depths=np.asarray([5.0, 10.0, 20.0, 80.0], dtype=np.float32),
        source_kinds=np.asarray(["sparse", "depthcov", "depthcov", "depthcov"]),
        config=_config(
            far_scale_control=True,
            far_depth_quantile=0.50,
            far_max_scale_expansion=1.5,
        ),
    )
    assert np.isinf(limits[0])
    assert np.isinf(limits[1])
    assert limits[2:].tolist() == [1.5, 1.5]
