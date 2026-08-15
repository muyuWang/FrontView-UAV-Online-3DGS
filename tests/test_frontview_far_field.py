import numpy as np
import pytest

from utils_new.frontview_far_field import (
    CausalRayResponsibilityAtlas,
    adaptive_log_depth_responsibility,
    budgeted_fallback_radius,
    budget_cell_parameters,
    far_field_responsibility_mask,
    matched_responsibility_shuffle,
    observability_footprint_trust_limits,
    projected_gaussian_radii,
    posterior_budget_refill_mask,
    projective_map_redundancy_mask,
    projective_radial_scale_factors,
    projective_survivor_mask,
    ray_aligned_quaternions,
    validate_front_view_far_field_config,
    visible_parallax_pixels,
)
from utils_new.frontview_projective_structure import (
    budget_normalized_information_radii,
    structure_aligned_covariances,
)


def test_far_field_config_rejects_invalid_depth():
    with pytest.raises(ValueError):
        validate_front_view_far_field_config({"enabled": True, "depth_m": 0.0})


def test_far_field_config_rejects_invalid_responsibility_basis():
    with pytest.raises(ValueError):
        validate_front_view_far_field_config(
            {"enabled": True, "responsibility_basis": "pixel_color"}
        )


def test_far_field_config_accepts_matched_certificate_shuffle():
    config = validate_front_view_far_field_config(
        {
            "enabled": True,
            "routing_mode": "adaptive_observability",
            "footprint_trust_mode": "certificate_odds",
            "footprint_trust_dynamic_update": True,
            "footprint_trust_dynamic_shuffle": True,
            "footprint_trust_dynamic_shuffle_mode": "certificate",
        }
    )
    assert config["footprint_trust_dynamic_shuffle_mode"] == "certificate"


def test_far_field_responsibility_can_route_by_persistent_identity():
    depths = [100.0, 100.0, 40.0]
    sparse_valid = [True, False, False]
    track_ids = [-1, 17, -1]

    source = far_field_responsibility_mask(
        depths,
        sparse_valid,
        track_ids,
        validate_front_view_far_field_config(
            {"enabled": True, "depth_m": 80.0}
        ),
    )
    identity = far_field_responsibility_mask(
        depths,
        sparse_valid,
        track_ids,
        validate_front_view_far_field_config(
            {
                "enabled": True,
                "depth_m": 80.0,
                "responsibility_basis": "persistent_identity",
            }
        ),
    )

    assert source.tolist() == [False, True, False]
    assert identity.tolist() == [True, False, False]


def test_projective_far_field_keeps_best_per_ray_depth_cell():
    keep = projective_survivor_mask(
        uv=np.asarray([[2.0, 2.0], [4.0, 4.0], [4.0, 4.0]], dtype=np.float32),
        depths=np.asarray([100.0, 100.0, 130.0], dtype=np.float32),
        scores=np.asarray([0.1, 0.3, 0.2], dtype=np.float32),
        config=validate_front_view_far_field_config({"enabled": True}),
    )
    assert keep.tolist() == [False, True, True]


def test_causal_observability_uses_gaussian_support_as_unit_resolution():
    config = validate_front_view_far_field_config(
        {"enabled": True, "routing_mode": "causal_observability"}
    )
    projective = far_field_responsibility_mask(
        depths=[10.0, 10.0, 10.0, 10.0],
        sparse_valid=[False, False, False, True],
        track_ids=[-1, -1, -1, -1],
        config=config,
        parallax_pixels=[0.5, 2.0, 20.0, 0.0],
        projected_radii=[1.0, 1.0, 1.0, 1.0],
        log_depth_stds=[0.01, 0.01, 0.10, 1.0],
    )

    assert projective.tolist() == [True, False, True, False]


def test_adaptive_responsibility_is_scale_invariant_and_selects_farthest_regime():
    depths = np.asarray(
        [2.0, 2.2, 2.4, 20.0, 22.0, 24.0, 200.0, 220.0, 240.0],
        dtype=np.float32,
    )
    eligible = np.ones_like(depths, dtype=np.bool_)
    selected, metadata = adaptive_log_depth_responsibility(depths, eligible)
    scaled, scaled_metadata = adaptive_log_depth_responsibility(
        depths * 7.0, eligible
    )

    assert selected.tolist() == scaled.tolist()
    assert np.flatnonzero(selected).tolist() == [6, 7, 8]
    assert metadata["regime_counts"] == [3, 3, 3]
    assert scaled_metadata["boundaries_m"] == pytest.approx(
        np.asarray(metadata["boundaries_m"]) * 7.0, rel=1.0e-5
    )


def test_adaptive_observability_protects_near_unresolved_candidates():
    config = validate_front_view_far_field_config(
        {"enabled": True, "routing_mode": "adaptive_observability"}
    )
    depths = np.asarray(
        [2.0, 2.2, 2.4, 20.0, 22.0, 24.0, 200.0, 220.0, 240.0],
        dtype=np.float32,
    )
    projective, metadata = far_field_responsibility_mask(
        depths,
        sparse_valid=np.zeros(9, dtype=np.bool_),
        track_ids=np.full(9, -1, dtype=np.int64),
        config=config,
        parallax_pixels=np.zeros(9, dtype=np.float32),
        projected_radii=np.ones(9, dtype=np.float32),
        log_depth_stds=np.full(9, 0.06, dtype=np.float32),
        return_metadata=True,
    )

    assert projective.tolist() == [False] * 6 + [True] * 3
    assert metadata["far_rows"] == 3


def test_adaptive_observability_keeps_metric_far_candidate_out_of_projective_route():
    config = validate_front_view_far_field_config(
        {"enabled": True, "routing_mode": "adaptive_observability"}
    )
    projective = far_field_responsibility_mask(
        depths=[2.0, 2.2, 2.4, 20.0, 22.0, 24.0, 200.0, 220.0, 240.0],
        sparse_valid=[False] * 9,
        track_ids=[-1] * 9,
        config=config,
        parallax_pixels=[0.0] * 8 + [20.0],
        projected_radii=[1.0] * 9,
        log_depth_stds=[0.01] * 9,
    )

    assert projective.tolist() == [False] * 6 + [True, True, False]


def test_matched_responsibility_shuffle_preserves_each_online_depth_domain():
    depths = np.asarray(
        [2.0, 2.2, 2.4, 20.0, 22.0, 24.0, 200.0, 220.0, 240.0],
        dtype=np.float32,
    )
    responsibility = np.asarray(
        [True, False, False, True, True, False, False, True, True],
        dtype=np.bool_,
    )
    shuffled = matched_responsibility_shuffle(
        responsibility,
        np.ones(9, dtype=np.bool_),
        depths,
        seed=17,
        mode="log_depth_regimes",
    )

    assert shuffled.tolist() != responsibility.tolist()
    for rows in (slice(0, 3), slice(3, 6), slice(6, 9)):
        assert int(np.count_nonzero(shuffled[rows])) == int(
            np.count_nonzero(responsibility[rows])
        )


def test_matched_responsibility_shuffle_never_assigns_ineligible_rows():
    responsibility = np.asarray([True, False, True, False], dtype=np.bool_)
    eligible = np.asarray([True, True, True, False], dtype=np.bool_)
    shuffled = matched_responsibility_shuffle(
        responsibility,
        eligible,
        depths=[1.0, 1.1, 1.2, 1.3],
        seed=3,
        mode="global",
    )

    assert int(np.count_nonzero(shuffled)) == 2
    assert not shuffled[3]


def test_projected_radius_and_visible_parallax_are_in_pixel_units():
    radii = projected_gaussian_radii(
        log_scales=np.log([[0.1], [0.2]]),
        depths=[10.0, 20.0],
        focal_pixels=100.0,
    )
    assert radii == pytest.approx([1.0, 1.0])

    current = np.eye(4, dtype=np.float32)
    reference = np.eye(4, dtype=np.float32)
    reference[0, 3] = -1.0
    intrinsic = np.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    parallax, support = visible_parallax_pixels(
        [[0.0, 0.0, 10.0]],
        current,
        [reference],
        [intrinsic],
        [(100, 100)],
        100.0,
    )
    assert parallax[0] == pytest.approx(9.95037, rel=1.0e-4)
    assert support.tolist() == [1]


def test_map_redundancy_gate_uses_opacity_odds_and_disparity_likelihood():
    redundant = projective_map_redundancy_mask(
        depths=[100.0, 100.0, 50.0],
        map_depths=[100.0, 100.0, 100.0],
        map_opacities=[0.8, 0.2, 0.9],
        parallax_pixels=[1.0, 1.0, 10.0],
        projected_radii=[1.0, 1.0, 1.0],
    )
    assert redundant.tolist() == [True, False, False]


def test_photometric_map_gate_retains_unexplained_geometry():
    redundant = projective_map_redundancy_mask(
        depths=[100.0, 100.0],
        map_depths=[100.0, 100.0],
        map_opacities=[0.9, 0.9],
        parallax_pixels=[1.0, 1.0],
        projected_radii=[1.0, 1.0],
        residuals=[0.01, 0.30],
        residual_scale=0.10,
    )

    assert redundant.tolist() == [True, False]


def test_photometric_map_mode_requires_the_redundancy_gate():
    with pytest.raises(ValueError):
        validate_front_view_far_field_config(
            {"map_redundancy_evidence": "photometric"}
        )


def test_posterior_refill_requires_map_gate_and_shuffled_control_requires_refill():
    with pytest.raises(ValueError):
        validate_front_view_far_field_config({"posterior_budget_refill": True})
    with pytest.raises(ValueError):
        validate_front_view_far_field_config({"shuffle_refill_evidence": True})


def test_ray_atlas_requires_causal_routing_and_its_shuffle_requires_atlas():
    with pytest.raises(ValueError, match="causal-observability"):
        validate_front_view_far_field_config({"ray_atlas_enabled": True})
    with pytest.raises(ValueError, match="requires ray_atlas_enabled"):
        validate_front_view_far_field_config(
            {"ray_atlas_shuffle_evidence": True}
        )
    with pytest.raises(ValueError, match="canonical_world"):
        validate_front_view_far_field_config(
            {
                "routing_mode": "causal_observability",
                "ray_atlas_enabled": True,
                "ray_atlas_competition_mode": "continuous_kernel",
            }
        )


def test_ray_atlas_rejects_cross_frame_duplicate_but_keeps_resolvable_layers():
    atlas = CausalRayResponsibilityAtlas(enabled=True)
    kwargs = dict(
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        focal_pixels=640.0,
        max_log_depth_std=0.06,
    )
    directions = np.asarray(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 0.995]], dtype=np.float32
    )
    keep, keys = atlas.admit(
        directions,
        np.asarray([100.0, 100.0], dtype=np.float32),
        frame_id=0,
        **kwargs,
    )
    assert keep.tolist() == [True, True]
    atlas.register([key for key, retain in zip(keys, keep) if retain], 0)

    keep, _ = atlas.admit(
        np.asarray(
            [[0.001, 0.0, 1.0], [0.0, 0.0, 1.0], [0.2, 0.0, 0.98]],
            dtype=np.float32,
        ),
        np.asarray([100.0, 130.0, 100.0], dtype=np.float32),
        frame_id=1,
        **kwargs,
    )
    assert keep.tolist() == [False, True, True]


def test_ray_atlas_memory_is_bounded_by_the_online_candidate_pool():
    atlas = CausalRayResponsibilityAtlas(enabled=True)
    kwargs = dict(
        image_size=(40, 20),
        birth_budget=2,
        pool_multiplier=2,
        focal_pixels=20.0,
        max_log_depth_std=0.06,
    )
    for frame_id, x in enumerate(np.linspace(-0.8, 0.8, 9)):
        direction = np.asarray([[x, 0.0, 1.0]], dtype=np.float32)
        keep, keys = atlas.admit(direction, [10.0], frame_id=frame_id, **kwargs)
        atlas.register([key for key, retain in zip(keys, keep) if retain], frame_id)
    summary = atlas.summary()
    assert summary["active_cells"] <= 4
    assert summary["capacity"] == 4
    assert summary["evicted_rows"] > 0


def test_shuffled_ray_atlas_is_deterministic_and_preserves_input_count():
    kwargs = dict(
        directions=np.asarray(
            [[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.4, 0.0, 1.0]],
            dtype=np.float32,
        ),
        depths=np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        focal_pixels=640.0,
        max_log_depth_std=0.06,
        frame_id=7,
    )
    first = CausalRayResponsibilityAtlas(
        enabled=True, shuffle_evidence=True, seed=43
    )
    second = CausalRayResponsibilityAtlas(
        enabled=True, shuffle_evidence=True, seed=43
    )
    first_keep, first_keys = first.admit(**kwargs)
    second_keep, second_keys = second.admit(**kwargs)
    assert first_keep.tolist() == second_keep.tolist()
    assert first_keys == second_keys
    assert len(first_keys) == 3


def test_canonical_ray_atlas_is_invariant_to_observer_translation():
    atlas = CausalRayResponsibilityAtlas(
        enabled=True,
        coordinate_mode="canonical_world",
    )
    kwargs = dict(
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        focal_pixels=640.0,
        max_log_depth_std=0.06,
    )
    point = np.asarray([[10.0, 0.0, 100.0]], dtype=np.float32)
    first_keep, claims = atlas.admit(
        point,
        np.linalg.norm(point, axis=1),
        world_points=point,
        camera_center=np.zeros(3, dtype=np.float32),
        evidence_scores=[1.0],
        frame_id=0,
        **kwargs,
    )
    atlas.register([claim for claim, keep in zip(claims, first_keep) if keep], 0)
    translated = np.asarray([5.0, 0.0, 0.0], dtype=np.float32)
    ray = point - translated[None, :]
    second_keep, _ = atlas.admit(
        ray,
        np.linalg.norm(ray, axis=1),
        world_points=point,
        camera_center=translated,
        evidence_scores=[1.0],
        frame_id=1,
        **kwargs,
    )
    assert first_keep.tolist() == [True]
    assert second_keep.tolist() == [False]


def test_continuous_canonical_competition_crosses_cell_boundaries():
    atlas = CausalRayResponsibilityAtlas(
        enabled=True,
        coordinate_mode="canonical_world",
        competition_mode="continuous_kernel",
    )
    kwargs = dict(
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        focal_pixels=640.0,
        max_log_depth_std=0.06,
        camera_center=np.zeros(3, dtype=np.float32),
    )
    points = np.asarray(
        [[0.0, 0.0, 100.0], [1.0, 0.0, 100.0], [20.0, 0.0, 100.0]],
        dtype=np.float32,
    )
    keep, claims = atlas.admit(
        points,
        np.linalg.norm(points, axis=1),
        world_points=points,
        evidence_scores=[0.5, 1.0, 0.25],
        projected_radii=[1.0, 1.0, 1.0],
        log_depth_stds=[0.06, 0.06, 0.06],
        frame_id=0,
        **kwargs,
    )
    assert keep.tolist() == [False, True, True]
    atlas.register([claim for claim, retain in zip(claims, keep) if retain], 0)
    assert atlas.summary()["active_cells"] == 2


def test_continuous_record_only_refines_on_new_evidence_record():
    atlas = CausalRayResponsibilityAtlas(
        enabled=True,
        coordinate_mode="canonical_world",
        competition_mode="continuous_record",
    )
    kwargs = dict(
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        focal_pixels=640.0,
        max_log_depth_std=0.06,
        camera_center=np.zeros(3, dtype=np.float32),
        projected_radii=[1.0],
        log_depth_stds=[0.06],
    )
    point = np.asarray([[0.0, 0.0, 100.0]], dtype=np.float32)
    depth = np.linalg.norm(point, axis=1)

    keep, claims = atlas.admit(
        point,
        depth,
        world_points=point,
        evidence_scores=[0.2],
        frame_id=0,
        **kwargs,
    )
    atlas.register([claims[0]], 0)
    assert keep.tolist() == [True]

    keep, _ = atlas.admit(
        point,
        depth,
        world_points=point,
        evidence_scores=[0.1],
        frame_id=1,
        **kwargs,
    )
    assert keep.tolist() == [False]

    keep, claims = atlas.admit(
        point,
        depth,
        world_points=point,
        evidence_scores=[0.3],
        frame_id=2,
        **kwargs,
    )
    assert keep.tolist() == [True]
    atlas.register([claims[0]], 2)
    summary = atlas.summary()
    assert summary["active_cells"] == 1
    assert summary["record_admissions"] == 1
    assert summary["record_rejections"] == 1
    assert summary["record_replacements"] == 1


def test_continuous_record_shuffle_changes_only_evidence_assignment():
    kwargs = dict(
        directions=np.asarray(
            [[0.0, 0.0, 10.0], [3.0, 0.0, 10.0], [6.0, 0.0, 10.0]],
            dtype=np.float32,
        ),
        depths=np.asarray([10.0, 10.44, 11.66], dtype=np.float32),
        world_points=np.asarray(
            [[0.0, 0.0, 10.0], [3.0, 0.0, 10.0], [6.0, 0.0, 10.0]],
            dtype=np.float32,
        ),
        camera_center=np.zeros(3, dtype=np.float32),
        evidence_scores=[0.1, 0.5, 0.9],
        projected_radii=[1.0, 1.0, 1.0],
        log_depth_stds=[0.06, 0.06, 0.06],
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        focal_pixels=640.0,
        max_log_depth_std=0.06,
        frame_id=3,
    )
    real = CausalRayResponsibilityAtlas(
        enabled=True,
        coordinate_mode="canonical_world",
        competition_mode="continuous_record",
        seed=7,
    )
    shuffled = CausalRayResponsibilityAtlas(
        enabled=True,
        coordinate_mode="canonical_world",
        competition_mode="continuous_record",
        shuffle_evidence=True,
        seed=7,
    )
    real_keep, real_claims = real.admit(**kwargs)
    shuffled_keep, shuffled_claims = shuffled.admit(**kwargs)

    assert real_keep.tolist() == shuffled_keep.tolist() == [True, True, True]
    assert [claim["key"] for claim in real_claims] == [
        claim["key"] for claim in shuffled_claims
    ]
    for real_claim, shuffled_claim in zip(real_claims, shuffled_claims):
        assert real_claim["unit"] == pytest.approx(shuffled_claim["unit"])
        assert real_claim["log_depth"] == pytest.approx(shuffled_claim["log_depth"])
    real_scores = [claim["score"] for claim in real_claims]
    shuffled_scores = [claim["score"] for claim in shuffled_claims]
    assert sorted(real_scores) == pytest.approx(sorted(shuffled_scores))
    assert real_scores != shuffled_scores


def test_continuous_record_eviction_keeps_buckets_consistent():
    atlas = CausalRayResponsibilityAtlas(
        enabled=True,
        coordinate_mode="canonical_world",
        competition_mode="continuous_record",
    )
    kwargs = dict(
        image_size=(40, 20),
        birth_budget=1,
        pool_multiplier=2,
        focal_pixels=20.0,
        max_log_depth_std=0.06,
        camera_center=np.zeros(3, dtype=np.float32),
        projected_radii=[0.1],
        log_depth_stds=[0.01],
    )
    for frame_id, x in enumerate((-8.0, 0.0, 8.0)):
        point = np.asarray([[x, 0.0, 10.0]], dtype=np.float32)
        keep, claims = atlas.admit(
            point,
            np.linalg.norm(point, axis=1),
            world_points=point,
            evidence_scores=[0.1 + 0.1 * frame_id],
            frame_id=frame_id,
            **kwargs,
        )
        assert keep.tolist() == [True]
        atlas.register([claims[0]], frame_id)

    bucket_owner_ids = set().union(*atlas._buckets.values())
    assert atlas.summary()["active_cells"] == 2
    assert bucket_owner_ids == set(atlas._owners)
    assert all(atlas._buckets.values())


def _run_dyadic_evidence_sequence(scores):
    atlas = CausalRayResponsibilityAtlas(
        enabled=True,
        coordinate_mode="canonical_world",
        competition_mode="continuous_dyadic",
    )
    kwargs = dict(
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        focal_pixels=640.0,
        max_log_depth_std=0.06,
        camera_center=np.zeros(3, dtype=np.float32),
        projected_radii=[1.0],
        log_depth_stds=[0.06],
    )
    point = np.asarray([[0.0, 0.0, 100.0]], dtype=np.float32)
    depth = np.linalg.norm(point, axis=1)
    decisions = []
    for frame_id, score in enumerate(scores):
        keep, claims = atlas.admit(
            point,
            depth,
            world_points=point,
            evidence_scores=[score],
            frame_id=frame_id,
            **kwargs,
        )
        decisions.append(bool(keep[0]))
        if keep[0]:
            atlas.register([claims[0]], frame_id)
    return decisions, atlas


def test_continuous_dyadic_admits_only_when_cumulative_evidence_doubles():
    decisions, atlas = _run_dyadic_evidence_sequence([1.0, 0.5, 0.5, 1.0, 1.0])

    assert decisions == [True, False, True, False, True]
    summary = atlas.summary()
    assert summary["active_cells"] == 1
    assert summary["dyadic_accumulations"] == 4
    assert summary["dyadic_admissions"] == 2
    assert summary["dyadic_rejections"] == 2
    assert summary["dyadic_level_sum"] == 2


def test_continuous_dyadic_is_invariant_to_positive_evidence_scaling():
    scores = [0.2, 0.1, 0.1, 0.2, 0.2, 0.4]
    reference, _ = _run_dyadic_evidence_sequence(scores)
    scaled, _ = _run_dyadic_evidence_sequence([17.0 * score for score in scores])

    assert scaled == reference


def test_continuous_dyadic_accumulates_at_most_once_per_frame():
    decisions, atlas = _run_dyadic_evidence_sequence([1.0])
    assert decisions == [True]
    point = np.asarray([[0.0, 0.0, 100.0]], dtype=np.float32)
    keep, _ = atlas.admit(
        point,
        np.linalg.norm(point, axis=1),
        world_points=point,
        camera_center=np.zeros(3, dtype=np.float32),
        evidence_scores=[100.0],
        projected_radii=[1.0],
        log_depth_stds=[0.06],
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        focal_pixels=640.0,
        max_log_depth_std=0.06,
        frame_id=0,
    )

    assert keep.tolist() == [False]
    summary = atlas.summary()
    assert summary["dyadic_accumulations"] == 0
    assert summary["dyadic_same_frame_rejections"] == 1


def test_posterior_refill_preserves_primary_and_uses_least_explained_reserve():
    keep, stats = posterior_budget_refill_mask(
        budget_primary=[True, True, False, False, False],
        sparse_valid=[True, False, False, False, False],
        map_log_odds=[-np.inf, -1.0, 0.8, -2.0, -2.0],
        residual_scores=[0.0, 0.1, 0.9, 0.2, 0.7],
        requested_refill_count=2,
        reserve_eligible=[False, False, True, True, True],
    )

    assert keep.tolist() == [True, True, False, True, True]
    assert stats == {"requested": 2, "reserves": 3, "selected": 2}


def test_posterior_refill_never_exceeds_rejected_primary_budget():
    keep, stats = posterior_budget_refill_mask(
        budget_primary=[True, False, False],
        sparse_valid=[False, False, False],
        map_log_odds=[-1.0, -3.0, -2.0],
        residual_scores=[0.1, 0.2, 0.3],
        requested_refill_count=1,
    )

    assert keep.tolist() == [True, True, False]
    assert int(np.sum(keep)) == 2
    assert stats["selected"] == 1


def test_budget_nms_gives_primary_rows_first_claim_on_a_cell():
    keep = projective_survivor_mask(
        uv=np.asarray([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32),
        depths=np.asarray([100.0, 100.0], dtype=np.float32),
        scores=np.asarray([0.1, 0.9], dtype=np.float32),
        config=validate_front_view_far_field_config(
            {"enabled": True, "projective_nms_mode": "budget_cells"}
        ),
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        max_log_depth_std=0.06,
        primary_mask=np.asarray([True, False]),
    )

    assert keep.tolist() == [True, False]


def test_support_aware_nms_preserves_resolvable_depth_layers():
    config = validate_front_view_far_field_config(
        {
            "enabled": True,
            "projective_nms_mode": "gaussian_support",
        }
    )
    keep = projective_survivor_mask(
        uv=np.asarray(
            [[0.0, 0.0], [0.5, 0.0], [0.5, 0.0], [5.0, 0.0]],
            dtype=np.float32,
        ),
        depths=np.asarray([100.0, 100.0, 130.0, 100.0], dtype=np.float32),
        scores=np.asarray([0.1, 0.3, 0.2, 0.1], dtype=np.float32),
        config=config,
        projected_radii=np.ones((4,), dtype=np.float32),
        log_depth_stds=np.full((4,), 0.05, dtype=np.float32),
    )
    assert keep.tolist() == [False, True, True, True]


def test_budget_cells_are_derived_from_resolution_budget_and_depth_noise():
    cell_px, log_depth_width = budget_cell_parameters(
        (1280, 720), 3200, 2, 0.06
    )
    assert cell_px == pytest.approx(12.0)
    assert log_depth_width == pytest.approx(np.sqrt(2.0) * 0.06)


def test_fallback_support_is_derived_from_resolution_and_birth_budget():
    assert budgeted_fallback_radius((1280, 720), 3200) == pytest.approx(
        0.5 * np.sqrt(1280 * 720 / 3200)
    )


def test_information_support_preserves_budget_and_shrinks_textured_regions():
    image = np.zeros((41, 81, 3), dtype=np.float32)
    image[:, 41:, :] = np.linspace(0.0, 1.0, 40, dtype=np.float32)[None, :, None]
    uv = np.asarray([[20.0, 20.0], [60.0, 20.0]], dtype=np.float32)
    factors, density = budget_normalized_information_radii(
        image, uv, [True, True], 4.0
    )

    assert factors[1] < factors[0]
    assert np.mean(1.0 / factors**2) == pytest.approx(1.0, abs=1.0e-6)
    assert np.mean(density) == pytest.approx(1.0, abs=1.0e-6)


def test_information_support_shuffle_preserves_radius_distribution():
    image = np.zeros((41, 81, 3), dtype=np.float32)
    image[:, 20:] = 0.25
    image[:, 40:] = 0.75
    image[:, 60:] = 1.0
    uv = np.asarray(
        [[10.0, 20.0], [30.0, 20.0], [50.0, 20.0], [70.0, 20.0]],
        dtype=np.float32,
    )
    ordered = budget_normalized_information_radii(
        image, uv, np.ones(4, dtype=np.bool_), 4.0
    )[0]
    shuffled = budget_normalized_information_radii(
        image,
        uv,
        np.ones(4, dtype=np.bool_),
        4.0,
        shuffle=True,
        seed=7,
    )[0]

    assert sorted(shuffled.tolist()) == pytest.approx(sorted(ordered.tolist()))


def test_structure_covariance_preserves_area_and_aligns_with_edge():
    image = np.zeros((9, 9, 3), dtype=np.float32)
    image[:, 5:, :] = 1.0
    factors, quaternions, anisotropy = structure_aligned_covariances(
        image,
        uv=np.asarray([[4.0, 4.0], [1.0, 1.0]], dtype=np.float32),
        view_directions=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),
        world_to_camera_rotation=np.eye(3, dtype=np.float32),
        eligible=np.asarray([True, False]),
    )

    assert np.prod(factors[0]) == pytest.approx(1.0, abs=1.0e-6)
    assert factors[0, 0] > 1.0
    assert factors[0, 1] < 1.0
    assert factors[1].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert anisotropy[0] > 1.0
    assert np.linalg.norm(quaternions, axis=1).tolist() == pytest.approx(
        [1.0, 1.0]
    )


def test_structure_shuffle_preserves_anisotropy_distribution():
    image = np.zeros((9, 9, 3), dtype=np.float32)
    image[:, 3:, :] = 0.25
    image[:, 6:, :] = 1.0
    kwargs = dict(
        image=image,
        uv=np.asarray([[2.0, 4.0], [5.0, 4.0], [7.0, 4.0]], dtype=np.float32),
        view_directions=np.tile(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (3, 1)
        ),
        world_to_camera_rotation=np.eye(3, dtype=np.float32),
        eligible=np.ones((3,), dtype=np.bool_),
    )
    ordered = structure_aligned_covariances(**kwargs)[2]
    shuffled = structure_aligned_covariances(
        **kwargs, shuffle=True, seed=4
    )[2]

    assert sorted(shuffled.tolist()) == pytest.approx(sorted(ordered.tolist()))


def test_certificate_structure_uses_budget_support_and_degenerates_isotropically():
    image = np.zeros((21, 21, 3), dtype=np.float32)
    image[:, 11:, :] = 1.0
    kwargs = dict(
        image=image,
        uv=np.asarray([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32),
        view_directions=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),
        world_to_camera_rotation=np.eye(3, dtype=np.float32),
        eligible=np.ones((2,), dtype=np.bool_),
        support_radius_pixels=3.0,
        certificate_strength=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    factors, quaternions, anisotropy = structure_aligned_covariances(**kwargs)

    assert anisotropy[0] > 1.0
    assert anisotropy[1] == pytest.approx(1.0)
    assert np.prod(factors, axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert np.linalg.norm(quaternions, axis=1).tolist() == pytest.approx(
        [1.0, 1.0]
    )


def test_certificate_structure_shuffle_preserves_strength_conditioned_distribution():
    image = np.zeros((21, 21, 3), dtype=np.float32)
    image[:, 5:, :] = 0.25
    image[:, 11:, :] = 0.75
    image[:, 16:, :] = 1.0
    kwargs = dict(
        image=image,
        uv=np.asarray([[4.0, 10.0], [10.0, 10.0], [15.0, 10.0]], dtype=np.float32),
        view_directions=np.tile(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (3, 1)
        ),
        world_to_camera_rotation=np.eye(3, dtype=np.float32),
        eligible=np.ones((3,), dtype=np.bool_),
        support_radius_pixels=2.0,
        certificate_strength=np.asarray([1.0, 0.75, 0.5], dtype=np.float32),
    )
    ordered = structure_aligned_covariances(**kwargs)[2]
    shuffled = structure_aligned_covariances(
        **kwargs, shuffle=True, seed=4
    )[2]

    assert sorted(shuffled.tolist()) == pytest.approx(sorted(ordered.tolist()))


def test_budget_cell_nms_keeps_best_per_derived_information_cell():
    config = validate_front_view_far_field_config(
        {"enabled": True, "projective_nms_mode": "budget_cells"}
    )
    keep = projective_survivor_mask(
        uv=np.asarray([[2.0, 2.0], [4.0, 4.0], [14.0, 4.0]], dtype=np.float32),
        depths=np.asarray([100.0, 103.0, 100.0], dtype=np.float32),
        scores=np.asarray([0.1, 0.3, 0.2], dtype=np.float32),
        config=config,
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        max_log_depth_std=0.06,
    )
    assert keep.tolist() == [False, True, True]


def test_projective_rank_factor_is_certificate_information_margin():
    factors = projective_radial_scale_factors(
        parallax_pixels=[0.5, 2.0, 20.0, 2.0],
        projected_radii=[1.0, 1.0, 1.0, 1.0],
        log_depth_stds=[0.01, 0.01, 0.10, 0.01],
        projective_mask=[True, False, True, False],
    )

    assert factors[0] == pytest.approx(np.sqrt(0.5))
    assert factors[1] == pytest.approx(1.0)
    assert factors[2] == pytest.approx(np.sqrt(0.5))
    assert factors[3] == pytest.approx(1.0)


def test_projective_rank_shuffle_preserves_factor_distribution():
    kwargs = dict(
        parallax_pixels=[0.25, 0.5, 0.75],
        projected_radii=[1.0, 1.0, 1.0],
        log_depth_stds=[0.01, 0.01, 0.01],
        projective_mask=[True, True, True],
    )
    ordered = projective_radial_scale_factors(**kwargs)
    shuffled = projective_radial_scale_factors(
        **kwargs, mode="observability_rank_shuffled", seed=7
    )

    assert sorted(shuffled.tolist()) == pytest.approx(sorted(ordered.tolist()))
    assert not np.allclose(shuffled, ordered)


def test_footprint_trust_is_scale_invariant_and_strict_without_parallax():
    kwargs = dict(
        parallax_pixels=[0.0, 2.0, 8.0],
        projected_radii=[1.0, 1.0, 1.0],
        log_depth_stds=[0.06, 0.06, 0.06],
        depths=[10.0, 30.0, 100.0],
        eligible=[True, True, True],
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
    )
    limits, information, metadata = observability_footprint_trust_limits(**kwargs)
    scaled, scaled_information, _ = observability_footprint_trust_limits(
        **{**kwargs, "depths": np.asarray(kwargs["depths"]) * 9.0}
    )

    assert limits[0] == pytest.approx(1.0)
    assert np.all(limits >= 1.0)
    assert limits == pytest.approx(scaled)
    assert information == pytest.approx(scaled_information)
    assert metadata["cell_px"] == pytest.approx(12.0)


def test_footprint_trust_shuffle_preserves_limits_inside_depth_regimes():
    depths = np.asarray(
        [2.0, 2.2, 2.4, 20.0, 22.0, 24.0, 200.0, 220.0, 240.0],
        dtype=np.float32,
    )
    kwargs = dict(
        parallax_pixels=np.arange(1.0, 10.0, dtype=np.float32),
        projected_radii=np.ones(9, dtype=np.float32),
        log_depth_stds=np.full(9, 0.06, dtype=np.float32),
        depths=depths,
        eligible=np.ones(9, dtype=np.bool_),
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
    )
    real, _, _ = observability_footprint_trust_limits(**kwargs)
    shuffled, _, metadata = observability_footprint_trust_limits(
        **kwargs, mode="information_shuffled", seed=7
    )

    for rows in (slice(0, 3), slice(3, 6), slice(6, 9)):
        assert sorted(shuffled[rows].tolist()) == pytest.approx(
            sorted(real[rows].tolist())
        )
    assert metadata["shuffled"] is True
    assert not np.allclose(shuffled, real)


def test_certificate_odds_recovers_unbounded_scale_when_resolved():
    limits, information, _ = observability_footprint_trust_limits(
        parallax_pixels=[0.0, 0.5, 1.0],
        projected_radii=[1.0, 1.0, 1.0],
        log_depth_stds=[0.06, 0.06, 0.06],
        depths=[10.0, 20.0, 30.0],
        eligible=[True, True, True],
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        mode="certificate_odds",
    )

    assert information == pytest.approx([0.0, 0.5, 1.0])
    assert limits[0] == pytest.approx(1.0)
    assert limits[1] == pytest.approx(6.0)
    assert np.isinf(limits[2])


def test_footprint_trust_accepts_frame_without_eligible_responsibility():
    limits, information, metadata = observability_footprint_trust_limits(
        parallax_pixels=[0.0, 1.0],
        projected_radii=[1.0, 1.0],
        log_depth_stds=[0.06, 0.06],
        depths=[10.0, 20.0],
        eligible=[False, False],
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        mode="certificate_odds",
    )

    assert np.all(np.isinf(limits))
    assert information.tolist() == [0.0, 0.0]
    assert metadata["rows"] == 0
    assert metadata["cell_px"] is None


def test_equal_area_certificate_derives_radius_from_budget_density():
    limits, information, metadata = observability_footprint_trust_limits(
        parallax_pixels=[0.5],
        projected_radii=[1.0],
        log_depth_stds=[0.06],
        depths=[20.0],
        eligible=[True],
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        mode="certificate_equal_area",
    )

    assert information == pytest.approx([0.5])
    assert metadata["cell_px"] == pytest.approx(12.0)
    assert limits == pytest.approx([12.0 / np.sqrt(np.pi)])


def test_bounded_area_certificate_interpolates_projected_support_area():
    limits, information, metadata = observability_footprint_trust_limits(
        parallax_pixels=[0.5, 1.0],
        projected_radii=[1.0, 1.0],
        log_depth_stds=[0.06, 0.06],
        depths=[20.0, 30.0],
        eligible=[True, True],
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        mode="certificate_residual_rd_visible_detail_bounded_area",
        responsibility_radius_factors=[1.0, 1.0],
    )

    assert information == pytest.approx([0.5, 1.0])
    assert metadata["cell_px"] == pytest.approx(12.0)
    assert limits[0] == pytest.approx(np.sqrt(0.5 + 0.5 * 36.0))
    assert limits[1] == pytest.approx(6.0)


def test_owner_area_certificate_only_expands_projective_responsibility_area():
    limits, information, metadata = observability_footprint_trust_limits(
        parallax_pixels=[0.5, 0.5],
        projected_radii=[1.0, 1.0],
        log_depth_stds=[0.06, 0.06],
        depths=[20.0, 20.0],
        eligible=[True, True],
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        mode="certificate_owner_area",
        projective_owner=[False, True],
    )

    assert information == pytest.approx([0.5, 0.5])
    assert limits == pytest.approx([6.0, 12.0 / np.sqrt(np.pi)])
    assert metadata["projective_owner_rows"] == 1


def test_owner_area_shuffle_jointly_preserves_evidence_and_owner_pairs():
    depths = np.asarray(
        [2.0, 2.2, 2.4, 20.0, 22.0, 24.0, 200.0, 220.0, 240.0],
        dtype=np.float32,
    )
    kwargs = dict(
        parallax_pixels=np.full(9, 0.5, dtype=np.float32),
        projected_radii=np.ones(9, dtype=np.float32),
        log_depth_stds=np.full(9, 0.06, dtype=np.float32),
        depths=depths,
        eligible=np.ones(9, dtype=np.bool_),
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        projective_owner=[False, True, False, True, False, True, True, True, False],
    )
    real, _, _ = observability_footprint_trust_limits(
        **kwargs, mode="certificate_owner_area", seed=19
    )
    shuffled, _, metadata = observability_footprint_trust_limits(
        **kwargs, mode="certificate_owner_area_shuffled", seed=19
    )

    for rows in (slice(0, 3), slice(3, 6), slice(6, 9)):
        assert sorted(shuffled[rows].tolist()) == pytest.approx(
            sorted(real[rows].tolist())
        )
    assert metadata["projective_owner_rows"] == 5
    assert not np.allclose(shuffled, real)


def test_residual_rate_distortion_shuffle_preserves_caps_per_depth_domain():
    depths = np.asarray(
        [2.0, 2.2, 2.4, 20.0, 22.0, 24.0, 200.0, 220.0, 240.0],
        dtype=np.float32,
    )
    kwargs = dict(
        parallax_pixels=np.full(9, 0.5, dtype=np.float32),
        projected_radii=np.ones(9, dtype=np.float32),
        log_depth_stds=np.full(9, 0.06, dtype=np.float32),
        depths=depths,
        eligible=np.ones(9, dtype=np.bool_),
        image_size=(1280, 720),
        birth_budget=3200,
        pool_multiplier=2,
        responsibility_radius_factors=np.linspace(0.5, 1.5, 9),
    )
    real, _, _ = observability_footprint_trust_limits(
        **kwargs, mode="certificate_residual_rd", seed=23
    )
    shuffled, _, metadata = observability_footprint_trust_limits(
        **kwargs, mode="certificate_residual_rd_shuffled", seed=23
    )

    for rows in (slice(0, 3), slice(3, 6), slice(6, 9)):
        assert sorted(shuffled[rows].tolist()) == pytest.approx(
            sorted(real[rows].tolist())
        )
    assert metadata["mean_radius_factor"] == pytest.approx(1.0)
    assert not np.allclose(shuffled, real)


def test_ray_aligned_quaternion_rotates_local_z_to_view_ray():
    directions = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    quaternions = ray_aligned_quaternions(directions)
    for quaternion, expected in zip(quaternions, directions):
        w, x, y, z = quaternion
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        assert rotation[:, 2] == pytest.approx(expected, abs=1.0e-6)


def test_projective_covariance_requires_causal_routing():
    with pytest.raises(ValueError, match="causal-observability"):
        validate_front_view_far_field_config(
            {"projective_covariance_mode": "observability_rank"}
        )
