import numpy as np
import pytest
import torch

from utils_new.frontview_scale_cover import (
    FrontViewScaleCover,
    validate_front_view_scale_cover_config,
)


def test_config_rejects_unknown_and_invalid_options():
    config = validate_front_view_scale_cover_config(
        {"shuffle_depth_edges_m": [10.0, 30.0]}
    )
    assert config["shuffle_depth_edges_m"] == [10.0, 30.0]

    for update in (
        {"unknown": True},
        {"radius_multiplier": 0.0},
        {"query_backend": "unknown"},
        {"frustum_conflict_budget_enabled": True},
        {"shuffle_frustum_conflict_budget": True},
    ):
        try:
            validate_front_view_scale_cover_config(update)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Invalid scale-cover options must fail closed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_pytorch3d_knn_matches_scipy_cover_decisions():
    pytest.importorskip("pytorch3d")
    rng = np.random.default_rng(23)
    registered = rng.normal(size=(257, 3)).astype(np.float32)
    registered_sizes = rng.uniform(0.2, 0.8, size=257).astype(np.float32)
    registered_colors = rng.uniform(0.0, 1.0, size=(257, 3)).astype(np.float32)
    registered_ranks = rng.integers(1, 3, size=257, dtype=np.int8)
    query = registered[:64] + rng.normal(scale=0.03, size=(64, 3)).astype(
        np.float32
    )
    query_sizes = registered_sizes[:64] * 0.9
    query_colors = registered_colors[:64]
    query_ranks = registered_ranks[:64]
    common = {
        "enabled": True,
        "dynamic_handoff_enabled": True,
        "source_priority_enabled": True,
        "color_distance_threshold": 0.2,
        "radius_multiplier": 0.5,
        "neighbors": 32,
        "rebuild_rows": 1,
    }

    outputs = []
    summaries = []
    for backend in ("scipy_kdtree", "pytorch3d_knn"):
        cover = FrontViewScaleCover({**common, "query_backend": backend})
        cover.register(
            registered,
            registered_sizes,
            colors=registered_colors,
            uids=np.arange(len(registered)),
            source_ranks=registered_ranks,
        )
        outputs.append(
            cover.occupied_with_parents(
                query,
                query_sizes,
                colors=query_colors,
                source_ranks=query_ranks,
            )
        )
        summaries.append(cover.summary())

    assert np.array_equal(outputs[0][0], outputs[1][0])
    assert np.array_equal(outputs[0][1], outputs[1][1])
    assert summaries[0]["scipy_query_calls"] == 1
    assert summaries[1]["gpu_query_calls"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_pytorch3d_single_neighbor_keeps_two_dimensional_query_shape():
    pytest.importorskip("pytorch3d")
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "query_backend": "pytorch3d_knn",
            "radius_multiplier": 1.0,
            "rebuild_rows": 100,
        }
    )
    cover.register([[0.0, 0.0, 0.0]], target_size=1.0)

    occupied = cover.occupied(
        [[0.1, 0.0, 0.0], [2.0, 0.0, 0.0]], target_size=1.0
    )

    assert occupied.tolist() == [True, False]


def test_coarse_birth_stops_duplicates_but_unlocks_finer_lod():
    cover = FrontViewScaleCover(
        {"enabled": True, "radius_multiplier": 1.0, "rebuild_rows": 100}
    )
    point = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    cover.register(point, target_size=1.0)

    assert cover.occupied(point, target_size=1.0).tolist() == [True]
    assert cover.occupied(point, target_size=0.5).tolist() == [False]
    assert cover.summary()["coarse_refinement_bypass_rows"] == 1


def test_continuous_cover_uses_metric_radius_without_voxel_keys():
    cover = FrontViewScaleCover(
        {"enabled": True, "radius_multiplier": 0.5, "rebuild_rows": 1}
    )
    cover.register([[0.0, 0.0, 0.0]], target_size=1.0)
    occupied = cover.occupied(
        [[0.49, 0.0, 0.0], [0.51, 0.0, 0.0]], target_size=1.0
    )

    assert occupied.tolist() == [True, False]
    assert cover.summary()["tree_rebuilds"] == 1


def test_candidate_footprints_give_near_and_far_queries_distinct_radii():
    cover = FrontViewScaleCover(
        {"enabled": True, "radius_multiplier": 0.5, "rebuild_rows": 1}
    )
    cover.register(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        target_size=[1.0, 1.0],
    )

    occupied = cover.occupied(
        [[0.4, 0.0, 0.0], [10.4, 0.0, 0.0]],
        target_size=[0.5, 1.0],
    )

    assert occupied.tolist() == [False, True]


def test_gaussian_support_cover_uses_covariance_overlap_without_metric_threshold():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "target_size_mode": "gaussian_support",
            "distance_mode": "gaussian_overlap",
            "rebuild_rows": 1,
        }
    )
    sizes = cover.candidate_target_sizes(
        depths=[10.0, 100.0],
        focal_pixels=100.0,
        camera_scale_rescalar=0.25,
        view_scale_size=0.1,
        sparse_valid=[False, False],
        frame_id=0,
        gaussian_scales=[1.0, 2.0],
    )
    assert sizes.tolist() == [1.0, 2.0]

    cover.register([[0.0, 0.0, 0.0]], target_size=[1.0])
    occupied = cover.occupied(
        [[1.3, 0.0, 0.0], [1.5, 0.0, 0.0]], target_size=[1.0, 1.0]
    )
    assert occupied.tolist() == [True, False]


def test_projective_footprints_clamp_ratios_and_shuffle_only_locations():
    config = {
        "enabled": True,
        "per_candidate_footprints": True,
        "footprint_ratio_min": 0.5,
        "footprint_ratio_max": 2.0,
        "shuffle_footprints": True,
        "shuffle_seed": 4,
    }
    cover = FrontViewScaleCover(config)
    sizes = cover.candidate_target_sizes(
        depths=[1.0, 10.0, 20.0, 100.0],
        focal_pixels=10.0,
        camera_scale_rescalar=2.0,
        view_scale_size=1.0,
        sparse_valid=[False, False, False, False],
        frame_id=2,
    )

    assert sorted(sizes.tolist()) == [0.5, 1.0, 2.0, 2.0]
    summary = cover.summary()
    assert summary["footprint_min_clamped_rows"] == 1
    assert summary["footprint_max_clamped_rows"] == 1
    assert summary["shuffled_footprint_rows"] == 4


def test_shuffled_cover_preserves_source_and_depth_band_counts():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "shuffle_occupancy": True,
            "shuffle_seed": 5,
            "shuffle_depth_edges_m": [20.0],
        }
    )
    occupied = np.asarray([True, False, True, False, True, False, True, False])
    depths = np.asarray([10.0, 10.0, 30.0, 30.0] * 2, dtype=np.float32)
    sparse = np.asarray([False] * 4 + [True] * 4)

    first = cover.shuffle(occupied, depths, sparse, frame_id=7)
    second_cover = FrontViewScaleCover(cover.config)
    second = second_cover.shuffle(occupied, depths, sparse, frame_id=7)

    assert np.array_equal(first, second)
    for source in (False, True):
        for band in (0, 1):
            rows = (sparse == source) & (np.digitize(depths, [20.0]) == band)
            assert int(first[rows].sum()) == int(occupied[rows].sum())
    assert cover.summary()["hash_calls_zero"] is True


def test_evidence_quota_routing_preserves_band_counts_and_sparse_rows():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "evidence_quota_routing": True,
            "shuffle_depth_edges_m": [20.0],
            "quota_unoccupied_bonus": 0.0,
        }
    )
    occupied = np.asarray([False, True, True, False, True, False])
    depths = np.asarray([10.0, 10.0, 10.0, 30.0, 30.0, 10.0])
    sparse = np.asarray([False, False, False, False, False, True])
    routed = cover.route_evidence_quota(
        occupied,
        depths,
        sparse,
        residual_scores=[0.1, 0.9, 0.2, 0.1, 0.8, 1.0],
        coverage_scores=np.zeros(6),
        depth_confidences=np.zeros(6),
        frame_id=4,
    )

    assert routed.tolist() == [True, False, True, True, False, False]
    for band in (0, 1):
        rows = (~sparse) & (np.digitize(depths, [20.0]) == band)
        assert int(routed[rows].sum()) == int(occupied[rows].sum())
    summary = cover.summary()
    assert summary["evidence_quota_selected_rows"] == 2
    assert summary["evidence_quota_reassigned_rows"] == 4


def test_multiview_support_routes_fixed_dense_quota_to_supported_candidates():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "evidence_quota_routing": True,
            "quota_residual_weight": 0.0,
            "quota_coverage_weight": 0.0,
            "quota_confidence_weight": 0.0,
            "quota_unoccupied_bonus": 0.0,
            "quota_multiview_support_weight": 1.0,
        }
    )
    occupied = np.asarray([False, True, True, False, False])
    depths = np.asarray([10.0, 10.0, 10.0, 10.0, 10.0])
    sparse = np.asarray([False, False, False, False, True])
    support = np.asarray([0.1, 0.9, 0.8, 0.2, 1.0], dtype=np.float32)

    routed = cover.route_evidence_quota(
        occupied,
        depths,
        sparse,
        residual_scores=np.zeros(5),
        coverage_scores=np.zeros(5),
        depth_confidences=np.zeros(5),
        frame_id=4,
        multiview_support_scores=support,
    )

    assert routed.tolist() == [True, False, False, True, False]
    assert int(np.sum(~routed[~sparse])) == int(np.sum(~occupied[~sparse]))
    summary = cover.summary()
    assert summary["multiview_support_rows"] == 4
    assert summary["multiview_support_positive_rows"] == 4
    assert summary["multiview_support_score_mean"] == pytest.approx(0.5)


def test_shuffled_multiview_support_is_deterministic_and_count_matched():
    config = {
        "enabled": True,
        "evidence_quota_routing": True,
        "quota_residual_weight": 0.0,
        "quota_coverage_weight": 0.0,
        "quota_confidence_weight": 0.0,
        "quota_unoccupied_bonus": 0.0,
        "quota_multiview_support_weight": 1.0,
        "shuffle_multiview_support": True,
        "shuffle_seed": 17,
        "shuffle_depth_edges_m": [20.0],
    }
    occupied = np.asarray([False, True, False, True, False, True, False, True])
    depths = np.asarray([10.0] * 4 + [30.0] * 4)
    sparse = np.zeros(8, dtype=np.bool_)
    support = np.linspace(0.0, 1.0, 8, dtype=np.float32)

    outputs = []
    for _ in range(2):
        cover = FrontViewScaleCover(config)
        outputs.append(
            cover.route_evidence_quota(
                occupied,
                depths,
                sparse,
                np.zeros(8),
                np.zeros(8),
                np.zeros(8),
                frame_id=3,
                multiview_support_scores=support,
            )
        )
        assert cover.summary()["shuffled_multiview_support_rows"] == 8

    assert np.array_equal(outputs[0], outputs[1])
    for band in (0, 1):
        rows = np.digitize(depths, [20.0]) == band
        assert int(np.sum(outputs[0][rows])) == int(np.sum(occupied[rows]))


def test_shuffled_evidence_quota_is_deterministic_and_count_matched():
    config = {
        "enabled": True,
        "evidence_quota_routing": True,
        "shuffle_evidence_quota": True,
        "shuffle_seed": 9,
        "shuffle_depth_edges_m": [20.0],
    }
    occupied = np.asarray([False, True, False, True, False, True])
    depths = np.asarray([10.0, 10.0, 10.0, 30.0, 30.0, 30.0])
    sparse = np.zeros(6, dtype=np.bool_)
    evidence = np.arange(6, dtype=np.float32)

    first_cover = FrontViewScaleCover(config)
    second_cover = FrontViewScaleCover(config)
    first = first_cover.route_evidence_quota(
        occupied, depths, sparse, evidence, evidence, evidence, frame_id=3
    )
    second = second_cover.route_evidence_quota(
        occupied, depths, sparse, evidence, evidence, evidence, frame_id=3
    )

    assert np.array_equal(first, second)
    for band in (0, 1):
        rows = np.digitize(depths, [20.0]) == band
        assert int(first[rows].sum()) == int(occupied[rows].sum())
    assert first_cover.summary()["shuffled_evidence_quota_rows"] == 6


def test_evidence_quota_depth_gate_leaves_near_rows_unchanged():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "evidence_quota_routing": True,
            "shuffle_depth_edges_m": [20.0, 50.0],
            "quota_min_depth_m": 20.0,
            "quota_unoccupied_bonus": 0.0,
        }
    )
    occupied = np.asarray([False, True, False, True, False, True])
    depths = np.asarray([10.0, 10.0, 30.0, 30.0, 60.0, 60.0])
    sparse = np.zeros(6, dtype=np.bool_)
    routed = cover.route_evidence_quota(
        occupied,
        depths,
        sparse,
        residual_scores=np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        coverage_scores=np.zeros(6),
        depth_confidences=np.zeros(6),
        frame_id=5,
    )

    assert routed[:2].tolist() == occupied[:2].tolist()
    assert routed.tolist() == [False, True, True, False, True, False]
    summary = cover.summary()
    assert summary["evidence_quota_routed_band_rows"] == [0, 2, 2]
    assert summary["evidence_quota_reassigned_band_rows"] == [0, 2, 2]


def test_projective_quota_round_robins_cells_and_routes_sparse_rows():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "evidence_quota_routing": True,
            "quota_route_sparse": True,
            "quota_projective_cell_px": 10.0,
            "quota_unoccupied_bonus": 0.0,
            "shuffle_depth_edges_m": [20.0],
        }
    )
    occupied = np.asarray([False, False, True, True])
    routed = cover.route_evidence_quota(
        occupied,
        depths=np.asarray([10.0] * 4),
        sparse_valid=np.asarray([True] * 4),
        residual_scores=np.asarray([1.0, 0.9, 0.8, 0.1]),
        coverage_scores=np.zeros(4),
        depth_confidences=np.zeros(4),
        frame_id=2,
        uv=np.asarray([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [21.0, 1.0]]),
    )

    assert routed.tolist() == [False, True, True, False]
    summary = cover.summary()
    assert summary["projective_quota_rows"] == 4
    assert summary["evidence_quota_reassigned_rows"] == 2


def _world_to_camera(center, yaw_degrees):
    yaw = np.deg2rad(float(yaw_degrees))
    camera_to_world = np.asarray(
        (
            (np.cos(yaw), 0.0, np.sin(yaw)),
            (0.0, 1.0, 0.0),
            (-np.sin(yaw), 0.0, np.cos(yaw)),
        ),
        dtype=np.float32,
    )
    rotation = camera_to_world.T
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ np.asarray(center, dtype=np.float32)
    return pose


def _revisit_cover(**updates):
    config = {
        "enabled": True,
        "evidence_quota_routing": True,
        "quota_projective_cell_px": 10.0,
        "quota_route_sparse": True,
        "projective_revisit_gate": "pose",
        "revisit_min_frame_gap": 2,
        "revisit_max_position_distance_m": 0.5,
        "revisit_max_view_angle_deg": 5.0,
        "revisit_min_cumulative_turn_deg": 90.0,
    }
    config.update(updates)
    return FrontViewScaleCover(config)


def test_stationary_and_straight_exploration_do_not_certify_revisit():
    stationary = _revisit_cover()
    for frame_id in range(4):
        assert not stationary.observe_raw_pose(
            _world_to_camera((0.0, 0.0, 0.0), 0.0), frame_id
        )

    straight = _revisit_cover()
    for frame_id in range(4):
        assert not straight.observe_raw_pose(
            _world_to_camera((float(frame_id), 0.0, 0.0), 0.0), frame_id
        )

    assert stationary.summary()["revisit_certificate_certified_frames"] == 0
    assert straight.summary()["revisit_certificate_certified_frames"] == 0


def test_leave_and_return_loop_certifies_revisit():
    cover = _revisit_cover(revisit_min_frame_gap=3, revisit_min_cumulative_turn_deg=180.0)
    trajectory = (
        ((0.0, 0.0, 0.0), 0.0),
        ((4.0, 0.0, 0.0), 90.0),
        ((4.0, 0.0, 4.0), 180.0),
        ((0.0, 0.0, 0.0), 0.0),
    )
    certificates = [
        cover.observe_raw_pose(_world_to_camera(center, yaw), frame_id)
        for frame_id, (center, yaw) in enumerate(trajectory)
    ]

    assert certificates == [False, False, False, True]
    summary = cover.summary()
    assert summary["revisit_certificate_first_frame"] == 3
    assert summary["revisit_support_distance_mean_m"] == 0.0
    assert summary["revisit_support_turn_mean_deg"] >= 180.0


def test_revisit_gate_preserves_tsc_routing_before_certification():
    cover = _revisit_cover()
    cover.observe_raw_pose(_world_to_camera((0.0, 0.0, 0.0), 0.0), frame_id=0)
    occupied = np.asarray([False, False, True, True])
    routed = cover.route_evidence_quota(
        occupied,
        depths=np.asarray([10.0] * 4),
        sparse_valid=np.asarray([True] * 4),
        residual_scores=np.asarray([1.0, 0.9, 0.8, 0.1]),
        coverage_scores=np.zeros(4),
        depth_confidences=np.ones(4),
        frame_id=0,
        uv=np.asarray([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [21.0, 1.0]]),
    )

    assert np.array_equal(routed, occupied)
    assert cover.summary()["revisit_gate_skipped_rows"] == 4


def test_certified_projective_routing_preserves_source_depth_quotas():
    cover = _revisit_cover(revisit_min_cumulative_turn_deg=180.0)
    trajectory = (
        ((0.0, 0.0, 0.0), 0.0),
        ((4.0, 0.0, 0.0), 90.0),
        ((0.0, 0.0, 0.0), 0.0),
    )
    for frame_id, (center, yaw) in enumerate(trajectory):
        certified = cover.observe_raw_pose(
            _world_to_camera(center, yaw), frame_id
        )
    assert certified

    occupied = np.asarray([False, True, True, False, False, True, True, False])
    depths = np.asarray([10.0] * 4 + [30.0] * 4)
    sparse = np.asarray([True, True, False, False] * 2)
    scores = np.asarray([1.0, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4])
    uv = np.asarray([[1.0 + 12.0 * row, 1.0] for row in range(8)])
    routed = cover.route_evidence_quota(
        occupied,
        depths,
        sparse,
        scores,
        np.zeros(8),
        np.ones(8),
        frame_id=2,
        uv=uv,
    )

    bands = np.digitize(depths, cover.config["shuffle_depth_edges_m"])
    for is_sparse in (False, True):
        for band in np.unique(bands):
            rows = (sparse == is_sparse) & (bands == band)
            assert int(np.sum(routed[rows])) == int(np.sum(occupied[rows]))


def test_geometry_shared_appearance_packets_keep_distinct_birth_colors():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "radius_multiplier": 1.0,
            "color_distance_threshold": 0.2,
        }
    )
    cover.register(
        [[0.0, 0.0, 0.0]],
        target_size=1.0,
        colors=[[1.0, 0.0, 0.0]],
    )
    occupied = cover.occupied(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        target_size=1.0,
        colors=[[0.9, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )

    assert occupied.tolist() == [True, False]
    assert cover.summary()["appearance_packet_bypass_rows"] == 1


def test_appearance_certificate_requires_depthcov_residual_and_confidence():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "appearance_certificate_enabled": True,
            "appearance_min_residual": 0.1,
            "appearance_min_depth_confidence": 0.5,
        }
    )

    eligible = cover.appearance_certificates(
        residual_scores=[0.2, 0.2, 0.05, 0.2],
        depth_confidences=[1.0, 1.0, 1.0, 0.4],
        sparse_valid=[False, True, False, False],
        depths=[10.0, 10.0, 30.0, 30.0],
        frame_id=3,
    )

    assert eligible.tolist() == [True, False, False, False]


def test_appearance_certificate_can_admit_residual_supported_sparse_track():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "appearance_certificate_enabled": True,
            "appearance_allow_sparse": True,
            "appearance_min_residual": 0.1,
        }
    )

    eligible = cover.appearance_certificates(
        residual_scores=[0.2, 0.0],
        depth_confidences=[1.0, 1.0],
        sparse_valid=[True, True],
        depths=[10.0, 10.0],
        frame_id=3,
    )

    assert eligible.tolist() == [True, False]


def test_uncertified_color_change_remains_covered():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "radius_multiplier": 1.0,
            "color_distance_threshold": 0.2,
        }
    )
    cover.register(
        [[0.0, 0.0, 0.0]],
        target_size=1.0,
        colors=[[1.0, 0.0, 0.0]],
    )

    occupied = cover.occupied(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        target_size=1.0,
        colors=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        appearance_eligible=[False, True],
    )

    assert occupied.tolist() == [True, False]


def test_shuffled_appearance_certificates_preserve_depth_band_counts():
    config = {
        "enabled": True,
        "appearance_certificate_enabled": True,
        "appearance_min_residual": 0.1,
        "shuffle_appearance_certificates": True,
        "shuffle_seed": 13,
        "shuffle_depth_edges_m": [20.0],
    }
    cover = FrontViewScaleCover(config)
    residual = np.asarray([0.2, 0.0, 0.2, 0.0, 0.2, 0.0])
    depths = np.asarray([10.0, 10.0, 10.0, 30.0, 30.0, 30.0])
    sparse = np.zeros(6, dtype=np.bool_)

    first = cover.appearance_certificates(
        residual, np.ones(6), sparse, depths, frame_id=5
    )
    second = FrontViewScaleCover(config).appearance_certificates(
        residual, np.ones(6), sparse, depths, frame_id=5
    )

    assert np.array_equal(first, second)
    original = residual >= 0.1
    for band in (0, 1):
        rows = np.digitize(depths, [20.0]) == band
        assert int(first[rows].sum()) == int(original[rows].sum())


def test_dynamic_cover_links_finer_birth_to_nearest_coarse_uid():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "radius_multiplier": 1.0,
            "dynamic_handoff_enabled": True,
            "rebuild_rows": 100,
        }
    )
    cover.register([[0.0, 0.0, 0.0]], 1.0, uids=[10])

    occupied, parents = cover.occupied_with_parents(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], 0.5
    )

    assert occupied.tolist() == [False, False]
    assert parents.tolist() == [10, -1]


def test_area_mature_handoff_fades_parent_only_when_projected_large():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "dynamic_handoff_enabled": True,
            "handoff_radius_start_px": 1.0,
            "handoff_radius_end_px": 2.0,
            "handoff_parent_floor": 0.0,
            "handoff_area_full": 0.25,
        }
    )
    cover.register([[0.0, 0.0, 10.0]], 1.0, uids=[0])
    cover.register([[0.0, 0.0, 10.0]], 0.5, uids=[1], parent_uids=[0])
    means = torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
    scales = torch.full((2, 3), 0.1)

    near = cover.render_handoff_multipliers(
        torch.tensor([0, 1]), means, scales, torch.eye(4), torch.tensor([100.0])
    )
    far = cover.render_handoff_multipliers(
        torch.tensor([0, 1]), means, scales, torch.eye(4), torch.tensor([10.0])
    )

    assert torch.allclose(near, torch.tensor([0.0, 1.0]), atol=1.0e-6)
    assert torch.allclose(far, torch.tensor([1.0, 1.0]), atol=1.0e-6)


def test_releasing_fine_child_restores_parent_responsibility():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "dynamic_handoff_enabled": True,
            "handoff_radius_start_px": 1.0,
            "handoff_radius_end_px": 2.0,
            "handoff_parent_floor": 0.0,
            "handoff_area_full": 0.25,
        }
    )
    cover.register([[0.0, 0.0, 10.0]], 1.0, uids=[0])
    cover.register([[0.0, 0.0, 10.0]], 0.5, uids=[1], parent_uids=[0])
    cover.release([1])

    multipliers = cover.render_handoff_multipliers(
        torch.tensor([0, 1]),
        torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]]),
        torch.full((2, 3), 0.1),
        torch.eye(4),
        torch.tensor([100.0]),
    )

    assert torch.allclose(multipliers, torch.ones(2), atol=1.0e-6)


def test_active_occupancy_release_removes_pruned_spatial_blocker():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "active_occupancy_enabled": True,
            "rebuild_rows": 1,
        }
    )
    point = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    cover.register(point, 1.0, uids=[4])
    assert cover.occupied(point, 1.0).tolist() == [True]

    assert cover.release([4]) == 1

    assert cover.occupied(point, 1.0).tolist() == [False]
    summary = cover.summary()
    assert summary["released_cover_rows"] == 1
    assert summary["active_cover_rows"] == 0


def test_shuffled_release_preserves_count_but_breaks_spatial_identity():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "active_occupancy_enabled": True,
            "shuffle_lifecycle_release": True,
            "shuffle_seed": 9,
            "rebuild_rows": 1,
        }
    )
    points = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    cover.register(points, 1.0, uids=[4, 5])

    cover.release([4])

    occupied = cover.occupied(points, 1.0)
    assert int(occupied.sum()) == 1
    assert cover.summary()["shuffled_release_rows"] == 1


def test_evidence_ordered_cover_lets_sparse_track_supersede_depth_proxy():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "source_priority_enabled": True,
            "rebuild_rows": 100,
        }
    )
    point = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    cover.register(point, 1.0, source_ranks=[1])

    assert cover.occupied(point, 1.0, source_ranks=[1]).tolist() == [True]
    assert cover.occupied(point, 1.0, source_ranks=[2]).tolist() == [False]

    cover.register(point, 1.0, source_ranks=[2])
    assert cover.occupied(point, 1.0, source_ranks=[1]).tolist() == [True]
    assert cover.occupied(point, 1.0, source_ranks=[2]).tolist() == [True]
    assert cover.summary()["priority_bypass_rows"] == 1


def test_shuffled_source_priority_preserves_rank_counts_per_depth_band():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "source_priority_enabled": True,
            "shuffle_source_priority": True,
            "shuffle_seed": 11,
            "shuffle_depth_edges_m": [20.0],
        }
    )
    sparse = np.asarray([True, False, True, False, True, False])
    depths = np.asarray([10.0, 10.0, 10.0, 30.0, 30.0, 30.0])

    first = cover.candidate_source_ranks(sparse, depths, frame_id=7)
    second = FrontViewScaleCover(cover.config).candidate_source_ranks(
        sparse, depths, frame_id=7
    )

    assert np.array_equal(first, second)
    original = np.where(sparse, 2, 1)
    for band in (0, 1):
        rows = np.digitize(depths, [20.0]) == band
        assert sorted(first[rows].tolist()) == sorted(original[rows].tolist())


def test_projected_handoff_activates_only_resolvable_visible_far_births():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "projected_handoff_enabled": True,
            "projected_handoff_radius_px": 0.5,
        }
    )
    uids = cover.allocate_uids(3)
    cover.stage_projected_handoff(
        np.asarray(
            ((0.0, 0.0, 10.0), (0.0, 0.0, 10.0), (20.0, 0.0, 10.0))
        ),
        np.asarray((1.0, 0.1, 1.0)),
        uids=uids,
    )

    activated = cover.activate_projected_handoff(
        np.eye(4, dtype=np.float32),
        focal_pixels=10.0,
        image_width=20,
        image_height=20,
        principal_x=10.0,
        principal_y=10.0,
        near=0.1,
        far=100.0,
        frame_id=1,
    )

    assert activated == 1
    assert cover.stats["projected_handoff_staged_rows"] == 3
    assert cover.stats["projected_handoff_activated_rows"] == 1
    assert len(cover._far_points) == 2
    assert cover.occupied(np.asarray(((0.0, 0.0, 10.0),)), 1.0).tolist() == [
        True
    ]
    assert (
        cover.activate_projected_handoff(
            np.eye(4, dtype=np.float32),
            10.0,
            20,
            20,
            10.0,
            10.0,
            0.1,
            100.0,
            1,
        )
        == 0
    )


def test_projected_handoff_shuffle_preserves_activation_count():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "projected_handoff_enabled": True,
            "projected_handoff_radius_px": 0.5,
            "shuffle_projected_handoff": True,
            "shuffle_seed": 9,
        }
    )
    uids = cover.allocate_uids(4)
    cover.stage_projected_handoff(
        np.asarray(
            (
                (0.0, 0.0, 10.0),
                (0.0, 0.0, 10.0),
                (20.0, 0.0, 10.0),
                (30.0, 0.0, 10.0),
            )
        ),
        np.asarray((1.0, 0.1, 1.0, 1.0)),
        uids=uids,
    )

    activated = cover.activate_projected_handoff(
        np.eye(4, dtype=np.float32),
        10.0,
        20,
        20,
        10.0,
        10.0,
        0.1,
        100.0,
        2,
    )

    assert activated == 1
    assert cover.stats["shuffled_projected_handoff_rows"] == 1
    assert len(cover._far_points) == 3


def test_sparse_track_identity_supersedes_spatial_proximity():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "sparse_track_identity_enabled": True,
            "rebuild_rows": 1,
        }
    )
    point = np.asarray(((0.0, 0.0, 0.0),), dtype=np.float32)
    cover.register(point, 1.0)
    spatial = cover.occupied(point, 1.0)
    assert spatial.tolist() == [True]

    first = cover.apply_sparse_track_identity(
        spatial, [17], [True], [10.0], frame_id=1
    )
    assert first.tolist() == [False]
    cover.register_sparse_tracks([17], [True])

    repeated = cover.apply_sparse_track_identity(
        spatial, [17], [True], [10.0], frame_id=2
    )
    distinct = cover.apply_sparse_track_identity(
        spatial, [18], [True], [10.0], frame_id=2
    )
    assert repeated.tolist() == [True]
    assert distinct.tolist() == [False]

    cover.release_sparse_tracks([17])
    rebirth = cover.apply_sparse_track_identity(
        spatial, [17], [True], [10.0], frame_id=3
    )
    assert rebirth.tolist() == [False]


def test_sparse_track_identity_rejects_duplicate_within_batch():
    cover = FrontViewScaleCover(
        {"enabled": True, "sparse_track_identity_enabled": True}
    )
    occupied = cover.apply_sparse_track_identity(
        [True, True, True],
        [4, 4, 5],
        [True, True, True],
        [10.0, 10.0, 10.0],
        frame_id=1,
    )

    assert occupied.tolist() == [False, True, False]


def test_shuffled_sparse_track_identity_preserves_repeat_count_per_band():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "sparse_track_identity_enabled": True,
            "shuffle_sparse_track_identity": True,
            "shuffle_seed": 12,
            "shuffle_depth_edges_m": [20.0],
        }
    )
    cover.register_sparse_tracks([1, 3], [True, True])
    occupied = cover.apply_sparse_track_identity(
        [True, True, True, True],
        [1, 2, 3, 4],
        [True, True, True, True],
        [10.0, 10.0, 30.0, 30.0],
        frame_id=5,
    )

    assert int(np.sum(occupied[:2])) == 1
    assert int(np.sum(occupied[2:])) == 1
    assert cover.summary()["shuffled_sparse_track_rows"] == 4


def test_directional_ownership_allows_novel_view_of_same_location():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "directional_ownership_enabled": True,
            "directional_max_angle_deg": 30.0,
            "rebuild_rows": 1,
        }
    )
    point = np.asarray(((0.0, 0.0, 0.0),), dtype=np.float32)
    cover.register(point, 1.0, view_directions=[[0.0, 0.0, 1.0]])

    same = cover.occupied(
        point, 1.0, view_directions=[[0.0, 0.0, 1.0]]
    )
    novel = cover.occupied(
        point, 1.0, view_directions=[[0.0, 0.0, -1.0]]
    )

    assert same.tolist() == [True]
    assert novel.tolist() == [False]
    assert cover.summary()["directional_bypass_rows"] == 1


def test_directional_shuffle_is_reproducible_and_preserves_vectors_per_band():
    config = {
        "enabled": True,
        "directional_ownership_enabled": True,
        "shuffle_directional_ownership": True,
        "shuffle_seed": 7,
        "shuffle_depth_edges_m": [20.0],
    }
    points = np.asarray(
        ((1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (-1.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    depths = np.asarray((10.0, 10.0, 30.0), dtype=np.float32)
    sparse = np.asarray((False, False, False))
    first_cover = FrontViewScaleCover(config)
    second_cover = FrontViewScaleCover(config)

    first = first_cover.candidate_view_directions(
        points, np.zeros(3), depths, sparse, frame_id=4
    )
    second = second_cover.candidate_view_directions(
        points, np.zeros(3), depths, sparse, frame_id=4
    )
    original = points / np.linalg.norm(points, axis=1, keepdims=True)

    assert np.array_equal(first, second)
    assert sorted(map(tuple, first[:2])) == sorted(map(tuple, original[:2]))
    assert np.allclose(first[2], original[2])
    assert first_cover.summary()["shuffled_directional_rows"] == 2


def test_frustum_conflict_budget_is_depthcov_only_and_residual_ranked():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "directional_ownership_enabled": True,
            "directional_max_angle_deg": 30.0,
            "frustum_conflict_budget_enabled": True,
            "frustum_conflict_min_fraction": 0.0,
            "frustum_conflict_min_rows": 1,
            "frustum_conflict_budget_rows": 1,
            "rebuild_rows": 1,
        }
    )
    owner = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    points = np.repeat(owner, 4, axis=0)
    cover.register(owner, 1.0, view_directions=[[0.0, 0.0, 1.0]])

    occupied = cover.occupied(
        points,
        1.0,
        view_directions=np.repeat([[0.0, 0.0, -1.0]], 4, axis=0),
        residual_scores=[1.0, 0.1, 0.9, 0.5],
        depth_confidences=[1.0, 1.0, 1.0, 1.0],
        sparse_valid=[True, False, False, False],
        depths=[10.0, 10.0, 10.0, 10.0],
        frame_id=7,
    )

    assert occupied.tolist() == [True, True, False, True]
    summary = cover.summary()
    assert summary["frustum_conflict_selected_rows"] == 1
    assert summary["frustum_conflict_sparse_suppressed_rows"] == 1
    assert summary["directional_bypass_rows"] == 1


def test_shuffled_frustum_conflict_preserves_count_per_depth_band():
    cover = FrontViewScaleCover(
        {
            "enabled": True,
            "directional_ownership_enabled": True,
            "directional_max_angle_deg": 30.0,
            "frustum_conflict_budget_enabled": True,
            "frustum_conflict_min_fraction": 0.0,
            "frustum_conflict_min_rows": 1,
            "frustum_conflict_budget_rows": 2,
            "shuffle_frustum_conflict_budget": True,
            "shuffle_depth_edges_m": [20.0],
            "shuffle_seed": 9,
            "rebuild_rows": 1,
        }
    )
    owner = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    points = np.repeat(owner, 4, axis=0)
    depths = np.asarray([10.0, 10.0, 30.0, 30.0], dtype=np.float32)
    cover.register(owner, 1.0, view_directions=[[0.0, 0.0, 1.0]])

    occupied = cover.occupied(
        points,
        1.0,
        view_directions=np.repeat([[0.0, 0.0, -1.0]], 4, axis=0),
        residual_scores=[0.9, 0.1, 0.8, 0.2],
        depth_confidences=np.ones(4),
        sparse_valid=np.zeros(4, dtype=np.bool_),
        depths=depths,
        frame_id=11,
    )
    selected = ~occupied

    assert int(np.sum(selected)) == 2
    assert int(np.sum(selected[depths < 20.0])) == 1
    assert int(np.sum(selected[depths >= 20.0])) == 1
    assert cover.summary()["shuffled_frustum_conflict_rows"] == 2
