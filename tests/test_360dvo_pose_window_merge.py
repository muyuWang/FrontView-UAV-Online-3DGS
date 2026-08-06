import numpy as np
from scipy.spatial.transform import Rotation

from scripts import merge_360dvo_orbslam3_pose_windows as merge


def yaw(degrees: float) -> np.ndarray:
    return Rotation.from_euler("y", degrees, degrees=True).as_matrix()


def segment(start: int, end: int, offset_deg: float) -> dict[int, np.ndarray]:
    offset = yaw(offset_deg)
    return {index: offset.T @ yaw(2.0 * index) for index in range(start, end + 1)}


def test_globally_aligned_overlapping_windows_are_blended():
    rotations, certificate = merge.merge_segment_rotations(
        [segment(0, 5, 0.0), segment(3, 9, 0.0), segment(7, 12, 0.0)],
        frame_count=13,
        blend_ramp_frames=3,
        max_overlap_disagreement_deg=15.0,
        max_gap_fill_frames=2,
        max_edge_fill_frames=2,
    )

    errors = [
        merge.rotation_angle_deg(rotation, yaw(2.0 * index))
        for index, rotation in enumerate(rotations)
    ]
    assert max(errors) < 1.0e-10
    assert certificate["segment_count"] == 3
    assert certificate["segments"][1]["overlap_frame_count"] == 3
    assert certificate["segments"][2]["overlap_frame_count"] == 3
    assert certificate["filled_internal_frames"] == 0


def test_small_uncovered_edges_are_certified_and_filled():
    rotations, certificate = merge.merge_segment_rotations(
        [segment(1, 5, 0.0)],
        frame_count=7,
        blend_ramp_frames=2,
        max_overlap_disagreement_deg=15.0,
        max_gap_fill_frames=1,
        max_edge_fill_frames=1,
    )

    np.testing.assert_allclose(rotations[0], rotations[1])
    np.testing.assert_allclose(rotations[6], rotations[5])
    assert certificate["filled_leading_frames"] == 1
    assert certificate["filled_trailing_frames"] == 1


def test_large_internal_gap_is_rejected():
    try:
        merge.merge_segment_rotations(
            [segment(0, 2, 0.0), segment(5, 8, 0.0)],
            frame_count=9,
            blend_ramp_frames=2,
            max_overlap_disagreement_deg=15.0,
            max_gap_fill_frames=1,
            max_edge_fill_frames=1,
        )
    except RuntimeError as error:
        assert "shared frames" in str(error)
    else:
        raise AssertionError("Expected disconnected pose windows to be rejected")


def test_inconsistent_overlap_is_rejected_without_rotating_gt_centers():
    try:
        merge.merge_segment_rotations(
            [segment(0, 5, 0.0), segment(3, 9, 25.0)],
            frame_count=10,
            blend_ramp_frames=3,
            max_overlap_disagreement_deg=15.0,
            max_gap_fill_frames=1,
            max_edge_fill_frames=1,
        )
    except RuntimeError as error:
        assert "overlap rotation p95" in str(error)
    else:
        raise AssertionError("Expected an inconsistent pose window to be rejected")


def test_covering_selector_ignores_redundant_failed_windows():
    selected, certificate = merge.select_covering_segments(
        [
            segment(0, 5, 0.0),
            segment(3, 7, 30.0),
            segment(3, 9, 0.0),
            segment(7, 12, 0.0),
        ],
        frame_count=13,
        max_overlap_disagreement_deg=15.0,
        max_gap_fill_frames=1,
        max_gap_rotation_deg=30.0,
        max_edge_fill_frames=1,
    )

    assert [(min(item), max(item)) for item in selected] == [(0, 5), (3, 9), (7, 12)]
    assert certificate["selected_segment_count"] == 3
    assert certificate["ignored_segment_count"] == 1


def test_covering_selector_accepts_only_small_rotation_certified_gaps():
    selected, certificate = merge.select_covering_segments(
        [segment(0, 5, 0.0), segment(7, 12, 0.0)],
        frame_count=13,
        max_overlap_disagreement_deg=15.0,
        max_gap_fill_frames=2,
        max_gap_rotation_deg=30.0,
        max_edge_fill_frames=1,
    )

    assert [(min(item), max(item)) for item in selected] == [(0, 5), (7, 12)]
    assert certificate["selected_path"][1]["connection"] == "certified_gap"
    assert certificate["selected_path"][1]["filled_gap_frames"] == 1


def test_covering_selector_accepts_a_certified_leading_edge_fill():
    selected, certificate = merge.select_covering_segments(
        [segment(42, 100, 0.0), segment(80, 250, 0.0)],
        frame_count=300,
        max_overlap_disagreement_deg=15.0,
        max_gap_fill_frames=60,
        max_gap_rotation_deg=180.0,
        max_edge_fill_frames=60,
    )

    assert [(min(item), max(item)) for item in selected] == [(42, 100), (80, 250)]
    assert certificate["selected_path"][0]["source_frame_start"] == 42


def test_source_rotation_prior_requires_dominant_visual_consensus():
    source = np.stack([yaw(2.0 * index) for index in range(20)])
    correction = Rotation.from_euler("x", 170.0, degrees=True).as_matrix()
    visual = {
        index: source[index] @ correction
        for index in range(15)
    }
    outliers = {
        index: source[index] @ yaw(60.0)
        for index in range(15, 20)
    }

    calibrated, certificate = merge.calibrate_source_rotation_prior(
        source,
        [visual, outliers],
        max_residual_deg=10.0,
        min_inlier_fraction=0.70,
        min_observations=10,
        min_frame_span_fraction=0.70,
    )

    expected = source @ correction
    errors = [
        merge.rotation_angle_deg(left, right)
        for left, right in zip(calibrated, expected)
    ]
    assert max(errors) < 1.0e-10
    assert certificate["inlier_count"] == 15
    assert certificate["inlier_fraction"] == 0.75


def test_source_rotation_prior_rejects_fragmented_visual_support():
    source = np.stack([yaw(2.0 * index) for index in range(12)])
    fragmented = {
        index: source[index] @ yaw(20.0 * index)
        for index in range(12)
    }

    try:
        merge.calibrate_source_rotation_prior(
            source,
            [fragmented],
            max_residual_deg=5.0,
            min_inlier_fraction=0.75,
            min_observations=4,
            min_frame_span_fraction=0.75,
        )
    except RuntimeError as error:
        assert "lacks visual consensus" in str(error)
    else:
        raise AssertionError("Expected a fragmented source rotation prior to be rejected")
