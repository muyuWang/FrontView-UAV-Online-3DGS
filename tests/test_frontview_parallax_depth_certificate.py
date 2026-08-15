import numpy as np

from utils_new.frontview_parallax_depth_certificate import (
    certificate_information_gain,
    consensus_triangulate_parallax_depth,
    project_world_point,
    triangulate_parallax_depth,
)


def _pose(center):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = -np.asarray(center, dtype=np.float64)
    return pose


def test_parallax_triangulation_recovers_metric_depth():
    intrinsics = np.asarray(
        [[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0.0, 0.0, 1.0]]
    )
    point = np.asarray([2.0, -1.0, 60.0])
    poses = [_pose([0.0, 0.0, 0.0]), _pose([4.0, 0.0, 0.0])]
    pixels = [project_world_point(pose, intrinsics, point)[0] for pose in poses]
    estimate = triangulate_parallax_depth(
        poses, intrinsics, pixels, current_view_index=1
    )
    assert estimate.finite
    assert np.allclose(estimate.world_point, point, atol=1.0e-7)
    assert abs(estimate.current_depth - 60.0) < 1.0e-7
    assert estimate.maximum_parallax_deg > 3.0
    assert estimate.log_depth_std < 0.02


def test_longer_baseline_reduces_depth_uncertainty():
    intrinsics = np.asarray(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    )
    point = np.asarray([0.0, 0.0, 80.0])
    short = [_pose([0.0, 0.0, 0.0]), _pose([0.5, 0.0, 0.0])]
    long = [_pose([0.0, 0.0, 0.0]), _pose([5.0, 0.0, 0.0])]
    short_pixels = [project_world_point(pose, intrinsics, point)[0] for pose in short]
    long_pixels = [project_world_point(pose, intrinsics, point)[0] for pose in long]
    short_estimate = triangulate_parallax_depth(short, intrinsics, short_pixels)
    long_estimate = triangulate_parallax_depth(long, intrinsics, long_pixels)
    assert long_estimate.log_depth_std < short_estimate.log_depth_std


def test_information_gain_is_scale_free_and_monotonic():
    weak = certificate_information_gain(0.1, 0.2)
    strong = certificate_information_gain(0.1, 0.02)
    scaled = certificate_information_gain(0.2, 0.04)
    assert strong > weak > 0.0
    assert np.isclose(strong, scaled)


def test_consensus_rejects_inconsistent_reference_and_recovers_depth():
    intrinsics = np.asarray(
        [[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0.0, 0.0, 1.0]]
    )
    point = np.asarray([1.0, 0.5, 70.0])
    current = _pose([0.0, 0.0, 0.0])
    references = [_pose([2.0, 0.0, 0.0]), _pose([4.0, 0.0, 0.0]), _pose([6.0, 0.0, 0.0])]
    current_pixel = project_world_point(current, intrinsics, point)[0]
    reference_pixels = [
        project_world_point(pose, intrinsics, point)[0] for pose in references
    ]
    reference_pixels[2] = reference_pixels[2] + np.asarray([25.0, 0.0])
    consensus = consensus_triangulate_parallax_depth(
        current,
        current_pixel,
        references,
        reference_pixels,
        intrinsics,
    )
    assert consensus is not None
    assert consensus.reference_indices == (0, 1)
    assert np.allclose(consensus.estimate.world_point, point, atol=1.0e-6)
    assert consensus.maximum_pairwise_chi2 < 3.841458820694124


def test_consensus_fails_closed_with_one_reference():
    intrinsics = np.eye(3)
    result = consensus_triangulate_parallax_depth(
        _pose([0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0]),
        [_pose([1.0, 0.0, 0.0])],
        [np.asarray([-0.1, 0.0])],
        intrinsics,
    )
    assert result is None
