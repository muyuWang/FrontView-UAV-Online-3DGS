import numpy as np
from scipy.spatial.transform import Rotation

from scripts import merge_360dvo_orbslam3_pose_segments as merge


def camera(source_index: int, center, yaw_deg: float) -> dict:
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
    c2w[:3, 3] = np.asarray(center, dtype=np.float64)
    return {
        "source_frame_index": source_index,
        "T_camera_world": np.linalg.inv(c2w).tolist(),
    }


def test_slerp_rotation_uses_both_endpoints():
    left = Rotation.from_euler("y", 10.0, degrees=True).as_matrix()
    right = Rotation.from_euler("y", 30.0, degrees=True).as_matrix()

    np.testing.assert_allclose(merge.slerp_rotation(left, right, 0.0), left)
    np.testing.assert_allclose(merge.slerp_rotation(left, right, 1.0), right)
    middle = merge.slerp_rotation(left, right, 0.5)
    assert np.isclose(merge.rotation_angle_deg(left, middle), 10.0)


def test_merge_uses_gt_centers_and_normalizes_once():
    source = [camera(index, [index, 0.0, 0.0], 90.0) for index in range(6)]
    prefix = [camera(index, [99.0, 0.0, 0.0], index * 2.0) for index in range(5)]
    suffix = [camera(index, [-99.0, 0.0, 0.0], index * 2.0 + 4.0) for index in range(2, 6)]

    indices, poses, certificate = merge.build_merged_poses(
        source, prefix, suffix, 0, 5, normalize_first=True
    )
    centers = np.linalg.inv(poses)[:, :3, 3]

    assert indices == list(range(6))
    np.testing.assert_allclose(poses[0], np.eye(4), atol=1.0e-12)
    np.testing.assert_allclose(centers[:, 0], np.arange(6), atol=1.0e-12)
    np.testing.assert_allclose(centers[:, 1:], 0.0, atol=1.0e-12)
    assert certificate["overlap_start"] == 2
    assert certificate["overlap_end_inclusive"] == 4
    assert certificate["assignment"] == {
        "prefix": 2,
        "blended_overlap": 3,
        "suffix": 1,
    }
    assert certificate["gt_center_preservation_max_error_m"] < 1.0e-12
