import numpy as np
from scipy.spatial.transform import Rotation

from scripts.audit_360dvo_pose_contract import (
    pose_conventions,
    score_candidate,
    signed_axis_rotations,
)
from scripts.select_360dvo_pose_contract import canonical_poses
from tools.convert_360dvo_frontview_to_online3dgs import (
    pose_row_to_t_camera_world,
)


def _project(rotation_c2w, center, points, intrinsic):
    camera_points = (points - center) @ rotation_c2w
    pixels = camera_points @ intrinsic.T
    return pixels[:, :2] / pixels[:, 2:3]


def _synthetic_rows_and_samples():
    centers = np.asarray([[0.0, 0.0, 0.0], [0.8, 0.0, 0.1], [1.6, 0.1, 0.2]])
    w2c_rotations = Rotation.from_euler(
        "zyx", [[0.0, 0.0, 0.0], [5.0, -2.0, 1.0], [10.0, -3.0, 2.0]], degrees=True
    )
    rows = np.concatenate(
        (centers, w2c_rotations.as_quat()),
        axis=1,
    )
    axis = np.diag([1.0, -1.0, -1.0])
    camera_to_world = w2c_rotations.as_matrix().transpose(0, 2, 1) @ axis
    intrinsic = np.asarray([[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0.0, 0.0, 1.0]])
    camera_points = np.asarray(
        [
            [-1.0, -0.5, 8.0],
            [0.2, 0.1, 10.0],
            [1.3, -0.2, 12.0],
            [0.5, 0.8, 9.0],
        ]
    )
    world_points = camera_points @ camera_to_world[0].T + centers[0]
    samples = []
    for left, right in ((0, 1), (1, 2), (0, 2)):
        samples.append(
            (
                left,
                right,
                _project(camera_to_world[left], centers[left], world_points, intrinsic),
                _project(
                    camera_to_world[right], centers[right], world_points, intrinsic
                ),
            )
        )
    return rows, samples, intrinsic, axis


def test_signed_axis_hypotheses_are_all_proper_rotations():
    hypotheses = signed_axis_rotations()

    assert len(hypotheses) == 24
    assert all(np.isclose(np.linalg.det(matrix), 1.0) for _, matrix in hypotheses)


def test_w2c_quaternion_with_front_axis_is_image_consistent():
    rows, samples, intrinsic, axis = _synthetic_rows_and_samples()
    conventions = {item.label: item for item in pose_conventions(rows)}

    correct = score_candidate(
        conventions["quat_w2c_xyz_center"],
        axis,
        samples,
        np.linalg.inv(intrinsic),
        threshold=1.5,
    )
    wrong = score_candidate(
        conventions["quat_c2w_xyz_center"],
        np.eye(3),
        samples,
        np.linalg.inv(intrinsic),
        threshold=1.5,
    )

    assert correct["inlier_fraction"] == 1.0
    assert correct["median_error_px"] < 1.0e-6
    assert wrong["median_error_px"] > correct["median_error_px"] + 1.0


def test_selected_pose_contract_is_normalized_without_changing_path_length():
    rows, _, _, axis = _synthetic_rows_and_samples()
    convention = {item.label: item for item in pose_conventions(rows)}[
        "quat_w2c_xyz_center"
    ]

    poses, image_indices, certificate = canonical_poses(convention, axis, reverse=False)

    expected_length = np.linalg.norm(np.diff(rows[:, :3], axis=0), axis=1).sum()
    assert np.allclose(poses[0], np.eye(4), atol=1.0e-9)
    assert np.array_equal(image_indices, np.arange(len(rows)))
    assert np.isclose(certificate["trajectory_length_m"], expected_length)


def test_frame_shift_keeps_only_image_pose_overlap():
    rows, _, _, axis = _synthetic_rows_and_samples()
    convention = {item.label: item for item in pose_conventions(rows)}[
        "quat_w2c_xyz_center"
    ]

    poses, image_indices, certificate = canonical_poses(
        convention, axis, reverse=False, shift=1
    )

    assert poses.shape[0] == len(rows) - 1
    assert np.array_equal(image_indices, np.asarray([0, 1]))
    assert certificate["source_pose_start"] == 1
    assert certificate["source_pose_end_inclusive"] == 2
    assert certificate["frame_shift"] == 1


def test_converter_default_contract_uses_w2c_quaternion_and_camera_center():
    center = np.asarray([1.0, 2.0, 3.0])
    rotation = Rotation.from_euler("zyx", [12.0, -4.0, 3.0], degrees=True)
    row = np.concatenate((center, rotation.as_quat()))

    pose = pose_row_to_t_camera_world(row, "frontview-w2c-center")
    expected_rotation = np.diag([1.0, -1.0, -1.0]) @ rotation.as_matrix()

    assert np.allclose(pose[:3, :3], expected_rotation)
    assert np.allclose(pose[:3, 3], -expected_rotation @ center)
    assert np.allclose(np.linalg.inv(pose)[:3, 3], center)
