from pathlib import Path

import numpy as np
import yaml

from scripts.build_panoair_learned_tracks import (
    UnionFind,
    camera_centers,
    encode_node,
    linear_triangulate,
    maximum_triangulation_angle_deg,
    projection_errors,
)
from scripts.convert_panoair_orbslam3_vi import load_t_cam_imu
from scripts.fuse_panoair_canonical_tracks import visible_point_ids


def make_pose(center):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = -np.asarray(center, dtype=np.float64)
    return pose


def project(pose, point):
    intrinsics = np.asarray(
        [[320.0, 0.0, 480.0], [0.0, 320.0, 320.0], [0.0, 0.0, 1.0]]
    )
    camera = pose[:3] @ np.append(point, 1.0)
    screen = intrinsics @ camera
    return screen[:2] / screen[2]


def test_linear_multiview_triangulation_recovers_world_point():
    point = np.asarray([0.4, -0.2, 8.0])
    poses = [make_pose(center) for center in ((0, 0, 0), (1, 0, 0), (0, 0.5, 0))]
    pixels = [project(pose, point) for pose in poses]
    estimated = linear_triangulate(poses, pixels)
    errors, depths = projection_errors(estimated, poses, pixels)
    assert np.linalg.norm(estimated - point) < 1.0e-7
    assert errors.max() < 1.0e-7
    assert np.all(depths > 0)


def test_triangulation_angle_distinguishes_weak_baseline():
    point = np.asarray([0.0, 0.0, 20.0])
    weak = camera_centers(np.asarray([make_pose((0, 0, 0)), make_pose((0.01, 0, 0))]))
    strong = camera_centers(np.asarray([make_pose((0, 0, 0)), make_pose((1.0, 0, 0))]))
    assert maximum_triangulation_angle_deg(point, weak) < 0.1
    assert maximum_triangulation_angle_deg(point, strong) > 2.0


def test_union_find_builds_persistent_track_identity():
    graph = UnionFind()
    nodes = [encode_node(frame, 7 + frame) for frame in range(4)]
    graph.union(nodes[0], nodes[1])
    graph.union(nodes[1], nodes[2])
    graph.union(nodes[2], nodes[3])
    assert len({graph.find(node) for node in nodes}) == 1


def test_visible_point_selection_rejects_behind_camera_and_caps_count():
    pose = np.eye(4, dtype=np.float64)
    points = np.asarray(
        [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [-0.1, 0.0, 3.0], [0.0, 0.0, -2.0]]
    )
    selected = visible_point_ids(pose, points, max_points=2, grid_size=8)
    assert len(selected) == 2
    assert 3 not in selected


def test_orbslam3_camera_imu_calibration_is_inverted():
    calibration = Path(
        "configs/panoair_preprocess/PanoAir_seq1_mono_inertial.yaml"
    )
    transform = load_t_cam_imu(calibration)
    expected_body_from_camera = np.asarray(
        [
            [0.01082668, -0.02293296, -0.99967838, -0.00886266],
            [-0.01255263, 0.99965508, -0.02306837, 0.00793965],
            [0.99986260, 0.01279835, 0.01053508, 0.01372886],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(
        transform, np.linalg.inv(expected_body_from_camera), atol=1.0e-7
    )


def test_kalibr_camera_imu_calibration_is_used_directly(tmp_path):
    expected = np.eye(4)
    expected[:3, 3] = [0.1, -0.2, 0.3]
    calibration = tmp_path / "camchain.yaml"
    calibration.write_text(
        yaml.safe_dump({"cam0": {"T_cam_imu": expected.tolist()}}),
        encoding="utf-8",
    )

    np.testing.assert_allclose(load_t_cam_imu(calibration), expected)
