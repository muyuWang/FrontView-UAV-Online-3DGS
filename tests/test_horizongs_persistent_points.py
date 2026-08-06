import numpy as np
import yaml

from tools.convert_airvln_to_online3dgs import project_points
from tools.convert_horizongs_real_to_online3dgs import visible_point_ids, write_config


def test_visible_point_ids_preserve_global_row_identity():
    intrinsics = np.eye(3, dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    points = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )

    point_ids = visible_point_ids(
        intrinsics, pose, points, width=2, height=2,
        min_depth=0.1, max_depth=10.0, max_points=-1,
    )

    np.testing.assert_array_equal(point_ids, np.asarray([0, 3]))
    np.testing.assert_array_equal(points[point_ids], points[[0, 3]])


def test_visible_point_ids_grid_cap_is_stable_and_spatially_balanced():
    intrinsics = np.asarray(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    points = np.asarray(
        [
            [0.1, 0.1, 1.0],
            [0.2, 0.2, 2.0],
            [2.0, 0.1, 1.0],
            [2.1, 0.2, 2.0],
            [0.1, 2.0, 1.0],
            [0.2, 2.1, 2.0],
        ],
        dtype=np.float32,
    )

    first = visible_point_ids(
        intrinsics, pose, points, width=32, height=32,
        min_depth=0.1, max_depth=10.0, max_points=3, grid_size=16,
    )
    second = visible_point_ids(
        intrinsics, pose, points, width=32, height=32,
        min_depth=0.1, max_depth=10.0, max_points=3, grid_size=16,
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, np.asarray([0, 2, 4]))
    uv, _ = project_points(intrinsics, pose, points[first])
    cells = np.floor_divide(uv.astype(np.int64), 16)
    assert len(np.unique(cells, axis=0)) == 3


def test_visible_point_ids_reject_nonfinite_projections():
    intrinsics = np.eye(3, dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    points = np.asarray(
        [[0.0, 0.0, 1.0], [np.nan, 0.0, 1.0]], dtype=np.float32
    )

    point_ids = visible_point_ids(
        intrinsics, pose, points, width=2, height=2,
        min_depth=0.1, max_depth=10.0, max_points=-1,
    )

    np.testing.assert_array_equal(point_ids, np.asarray([0]))


def test_generated_config_applies_point_budget_to_train_and_test(tmp_path):
    path = write_config(
        tmp_path,
        tmp_path / "dataset",
        "test-scene",
        "test_scene",
        100.0,
        101.0,
        50.0,
        40.0,
        100,
        80,
        max_points_per_frame=1234,
    )

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section in ("Dataset", "Testset"):
        assert config[section]["max_pts_num"] == 1234
        assert config[section]["num_threads"] == 0
