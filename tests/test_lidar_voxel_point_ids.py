import numpy as np
import pytest

from scripts.build_lidar_voxel_point_ids import voxel_ids


def test_voxel_ids_are_stable_and_collision_free():
    points = np.asarray(
        [[0.01, 0.02, 0.03], [0.04, -0.02, 0.01], [0.11, 0.0, 0.0]],
        dtype=np.float32,
    )
    identities = voxel_ids(points, 0.10)
    assert identities[0] == identities[1]
    assert identities[0] != identities[2]
    np.testing.assert_array_equal(identities, voxel_ids(points.copy(), 0.10))


def test_voxel_ids_reject_nonfinite_points():
    with pytest.raises(ValueError, match="non-finite"):
        voxel_ids(np.asarray([[np.nan, 0.0, 0.0]]), 0.10)
