import numpy as np
import torch

from utils_new.camera_utils import Camera


def test_downsample_point_ids_selects_identity_nearest_pooled_depth():
    depth = torch.tensor([[1.0, 3.0], [0.0, 0.0]])
    point_ids = torch.tensor([[10, 30], [-1, -1]])
    pooled = Camera.downsample(depth, mode="depth")

    result = Camera.downsample_point_ids(depth, point_ids, pooled)

    assert pooled.item() == 2.0
    assert result.item() == 10


def test_project_pts_track_raster_matches_sparse_depth_overwrite():
    camera = Camera.__new__(Camera)
    camera.height = 4
    camera.width = 4
    camera.raw_pose = torch.eye(4)
    camera.K = torch.eye(3)
    points = np.asarray([[1.2, 1.2, 1.0], [2.4, 2.4, 2.0]], dtype=np.float32)
    colors = np.ones((4, 4, 3), dtype=np.float32)

    result = camera.project_pts(points, colors, point_ids=np.asarray([7, 9]))
    sparse_depth = result[2]
    sparse_point_ids = result[6]

    assert sparse_depth[1, 1] == 2.0
    assert sparse_point_ids[1, 1] == 9
