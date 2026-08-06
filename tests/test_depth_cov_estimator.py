import numpy as np
import pytest
import torch

from depth_cov.depth_cov_estimator import DepthCovEstimator


class _ShapeRecordingModel:
    def __init__(self):
        self.shapes = []

    def __call__(self, rgb):
        self.shapes.append(tuple(rgb.shape[-2:]))
        return [torch.zeros((1, 4, *rgb.shape[-2:]), dtype=rgb.dtype)]

    def est_sparse_depth(
        self, gaussian_covs, train_coords, sparse_depth, mean_depth, test_coords, level=-1
    ):
        self.shapes.append(tuple(gaussian_covs[level].shape[-2:]))
        count = test_coords.shape[1]
        return (
            torch.zeros((1, 1, count), dtype=torch.float32),
            torch.full((1, 1, count), 0.01, dtype=torch.float32),
        )


def _estimator():
    estimator = DepthCovEstimator.__new__(DepthCovEstimator)
    estimator.level = -1
    estimator.device = "cpu"
    estimator.std_valid_threshold = 0.2
    estimator.model = _ShapeRecordingModel()
    return estimator


@pytest.mark.parametrize("use_tensor_query", (False, True))
def test_depth_cov_preserves_non_square_height_width(use_tensor_query):
    estimator = _estimator()
    rgb = torch.zeros((65, 97, 3), dtype=torch.float32)
    train_xy = np.asarray([[10.0, 12.0], [20.0, 24.0]], dtype=np.float32)
    test_xy = np.asarray([[30.0, 32.0]], dtype=np.float32)
    sparse_depth = np.asarray([5.0, 6.0], dtype=np.float32)

    if use_tensor_query:
        estimator.query_tensor(
            rgb,
            torch.from_numpy(sparse_depth),
            torch.from_numpy(train_xy),
            torch.from_numpy(test_xy),
        )
    else:
        estimator.query(rgb, sparse_depth, train_xy, test_xy)

    assert estimator.model.shapes == [(64, 96), (65, 97)]
