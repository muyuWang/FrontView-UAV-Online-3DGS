from pathlib import Path
import sys

import torch


DEPTHCOV_ROOT = Path(__file__).resolve().parents[1] / "DepthCov-Modified"
sys.path.insert(0, str(DEPTHCOV_ROOT))

from depth_cov.depth_cov_estimator import DepthCovEstimator  # noqa: E402


class _FakeDepthModel:
    def __call__(self, rgb):
        batch, _, height, width = rgb.shape
        return [torch.ones((batch, 3, height, width), device=rgb.device)]

    def est_sparse_depth(
        self,
        gaussian_covs,
        train_coords,
        sparse_log_depth,
        mean_depth,
        test_coords,
        level,
    ):
        count = test_coords.shape[1]
        mean = mean_depth.expand(1, count, 1).clone()
        variance = torch.full_like(mean, 0.01)
        return mean, variance


def test_prepared_query_matches_query_tensor_and_reuses_image_features():
    estimator = DepthCovEstimator.__new__(DepthCovEstimator)
    estimator.level = -1
    estimator.device = "cpu"
    estimator.model = _FakeDepthModel()
    estimator.std_valid_threshold = 0.2
    image = torch.zeros((64, 96, 3), dtype=torch.float32)
    train_uv = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    train_depth = torch.tensor([10.0, 40.0])
    query_uv = torch.tensor([[15.0, 25.0], [35.0, 45.0]])

    direct = estimator.query_tensor(
        image, train_depth, train_uv, query_uv, return_std=True
    )
    context = estimator.prepare_tensor_context(image)
    prepared = estimator.query_prepared_tensor(
        context, train_depth, train_uv, query_uv
    )

    for direct_value, prepared_value in zip(direct, prepared):
        torch.testing.assert_close(direct_value, prepared_value)
