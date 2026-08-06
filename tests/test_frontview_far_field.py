import numpy as np
import pytest

from utils_new.frontview_far_field import (
    projective_survivor_mask,
    validate_front_view_far_field_config,
)


def test_far_field_config_rejects_invalid_depth():
    with pytest.raises(ValueError):
        validate_front_view_far_field_config({"enabled": True, "depth_m": 0.0})


def test_projective_far_field_keeps_best_per_ray_depth_cell():
    keep = projective_survivor_mask(
        uv=np.asarray([[2.0, 2.0], [4.0, 4.0], [4.0, 4.0]], dtype=np.float32),
        depths=np.asarray([100.0, 100.0, 130.0], dtype=np.float32),
        scores=np.asarray([0.1, 0.3, 0.2], dtype=np.float32),
        config=validate_front_view_far_field_config({"enabled": True}),
    )
    assert keep.tolist() == [False, True, True]
