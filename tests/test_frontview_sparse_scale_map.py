import numpy as np
import pytest

from utils_new.frontview_sparse_scale_map import (
    FrontViewSparseScaleMap,
    validate_front_view_sparse_scale_map_config,
)


def test_sparse_scale_map_rejects_invalid_config_and_coordinate_overflow():
    with pytest.raises(ValueError, match="Unknown"):
        validate_front_view_sparse_scale_map_config({"unknown": True})
    scale_map = FrontViewSparseScaleMap(
        {"enabled": True, "start_scale": 1.0, "levels": 1, "coordinate_bits": 3}
    )
    with pytest.raises(OverflowError):
        scale_map.register([[4.0, 0.0, 0.0]])


def test_sparse_scale_map_preserves_coarse_to_fine_query_semantics():
    scale_map = FrontViewSparseScaleMap(
        {"enabled": True, "start_scale": 1.0, "levels": 3}
    )
    point = np.asarray([[0.1, 0.1, 0.1]], dtype=np.float32)
    scale_map.register(point, target_size=0.5)

    assert scale_map.occupied(point, target_size=0.5).tolist() == [True]
    assert scale_map.occupied(point, target_size=0.25).tolist() == [False]

    scale_map.register(point, target_size=0.25)
    assert scale_map.occupied(point, target_size=0.25).tolist() == [True]


def test_exact_keys_do_not_alias_like_fixed_modulo_hash_cells():
    scale_map = FrontViewSparseScaleMap(
        {"enabled": True, "start_scale": 1.0, "levels": 1}
    )
    scale_map.register([[0.1, 0.0, 0.0]])

    occupied = scale_map.occupied([[0.1, 0.0, 0.0], [512.1, 0.0, 0.0]])

    assert occupied.tolist() == [True, False]
    assert scale_map.summary()["hash_calls_zero"] is True


def test_log_structured_runs_merge_without_losing_keys():
    scale_map = FrontViewSparseScaleMap(
        {"enabled": True, "start_scale": 1.0, "levels": 1}
    )
    for value in range(7):
        scale_map.register([[value + 0.1, 0.0, 0.0]])

    query = [[value + 0.1, 0.0, 0.0] for value in range(7)]
    assert scale_map.occupied(query).tolist() == [True] * 7
    summary = scale_map.summary()
    assert summary["run_merges"] > 0
    assert summary["stored_keys"] == 7
