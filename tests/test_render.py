from pathlib import Path

import pytest

from utils_new.render_utils import select_gaussian_ply


def test_select_gaussian_ply_prefers_complete_progressive_export(tmp_path: Path):
    baseline = tmp_path / "point_cloud.ply"
    progressive = tmp_path / "point_cloud_progressive_full.ply"
    baseline.touch()
    progressive.touch()

    assert select_gaussian_ply(
        tmp_path, {"ProgressiveMapping": {"enabled": True}}
    ) == progressive
    assert select_gaussian_ply(
        tmp_path, {"ProgressiveMapping": {"enabled": False}}
    ) == baseline

    progressive.unlink()
    assert select_gaussian_ply(
        tmp_path, {"ProgressiveMapping": {"enabled": True}}
    ) == baseline


def test_select_gaussian_ply_requires_baseline_fallback(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="point_cloud.ply"):
        select_gaussian_ply(tmp_path, {"ProgressiveMapping": {"enabled": True}})


def test_ply_render_disables_compute_routing_without_degree_metadata():
    from render import prepare_render_config

    config = {
        "Model": {"DepthCovEstimator": {}, "device": "cuda:7"},
        "Mapper": {"device": "cuda:7"},
        "StreamingAppearanceLOD": {"enabled": True, "compute_routing": True},
    }

    prepared = prepare_render_config(config, "cuda:0")

    assert prepared["StreamingAppearanceLOD"]["compute_routing"] is False
    assert config["StreamingAppearanceLOD"]["compute_routing"] is True
    assert "DepthCovEstimator" not in prepared["Model"]
