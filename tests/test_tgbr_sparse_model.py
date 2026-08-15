import numpy as np
import pytest
import torch
from plyfile import PlyData

from utils_new.gaussian_models import GaussianModel, Gaussians
from utils_new.streaming_appearance_lod import (
    validate_streaming_appearance_lod_config,
)
from utils_new.tgbr_sparse_model import (
    compact_payload_bytes,
    decode_tgbr_sparse_sh,
    dense_payload_bytes,
    is_tgbr_sparse_ply,
    write_tgbr_sparse_ply,
)


def make_arrays(count=8):
    generator = np.random.default_rng(7)
    active = np.zeros(count, dtype=bool)
    active[[1, 3, 4, 7]] = True
    shN = generator.normal(size=(count, 15, 3)).astype(np.float32)
    shN[~active, 8:15] = 0.0
    return {
        "means": generator.normal(size=(count, 3)).astype(np.float32),
        "sh0": generator.normal(size=(count, 1, 3)).astype(np.float32),
        "shN": shN,
        "opacities": generator.normal(size=count).astype(np.float32),
        "scales": generator.normal(size=(count, 3)).astype(np.float32),
        "quats": generator.normal(size=(count, 4)).astype(np.float32),
        "active_mask": active,
    }


def test_sparse_tgbr_ply_roundtrip_is_bitwise_exact(tmp_path):
    arrays = make_arrays()
    path = tmp_path / "point_cloud.ply"
    stats = write_tgbr_sparse_ply(
        path, **arrays, base_degree=2, target_degree=3
    )

    plydata = PlyData.read(path)
    restored, degrees, metadata = decode_tgbr_sparse_sh(plydata, 3)

    assert is_tgbr_sparse_ply(plydata)
    assert np.array_equal(restored, arrays["shN"])
    assert np.array_equal(degrees == 3, arrays["active_mask"])
    assert metadata["active_high_band_count"] == 4
    assert stats["active_high_band_count"] == 4


def test_sparse_tgbr_ply_preserves_depth_confidences(tmp_path):
    arrays = make_arrays()
    confidence = np.linspace(0.0, 1.0, len(arrays["means"]), dtype=np.float32)
    uncertainty = np.linspace(1.0, 0.0, len(arrays["means"]), dtype=np.float32)
    path = tmp_path / "point_cloud_metric.ply"
    write_tgbr_sparse_ply(
        path,
        **arrays,
        metric_confidences=confidence,
        uncertainty_confidences=uncertainty,
        base_degree=2,
        target_degree=3,
    )

    group = Gaussians(BS=1, scene_scale=1.0, max_sh_degree=3)
    group.to_device("cpu")
    group.load_from_ply(path, max_sh_degree=3)
    assert torch.equal(
        group.non_trainable_params["metric_confidences"],
        torch.from_numpy(confidence),
    )
    assert torch.equal(
        group.non_trainable_params["uncertainty_confidences"],
        torch.from_numpy(uncertainty),
    )


def test_gaussian_loader_restores_sparse_tgbr_degrees(tmp_path):
    arrays = make_arrays()
    path = tmp_path / "point_cloud.ply"
    write_tgbr_sparse_ply(path, **arrays, base_degree=2, target_degree=3)

    group = Gaussians(BS=1, scene_scale=1.0, max_sh_degree=3)
    group.to_device("cpu")
    group.load_from_ply(path, max_sh_degree=3)

    assert torch.equal(
        group.splats["shN"].detach(), torch.from_numpy(arrays["shN"])
    )
    assert torch.equal(
        group.non_trainable_params["appearance_sh_degree"] == 3,
        torch.from_numpy(arrays["active_mask"]),
    )


def test_gaussian_model_sparse_export_uses_allocated_degrees(tmp_path):
    arrays = make_arrays()
    group = Gaussians(BS=1, scene_scale=1.0, max_sh_degree=3)
    group.to_device("cpu")
    group.replace_gaussians(
        {
            "means": torch.from_numpy(arrays["means"]),
            "sh0": torch.from_numpy(arrays["sh0"]),
            "shN": torch.from_numpy(arrays["shN"]),
            "opacities": torch.from_numpy(arrays["opacities"]),
            "scales": torch.from_numpy(arrays["scales"]),
            "quats": torch.from_numpy(arrays["quats"]),
        }
    )
    group.non_trainable_params["appearance_sh_degree"].copy_(
        torch.where(
            torch.from_numpy(arrays["active_mask"]),
            torch.tensor(3, dtype=torch.uint8),
            torch.tensor(2, dtype=torch.uint8),
        )
    )
    model = GaussianModel.__new__(GaussianModel)
    model.max_sh_degree = 3
    model.valid_groups = [0]
    model.gaussian_groups = [group]
    model.streaming_appearance_lod_config = {
        "enabled": True,
        "birth_degree": 2,
        "target_degree": 3,
    }
    model.tgbr_sparse_model_stats = None

    stats = model.save_as_tgbr_sparse_ply(tmp_path / "model.ply")

    assert stats["active_high_band_count"] == 4
    assert model.tgbr_sparse_model_stats == stats


def test_sparse_tgbr_payload_exceeds_eight_percent_at_cap75():
    dense = dense_payload_bytes(10_000, target_degree=3)
    compact = compact_payload_bytes(
        10_000, 7_500, base_degree=2, target_degree=3
    )
    assert 1.0 - compact / dense > 0.08


def test_sparse_tgbr_export_rejects_nonzero_inactive_band(tmp_path):
    arrays = make_arrays()
    arrays["shN"][0, 8, 0] = 1.0
    with pytest.raises(ValueError, match="Inactive TGBR rows"):
        write_tgbr_sparse_ply(
            tmp_path / "invalid.ply", **arrays, base_degree=2, target_degree=3
        )


def test_sparse_model_export_config_must_be_boolean():
    with pytest.raises(TypeError, match="sparse_model_export"):
        validate_streaming_appearance_lod_config(
            {"sparse_model_export": "yes"}
        )


def test_sparse_model_export_requires_enabled_tgbr():
    with pytest.raises(ValueError, match="requires StreamingAppearanceLOD.enabled"):
        validate_streaming_appearance_lod_config({"sparse_model_export": True})
