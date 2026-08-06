import pytest
import torch

from utils_new.frontview_sampling import (
    projective_coverage_indices,
    validate_front_view_sampling_config,
)


def test_projective_coverage_is_deterministic_and_depth_balanced():
    uv = torch.tensor(
        [[1.0, 1.0], [2.0, 2.0], [21.0, 1.0], [22.0, 2.0]] * 3
    )
    depths = torch.tensor([10.0] * 4 + [30.0] * 4 + [60.0] * 4)
    confidence = torch.linspace(0.1, 1.0, 12)

    first = projective_coverage_indices(
        uv,
        depths,
        confidence,
        6,
        [20.0, 50.0],
        [1.0 / 3.0] * 3,
        image_width=40,
        cell_px=10,
    )
    second = projective_coverage_indices(
        uv,
        depths,
        confidence,
        6,
        [20.0, 50.0],
        [1.0 / 3.0] * 3,
        image_width=40,
        cell_px=10,
        seed=999,
    )

    assert torch.equal(first, second)
    selected_bands = torch.bucketize(depths[first], torch.tensor([20.0, 50.0]))
    assert torch.bincount(selected_bands, minlength=3).tolist() == [2, 2, 2]


def test_projective_coverage_keeps_highest_confidence_per_cell():
    selected = projective_coverage_indices(
        torch.tensor([[1.0, 1.0], [2.0, 2.0], [21.0, 1.0]]),
        torch.tensor([10.0, 10.0, 10.0]),
        torch.tensor([0.1, 0.9, 0.5]),
        2,
        [20.0, 50.0],
        [1.0, 0.0, 0.0],
        image_width=40,
        cell_px=10,
    )

    assert selected.tolist() == [1, 2]


def test_shuffled_projective_control_preserves_band_counts():
    uv = torch.stack(
        (torch.arange(12, dtype=torch.float32) * 10.0, torch.zeros(12)), dim=1
    )
    depths = torch.tensor([10.0] * 4 + [30.0] * 4 + [60.0] * 4)
    kwargs = dict(
        uv=uv,
        depths=depths,
        confidences=torch.ones(12),
        budget=6,
        edges=[20.0, 50.0],
        fractions=[1.0 / 3.0] * 3,
        image_width=120,
        cell_px=10,
        shuffle=True,
        seed=7,
    )

    first = projective_coverage_indices(**kwargs)
    second = projective_coverage_indices(**kwargs)
    selected_bands = torch.bucketize(depths[first], torch.tensor([20.0, 50.0]))

    assert torch.equal(first, second)
    assert torch.bincount(selected_bands, minlength=3).tolist() == [2, 2, 2]


def test_sampling_config_validates_projective_mode():
    config = validate_front_view_sampling_config(
        {"selection_mode": "projective_coverage", "projective_cell_px": 8}
    )
    assert config["selection_mode"] == "projective_coverage"
    with pytest.raises(ValueError, match="projective_cell_px"):
        validate_front_view_sampling_config({"projective_cell_px": 0})
