import pytest
import torch

from utils_new.frontview_depth_transport import (
    split_depth_anchors,
    transport_candidate_depths,
    validate_front_view_depth_transport_config,
)


def test_depth_transport_rejects_half_or_larger_calibration_split():
    with pytest.raises(ValueError):
        validate_front_view_depth_transport_config({"calibration_fraction": 0.5})


def test_depth_anchor_split_is_disjoint_and_deterministic():
    first = split_depth_anchors(100, 0.1, 32, 8, seed=7, device="cpu")
    second = split_depth_anchors(100, 0.1, 32, 8, seed=7, device="cpu")
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert len(first[0]) == 90
    assert len(first[1]) == 10
    assert not set(first[0].tolist()) & set(first[1].tolist())


def test_depth_anchor_split_falls_back_without_leaking_small_sets():
    training, calibration = split_depth_anchors(
        39, 0.1, 32, 8, seed=7, device="cpu"
    )
    assert torch.equal(training, torch.arange(39))
    assert len(calibration) == 0


def test_transport_applies_out_of_sample_log_depth_residual():
    image = torch.zeros((8, 8, 3), dtype=torch.float32)
    candidate_coords = torch.tensor([[3.5, 3.5]])
    calibration_coords = torch.tensor(
        [[1.5, 1.5], [5.5, 1.5], [1.5, 5.5], [5.5, 5.5]]
    )
    corrected, correction = transport_candidate_depths(
        candidate_coords,
        torch.tensor([10.0]),
        calibration_coords,
        torch.full((4,), 10.0),
        torch.full((4,), 8.0),
        image,
        neighbors=4,
        clip_quantiles=[0.0, 1.0],
    )
    assert corrected.item() == pytest.approx(8.0, abs=1.0e-5)
    assert correction.item() == pytest.approx(torch.log(torch.tensor(0.8)).item())


def test_shuffled_transport_is_deterministic():
    image = torch.zeros((8, 8, 3), dtype=torch.float32)
    kwargs = dict(
        candidate_coords=torch.tensor([[2.5, 2.5], [4.5, 4.5]]),
        candidate_depths=torch.tensor([10.0, 10.0]),
        calibration_coords=torch.tensor(
            [[1.5, 1.5], [5.5, 1.5], [1.5, 5.5], [5.5, 5.5]]
        ),
        calibration_pred_depths=torch.full((4,), 10.0),
        calibration_true_depths=torch.tensor([7.0, 8.0, 9.0, 10.0]),
        image=image,
        neighbors=2,
        clip_quantiles=[0.0, 1.0],
        shuffle_residual_locations=True,
        seed=11,
    )
    first = transport_candidate_depths(**kwargs)
    second = transport_candidate_depths(**kwargs)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
