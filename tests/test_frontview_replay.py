import numpy as np
import pytest
import torch

from utils_new.kf_graph import frontview_stratified_replay_indices
from utils_new.loss_utils import blend_mse_tail_loss


def test_range_replay_selects_one_unique_frame_per_stratum():
    positions = np.stack(
        (np.arange(9, dtype=np.float32), np.zeros(9), np.zeros(9)), axis=1
    )
    errors = np.asarray([0.1, 0.2, 0.9, 0.1, 0.4, 0.8, 0.2, 0.7, 0.0])
    selected = frontview_stratified_replay_indices(
        positions, errors, 4, "frontview_range_error"
    )
    assert len(selected) == 4
    assert len(np.unique(selected)) == 4
    assert 8 not in selected


def test_shuffled_range_control_is_deterministic_per_call():
    positions = np.stack(
        (np.arange(10, dtype=np.float32), np.zeros(10), np.zeros(10)), axis=1
    )
    errors = np.arange(10, dtype=np.float32)
    first = frontview_stratified_replay_indices(
        positions,
        errors,
        4,
        "frontview_range_shuffled",
        seed=7,
        call_index=3,
    )
    second = frontview_stratified_replay_indices(
        positions,
        errors,
        4,
        "frontview_range_shuffled",
        seed=7,
        call_index=3,
    )
    assert np.array_equal(first, second)


def test_cyclic_range_replay_covers_all_strata_across_steps():
    positions = np.stack(
        (np.arange(9, dtype=np.float32), np.zeros(9), np.zeros(9)), axis=1
    )
    errors = np.zeros(9, dtype=np.float32)
    selected = [
        int(
            frontview_stratified_replay_indices(
                positions,
                errors,
                1,
                "frontview_range_cyclic",
                call_index=step,
                strata_count=4,
            )[0]
        )
        for step in range(4)
    ]
    assert len(set(selected)) == 4


def test_range_round_robin_preserves_strata_and_rotates_members():
    positions = np.stack(
        (np.arange(13, dtype=np.float32), np.zeros(13), np.zeros(13)), axis=1
    )
    errors = np.zeros(13, dtype=np.float32)
    first = frontview_stratified_replay_indices(
        positions, errors, 4, "frontview_range_round_robin", call_index=0
    )
    second = frontview_stratified_replay_indices(
        positions, errors, 4, "frontview_range_round_robin", call_index=1
    )
    assert len(first) == len(second) == 4
    assert np.all(first != second)


def test_mse_tail_blend_preserves_endpoints():
    rgb = torch.tensor([[[[0.0, 0.5, 1.0]]]])
    gt = torch.tensor([[[[1.0, 0.5, 0.0]]]])
    base = torch.tensor(0.25)
    assert blend_mse_tail_loss(base, rgb, gt, 0.0).item() == pytest.approx(0.25)
    assert blend_mse_tail_loss(base, rgb, gt, 1.0).item() == pytest.approx(
        2.0 / 3.0
    )
