import pytest
import torch

from utils_new.streaming_appearance_lod import (
    bias_corrected_gradient_utility,
    persistent_gradient_utility,
    projective_evidence_measurements,
    select_gradient_agreement_promotions,
    select_gradient_promotions,
    select_monotonic_promotions,
    sh_band_bounds,
    validate_streaming_appearance_lod_config,
)


def test_streaming_appearance_lod_is_disabled_and_fail_closed_by_default():
    config = validate_streaming_appearance_lod_config(None)
    assert config["enabled"] is False
    assert config["birth_degree"] == 1
    assert config["target_degree"] == 2

    with pytest.raises(TypeError, match="Unknown"):
        validate_streaming_appearance_lod_config({"unknown": 1})
    with pytest.raises(ValueError, match="birth_degree"):
        validate_streaming_appearance_lod_config(
            {"birth_degree": 2, "target_degree": 2}
        )
    assert validate_streaming_appearance_lod_config(
        {"birth_degree": 2, "target_degree": 3}
    )["target_degree"] == 3
    with pytest.raises(ValueError, match="up to SH3"):
        validate_streaming_appearance_lod_config(
            {"birth_degree": 2, "target_degree": 4}
        )


def test_projective_evidence_rewards_supported_large_directional_observations():
    count = torch.tensor([1, 4, 4])
    radius_sum = torch.tensor([4.0, 4.0, 16.0])
    directions = torch.tensor(
        [[1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    radius, dispersion, score = projective_evidence_measurements(
        count, radius_sum, directions
    )

    assert radius.tolist() == pytest.approx([4.0, 1.0, 4.0])
    assert dispersion.tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert score[2] > score[0]
    assert score[0].item() == pytest.approx(score[1].item())


def test_evidence_and_shuffled_controls_match_promotion_budget():
    degrees = torch.ones(8, dtype=torch.uint8)
    count = torch.arange(1, 9)
    radius_sum = count.float() * 2.0
    directions = torch.zeros((8, 3))
    common = {
        "enabled": True,
        "min_views": 2,
        "max_target_fraction": 0.5,
    }

    evidence, evidence_rows = select_monotonic_promotions(
        degrees, count, radius_sum, directions, common
    )
    shuffled, shuffled_rows = select_monotonic_promotions(
        degrees,
        count,
        radius_sum,
        directions,
        {**common, "selection_mode": "shuffled", "shuffle_seed": 17},
    )

    assert evidence_rows.numel() == shuffled_rows.numel() == 4
    assert int((evidence == 2).sum()) == int((shuffled == 2).sum()) == 4
    assert torch.equal(evidence_rows.sort().values, torch.tensor([4, 5, 6, 7]))
    assert not torch.equal(evidence, shuffled)


def test_promotions_are_monotonic_and_do_not_exceed_capacity():
    degrees = torch.tensor([2, 1, 1, 1, 1], dtype=torch.uint8)
    count = torch.full((5,), 5)
    radius_sum = torch.full((5,), 10.0)
    directions = torch.zeros((5, 3))
    config = {"enabled": True, "max_target_fraction": 0.4}

    result, selected = select_monotonic_promotions(
        degrees, count, radius_sum, directions, config
    )

    assert selected.numel() == 1
    assert result[0] == 2
    assert int((result == 2).sum()) == 2


def test_global_shuffle_matches_count_without_reusing_evidence_eligibility():
    degrees = torch.ones(10, dtype=torch.uint8)
    count = torch.tensor([5, 5] + [0] * 8)
    radius_sum = torch.tensor([10.0, 10.0] + [0.0] * 8)
    directions = torch.zeros((10, 3))
    config = {
        "enabled": True,
        "max_target_fraction": 1.0,
        "selection_mode": "shuffled_global",
        "shuffle_seed": 7,
    }

    result, selected = select_monotonic_promotions(
        degrees, count, radius_sum, directions, config
    )

    assert selected.numel() == 2
    assert int((result == 2).sum()) == 2
    assert torch.any(selected >= 2)


def test_sh_band_bounds_exclude_dc_and_cover_each_degree_once():
    assert sh_band_bounds(1) == (0, 3)
    assert sh_band_bounds(2) == (3, 8)
    assert sh_band_bounds(3) == (8, 15)
    with pytest.raises(ValueError, match="positive"):
        sh_band_bounds(0)


def test_gradient_utility_is_bias_corrected_and_selects_largest_reduction():
    decay = 0.9
    count = torch.tensor([1, 2, 3, 4])
    true_utility = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ema = true_utility * (1.0 - decay ** count.float())

    corrected = bias_corrected_gradient_utility(ema, count, decay)
    degrees, selected = select_gradient_promotions(
        torch.full((4,), 2, dtype=torch.uint8),
        ema,
        count,
        {
            "enabled": True,
            "birth_degree": 2,
            "target_degree": 3,
            "min_views": 1,
            "max_target_fraction": 0.5,
            "selection_mode": "gradient",
            "utility_ema_decay": decay,
        },
    )

    assert corrected.tolist() == pytest.approx(true_utility.tolist())
    assert torch.equal(selected.sort().values, torch.tensor([2, 3]))
    assert int((degrees == 3).sum()) == 2


def test_gradient_shuffle_matches_eligible_budget_without_reusing_utility_rank():
    common = {
        "enabled": True,
        "birth_degree": 2,
        "target_degree": 3,
        "min_views": 2,
        "max_target_fraction": 0.5,
        "utility_ema_decay": 0.0,
    }
    degrees = torch.full((8,), 2, dtype=torch.uint8)
    utility = torch.arange(1, 9, dtype=torch.float32)
    count = torch.tensor([0, 1, 2, 2, 2, 2, 2, 2])

    evidence, evidence_rows = select_gradient_promotions(
        degrees,
        utility,
        count,
        {**common, "selection_mode": "gradient"},
    )
    shuffled, shuffled_rows = select_gradient_promotions(
        degrees,
        utility,
        count,
        {
            **common,
            "selection_mode": "gradient_shuffled",
            "shuffle_seed": 17,
        },
    )

    assert evidence_rows.numel() == shuffled_rows.numel() == 4
    assert int((evidence == 3).sum()) == int((shuffled == 3).sum()) == 4
    assert torch.all(evidence_rows >= 2)
    assert torch.all(shuffled_rows >= 2)
    assert not torch.equal(evidence, shuffled)


def test_persistent_gradient_agreement_rejects_cancelling_energy():
    count = torch.tensor([2, 2, 2])
    gradient_ema = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 1.0],
        ]
    )
    utility = persistent_gradient_utility(gradient_ema, count, decay=0.0)

    degrees, selected = select_gradient_agreement_promotions(
        torch.full((3,), 2, dtype=torch.uint8),
        gradient_ema,
        count,
        {
            "enabled": True,
            "birth_degree": 2,
            "target_degree": 3,
            "min_views": 2,
            "max_target_fraction": 1 / 3,
            "selection_mode": "gradient_agreement",
            "utility_ema_decay": 0.0,
        },
    )

    assert utility.tolist() == pytest.approx([0.0, 0.25, 1.0])
    assert selected.tolist() == [2]
    assert degrees.tolist() == [2, 2, 3]
