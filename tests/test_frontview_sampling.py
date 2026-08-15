import pytest
import torch

from utils_new.frontview_sampling import (
    adaptive_log_depth_indices,
    projective_coverage_indices,
    rate_distortion_density_weights,
    residual_rate_distortion_radius_factors,
    residual_importance_indices,
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


def test_residual_importance_is_deterministic_and_rejects_zero_gradient_rows():
    residuals = torch.tensor([1.0, 0.5, 0.8, 0.6])
    confidences = torch.tensor([0.0, 0.0, 1.0, 1.0])
    first = residual_importance_indices(residuals, confidences, 2, seed=17)
    second = residual_importance_indices(residuals, confidences, 2, seed=17)

    assert torch.equal(first, second)
    assert first.tolist() == [2, 3]
    assert validate_front_view_sampling_config(
        {"selection_mode": "residual_importance"}
    )["selection_mode"] == "residual_importance"


def test_adaptive_log_depth_balances_learned_relative_depth_regimes():
    depths = torch.tensor([1.0, 1.1, 1.2, 10.0, 11.0, 12.0, 100.0, 110.0, 120.0])
    selected, metadata = adaptive_log_depth_indices(
        depths,
        torch.ones_like(depths),
        torch.ones_like(depths),
        6,
        seed=5,
    )

    assert selected.numel() == 6
    assert metadata["pool_counts"] == [3, 3, 3]
    assert metadata["selected_counts"] == [2, 2, 2]
    assert 1.2 < metadata["boundaries_m"][0] < 10.0
    assert 12.0 < metadata["boundaries_m"][1] < 100.0


def test_adaptive_log_depth_retains_capacity_limited_far_regime():
    depths = torch.tensor(
        [1.0, 1.1, 1.2, 1.3, 8.0, 8.5, 9.0, 9.5, 100.0]
    )
    selected, metadata = adaptive_log_depth_indices(
        depths,
        torch.ones_like(depths),
        torch.ones_like(depths),
        5,
        seed=9,
    )

    assert selected.numel() == 5
    assert 8 in selected.tolist()
    assert metadata["selected_counts"][2] == metadata["pool_counts"][2] == 1
    assert sum(metadata["quotas"]) == 5


def test_adaptive_log_depth_is_scale_invariant():
    depths = torch.tensor([2.0, 2.5, 3.0, 12.0, 15.0, 18.0, 70.0, 90.0, 110.0])
    kwargs = dict(
        confidences=torch.ones_like(depths),
        residuals=torch.ones_like(depths),
        budget=6,
        seed=13,
    )
    selected, metadata = adaptive_log_depth_indices(depths, **kwargs)
    scaled, scaled_metadata = adaptive_log_depth_indices(depths * 7.0, **kwargs)

    assert torch.equal(selected, scaled)
    assert scaled_metadata["boundaries_m"] == pytest.approx(
        [value * 7.0 for value in metadata["boundaries_m"]], rel=1.0e-5
    )


@pytest.mark.parametrize(
    "mode",
    [
        "adaptive_log_depth_random",
        "adaptive_log_depth_importance",
        "adaptive_log_depth_shuffled",
    ],
)
def test_sampling_config_accepts_adaptive_log_depth_modes(mode):
    assert validate_front_view_sampling_config({"selection_mode": mode})[
        "selection_mode"
    ] == mode


def test_adaptive_log_depth_coverage_uses_distinct_projective_cells():
    depths = torch.tensor([2.0] * 4 + [20.0] * 4 + [200.0] * 4)
    uv = torch.tensor(
        [[1.0, 1.0], [2.0, 2.0], [21.0, 1.0], [22.0, 2.0]] * 3
    )
    selected, metadata = adaptive_log_depth_indices(
        depths,
        torch.ones_like(depths),
        torch.ones_like(depths),
        6,
        uv=uv,
        image_size=(40, 20),
        pool_multiplier=1,
        coverage_priority="confidence",
        seed=3,
    )

    assert metadata["selected_counts"] == [2, 2, 2]
    for regime in range(3):
        rows = selected[depths[selected] == depths[regime * 4]]
        cells = torch.floor(uv[rows, 0] / metadata["coverage_cell_px"])
        assert torch.unique(cells).numel() == 2


def test_adaptive_log_depth_coverage_is_resolution_invariant():
    depths = torch.tensor([2.0, 2.5, 3.0, 12.0, 15.0, 18.0, 70.0, 90.0, 110.0])
    uv = torch.tensor(
        [[1.0, 1.0], [4.0, 2.0], [8.0, 3.0]] * 3
    )
    kwargs = dict(
        depths=depths,
        confidences=torch.linspace(0.1, 0.9, 9),
        residuals=torch.ones_like(depths),
        budget=6,
        pool_multiplier=2,
        coverage_priority="confidence",
        seed=11,
    )
    selected, metadata = adaptive_log_depth_indices(
        uv=uv, image_size=(20, 10), **kwargs
    )
    scaled, scaled_metadata = adaptive_log_depth_indices(
        uv=uv * 4.0, image_size=(80, 40), **kwargs
    )

    assert torch.equal(selected, scaled)
    assert scaled_metadata["coverage_cell_px"] == pytest.approx(
        metadata["coverage_cell_px"] * 4.0
    )


def test_rate_distortion_density_prefers_structure_and_rejects_uncertain_depth():
    image = torch.zeros((20, 40, 3), dtype=torch.float32)
    image[:, 20:, :] = 1.0
    uv = torch.tensor([[5.0, 10.0], [20.0, 10.0], [21.0, 10.0]])
    confidence = torch.tensor([1.0, 1.0, 0.0])
    density, metadata = rate_distortion_density_weights(
        image,
        uv,
        confidence,
        image_size=(40, 20),
        budget=20,
        pool_multiplier=2,
    )

    assert density[1] > density[0]
    assert density[2] == 0.0
    assert metadata["cell_px"] == pytest.approx((40 * 20 / 40) ** 0.5)


def test_residual_rate_distortion_shrinks_structure_and_preserves_area():
    residual = torch.zeros((20, 40), dtype=torch.float32)
    residual[:, 20:] = 1.0
    uv = torch.tensor([[5.0, 10.0], [20.0, 10.0], [35.0, 10.0]])
    eligible = torch.tensor([True, True, False])
    factors = residual_rate_distortion_radius_factors(
        residual,
        uv,
        eligible,
        image_size=(40, 20),
        budget=20,
        pool_multiplier=2,
    )

    assert factors[1] < factors[0]
    assert factors[2] == pytest.approx(1.0)
    assert torch.mean(factors[eligible].square()) == pytest.approx(1.0)


def test_residual_detail_protection_never_expands_stage25_footprint():
    residual = torch.zeros((20, 40), dtype=torch.float32)
    residual[:, 20:] = 1.0
    uv = torch.tensor([[5.0, 10.0], [20.0, 10.0], [35.0, 10.0]])
    eligible = torch.tensor([True, True, False])
    factors = residual_rate_distortion_radius_factors(
        residual,
        uv,
        eligible,
        image_size=(40, 20),
        budget=20,
        pool_multiplier=2,
        detail_protection=True,
    )

    assert factors[1] < factors[0]
    assert bool(torch.all(factors <= 1.0))
    assert factors[2] == pytest.approx(1.0)


def test_visible_residual_detail_ignores_uncovered_structure():
    residual = torch.zeros((20, 40), dtype=torch.float32)
    residual[:, 10:] = 1.0
    residual[:, 30:] = 0.0
    visibility = torch.ones((20, 40), dtype=torch.float32)
    visibility[:, :20] = 0.0
    uv = torch.tensor([[10.0, 10.0], [30.0, 10.0]])
    factors = residual_rate_distortion_radius_factors(
        residual,
        uv,
        torch.ones(2, dtype=torch.bool),
        image_size=(40, 20),
        budget=20,
        pool_multiplier=2,
        visibility=visibility,
        detail_protection=True,
    )

    assert factors[0] == pytest.approx(1.0)
    assert factors[1] < 1.0


def test_rate_distortion_shuffle_preserves_budget_and_depth_histogram():
    depths = torch.tensor([2.0] * 6 + [20.0] * 6 + [200.0] * 6)
    density = torch.arange(1, 19, dtype=torch.float32)
    kwargs = dict(
        depths=depths,
        confidences=torch.ones_like(depths),
        residuals=torch.ones_like(depths),
        budget=9,
        density_weights=density,
        seed=17,
    )
    real, real_metadata = adaptive_log_depth_indices(**kwargs)
    shuffled, shuffled_metadata = adaptive_log_depth_indices(
        **kwargs, shuffle_density=True
    )

    assert real.numel() == shuffled.numel() == 9
    assert real_metadata["selected_counts"] == [3, 3, 3]
    assert shuffled_metadata["selected_counts"] == [3, 3, 3]
    assert real_metadata["density_weighted"]
    assert shuffled_metadata["density_shuffled"]


@pytest.mark.parametrize(
    "mode",
    [
        "adaptive_log_depth_coverage",
        "adaptive_log_depth_residual_coverage",
        "adaptive_log_depth_coverage_shuffled",
        "adaptive_log_depth_rate_distortion",
        "adaptive_log_depth_rate_distortion_shuffled",
    ],
)
def test_sampling_config_accepts_adaptive_coverage_modes(mode):
    assert validate_front_view_sampling_config({"selection_mode": mode})[
        "selection_mode"
    ] == mode
