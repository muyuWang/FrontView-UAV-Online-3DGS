import pytest
import torch

from utils_new.streaming_appearance_lod import (
    bias_corrected_gradient_utility,
    directional_compact_view_steps,
    directional_step_budget,
    exact_replay_microbatch_ranges,
    frequency_consistent_step_levels,
    merge_replay_projection_info,
    optimization_tail_certificate,
    persistent_gradient_utility,
    projective_evidence_measurements,
    select_gradient_agreement_promotions,
    select_gradient_promotions,
    select_monotonic_promotions,
    sh_band_bounds,
    spectral_replay_microbatch_limit,
    spectral_residency_limit,
    validate_streaming_appearance_lod_config,
    view_direction_novelty_degrees,
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
    assert config["compute_routing"] is False
    assert config["bounded_replay_residency_enabled"] is False
    assert config["gradient_ema_dtype"] == "float32"
    assert config["frequency_schedule_enabled"] is False
    assert config["optimization_budget_routing_enabled"] is False
    assert config["spectral_replay_enabled"] is False
    assert config["spectral_residency_enabled"] is False
    assert config["spectral_residency_basis_budget"] == pytest.approx(5472.0)
    assert config["spectral_residency_max_views"] == 512
    routed = validate_streaming_appearance_lod_config(
        {
            "compute_routing": True,
            "compute_routing_warmup_evidence_updates": 10,
            "gradient_ema_dtype": "float16",
            "gradient_ema_scale": 4096.0,
        }
    )
    assert routed["compute_routing"] is True
    assert routed["compute_routing_warmup_evidence_updates"] == 10
    assert routed["gradient_ema_dtype"] == "float16"
    assert routed["gradient_ema_scale"] == 4096.0
    bounded = validate_streaming_appearance_lod_config(
        {"enabled": True, "bounded_replay_residency_enabled": True}
    )
    assert bounded["bounded_replay_residency_enabled"] is True
    with pytest.raises(ValueError, match="bounded_replay_residency_enabled"):
        validate_streaming_appearance_lod_config(
            {"bounded_replay_residency_enabled": True}
        )
    with pytest.raises(ValueError, match="gradient_ema_dtype"):
        validate_streaming_appearance_lod_config(
            {"gradient_ema_dtype": "bfloat16"}
        )
    with pytest.raises(ValueError, match="up to SH3"):
        validate_streaming_appearance_lod_config(
            {"birth_degree": 2, "target_degree": 4}
        )


def test_spectral_replay_requires_tgbr_and_excludes_fixed_microbatching():
    with pytest.raises(ValueError, match="requires StreamingAppearanceLOD.enabled"):
        validate_streaming_appearance_lod_config(
            {"spectral_replay_enabled": True}
        )
    with pytest.raises(ValueError, match="cannot be enabled together"):
        validate_streaming_appearance_lod_config(
            {
                "enabled": True,
                "spectral_replay_enabled": True,
                "exact_replay_microbatch_size": 3,
            }
        )


def test_spectral_replay_routes_views_from_active_tgbr_basis_load():
    config = {
        "enabled": True,
        "birth_degree": 2,
        "target_degree": 3,
        "spectral_replay_enabled": True,
        "spectral_replay_basis_budget": 48.0,
        "spectral_replay_max_views": 6,
    }

    cold_limit, cold_terms = spectral_replay_microbatch_limit(0.0, config)
    routed_limit, routed_terms = spectral_replay_microbatch_limit(0.75, config)

    assert cold_terms == pytest.approx(9.0)
    assert routed_terms == pytest.approx(14.25)
    assert cold_limit == 5
    assert routed_limit == 3


def test_spectral_residency_shares_tgbr_active_basis_budget():
    config = {
        "enabled": True,
        "birth_degree": 2,
        "target_degree": 3,
        "spectral_residency_enabled": True,
        "spectral_residency_basis_budget": 228.0,
        "spectral_residency_max_views": 32,
    }

    cold_limit, cold_terms = spectral_residency_limit(0.0, config)
    routed_limit, routed_terms = spectral_residency_limit(0.75, config)

    assert cold_terms == pytest.approx(9.0)
    assert routed_terms == pytest.approx(14.25)
    assert cold_limit == 25
    assert routed_limit == 16


def test_recommended_sh2_to_sh3_residency_routes_512_views_to_k384():
    config = validate_streaming_appearance_lod_config(
        {
            "enabled": True,
            "birth_degree": 2,
            "target_degree": 3,
            "spectral_residency_enabled": True,
        }
    )

    cold_limit, cold_terms = spectral_residency_limit(0.0, config)
    routed_limit, routed_terms = spectral_residency_limit(0.75, config)

    assert cold_terms == pytest.approx(9.0)
    assert routed_terms == pytest.approx(14.25)
    assert cold_limit == 512
    assert routed_limit == 384


def test_optimization_budget_routing_requires_tgbr_and_exclusive_budget():
    with pytest.raises(ValueError, match="requires StreamingAppearanceLOD.enabled"):
        validate_streaming_appearance_lod_config(
            {"optimization_budget_routing_enabled": True}
        )
    with pytest.raises(ValueError, match="directional step"):
        validate_streaming_appearance_lod_config(
            {
                "enabled": True,
                "optimization_budget_routing_enabled": True,
                "directional_step_budget_enabled": True,
            }
        )
    with pytest.raises(ValueError, match="directional view"):
        validate_streaming_appearance_lod_config(
            {
                "enabled": True,
                "optimization_budget_routing_enabled": True,
                "directional_view_budget_enabled": True,
            }
        )
    with pytest.raises(ValueError, match="frequency"):
        validate_streaming_appearance_lod_config(
            {
                "enabled": True,
                "optimization_budget_routing_enabled": True,
                "frequency_schedule_enabled": True,
            }
        )
    with pytest.raises(ValueError, match="collapse mode"):
        validate_streaming_appearance_lod_config(
            {"optimization_budget_tail_collapse_mode": "invalid"}
        )


def test_optimization_tail_certificate_bounds_geometric_improvement():
    config = validate_streaming_appearance_lod_config(
        {
            "enabled": True,
            "optimization_budget_routing_enabled": True,
            "optimization_budget_max_relative_tail": 0.02,
            "optimization_budget_max_high_band_ratio": 1.25,
            "optimization_budget_decay_cap": 0.9,
        }
    )
    certificate = optimization_tail_certificate(
        [1.0, 0.99, 0.985],
        high_band_gradient_ratio=0.1,
        remaining_steps=2,
        config=config,
    )

    assert certificate["decay_ratio"] == pytest.approx(0.5)
    assert certificate["relative_tail_bound"] == pytest.approx(
        (0.005 * 0.5 * (1.0 - 0.5**2) / (1.0 - 0.5)) / 0.985
    )
    assert certificate["stop"] is True
    assert certificate["reason"] == "certified"


def test_optimization_tail_certificate_protects_loss_and_high_band_demand():
    config = validate_streaming_appearance_lod_config(
        {
            "enabled": True,
            "optimization_budget_routing_enabled": True,
            "optimization_budget_max_relative_tail": 1.0e-3,
            "optimization_budget_max_high_band_ratio": 1.25,
        }
    )

    high_tail = optimization_tail_certificate(
        [1.0, 0.9, 0.82], 0.1, 2, config
    )
    high_band = optimization_tail_certificate(
        [1.0, 0.9999, 0.99985], 2.0, 2, config
    )
    invalid = optimization_tail_certificate(
        [1.0, float("nan"), 0.9], 0.1, 2, config
    )
    nondecaying = optimization_tail_certificate(
        [1.0, 0.99, 0.979], 0.1, 2, config
    )

    assert high_tail["stop"] is False
    assert high_tail["reason"] == "tail_bound"
    assert high_band["stop"] is False
    assert high_band["reason"] == "high_band_gradient"
    assert invalid["stop"] is False
    assert invalid["reason"] == "invalid_loss"
    assert nondecaying["stop"] is False
    assert nondecaying["reason"] == "nondecaying_tail"


def test_frequency_schedule_reallocates_full_steps_from_inactive_target_band():
    config = {
        "enabled": True,
        "birth_degree": 2,
        "target_degree": 3,
        "frequency_schedule_enabled": True,
        "frequency_schedule_warmup_evidence_updates": 10,
        "frequency_schedule_base_level_weights": [1.0, 0.0, 0.0, 0.0],
        "frequency_schedule_inactive_gain": 0.75,
        "frequency_schedule_min_full_fraction": 0.7,
        "frequency_schedule_reallocation_level": 2,
    }

    warmup = frequency_consistent_step_levels(10, 9, 0.0, config)
    routed = frequency_consistent_step_levels(10, 10, 0.6, config)

    assert [warmup.count(level) for level in range(4)] == [10, 0, 0, 0]
    assert [routed.count(level) for level in range(4)] == [7, 0, 3, 0]
    assert routed[0] == 0
    assert routed[-2:] == [0, 0]


def test_exact_replay_microbatches_preserve_all_views_and_projection_indices():
    assert exact_replay_microbatch_ranges(6, 3) == [(0, 3), (3, 6)]
    assert exact_replay_microbatch_ranges(5, 3) == [(0, 3), (3, 5)]
    assert exact_replay_microbatch_ranges(5, 0) == [(0, 5)]
    merged = merge_replay_projection_info(
        [
            (
                0,
                {
                    "gaussian_ids": torch.tensor([2, 3]),
                    "camera_ids": torch.tensor([0, 1]),
                    "radii": torch.tensor([[4.0, 3.0], [2.0, 5.0]]),
                    "depths": torch.tensor([6.0, 7.0]),
                },
            ),
            (
                2,
                {
                    "gaussian_ids": torch.tensor([8]),
                    "camera_ids": torch.tensor([0]),
                    "radii": torch.tensor([9.0]),
                    "depths": torch.tensor([10.0]),
                },
            ),
        ]
    )
    assert merged["gaussian_ids"].tolist() == [2, 3, 8]
    assert merged["camera_ids"].tolist() == [0, 1, 2]
    assert merged["radii"].tolist() == [4.0, 5.0, 9.0]


def test_frequency_schedule_preserves_budget_and_fails_closed_off_level_zero():
    config = {
        "frequency_schedule_enabled": True,
        "frequency_schedule_base_level_weights": [0.4, 0.3, 0.2, 0.1],
    }

    assert frequency_consistent_step_levels(3, 20, 0.0, config, base_level=1) == [
        1,
        1,
        1,
    ]
    schedule = frequency_consistent_step_levels(3, 20, 0.0, config)
    assert len(schedule) == 3
    assert schedule[0] == 0
    assert schedule[-2:] == [0, 0]
    with pytest.raises(ValueError, match="sum to one"):
        validate_streaming_appearance_lod_config(
            {"frequency_schedule_base_level_weights": [0.4, 0.3, 0.2, 0.2]}
        )


def test_directional_novelty_combines_rotation_and_depth_normalized_baseline():
    previous = torch.eye(4)
    current = torch.eye(4)
    angle = torch.deg2rad(torch.tensor(1.0))
    current[:3, :3] = torch.tensor(
        [
            [torch.cos(angle), 0.0, torch.sin(angle)],
            [0.0, 1.0, 0.0],
            [-torch.sin(angle), 0.0, torch.cos(angle)],
        ]
    )
    current[0, 3] = -1.0

    novelty, rotation, parallax = view_direction_novelty_degrees(
        previous, current, median_depth=100.0
    )

    assert rotation == pytest.approx(1.0, abs=1.0e-3)
    assert parallax == pytest.approx(0.57294, abs=1.0e-3)
    assert novelty == pytest.approx(rotation + parallax)


def test_directional_step_budget_keeps_warmup_and_routes_low_novelty():
    config = {
        "directional_step_budget_enabled": True,
        "directional_step_budget_warmup_keyframes": 8,
        "directional_step_budget_very_low_degrees": 0.5,
        "directional_step_budget_low_degrees": 1.5,
        "directional_step_budget_very_low_steps": 6,
        "directional_step_budget_low_steps": 8,
    }

    assert directional_step_budget(0.1, 10, 7, config) == (10, "full")
    assert directional_step_budget(0.1, 10, 8, config) == (6, "very_low")
    assert directional_step_budget(1.0, 10, 8, config) == (8, "low")
    assert directional_step_budget(2.0, 10, 8, config) == (10, "full")
    assert directional_step_budget(float("inf"), 10, 8, config) == (10, "full")


def test_directional_step_budget_routes_causal_novelty_percentiles():
    config = {
        "directional_step_budget_enabled": True,
        "directional_step_budget_selection_mode": "percentile",
        "directional_step_budget_warmup_keyframes": 8,
        "directional_step_budget_very_low_steps": 6,
        "directional_step_budget_low_steps": 8,
        "directional_step_budget_percentile_very_low_fraction": 0.3,
        "directional_step_budget_percentile_low_fraction": 0.4,
    }

    assert directional_step_budget(
        100.0, 10, 8, config, novelty_percentile=0.1
    ) == (6, "very_low")
    assert directional_step_budget(
        0.1, 10, 8, config, novelty_percentile=0.5
    ) == (8, "low")
    assert directional_step_budget(
        0.1, 10, 8, config, novelty_percentile=0.9
    ) == (10, "full")


def test_directional_view_budget_preserves_steps_and_compacts_middle_updates():
    config = {
        "directional_view_budget_enabled": True,
        "directional_view_budget_warmup_keyframes": 8,
        "directional_view_budget_very_low_degrees": 0.5,
        "directional_view_budget_low_degrees": 1.5,
        "directional_view_budget_very_low_compact_steps": 4,
        "directional_view_budget_low_compact_steps": 2,
        "directional_view_budget_full_anchor_steps": 1,
        "directional_view_budget_full_tail_steps": 2,
    }

    assert directional_compact_view_steps(0.1, 10, 7, config) == (0, "full")
    assert directional_compact_view_steps(0.1, 10, 8, config) == (
        4,
        "very_low",
    )
    assert directional_compact_view_steps(1.0, 10, 8, config) == (2, "low")
    assert directional_compact_view_steps(2.0, 10, 8, config) == (0, "full")

    percentile_config = {
        **config,
        "directional_view_budget_selection_mode": "percentile",
        "directional_view_budget_percentile_very_low_fraction": 0.3,
        "directional_view_budget_percentile_low_fraction": 0.4,
    }
    assert directional_compact_view_steps(
        100.0, 10, 8, percentile_config, novelty_percentile=0.1
    ) == (4, "very_low")
    assert directional_compact_view_steps(
        0.1, 10, 8, percentile_config, novelty_percentile=0.5
    ) == (2, "low")
    assert directional_compact_view_steps(
        0.1, 10, 8, percentile_config, novelty_percentile=0.9
    ) == (0, "full")


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
