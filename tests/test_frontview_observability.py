import math

import pytest
import torch

from utils_new.gaussian_models import GaussianModel, Gaussians
from utils_new.frontview_observability import (
    parallax_learning_scale,
    matched_events_within_log_depth_regimes,
    posterior_information_scale,
    precondition_raywise_gradient,
    project_raywise_update,
    release_owned_scale_caps,
    resolved_footprint_mask,
    resolution_information_scale,
    shuffle_within_log_depth_regimes,
    validate_front_view_observability_config,
)


def test_frontview_observability_config_rejects_invalid_scale():
    with pytest.raises(ValueError):
        validate_front_view_observability_config({"min_ray_lr_scale": 1.1})


def test_frontview_observability_accepts_resolution_information_mode():
    config = validate_front_view_observability_config(
        {"learning_scale_mode": "resolution_information"}
    )
    assert config["learning_scale_mode"] == "resolution_information"


def test_frontview_observability_accepts_posterior_information_mode():
    config = validate_front_view_observability_config(
        {"learning_scale_mode": "posterior_information"}
    )
    assert config["learning_scale_mode"] == "posterior_information"


def test_frontview_observability_accepts_projective_post_step_mode():
    config = validate_front_view_observability_config(
        {
            "optimization_mode": "post_step_projection",
            "responsibility_scope": "projective_only",
        }
    )
    assert config["optimization_mode"] == "post_step_projection"
    assert config["responsibility_scope"] == "projective_only"


def test_parallax_scale_unlocks_smoothly():
    threshold = math.sin(math.radians(4.0)) ** 2
    scales = parallax_learning_scale(
        torch.tensor([0.0, threshold * 0.5, threshold]), 0.1, 4.0
    )
    assert torch.allclose(scales, torch.tensor([0.1, 0.55, 1.0]), atol=1.0e-6)


def test_raywise_preconditioner_only_scales_radial_component():
    gradients = torch.tensor([[2.0, 3.0, 4.0]])
    rays = torch.tensor([[1.0, 0.0, 0.0]])
    adjusted = precondition_raywise_gradient(
        gradients, rays, torch.tensor([0.25])
    )
    assert torch.allclose(adjusted, torch.tensor([[0.5, 3.0, 4.0]]))


def test_post_step_projection_controls_realized_radial_update():
    previous = torch.tensor([[10.0, 20.0, 30.0]])
    updated = torch.tensor([[12.0, 23.0, 34.0]])
    projected = project_raywise_update(
        previous,
        updated,
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([0.25]),
    )
    assert torch.allclose(projected, torch.tensor([[10.5, 23.0, 34.0]]))


def test_post_step_projection_is_relative_to_previous_not_birth_anchor():
    previous = torch.tensor([[8.0, 0.0, 0.0]])
    updated = torch.tensor([[10.0, 0.0, 0.0]])
    projected = project_raywise_update(
        previous,
        updated,
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([0.5]),
    )
    assert torch.allclose(projected, torch.tensor([[9.0, 0.0, 0.0]]))


def test_resolution_information_scale_uses_parallax_to_support_ratio():
    scales = resolution_information_scale(
        torch.tensor([0.0, 0.01, 0.04]),
        torch.tensor([[1.0, 0.5, 0.25]]).repeat(3, 1),
        torch.tensor([10.0, 10.0, 10.0]),
    )
    assert torch.allclose(scales, torch.tensor([0.0, 0.5, 0.8]), atol=1.0e-6)


def test_resolution_information_scale_is_similarity_invariant():
    original = resolution_information_scale(
        torch.tensor([0.01]), torch.tensor([[2.0, 1.0, 0.5]]), torch.tensor([20.0])
    )
    rescaled = resolution_information_scale(
        torch.tensor([0.01]), torch.tensor([[20.0, 10.0, 5.0]]), torch.tensor([200.0])
    )
    assert torch.allclose(original, rescaled)


def test_posterior_information_is_kalman_gain_and_similarity_invariant():
    gain = posterior_information_scale(
        torch.tensor([0.01]),
        torch.tensor([[1.0, 0.5, 0.25]]),
        torch.tensor([10.0]),
        torch.tensor([0.1]),
    )
    rescaled = posterior_information_scale(
        torch.tensor([0.01]),
        torch.tensor([[10.0, 5.0, 2.5]]),
        torch.tensor([100.0]),
        torch.tensor([0.1]),
    )
    assert torch.allclose(gain, torch.tensor([1.0 / 101.0]), atol=1.0e-6)
    assert torch.allclose(gain, rescaled)


def test_posterior_information_keeps_exact_birth_depth_fixed_without_evidence():
    gain = posterior_information_scale(
        torch.tensor([0.0, 0.1]),
        torch.ones((2, 3)),
        torch.full((2,), 10.0),
        torch.zeros(2),
    )
    assert torch.equal(gain, torch.zeros(2))


def test_footprint_resolution_requires_span_and_depth_precision():
    resolved = resolved_footprint_mask(
        torch.tensor([0.04**2, 0.10**2, 0.10**2]),
        torch.tensor([0.5, 0.5, 0.5]),
        torch.tensor([10.0, 10.0, 10.0]),
        torch.tensor([0.1, 0.1, 1.0]),
    )
    assert torch.equal(resolved, torch.tensor([False, True, False]))


def test_footprint_resolution_is_similarity_invariant():
    original = resolved_footprint_mask(
        torch.tensor([0.10**2]),
        torch.tensor([0.5]),
        torch.tensor([10.0]),
        torch.tensor([0.1]),
    )
    rescaled = resolved_footprint_mask(
        torch.tensor([0.10**2]),
        torch.tensor([5.0]),
        torch.tensor([100.0]),
        torch.tensor([0.1]),
    )
    assert torch.equal(original, rescaled)


def test_log_depth_shuffle_preserves_each_regime_distribution():
    values = torch.arange(9, dtype=torch.float32)
    depths = torch.tensor([2.0, 2.2, 2.4, 20.0, 22.0, 24.0, 200.0, 220.0, 240.0])
    shuffled = shuffle_within_log_depth_regimes(values, depths, seed=7)
    for rows in (slice(0, 3), slice(3, 6), slice(6, 9)):
        assert torch.equal(torch.sort(shuffled[rows]).values, values[rows])
    assert not torch.equal(shuffled, values)


def test_matched_certificate_shuffle_preserves_event_count_per_regime():
    events = torch.tensor(
        [True, False, False, True, True, False, False, False, True]
    )
    depths = torch.tensor([2.0, 2.2, 2.4, 20.0, 22.0, 24.0, 200.0, 220.0, 240.0])
    shuffled = matched_events_within_log_depth_regimes(
        events, torch.ones_like(events), depths, seed=7
    )
    for rows in (slice(0, 3), slice(3, 6), slice(6, 9)):
        assert int(shuffled[rows].sum()) == int(events[rows].sum())
    assert not torch.equal(shuffled, events)


def test_matched_certificate_can_transfer_event_from_released_row():
    events = torch.tensor(
        [True, False, False, False, False, False, False, False, False]
    )
    eligible = torch.tensor(
        [False, True, True, True, True, True, True, True, True]
    )
    depths = torch.tensor(
        [2.0, 2.1, 2.2, 20.0, 21.0, 22.0, 200.0, 210.0, 220.0]
    )
    shuffled = matched_events_within_log_depth_regimes(
        events, eligible, depths, seed=7
    )
    assert int(shuffled.sum()) == 1
    assert not shuffled[0]


def test_footprint_release_restores_other_owner_and_is_one_way():
    limits, ownership = release_owned_scale_caps(
        torch.tensor([2.0, 3.0, 4.0]),
        torch.tensor([float("inf"), 2.5, float("inf")]),
        torch.tensor([True, True, False]),
        torch.tensor([False, True, True]),
    )
    assert torch.equal(limits, torch.tensor([2.0, 2.5, 4.0]))
    assert torch.equal(ownership, torch.tensor([True, False, False]))

    repeated, repeated_ownership = release_owned_scale_caps(
        limits,
        torch.tensor([float("inf"), 2.5, float("inf")]),
        ownership,
        torch.ones(3, dtype=torch.bool),
    )
    assert torch.isinf(repeated[0])
    assert repeated[1:].tolist() == [2.5, 4.0]
    assert not torch.any(repeated_ownership)


class _TrustCamera:
    near = 0.01
    far = 100.0

    def __init__(self, center_x):
        self._pose = torch.eye(4)
        self._pose[0, 3] = -float(center_x)

    def get_pose(self):
        return self._pose

    def get_int_mat(self, _level):
        return torch.tensor(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        )

    def get_width(self, _level):
        return 100

    def get_height(self, _level):
        return 100


def test_dynamic_footprint_release_is_causal_owned_and_monotonic():
    group = Gaussians(BS=1, scene_scale=1.0, max_sh_degree=0)
    group.to_device("cpu")
    group.non_trainable_params["mean_anchors"][0] = torch.tensor([0.0, 0.0, 10.0])
    group.non_trainable_params["reference_camera_centers"][0].zero_()
    group.non_trainable_params["reference_rays"][0] = torch.tensor([0.0, 0.0, 1.0])
    group.non_trainable_params["birth_log_depth_stds"][0] = 0.1
    group.non_trainable_params["footprint_target_scales"][0] = 0.5
    group.non_trainable_params["max_scale_expansions"][0] = 2.0
    group.non_trainable_params["footprint_release_scale_expansions"][0] = 3.0
    group.non_trainable_params["footprint_trust_mask"][0] = True
    group.non_trainable_params["footprint_evidence_pending_mask"][0] = True

    model = GaussianModel.__new__(GaussianModel)
    model.MAX_LEVEL = 1
    model.gaussian_groups = [group]
    model.active_gaussian_groups = {0: [0]}
    model.progressive_group_ids = set()
    model.frontview_far_field_config = {
        "footprint_trust_dynamic_update": True,
        "footprint_trust_dynamic_shuffle": False,
        "visibility_margin_px": 0.0,
        "shuffle_seed": 42,
    }
    model.frontview_far_field_stats = {
        "footprint_trust_dynamic_calls": 0,
        "footprint_trust_dynamic_rows": 0,
        "footprint_trust_dynamic_released_rows": 0,
        "footprint_trust_dynamic_shuffled_calls": 0,
    }

    assert model.update_dynamic_footprint_trust([_TrustCamera(0.1)]) == 0
    assert group.non_trainable_params["max_scale_expansions"][0] == 2.0
    assert group.non_trainable_params["footprint_trust_mask"][0]

    assert model.update_dynamic_footprint_trust([_TrustCamera(1.0)]) == 1
    assert group.non_trainable_params["max_scale_expansions"][0] == 3.0
    assert not group.non_trainable_params["footprint_trust_mask"][0]
    assert not group.non_trainable_params["footprint_evidence_pending_mask"][0]

    assert model.update_dynamic_footprint_trust([_TrustCamera(0.0)]) == 0
    assert group.non_trainable_params["max_scale_expansions"][0] == 3.0
