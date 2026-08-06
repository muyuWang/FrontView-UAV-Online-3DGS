import math

import pytest
import torch

from utils_new.frontview_observability import (
    parallax_learning_scale,
    precondition_raywise_gradient,
    validate_front_view_observability_config,
)


def test_frontview_observability_config_rejects_invalid_scale():
    with pytest.raises(ValueError):
        validate_front_view_observability_config({"min_ray_lr_scale": 1.1})


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
