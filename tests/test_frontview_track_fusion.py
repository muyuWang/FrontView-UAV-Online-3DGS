import pytest
import torch

from utils_new.frontview_track_fusion import (
    robust_color_ema,
    validate_front_view_track_fusion_config,
)


def test_robust_color_ema_bounds_outlier_update():
    current = torch.zeros((1, 3))
    observed = torch.tensor([[3.0, 0.0, 0.0]])

    fused = robust_color_ema(current, observed, alpha=0.5, max_color_step=0.2)

    assert torch.allclose(fused, torch.tensor([[0.1, 0.0, 0.0]]))


def test_track_fusion_config_fails_closed():
    config = validate_front_view_track_fusion_config(
        {"enabled": True, "color_ema": 0.1}
    )
    assert config["enabled"] is True
    with pytest.raises(ValueError, match="color_ema"):
        validate_front_view_track_fusion_config({"color_ema": 0.0})
    with pytest.raises(ValueError, match="Unknown"):
        validate_front_view_track_fusion_config({"unknown": True})
