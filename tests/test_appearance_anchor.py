from types import SimpleNamespace

import pytest
import torch

from utils_new.appearance_anchor import (
    AppearanceProximalAnchor,
    validate_appearance_anchor_config,
)


def _model():
    splats = torch.nn.ParameterDict(
        {
            "sh0": torch.nn.Parameter(torch.tensor([[[1.0, 2.0, 3.0]]])),
            "shN": torch.nn.Parameter(torch.zeros((1, 3, 3))),
            "opacities": torch.nn.Parameter(torch.tensor([0.5])),
        }
    )
    group = SimpleNamespace(splats=splats, get_num=1)
    return SimpleNamespace(valid_groups=[0], gaussian_groups=[group])


def test_validate_appearance_anchor_config_is_disabled_by_default():
    config = validate_appearance_anchor_config(None)

    assert config == {
        "sh0_weight": 0.0,
        "shN_weight": 0.0,
        "opacity_weight": 0.0,
        "enabled": False,
    }


def test_appearance_anchor_uses_frozen_snapshot_and_backpropagates():
    model = _model()
    anchor = AppearanceProximalAnchor(
        model,
        {"sh0_weight": 2.0, "shN_weight": 0.0, "opacity_weight": 0.5},
    )
    with torch.no_grad():
        model.gaussian_groups[0].splats["sh0"].add_(1.0)
        model.gaussian_groups[0].splats["opacities"].add_(2.0)

    loss, components = anchor.loss(model)
    loss.backward()

    assert components["sh0"].item() == pytest.approx(1.0)
    assert components["opacities"].item() == pytest.approx(4.0)
    assert loss.item() == pytest.approx(4.0)
    assert torch.all(model.gaussian_groups[0].splats["sh0"].grad > 0.0)
    assert model.gaussian_groups[0].splats["opacities"].grad.item() > 0.0
    assert model.gaussian_groups[0].splats["shN"].grad is None


def test_appearance_anchor_rejects_negative_weight():
    with pytest.raises(ValueError, match="non-negative"):
        validate_appearance_anchor_config({"opacity_weight": -1.0})
