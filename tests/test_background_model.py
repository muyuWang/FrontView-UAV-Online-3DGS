import numpy as np
import torch

from utils_new.background_model import (
    DirectionalSkyBackgroundModel,
    SkyBackgroundModel,
)


def test_sky_background_only_fills_top_transparency():
    model = SkyBackgroundModel((0.7, 0.8, 0.9), render_top_fraction=0.5)
    render = torch.zeros((4, 3, 3), dtype=torch.float32)
    opacity = torch.zeros((4, 3, 1), dtype=torch.float32)
    opacity[0, 0] = 1.0

    output = model.composite(render, opacity)

    assert torch.allclose(output[0, 0], torch.zeros(3))
    assert torch.allclose(output[0, 1], torch.tensor((0.7, 0.8, 0.9)))
    assert torch.count_nonzero(output[2:]) == 0


def test_sky_background_round_trip():
    model = SkyBackgroundModel((0.1, 0.2, 0.3), render_top_fraction=0.4)
    restored = SkyBackgroundModel.from_dict(model.to_dict(sampled_frames=5))
    assert np.allclose(restored.rgb, model.rgb)
    assert restored.render_top_fraction == model.render_top_fraction


def test_directional_sky_requires_world_ray_support():
    # A fully valid grid isolates camera/ray math from fitting in this unit test.
    model = DirectionalSkyBackgroundModel(
        rgb=(0.7, 0.8, 0.9),
        grid_shape=(8, 16),
        valid_indices=np.arange(8 * 16),
        min_support_frames=3,
    )
    camera = type(
        "CameraStub",
        (),
        {
            "get_int_mat": lambda self, level: torch.tensor(
                ((2.0, 0.0, 1.0), (0.0, 2.0, 1.0), (0.0, 0.0, 1.0))
            ),
            "get_pose": lambda self: torch.eye(4),
        },
    )()
    output = model.composite(
        torch.zeros((2, 2, 3)), torch.zeros((2, 2, 1)), camera
    )
    assert torch.allclose(output, torch.tensor((0.7, 0.8, 0.9)).expand_as(output))
