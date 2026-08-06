import torch

from utils_new.appearance_lod import (
    AppearanceLODEvidence,
    camera_centers_from_viewmats,
)
from utils_new.gaussian_models import GaussianModel, Gaussians


def viewmat_from_center(center):
    viewmat = torch.eye(4)
    viewmat[:3, 3] = -torch.as_tensor(center)
    return viewmat


def test_camera_centers_round_trip_translation_only():
    centers = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])
    viewmats = torch.stack([viewmat_from_center(center) for center in centers])
    torch.testing.assert_close(camera_centers_from_viewmats(viewmats), centers)


def test_evidence_selects_nested_appearance_degrees_and_shuffle_preserves_count():
    means = torch.zeros((3, 3))
    centers = torch.tensor(
        [
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 2.0],
            [3.0, 0.0, 2.0],
        ]
    )
    viewmats = torch.stack([viewmat_from_center(center) for center in centers])
    info = {
        "gaussian_ids": torch.tensor([0] * 6 + [1] * 3),
        "camera_ids": torch.tensor(list(range(6)) + [0, 2, 4]),
        "radii": torch.tensor([[3.0, 2.5]] * 6 + [[1.5, 1.0]] * 3),
        "depths": torch.ones(9),
    }
    evidence = AppearanceLODEvidence(gaussian_count=3, device="cpu")
    assert evidence.observe(info, means, viewmats) == 9
    config = {
        "min_views_sh1": 3,
        "min_views_sh2": 6,
        "min_mean_radius_sh1": 1.0,
        "min_mean_radius_sh2": 2.0,
        "min_angular_dispersion_sh1": 1.0e-5,
        "min_angular_dispersion_sh2": 1.0e-5,
        "max_sh1_fraction": 1.0,
        "max_sh2_fraction": 1.0,
    }
    degrees, stats = evidence.select_degrees(config)
    assert degrees.tolist() == [2, 1, 0]
    assert stats["degree_counts"] == {"sh0": 1, "sh1": 1, "sh2": 1}
    assert sum(stats["view_count_histogram"].values()) == 2

    shuffled, shuffled_stats = evidence.select_degrees(
        {**config, "selection_mode": "shuffled", "shuffle_seed": 7}
    )
    assert torch.sort(shuffled).values.tolist() == [0, 1, 2]
    assert shuffled_stats["degree_counts"] == stats["degree_counts"]


def test_gaussian_model_masks_inactive_sh_bands_and_gradients():
    model = GaussianModel.__new__(GaussianModel)
    model.device = "cpu"
    model.max_sh_degree = 2
    model.MAX_LEVEL = 4
    model.active_gaussian_groups = {level: [level] for level in range(4)}
    model.gaussian_groups = [
        Gaussians(BS=1, scene_scale=1.0, max_sh_degree=2) for _ in range(4)
    ]
    model.sh_degree_masks = {}
    for group in model.gaussian_groups:
        group.to_device("cpu")
        group.splats["shN"].data.fill_(1.0)

    model.configure_sh_degree_masks(torch.tensor([0, 1, 2, 0]))
    assert torch.count_nonzero(model.gaussian_groups[0].splats["shN"]) == 0
    assert torch.count_nonzero(model.gaussian_groups[1].splats["shN"]) == 9
    assert torch.count_nonzero(model.gaussian_groups[2].splats["shN"]) == 24

    for group in model.gaussian_groups:
        group.splats["shN"].grad = torch.ones_like(group.splats["shN"])
    model.mask_sh_degree_gradients()
    assert torch.count_nonzero(model.gaussian_groups[0].splats["shN"].grad) == 0
    assert torch.count_nonzero(model.gaussian_groups[1].splats["shN"].grad) == 9
    assert torch.count_nonzero(model.gaussian_groups[2].splats["shN"].grad) == 24
