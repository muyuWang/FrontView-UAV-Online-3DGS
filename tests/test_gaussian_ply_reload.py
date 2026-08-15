import numpy as np
import torch
from plyfile import PlyData, PlyElement

from utils_new.gaussian_models import GaussianModel, Gaussians


def make_model():
    model = GaussianModel.__new__(GaussianModel)
    model.max_sh_degree = 0
    model.MAX_LEVEL = 4
    model.active_gaussian_groups = {level: [level] for level in range(4)}
    model.valid_groups = list(range(4))
    model.gaussian_groups = [
        Gaussians(BS=1, scene_scale=1.0, max_sh_degree=0) for _ in range(4)
    ]
    for group in model.gaussian_groups:
        group.to_device("cpu")
    return model


def write_test_ply(path):
    names = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    rows = np.zeros(2, dtype=[(name, "f4") for name in names])
    rows["x"] = [1.0, 2.0]
    rows["y"] = [3.0, 4.0]
    rows["z"] = [5.0, 6.0]
    rows["opacity"] = 1.0
    rows["scale_0"] = rows["scale_1"] = rows["scale_2"] = -2.0
    rows["rot_0"] = 1.0
    PlyData([PlyElement.describe(rows, "vertex")]).write(path)


def test_ply_reload_does_not_accumulate_seed_gaussians(tmp_path):
    source = tmp_path / "source.ply"
    first_roundtrip = tmp_path / "first_roundtrip.ply"
    second_roundtrip = tmp_path / "second_roundtrip.ply"
    write_test_ply(source)

    first = make_model()
    first.load_from_ply(str(source))
    assert first.get_num_gaussians == 2
    assert [group.get_num for group in first.gaussian_groups] == [2, 0, 0, 0]
    first.save_as_ply(str(first_roundtrip))

    second = make_model()
    second.load_from_ply(str(first_roundtrip))
    assert second.get_num_gaussians == 2
    second.save_as_ply(str(second_roundtrip))

    assert len(PlyData.read(first_roundtrip)["vertex"].data) == 2
    assert len(PlyData.read(second_roundtrip)["vertex"].data) == 2


def test_depth_confidences_survive_extend_prune_and_standard_ply(tmp_path):
    model = make_model()
    model.valid_groups = [0]
    group = model.gaussian_groups[0]
    group.replace_gaussians(
        {
            "means": torch.zeros((2, 3)),
            "scales": torch.zeros((2, 3)),
            "quats": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            ),
            "opacities": torch.zeros(2),
            "sh0": torch.zeros((2, 1, 3)),
            "shN": torch.zeros((2, 0, 3)),
            "metric_confidences": torch.tensor([0.0, 0.4]),
            "uncertainty_confidences": torch.tensor([0.7, 0.8]),
        }
    )
    group.extend_gaussians_from_color_points(
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
        metric_confidences=np.asarray([0.9], dtype=np.float32),
        uncertainty_confidences=np.asarray([0.95], dtype=np.float32),
    )
    group.prune_with_mask([False, True, True])
    expected = np.asarray([0.4, 0.9], dtype=np.float32)
    expected_uncertainty = np.asarray([0.8, 0.95], dtype=np.float32)
    assert np.allclose(
        group.non_trainable_params["metric_confidences"].numpy(), expected
    )
    assert np.allclose(
        group.non_trainable_params["uncertainty_confidences"].numpy(),
        expected_uncertainty,
    )

    path = tmp_path / "metric_confidence.ply"
    model.save_as_ply(path)
    restored = make_model()
    restored.load_from_ply(path)
    assert np.array_equal(
        restored.gaussian_groups[0]
        .non_trainable_params["metric_confidences"]
        .numpy(),
        expected,
    )
    assert np.array_equal(
        restored.gaussian_groups[0]
        .non_trainable_params["uncertainty_confidences"]
        .numpy(),
        expected_uncertainty,
    )


def test_disabled_dual_responsibility_keeps_standard_render_depth(monkeypatch):
    model = make_model()
    model.device = "cpu"
    model.gaussian_type = "3dgs"
    model.use_anti_aliasing = False
    model.radius_clip = 0.0
    model.render_mode = "RGB+ED"
    model.active_sh_degree = 0
    model.scene_exposure_gain = 1.0
    model.vignette_imgs = []
    model.causal_dual_responsibility_config = {
        "enabled": False,
        "directional_use_metric_depth": True,
    }
    model.streaming_appearance_lod_config = {"enabled": False}
    captured = {}

    class DirectionalLayer:
        def composite(self, _cam, colors, depth, _opacity):
            captured["depth"] = depth
            return colors

    class Camera:
        exposure_gain = 1.0
        near = 0.01
        far = 1000.0

        @staticmethod
        def get_pose():
            return torch.eye(4)

        @staticmethod
        def get_int_mat(_level):
            return torch.eye(3)

        @staticmethod
        def get_width(_level):
            return 2

        @staticmethod
        def get_height(_level):
            return 2

    model.frontview_directional_layer = DirectionalLayer()
    monkeypatch.setattr(model, "_frontview_handoff_opacities", lambda x, *_: x)
    monkeypatch.setattr(model, "_streaming_appearance_render_degrees", lambda _: None)

    def fake_rasterization(**_kwargs):
        rendered = torch.zeros((1, 2, 2, 4))
        rendered[..., 3] = 7.0
        return rendered, torch.ones((1, 2, 2, 1)), {}

    monkeypatch.setattr("utils_new.gaussian_models.rasterization", fake_rasterization)
    result = model.render_3dgs(Camera())

    assert captured["depth"] is result["depth"]
    assert torch.all(result["depth"] == 7.0)
