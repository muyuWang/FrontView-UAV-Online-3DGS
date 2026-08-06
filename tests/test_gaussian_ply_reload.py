import numpy as np
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
