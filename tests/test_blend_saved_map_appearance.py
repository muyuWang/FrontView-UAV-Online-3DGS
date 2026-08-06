import numpy as np
import pytest

from scripts.blend_saved_map_appearance import (
    GEOMETRY_FIELDS,
    blend_vertex_components,
    blend_vertex_data,
)


def _rows(include_rest, x_offset=0.0):
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
    ]
    if include_rest:
        names += ["f_rest_0", "f_rest_1", "f_rest_2"]
    names += [
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
    for index, name in enumerate(GEOMETRY_FIELDS):
        rows[name] = np.asarray([index + 1.0, index + 2.0], dtype=np.float32)
    rows["x"] += np.float32(x_offset)
    rows["f_dc_0"] = [1.0, 2.0]
    rows["opacity"] = [-1.0, 1.0]
    if include_rest:
        rows["f_dc_0"] = [3.0, 4.0]
        rows["f_rest_0"] = [2.0, 6.0]
        rows["opacity"] = [1.0, 3.0]
    return rows


def test_blend_vertex_data_preserves_geometry_and_blends_missing_sh_as_zero():
    source = _rows(include_rest=False)
    refined = _rows(include_rest=True)

    output, report = blend_vertex_data(source, refined, 0.25)

    assert all(report.values())
    for name in GEOMETRY_FIELDS:
        assert np.array_equal(output[name].view(np.uint32), source[name].view(np.uint32))
    assert output["f_dc_0"] == pytest.approx([1.5, 2.5])
    assert output["f_rest_0"] == pytest.approx([0.5, 1.5])
    assert output["opacity"] == pytest.approx([-0.5, 1.5])


def test_blend_vertex_data_rejects_geometry_mismatch():
    source = _rows(include_rest=False)
    refined = _rows(include_rest=True, x_offset=0.5)

    with pytest.raises(ValueError, match="Geometry differs in field x"):
        blend_vertex_data(source, refined, 0.5)


def test_blend_vertex_components_separates_sh_and_opacity_responsibility():
    source = _rows(include_rest=False)
    refined = _rows(include_rest=True)

    output, _ = blend_vertex_components(
        source,
        refined,
        dc_alpha=0.0,
        rest_alpha=0.5,
        opacity_alpha=1.0,
    )

    assert output["f_dc_0"] == pytest.approx([1.0, 2.0])
    assert output["f_rest_0"] == pytest.approx([1.0, 3.0])
    assert output["opacity"] == pytest.approx([1.0, 3.0])
