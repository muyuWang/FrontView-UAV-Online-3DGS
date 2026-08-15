"""Sparse persistent storage for evidence-routed SH bands."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


FORMAT_VERSION = 1
MASK_ELEMENT = "tgbr_mask"
HIGH_BAND_ELEMENT = "tgbr_high_sh"
MASK_PROPERTY = "bits"


def sh_non_dc_coefficient_count(degree):
    degree = int(degree)
    if degree < 0:
        raise ValueError("SH degree must be non-negative")
    return (degree + 1) ** 2 - 1


def _sorted_properties(element, prefix):
    names = [prop.name for prop in element.properties if prop.name.startswith(prefix)]
    return sorted(names, key=lambda name: int(name.rsplit("_", 1)[1]))


def _metadata_comments(base_degree, target_degree):
    return [
        "tgbr_sparse_sh_version {}".format(FORMAT_VERSION),
        "tgbr_base_degree {}".format(int(base_degree)),
        "tgbr_target_degree {}".format(int(target_degree)),
        "tgbr_mask_bitorder little",
    ]


def _parse_metadata(plydata):
    metadata = {}
    for comment in plydata.comments:
        fields = comment.split(maxsplit=1)
        if len(fields) == 2 and fields[0].startswith("tgbr_"):
            metadata[fields[0]] = fields[1]
    if metadata.get("tgbr_sparse_sh_version") != str(FORMAT_VERSION):
        raise ValueError("Unsupported or missing TGBR sparse SH format version")
    if metadata.get("tgbr_mask_bitorder") != "little":
        raise ValueError("Unsupported TGBR sparse SH mask bit order")
    try:
        base_degree = int(metadata["tgbr_base_degree"])
        target_degree = int(metadata["tgbr_target_degree"])
    except (KeyError, ValueError) as error:
        raise ValueError("Invalid TGBR sparse SH degree metadata") from error
    if base_degree < 0 or target_degree <= base_degree:
        raise ValueError("Invalid TGBR sparse SH degree interval")
    return base_degree, target_degree


def is_tgbr_sparse_ply(plydata):
    names = {element.name for element in plydata.elements}
    return MASK_ELEMENT in names or HIGH_BAND_ELEMENT in names


def _flatten_channel_major(values):
    values = np.asarray(values, dtype=np.float32)
    return values.transpose(0, 2, 1).reshape(values.shape[0], -1)


def _structured_rows(field_names, values, dtype="f4"):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(field_names):
        raise ValueError("Structured PLY values do not match their fields")
    rows = np.empty(values.shape[0], dtype=[(name, dtype) for name in field_names])
    for index, name in enumerate(field_names):
        rows[name] = values[:, index]
    return rows


def compact_payload_bytes(
    gaussian_count,
    active_count,
    base_degree,
    target_degree,
    scale_dimensions=3,
    quaternion_dimensions=4,
    auxiliary_float_fields=0,
):
    gaussian_count = int(gaussian_count)
    active_count = int(active_count)
    base_coefficients = sh_non_dc_coefficient_count(base_degree)
    target_coefficients = sh_non_dc_coefficient_count(target_degree)
    common_float_fields = (
        3
        + 3
        + 3
        + 1
        + int(scale_dimensions)
        + int(quaternion_dimensions)
        + int(auxiliary_float_fields)
    )
    base_bytes = gaussian_count * (common_float_fields + 3 * base_coefficients) * 4
    mask_bytes = math.ceil(gaussian_count / 8)
    high_bytes = active_count * 3 * (target_coefficients - base_coefficients) * 4
    return base_bytes + mask_bytes + high_bytes


def dense_payload_bytes(
    gaussian_count,
    target_degree,
    scale_dimensions=3,
    quaternion_dimensions=4,
    auxiliary_float_fields=0,
):
    gaussian_count = int(gaussian_count)
    target_coefficients = sh_non_dc_coefficient_count(target_degree)
    common_float_fields = (
        3
        + 3
        + 3
        + 1
        + int(scale_dimensions)
        + int(quaternion_dimensions)
        + int(auxiliary_float_fields)
    )
    return gaussian_count * (common_float_fields + 3 * target_coefficients) * 4


def write_tgbr_sparse_ply(
    path,
    *,
    means,
    sh0,
    shN,
    opacities,
    scales,
    quats,
    active_mask,
    base_degree,
    target_degree,
    metric_confidences=None,
    uncertainty_confidences=None,
):
    """Write one self-contained PLY with a dense base and a sparse high-SH bank."""

    path = Path(path)
    means = np.asarray(means, dtype=np.float32)
    sh0 = np.asarray(sh0, dtype=np.float32)
    shN = np.asarray(shN, dtype=np.float32)
    opacities = np.asarray(opacities, dtype=np.float32).reshape(-1, 1)
    scales = np.asarray(scales, dtype=np.float32)
    quats = np.asarray(quats, dtype=np.float32)
    active_mask = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
    if metric_confidences is not None:
        metric_confidences = np.asarray(
            metric_confidences, dtype=np.float32
        ).reshape(-1, 1)
    if uncertainty_confidences is not None:
        uncertainty_confidences = np.asarray(
            uncertainty_confidences, dtype=np.float32
        ).reshape(-1, 1)

    count = means.shape[0]
    if means.shape != (count, 3) or sh0.shape != (count, 1, 3):
        raise ValueError("Invalid Gaussian means or SH0 shape")
    if any(values.shape[0] != count for values in (shN, opacities, scales, quats)):
        raise ValueError("All Gaussian arrays must have the same row count")
    if active_mask.shape != (count,):
        raise ValueError("TGBR active mask must have one value per Gaussian")
    if metric_confidences is not None and metric_confidences.shape != (count, 1):
        raise ValueError("Metric confidences must have one value per Gaussian")
    if (
        uncertainty_confidences is not None
        and uncertainty_confidences.shape != (count, 1)
    ):
        raise ValueError("Uncertainty confidences must have one value per Gaussian")

    base_coefficients = sh_non_dc_coefficient_count(base_degree)
    target_coefficients = sh_non_dc_coefficient_count(target_degree)
    if shN.shape != (count, target_coefficients, 3):
        raise ValueError("SH tensor does not match the TGBR target degree")
    inactive_high = shN[~active_mask, base_coefficients:target_coefficients]
    if np.any(inactive_high != 0.0):
        raise ValueError("Inactive TGBR rows contain non-zero high-band coefficients")

    normals = np.zeros_like(means)
    base_sh = _flatten_channel_major(shN[:, :base_coefficients])
    vertex_fields = ["x", "y", "z", "nx", "ny", "nz"]
    vertex_fields += ["f_dc_{}".format(index) for index in range(3)]
    vertex_fields += [
        "f_rest_{}".format(index) for index in range(base_sh.shape[1])
    ]
    vertex_fields += ["opacity"]
    vertex_fields += ["scale_{}".format(index) for index in range(scales.shape[1])]
    vertex_fields += ["rot_{}".format(index) for index in range(quats.shape[1])]
    if metric_confidences is not None:
        vertex_fields.append("metric_confidence")
    if uncertainty_confidences is not None:
        vertex_fields.append("uncertainty_confidence")
    vertex_values = np.concatenate(
        (
            means,
            normals,
            sh0.reshape(count, 3),
            base_sh,
            opacities,
            scales,
            quats,
            *(() if metric_confidences is None else (metric_confidences,)),
            *(
                ()
                if uncertainty_confidences is None
                else (uncertainty_confidences,)
            ),
        ),
        axis=1,
    )
    vertex_rows = _structured_rows(vertex_fields, vertex_values)

    packed_mask = np.packbits(active_mask.astype(np.uint8), bitorder="little")
    mask_rows = np.empty(packed_mask.shape[0], dtype=[(MASK_PROPERTY, "u1")])
    mask_rows[MASK_PROPERTY] = packed_mask

    high_sh = _flatten_channel_major(
        shN[active_mask, base_coefficients:target_coefficients]
    )
    high_fields = ["f_high_{}".format(index) for index in range(high_sh.shape[1])]
    high_rows = _structured_rows(high_fields, high_sh)

    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [
            PlyElement.describe(vertex_rows, "vertex"),
            PlyElement.describe(mask_rows, MASK_ELEMENT),
            PlyElement.describe(high_rows, HIGH_BAND_ELEMENT),
        ],
        text=False,
        byte_order="<",
        comments=_metadata_comments(base_degree, target_degree),
    ).write(path)

    compact_payload = compact_payload_bytes(
        count,
        int(active_mask.sum()),
        base_degree,
        target_degree,
        scales.shape[1],
        quats.shape[1],
        auxiliary_float_fields=(
            int(metric_confidences is not None)
            + int(uncertainty_confidences is not None)
        ),
    )
    dense_payload = dense_payload_bytes(
        count,
        target_degree,
        scales.shape[1],
        quats.shape[1],
        auxiliary_float_fields=(
            int(metric_confidences is not None)
            + int(uncertainty_confidences is not None)
        ),
    )
    return {
        "format": "tgbr_sparse_sh_v{}".format(FORMAT_VERSION),
        "gaussian_count": count,
        "active_high_band_count": int(active_mask.sum()),
        "active_high_band_fraction": float(active_mask.mean()) if count else 0.0,
        "base_degree": int(base_degree),
        "target_degree": int(target_degree),
        "file_bytes": path.stat().st_size,
        "compact_payload_bytes": compact_payload,
        "dense_payload_bytes": dense_payload,
        "payload_reduction_fraction": (
            1.0 - float(compact_payload) / float(dense_payload)
            if dense_payload
            else 0.0
        ),
    }


def decode_tgbr_sparse_sh(plydata, max_sh_degree):
    """Decode the sparse band bank into the renderer's dense SH tensor."""

    if not is_tgbr_sparse_ply(plydata):
        raise ValueError("PLY does not contain a TGBR sparse SH bank")
    base_degree, target_degree = _parse_metadata(plydata)
    if int(max_sh_degree) < target_degree:
        raise ValueError("Model SH degree is lower than the sparse PLY target degree")

    vertex = plydata["vertex"]
    count = len(vertex.data)
    base_coefficients = sh_non_dc_coefficient_count(base_degree)
    target_coefficients = sh_non_dc_coefficient_count(target_degree)
    max_coefficients = sh_non_dc_coefficient_count(max_sh_degree)

    base_names = _sorted_properties(vertex, "f_rest_")
    expected_base_width = base_coefficients * 3
    if len(base_names) != expected_base_width:
        raise ValueError("Sparse PLY base SH width does not match its metadata")
    base_flat = np.stack(
        [np.asarray(vertex[name], dtype=np.float32) for name in base_names], axis=1
    )
    base_sh = base_flat.reshape(count, 3, base_coefficients).transpose(0, 2, 1)

    packed_mask = np.asarray(
        plydata[MASK_ELEMENT][MASK_PROPERTY], dtype=np.uint8
    ).reshape(-1)
    expected_mask_bytes = math.ceil(count / 8)
    if packed_mask.shape[0] != expected_mask_bytes:
        raise ValueError("Sparse PLY mask length does not match its Gaussian count")
    active_mask = np.unpackbits(
        packed_mask, bitorder="little", count=count
    ).astype(np.bool_)

    high_element = plydata[HIGH_BAND_ELEMENT]
    high_names = _sorted_properties(high_element, "f_high_")
    high_coefficients = target_coefficients - base_coefficients
    if len(high_names) != high_coefficients * 3:
        raise ValueError("Sparse PLY high-band width does not match its metadata")
    if len(high_element.data) != int(active_mask.sum()):
        raise ValueError("Sparse PLY mask population does not match its high-band rows")
    high_flat = np.stack(
        [np.asarray(high_element[name], dtype=np.float32) for name in high_names],
        axis=1,
    )
    high_sh = high_flat.reshape(-1, 3, high_coefficients).transpose(0, 2, 1)

    shN = np.zeros((count, max_coefficients, 3), dtype=np.float32)
    shN[:, :base_coefficients] = base_sh
    shN[active_mask, base_coefficients:target_coefficients] = high_sh
    degrees = np.full(count, base_degree, dtype=np.uint8)
    degrees[active_mask] = target_degree
    return shN, degrees, {
        "base_degree": base_degree,
        "target_degree": target_degree,
        "gaussian_count": count,
        "active_high_band_count": int(active_mask.sum()),
    }


def read_dense_gaussian_ply(path, target_degree):
    """Read the repository's standard dense Gaussian PLY into numpy arrays."""

    plydata = PlyData.read(path)
    if is_tgbr_sparse_ply(plydata):
        raise ValueError("Expected a standard dense Gaussian PLY")
    vertex = plydata["vertex"]
    count = len(vertex.data)

    def stack(names):
        return np.stack(
            [np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1
        )

    means = stack(("x", "y", "z"))
    sh0 = stack(("f_dc_0", "f_dc_1", "f_dc_2")).reshape(count, 1, 3)
    rest_names = _sorted_properties(vertex, "f_rest_")
    coefficient_count = sh_non_dc_coefficient_count(target_degree)
    if len(rest_names) != coefficient_count * 3:
        raise ValueError("Dense PLY SH width does not match target_degree")
    shN = stack(rest_names).reshape(count, 3, coefficient_count).transpose(0, 2, 1)
    opacity = np.asarray(vertex["opacity"], dtype=np.float32)
    vertex_names = {prop.name for prop in vertex.properties}
    metric_confidences = (
        np.asarray(vertex["metric_confidence"], dtype=np.float32)
        if "metric_confidence" in vertex_names
        else None
    )
    uncertainty_confidences = (
        np.asarray(vertex["uncertainty_confidence"], dtype=np.float32)
        if "uncertainty_confidence" in vertex_names
        else metric_confidences.copy()
        if metric_confidences is not None
        else None
    )
    scale_names = _sorted_properties(vertex, "scale_")
    rotation_names = _sorted_properties(vertex, "rot_")
    result = {
        "means": means,
        "sh0": sh0,
        "shN": shN,
        "opacities": opacity,
        "scales": stack(scale_names),
        "quats": stack(rotation_names),
    }
    if metric_confidences is not None:
        result["metric_confidences"] = metric_confidences
    if uncertainty_confidences is not None:
        result["uncertainty_confidences"] = uncertainty_confidences
    return result


def convert_dense_tgbr_ply(input_path, output_path, base_degree=2, target_degree=3):
    """Losslessly compact a final TGBR map by inferring its non-zero high band."""

    input_path = Path(input_path)
    arrays = read_dense_gaussian_ply(input_path, target_degree)
    base_coefficients = sh_non_dc_coefficient_count(base_degree)
    target_coefficients = sh_non_dc_coefficient_count(target_degree)
    active_mask = np.any(
        arrays["shN"][:, base_coefficients:target_coefficients] != 0.0,
        axis=(1, 2),
    )
    stats = write_tgbr_sparse_ply(
        output_path,
        **arrays,
        active_mask=active_mask,
        base_degree=base_degree,
        target_degree=target_degree,
    )
    compact_ply = PlyData.read(output_path)
    restored_shN, _, _ = decode_tgbr_sparse_sh(compact_ply, target_degree)
    if not np.array_equal(restored_shN, arrays["shN"]):
        raise RuntimeError("TGBR sparse SH conversion was not bitwise exact")
    stats.update(
        {
            "source_file_bytes": input_path.stat().st_size,
            "file_reduction_fraction": 1.0
            - float(Path(output_path).stat().st_size)
            / float(input_path.stat().st_size),
            "bitwise_sh_roundtrip": True,
        }
    )
    return stats
