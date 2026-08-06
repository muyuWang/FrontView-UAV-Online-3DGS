#!/usr/bin/env python3
"""Create geometry-exact appearance blends between two saved Gaussian maps."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils_new.render_utils import select_gaussian_ply


GEOMETRY_FIELDS = (
    "x",
    "y",
    "z",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--refined-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        help="Refined-appearance fractions in [0, 1].",
    )
    parser.add_argument(
        "--specs",
        nargs="+",
        help=(
            "Named component blends as name:dc_alpha:rest_alpha:opacity_alpha. "
            "Use this to audit which appearance family provides the gain."
        ),
    )
    return parser.parse_args()


def _field_names(rows):
    return tuple(rows.dtype.names or ())


def _bits(values):
    values = np.ascontiguousarray(values)
    return values.view(np.uint8).reshape(values.shape[0], -1)


def blend_vertex_components(
    source, refined, dc_alpha, rest_alpha, opacity_alpha
):
    """Blend appearance families while proving Gaussian geometry is unchanged."""

    component_alphas = (dc_alpha, rest_alpha, opacity_alpha)
    if any(alpha < 0.0 or alpha > 1.0 for alpha in component_alphas):
        raise ValueError("component alphas must be in [0, 1]")
    if len(source) != len(refined):
        raise ValueError(
            "Gaussian count mismatch: {} != {}".format(len(source), len(refined))
        )

    source_fields = set(_field_names(source))
    refined_fields = set(_field_names(refined))
    missing_geometry = [
        name
        for name in GEOMETRY_FIELDS
        if name not in source_fields or name not in refined_fields
    ]
    if missing_geometry:
        raise ValueError("Missing geometry fields: {}".format(missing_geometry))

    geometry_report = {}
    for name in GEOMETRY_FIELDS:
        exact = bool(np.array_equal(_bits(source[name]), _bits(refined[name])))
        geometry_report[name + "_exactly_equal"] = exact
        if not exact:
            raise ValueError("Geometry differs in field {}".format(name))

    output = refined.copy()
    for name in ("nx", "ny", "nz"):
        if name in source_fields and name in refined_fields:
            output[name] = source[name]

    appearance_fields = sorted(
        name
        for name in refined_fields
        if name.startswith("f_dc_") or name.startswith("f_rest_")
    )
    if "opacity" not in source_fields or "opacity" not in refined_fields:
        raise ValueError("Both maps must contain opacity")
    if not appearance_fields:
        raise ValueError("Refined map does not contain SH coefficients")

    for name in appearance_fields:
        alpha = dc_alpha if name.startswith("f_dc_") else rest_alpha
        source_value = (
            source[name].astype(np.float64)
            if name in source_fields
            else np.zeros(len(source), dtype=np.float64)
        )
        refined_value = refined[name].astype(np.float64)
        output[name] = ((1.0 - alpha) * source_value + alpha * refined_value).astype(
            output.dtype[name]
        )

    source_opacity = source["opacity"].astype(np.float64)
    refined_opacity = refined["opacity"].astype(np.float64)
    output["opacity"] = (
        (1.0 - opacity_alpha) * source_opacity
        + opacity_alpha * refined_opacity
    ).astype(output.dtype["opacity"])

    for name in GEOMETRY_FIELDS:
        output[name] = source[name]
        if not np.array_equal(_bits(output[name]), _bits(source[name])):
            raise RuntimeError("Output geometry changed in field {}".format(name))

    return output, geometry_report


def blend_vertex_data(source, refined, alpha):
    """Blend all appearance families by one common refined fraction."""

    return blend_vertex_components(source, refined, alpha, alpha, alpha)


def _load_run(run_dir):
    run_dir = run_dir.resolve()
    config_path = run_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ply_path = select_gaussian_ply(run_dir, config).resolve()
    rows = PlyData.read(str(ply_path))["vertex"].data
    return run_dir, config, ply_path, rows


def _alpha_slug(alpha):
    return ("{:.3f}".format(alpha).rstrip("0").rstrip(".")).replace(".", "p")


def _copy_if_present(source, destination):
    if source.is_file():
        shutil.copy2(source, destination)


def _parse_specs(alphas, specs):
    parsed = []
    for alpha in alphas or ():
        parsed.append(
            (
                "alpha_" + _alpha_slug(alpha),
                float(alpha),
                float(alpha),
                float(alpha),
            )
        )
    for raw in specs or ():
        pieces = raw.split(":")
        if len(pieces) != 4 or not pieces[0]:
            raise ValueError(
                "Invalid --spec {!r}; expected name:dc:rest:opacity".format(raw)
            )
        parsed.append(
            (pieces[0], float(pieces[1]), float(pieces[2]), float(pieces[3]))
        )
    if not parsed:
        raise ValueError("At least one --alphas or --specs entry is required")
    names = [entry[0] for entry in parsed]
    if len(set(names)) != len(names):
        raise ValueError("Output blend names must be unique")
    if any(
        alpha < 0.0 or alpha > 1.0
        for entry in parsed
        for alpha in entry[1:]
    ):
        raise ValueError("All blend coefficients must be in [0, 1]")
    return parsed


def main():
    args = parse_args()
    blend_specs = _parse_specs(args.alphas, args.specs)

    source_run, source_config, source_ply, source_rows = _load_run(args.source_run)
    refined_run, refined_config, refined_ply, refined_rows = _load_run(
        args.refined_run
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    created = []
    for name, dc_alpha, rest_alpha, opacity_alpha in blend_specs:
        output_run = output_root / name
        if output_run.exists() and any(output_run.iterdir()):
            raise FileExistsError("Output run is not empty: {}".format(output_run))
        output_run.mkdir(parents=True, exist_ok=True)

        blended, geometry_report = blend_vertex_components(
            source_rows,
            refined_rows,
            float(dc_alpha),
            float(rest_alpha),
            float(opacity_alpha),
        )
        PlyData([PlyElement.describe(blended, "vertex")], text=False).write(
            str(output_run / "point_cloud.ply")
        )

        config = dict(refined_config)
        config["Results"] = dict(refined_config["Results"])
        config["Results"]["save_dir"] = str(output_run)
        (output_run / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        tracked_source = refined_run / "tracked_info.json"
        if not tracked_source.is_file():
            tracked_source = source_run / "tracked_info.json"
        _copy_if_present(tracked_source, output_run / "tracked_info.json")

        results = {}
        results_path = refined_run / "results.json"
        if results_path.is_file():
            results = json.loads(results_path.read_text(encoding="utf-8"))
        results["offline_appearance_blend"] = {
            "source_run": str(source_run),
            "source_ply": str(source_ply),
            "refined_run": str(refined_run),
            "refined_ply": str(refined_ply),
            "dc_refined_fraction": float(dc_alpha),
            "rest_refined_fraction": float(rest_alpha),
            "opacity_refined_fraction": float(opacity_alpha),
            "gaussian_count": int(len(blended)),
            "geometry_exactly_equal": bool(all(geometry_report.values())),
            "geometry_fields": geometry_report,
            "evaluation_status": "pending_render",
        }
        (output_run / "results.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        created.append(str(output_run))

    print(json.dumps({"created": created}, indent=2))


if __name__ == "__main__":
    main()
