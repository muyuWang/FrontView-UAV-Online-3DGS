#!/usr/bin/env python3
"""Collect comparable MODP/AeroCommit run metrics without rerunning mapping."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml


def read_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(records, key, percentile):
    values = [float(record.get(key, 0.0)) for record in records]
    return float(np.percentile(values, percentile)) if values else None


def git_value(repo, *args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_names():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted(set(line.strip() for line in output.splitlines() if line.strip()))


def collect_run(label, run_dir, cold_jit_seconds):
    run_dir = run_dir.resolve()
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    results = read_json(run_dir / "results.json")
    validation = read_json(run_dir / "validation_metrics.json")
    records = read_jsonl(run_dir / "aerocommit_stats.jsonl")
    trajectory = validation.get("trajectory_metrics", {}).get("mean", {})
    aero = config.get("AeroCommit", {})
    final = records[-1] if records else {}
    total_seconds = results.get("online_recon_time")
    steady_seconds = None
    if total_seconds is not None:
        steady_seconds = max(0.0, float(total_seconds) - cold_jit_seconds)
    dataset = config.get("Dataset", {})
    calibration = dataset.get("Calibration", {})
    post = config.get("Mapper", {}).get("post_refinement", {})
    return {
        "label": label,
        "run_dir": str(run_dir),
        "config_path": str(run_dir / "config.yaml"),
        "mode": aero.get("mode", "baseline"),
        "admission_policy": aero.get("admission", {}).get("policy", "immediate"),
        "random_seed": config.get("Results", {}).get("random_seed"),
        "scene": dataset.get("dataset_path"),
        "frames": results.get("num_processed_frames"),
        "resolution": {
            "width": calibration.get("width"),
            "height": calibration.get("height"),
        },
        "online_reconstruction_seconds": total_seconds,
        "cold_jit_seconds_assumed": cold_jit_seconds,
        "steady_reconstruction_seconds_estimate": steady_seconds,
        "average_seconds_per_frame": (
            float(total_seconds) / results["num_processed_frames"]
            if total_seconds is not None and results.get("num_processed_frames")
            else None
        ),
        "admission_processing_ms": {
            "mean": (
                float(np.mean([row.get("frame_total_ms", 0.0) for row in records]))
                if records
                else None
            ),
            "p50": percentile(records, "frame_total_ms", 50),
            "p95": percentile(records, "frame_total_ms", 95),
            "scope": "proposal/admission/refinement/archive only; excludes base mapper",
        },
        "quality": trajectory,
        "gaussians": {
            "active": results.get("num_gaussians"),
            "full_export": results.get("num_gaussians_full", results.get("num_gaussians")),
            "archived": final.get("num_archived_gaussians", 0),
        },
        "memory_bytes": {
            "parameters_final": final.get("parameter_bytes"),
            "gradients_final": final.get("gradient_bytes"),
            "optimizer_final": final.get("optimizer_bytes"),
            "candidates_final": final.get("candidate_bytes"),
            "archive_cpu_final": final.get("archive_cpu_bytes"),
            "cuda_peak_allocated": max(
                (int(row.get("cuda_peak_allocated", 0)) for row in records),
                default=None,
            ),
        },
        "post_refinement": {
            "enabled": int(post.get("max_steps", 0)) > 0,
            "max_steps": post.get("max_steps"),
            "optimize_camera": post.get("opt_cam", False),
            "uses_all_frames": post.get("use_all_frames", False),
            "measured_seconds": None,
        },
        "causal_online_inputs_only": not bool(post.get("use_all_frames", False)),
        "aerocommit_totals": {
            "risk_evaluations": sum(row.get("num_risk_evaluations", 0) for row in records),
            "committed_candidates": sum(
                row.get("num_committed_candidates", 0) for row in records
            ),
            "detail_splits": sum(row.get("num_detail_splits", 0) for row in records),
            "side_detail_splits": sum(
                row.get("num_side_detail_splits", 0) for row in records
            ),
            "depth_confidence_fast_path": sum(
                row.get("num_depth_confidence_fast_path_gaussians", 0)
                for row in records
            ),
            "expired_candidates": sum(row.get("num_expired", 0) for row in records),
        },
        "unavailable_metrics": [
            "base_mapper_ms",
            "render_ms",
            "backward_ms",
            "mapping_only_ms",
            "lpips",
        ],
    }


def parse_run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=RUN_DIR")
    label, path = value.split("=", 1)
    return label, Path(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cold_jit_seconds",
        type=float,
        default=0.0,
        help="Common one-time gsplat JIT time to subtract for a steady estimate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cold_jit_seconds < 0.0:
        raise ValueError("cold_jit_seconds must be non-negative")
    repo = Path(__file__).resolve().parents[1]
    runs = [
        collect_run(label, path, args.cold_jit_seconds)
        for label, path in args.run
    ]
    baseline = runs[0]
    baseline_total = baseline["online_reconstruction_seconds"]
    baseline_steady = baseline["steady_reconstruction_seconds_estimate"]
    baseline_psnr = baseline["quality"].get("psnr")
    for run in runs:
        total = run["online_reconstruction_seconds"]
        steady = run["steady_reconstruction_seconds_estimate"]
        psnr = run["quality"].get("psnr")
        run["relative_to_first_run"] = {
            "runtime_ratio": (
                float(total) / baseline_total if total is not None and baseline_total else None
            ),
            "steady_runtime_ratio_estimate": (
                float(steady) / baseline_steady
                if steady is not None and baseline_steady
                else None
            ),
            "psnr_delta_db": (
                float(psnr) - baseline_psnr
                if psnr is not None and baseline_psnr is not None
                else None
            ),
        }
    payload = {
        "git": {
            "branch": git_value(repo, "branch", "--show-current"),
            "commit": git_value(repo, "rev-parse", "HEAD"),
            "dirty": bool(git_value(repo, "status", "--porcelain")),
        },
        "gpu_models_visible": gpu_names(),
        "comparison_baseline": runs[0]["label"],
        "runs": runs,
        "notes": [
            "Quality is measured from decoded render_vs_gt.mp4 frames.",
            "Cold JIT subtraction is an explicit user-supplied estimate, not inferred.",
            "Post-refinement uses only stored online keyframes unless uses_all_frames is true.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
