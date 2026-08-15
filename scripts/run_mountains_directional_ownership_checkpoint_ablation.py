#!/usr/bin/env python3
"""Evaluate directional ownership gates on one fixed Mountains checkpoint."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    ROOT
    / "Logs_mountains_far_depth_goal_8_13/source_ablation_620"
    / "batch_20260813_153626/C_no_dropout_depth_fallback"
    / "360DVO-CRR-C_no_dropout_depth_fallback-mountains"
    / "2026-08-13-15-36-32_mountains_crr_C_no_dropout_depth_fallback_seed43_gpu6_20260813_153626"
)
DEFAULT_OUTPUT = (
    ROOT / "Logs_mountains_far_depth_goal_8_13/directional_ownership_checkpoint"
)
DEFAULT_MASKS = (
    ROOT
    / "Logs_mountains_adaptive_goal_8_12_8_13/evaluation"
    / "fixed_far_masks_q80_545_619_v2"
)
SHARED_GSPLAT_CACHE = Path.home() / ".cache/torch_extensions/online3dgs_dual_shared"
GSPLAT_PREBUILT_EXTENSION = SHARED_GSPLAT_CACHE / "gsplat_cuda/gsplat_cuda.so"
GSPLAT_PREBUILT_SITE = Path(__file__).resolve().parent / "gsplat_prebuilt_site"


VARIANTS = (
    ("A_owner_off", ("--skip_directional_layer",)),
    (
        "B_legacy_all_w075",
        ("--directional_geometry_gate_mode", "none", "--directional_blend_weight", "0.75"),
    ),
    (
        "C_opacity050_w075",
        (
            "--directional_geometry_gate_mode",
            "opacity",
            "--directional_low_opacity_threshold",
            "0.50",
            "--directional_blend_weight",
            "0.75",
        ),
    ),
    (
        "D_opacity075_w075",
        (
            "--directional_geometry_gate_mode",
            "opacity",
            "--directional_low_opacity_threshold",
            "0.75",
            "--directional_blend_weight",
            "0.75",
        ),
    ),
    (
        "E_opacity090_w075",
        (
            "--directional_geometry_gate_mode",
            "opacity",
            "--directional_low_opacity_threshold",
            "0.90",
            "--directional_blend_weight",
            "0.75",
        ),
    ),
    (
        "F_opacity075_w100",
        (
            "--directional_geometry_gate_mode",
            "opacity",
            "--directional_low_opacity_threshold",
            "0.75",
            "--directional_blend_weight",
            "1.00",
        ),
    ),
    (
        "G_metric_transmittance_w075",
        (
            "--directional_geometry_gate_mode",
            "metric_transmittance",
            "--directional_blend_weight",
            "0.75",
        ),
    ),
    (
        "H_metric_transmittance_w100",
        (
            "--directional_geometry_gate_mode",
            "metric_transmittance",
            "--directional_blend_weight",
            "1.00",
        ),
    ),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixed-far-mask-dir", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--gpus", default="6,7")
    return parser.parse_args()


def metric_mean(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["mean"]


def run_queue(gpu, jobs, args, results, lock):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["CUDA_HOME"] = "/usr/local/cuda-11.8"
    env["PATH"] = f"/usr/local/cuda-11.8/bin:{env.get('PATH', '')}"
    env["TORCH_EXTENSIONS_DIR"] = str(SHARED_GSPLAT_CACHE)
    env["GSPLAT_PREBUILT_EXTENSION"] = str(GSPLAT_PREBUILT_EXTENSION)
    env["PYTHONPATH"] = f"{GSPLAT_PREBUILT_SITE}:{env.get('PYTHONPATH', '')}"
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env.setdefault("OMP_NUM_THREADS", "2")
    for name, overrides in jobs:
        output_dir = args.output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "render.py",
            "--run_dir",
            str(args.run_dir),
            "--output_dir",
            str(output_dir),
            "--device",
            "cuda:0",
            "--render_begin",
            "545",
            "--render_end",
            "619",
            "--fixed_far_mask_dir",
            str(args.fixed_far_mask_dir),
            "--skip_lpips",
            "--skip_novel",
            "--skip_primitives",
            "--skip_depth",
            "--ignore_cached_renders",
            *overrides,
        ]
        log_path = output_dir / "render.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        row = {"variant": name, "gpu": gpu, "returncode": completed.returncode}
        metrics_path = output_dir / "render_metrics.json"
        if completed.returncode == 0 and metrics_path.is_file():
            row.update(metric_mean(metrics_path))
            modalities_path = output_dir / "render_modalities.json"
            if modalities_path.is_file():
                with modalities_path.open("r", encoding="utf-8") as handle:
                    row["directional_layer"] = json.load(handle).get(
                        "frontview_directional_layer", {}
                    )
        with lock:
            results.append(row)
            with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump({"results": sorted(results, key=lambda item: item["variant"])}, handle, indent=2)


def main():
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.fixed_far_mask_dir = args.fixed_far_mask_dir.expanduser().resolve()
    for path in (args.run_dir, args.fixed_far_mask_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not GSPLAT_PREBUILT_EXTENSION.is_file():
        raise FileNotFoundError(GSPLAT_PREBUILT_EXTENSION)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queues = [[] for _ in gpus]
    for index, variant in enumerate(VARIANTS):
        queues[index % len(gpus)].append(variant)
    results = []
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=run_queue, args=(gpu, jobs, args, results, lock), daemon=False
        )
        for gpu, jobs in zip(gpus, queues)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = [row for row in results if row["returncode"] != 0]
    return 1 if failed or len(results) != len(VARIANTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
