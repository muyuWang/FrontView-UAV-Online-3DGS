#!/usr/bin/env python3
"""Evaluate saved mountains PIR checkpoints with PSNR/SSIM only."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "Logs_mountains_pir_ablation_8_6"
DEFAULT_PYTHON = Path("/home/wmy/anaconda3/envs/worldvln/bin/python")
VARIANT_GPUS = {
    "A_source_raster": "4",
    "B_source_exact": "5",
    "C_identity_exact": "6",
    "D_identity_shuffled": "7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--fps", type=float, default=24.0)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def find_run_dir(root: Path, variant: str) -> Path:
    matches = sorted(
        (root / variant).glob("*/*"), key=lambda path: path.stat().st_mtime
    )
    matches = [path for path in matches if (path / "point_cloud.ply").is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one saved checkpoint for {variant}, got {matches}")
    return matches[0]


def process_env(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    cuda_home = Path("/usr/local/cuda-11.8")
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["CUDA_HOME"] = str(cuda_home)
    env["PATH"] = f"{cuda_home / 'bin'}:{env.get('PATH', '')}"
    env["TORCH_EXTENSIONS_DIR"] = str(
        Path.home() / ".cache" / "torch_extensions" / f"online3dgs_gpu{gpu}"
    )
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def parse_online_log(root: Path, variant: str) -> dict[str, float | int]:
    text = (root / "launcher_logs" / f"{variant}.log").read_text(
        encoding="utf-8", errors="replace"
    )
    patterns = {
        "online_recon_time": (r"Total reconstruction time:\s+([0-9.]+) s", float),
        "num_processed_frames": (r"Total Frames:\s+([0-9]+)", int),
        "num_gaussians": (r"Total Gaussians:\s+([0-9]+)", int),
        "num_keyframes": (r"Total keyframes:\s+([0-9]+)", int),
    }
    result: dict[str, float | int] = {}
    for name, (pattern, convert) in patterns.items():
        matches = re.findall(pattern, text)
        if not matches:
            raise RuntimeError(f"Missing {name} in {variant} log")
        result[name] = convert(matches[-1])
    result["online_fps"] = (
        float(result["num_processed_frames"]) / float(result["online_recon_time"])
    )
    return result


def evaluate_variant(
    root: Path,
    python: Path,
    fps: float,
    variant: str,
    gpu: str,
) -> dict[str, Any]:
    run_dir = find_run_dir(root, variant)
    output_dir = run_dir / "videos_psnr_ssim"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "render.py",
        "--run_dir",
        str(run_dir),
        "--output_dir",
        str(output_dir),
        "--fps",
        str(fps),
        "--max_frames",
        "-1",
        "--device",
        "cuda:0",
        "--far_gs_depth_threshold",
        "80",
        "--skip_lpips",
        "--skip_novel",
        "--skip_primitives",
        "--ignore_cached_renders",
    ]
    log_path = root / "launcher_logs" / f"{variant}_psnr_ssim.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=process_env(gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"{variant} render exited {result.returncode}: {log_path}")

    metrics_path = output_dir / "render_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics_payload = json.load(handle)
    mean = metrics_payload["mean"]
    if "lpips" in mean:
        raise RuntimeError(f"LPIPS unexpectedly present in {metrics_path}")
    row = {
        "status": "success",
        "variant": variant,
        "gpu": gpu,
        "run_dir": str(run_dir),
        "metrics": {"psnr": float(mean["psnr"]), "ssim": float(mean["ssim"])},
        "frame_count": int(metrics_payload["frame_count"]),
        "evaluation_wall_time": time.monotonic() - started,
        "metrics_path": str(metrics_path),
        "render_vs_gt": str(output_dir / "render_vs_gt.mp4"),
        "render_depth": str(output_dir / "render_depth.mp4"),
    }
    row.update(parse_online_log(root, variant))
    return row


def metric_delta(results: dict[str, dict[str, Any]], lhs: str, rhs: str):
    return {
        key: results[lhs]["metrics"][key] - results[rhs]["metrics"][key]
        for key in ("psnr", "ssim")
    }


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                evaluate_variant, root, python, args.fps, variant, gpu
            ): variant
            for variant, gpu in VARIANT_GPUS.items()
        }
        for future in as_completed(futures):
            variant = futures[future]
            try:
                results[variant] = future.result()
                print(f"{variant}: success", flush=True)
            except Exception as error:  # noqa: BLE001 - collect all batch failures.
                failures[variant] = str(error)
                print(f"{variant}: failed: {error}", flush=True)

    successful = not failures and len(results) == len(VARIANT_GPUS)
    summary: dict[str, Any] = {
        "status": "success" if successful else "failed",
        "metric_protocol": "saved checkpoint render, PSNR/SSIM only, no LPIPS",
        "results": results,
        "failures": failures,
    }
    if successful:
        summary["comparisons"] = {
            "B_minus_A_exact_geometry": metric_delta(
                results, "B_source_exact", "A_source_raster"
            ),
            "C_minus_B_identity_responsibility": metric_delta(
                results, "C_identity_exact", "B_source_exact"
            ),
            "C_minus_D_true_vs_shuffled_identity": metric_delta(
                results, "C_identity_exact", "D_identity_shuffled"
            ),
        }
    write_json(root / "summary_psnr_ssim.json", summary)
    print(f"Summary: {root / 'summary_psnr_ssim.json'}")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
