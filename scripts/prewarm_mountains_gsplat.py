#!/usr/bin/env python3
"""Build the shared gsplat extension once through the production SLAM path."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import run_mountains_crr_ablation as runner


BASE_CONFIG = (
    runner.ROOT
    / "Logs_mountains_adaptive_goal_8_12_8_13"
    / "final/stage35_full_765/batch_20260813_095909"
    / "runtime_configs/A_visible_residual_detail_real.yaml"
)
OUTPUT_ROOT = runner.ROOT / "Logs_mountains_far_depth_goal_8_13/gsplat_prewarm"
SHARED_CACHE = Path.home() / ".cache/torch_extensions/online3dgs_dual_shared"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="4")
    args = parser.parse_args()
    base = runner.load_config(str(BASE_CONFIG))
    config = runner.build_config(base, "gsplat_prewarm", {}, OUTPUT_ROOT, 1)
    config_path = OUTPUT_ROOT / "gsplat_prewarm.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    env = runner.process_env(args.gpu, 1)
    env["TORCH_EXTENSIONS_DIR"] = str(SHARED_CACHE)
    log_path = OUTPUT_ROOT / "gsplat_prewarm.log"
    with log_path.open("w", encoding="utf-8") as log:
        runner.run_command(
            [
                str(runner.DEFAULT_PYTHON),
                "slam_new.py",
                "--config",
                str(config_path),
                "--exp_name",
                "mountains_gsplat_prewarm",
                "--seed",
                "43",
                "--cpu_threads",
                "1",
            ],
            log,
            env,
        )
    library = SHARED_CACHE / "gsplat_cuda/gsplat_cuda.so"
    if not library.is_file() or library.stat().st_size == 0:
        raise RuntimeError(f"Shared gsplat extension was not built: {library}")
    print(library)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
