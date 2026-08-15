#!/usr/bin/env python3
"""Run one decision-preserving Stage35 opacity-pruning audit on Mountains."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

import run_mountains_landmark_admitted_mean as shared


runner = shared.runner
BASE_CONFIG = shared.BASE_CONFIG
OUTPUT_ROOT = (
    runner.ROOT / "Logs_mountains_far_depth_goal_8_13/pruning_audit"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = OUTPUT_ROOT / f"batch_{tag}"
    variant = "A_stage35_read_only_pruning_audit"
    variant_root = batch_root / variant
    baseline = runner.load_config(str(BASE_CONFIG))
    config = runner.build_config(baseline, variant, {}, variant_root, 765)
    config["CausalDepthAudit"] = {
        "enabled": True,
        "start_frame": 0,
        "audit_opacity_pruning": True,
    }
    config_path = batch_root / "runtime_configs" / f"{variant}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=False)
    with config_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "protocol": (
            "Stage35 decisions unchanged; read-only summaries are captured before "
            "each opacity-pruning call; no LPIPS or render evaluation"
        ),
        "gpu": str(args.gpu),
        "seed": int(args.seed),
        "config": str(config_path),
    }
    runner.write_json(batch_root / "manifest.json", manifest)
    print(batch_root, flush=True)
    if args.dry_run:
        return 0

    exp_name = f"mountains_pruning_audit_seed{args.seed}_gpu{args.gpu}_{tag}"
    log_path = batch_root / "launcher.log"
    env = runner.process_env(str(args.gpu), args.cpu_threads)
    with log_path.open("x", encoding="utf-8") as log:
        runner.run_command(
            [
                str(runner.DEFAULT_PYTHON),
                "slam_new.py",
                "--config",
                str(config_path),
                "--exp_name",
                exp_name,
                "--seed",
                str(args.seed),
                "--cpu_threads",
                str(args.cpu_threads),
            ],
            log,
            env,
        )
    run_dir = runner.find_run_dir(
        variant_root, config["Dataset"]["name"], exp_name
    )
    with (run_dir / "results.json").open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    summary = {
        "status": "success",
        "run_dir": str(run_dir),
        "num_processed_frames": int(result["num_processed_frames"]),
        "num_gaussians": int(result["num_gaussians"]),
        "online_recon_time": float(result["online_recon_time"]),
        "causal_depth_audit": result["causal_depth_audit"],
    }
    runner.write_json(batch_root / "summary.json", summary)
    manifest.update(status="success", summary=str(batch_root / "summary.json"))
    runner.write_json(batch_root / "manifest.json", manifest)
    print(batch_root / "summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
