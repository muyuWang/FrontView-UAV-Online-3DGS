#!/usr/bin/env python3
"""Run village1-4 on fixed GPU 5/6/7 queues and aggregate two-stage metrics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_mydata_scene_best_tgbr75.sh"
LOG_ROOT = ROOT / "Logs_mydata_villages_best_tgbr75"
QUEUES = {
    5: ["village1"],
    6: ["village2", "village3"],
    7: ["village4"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=[f"village{i}" for i in range(1, 5)],
        default=[f"village{i}" for i in range(1, 5)],
    )
    parser.add_argument("--batch-name", default="")
    return parser.parse_args()


def run_queue(gpu: int, scenes: list[str], batch_dir: Path, seed: int, dry_run: bool):
    rows = []
    for scene in scenes:
        result_path = batch_dir / f"{scene}.json"
        log_path = batch_dir / f"{scene}.log"
        exp_name = f"{batch_dir.name}_{scene}_seed{seed}_gpu{gpu}"
        command = [
            "bash",
            str(RUNNER),
            "--scene",
            scene,
            "--gpu",
            str(gpu),
            "--seed",
            str(seed),
            "--exp-name",
            exp_name,
            "--result-json",
            str(result_path),
        ]
        if dry_run:
            command.append("--dry-run")
        print(f"[gpu{gpu}] start {scene}; log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{scene} failed on GPU {gpu} with code {completed.returncode}; "
                f"see {log_path}"
            )
        if not dry_run:
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
        print(f"[gpu{gpu}] complete {scene}", flush=True)
    return rows


def write_summary(batch_dir: Path, rows: list[dict]) -> None:
    rows.sort(key=lambda row: row["scene"])
    payload = {
        "schema_version": 1,
        "batch": batch_dir.name,
        "metric_protocol": "same-trajectory reconstruction PSNR/SSIM; LPIPS disabled",
        "scenes": rows,
    }
    (batch_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (batch_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "scene", "frame_count", "online_mapping_seconds", "online_fps",
                "online_psnr", "online_ssim", "post_refinement_seconds",
                "post_psnr", "post_ssim", "psnr_delta", "ssim_delta", "run_dir",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scene": row["scene"],
                    "frame_count": row["frame_count"],
                    "online_mapping_seconds": row["online_mapping_seconds"],
                    "online_fps": row["online_fps"],
                    "online_psnr": row["online"]["psnr"],
                    "online_ssim": row["online"]["ssim"],
                    "post_refinement_seconds": row["post_refinement_seconds"],
                    "post_psnr": row["post_refinement"]["psnr"],
                    "post_ssim": row["post_refinement"]["ssim"],
                    "psnr_delta": row["post_refinement"]["psnr"] - row["online"]["psnr"],
                    "ssim_delta": row["post_refinement"]["ssim"] - row["online"]["ssim"],
                    "run_dir": row["run_dir"],
                }
            )


def main() -> int:
    args = parse_args()
    selected = set(args.scenes)
    stamp = args.batch_name or datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    batch_dir = LOG_ROOT / "_batch" / stamp
    batch_dir.mkdir(parents=True, exist_ok=False)
    active_queues = {
        gpu: [scene for scene in scenes if scene in selected]
        for gpu, scenes in QUEUES.items()
    }
    active_queues = {gpu: scenes for gpu, scenes in active_queues.items() if scenes}
    manifest = {
        "batch": stamp,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "queues": active_queues,
    }
    (batch_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    all_rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=len(active_queues)) as executor:
        futures = {
            executor.submit(
                run_queue, gpu, scenes, batch_dir, args.seed, args.dry_run
            ): gpu
            for gpu, scenes in active_queues.items()
        }
        for future in as_completed(futures):
            try:
                all_rows.extend(future.result())
            except Exception as error:
                failures.append(str(error))
                print(f"ERROR: {error}", file=sys.stderr, flush=True)
    if failures:
        (batch_dir / "failures.json").write_text(
            json.dumps(failures, indent=2) + "\n", encoding="utf-8"
        )
        return 1
    if not args.dry_run:
        write_summary(batch_dir, all_rows)
        print(f"Summary: {batch_dir / 'summary.json'}")
    else:
        print(f"Dry-run complete: {batch_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
