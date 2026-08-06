#!/usr/bin/env python3
"""Run the current hash-free TGBR-75 method on all prepared 360DVO scenes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "Online3DGS_360DVO"
DEFAULT_SAVE_ROOT = ROOT / "Logs_360dvo_tgbr75"
DEFAULT_CONFIG_DIR = ROOT / "configs" / "360dvo_tgbr75_runtime"
DEFAULT_PYTHON = Path("/home/wmy/anaconda3/envs/worldvln/bin/python")
DEFAULT_CUDA_HOME = Path("/usr/local/cuda-11.8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--cuda-home", type=Path, default=DEFAULT_CUDA_HOME)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--scenes", default=None, help="Comma-separated scene names; defaults to all.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--extra-points", type=int, default=3200)
    parser.add_argument("--skip-existing-success", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    error_group = parser.add_mutually_exclusive_group()
    error_group.add_argument("--continue-on-error", action="store_true", default=True)
    error_group.add_argument("--stop-on-error", action="store_false", dest="continue_on_error")
    return parser.parse_args()


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern)) if path.is_dir() else 0


def parse_csv_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("A comma-separated option was provided without any values")
    return values


def discover_scenes(data_root: Path, scene_filter: str | None) -> list[dict[str, Any]]:
    requested = parse_csv_names(scene_filter)
    candidates = requested or sorted(
        path.name for path in data_root.iterdir() if path.is_dir() and not path.name.startswith("_")
    )
    scenes: list[dict[str, Any]] = []
    for scene in candidates:
        scene_dir = data_root / scene
        stats_path = scene_dir / "conversion_stats.json"
        if not stats_path.is_file():
            raise FileNotFoundError(f"Missing conversion_stats.json for {scene}: {stats_path}")
        stats = read_json(stats_path)
        frame_count = int(stats["frame_count"])
        image_count = count_files(scene_dir / "rectified", "aria_*.jpg")
        point_count = count_files(scene_dir / "orb_point_clouds", "point_cloud_*.txt")
        if image_count != frame_count or point_count != frame_count:
            raise RuntimeError(
                f"Incomplete {scene}: expected={frame_count}, images={image_count}, points={point_count}"
            )
        for filename in ("trajectory_orb.json", "trajectory.json", "image_list.txt"):
            if not (scene_dir / filename).is_file():
                raise FileNotFoundError(f"Missing {filename} for {scene}: {scene_dir / filename}")
        source_config = ROOT / "configs" / "360dvo" / f"360DVO_{scene}_orb.yaml"
        if not source_config.is_file():
            raise FileNotFoundError(f"Missing source config for {scene}: {source_config}")
        scenes.append(
            {
                "scene": scene,
                "scene_dir": str(scene_dir.resolve()),
                "source_config": str(source_config.resolve()),
                "frame_count": frame_count,
            }
        )
    return sorted(scenes, key=lambda item: (-item["frame_count"], item["scene"]))


def dataset_name(scene: str) -> str:
    return f"360DVO-TGBR75-{scene.replace('_', '-')}"


def write_runtime_config(scene_info: dict[str, Any], args: argparse.Namespace) -> Path:
    scene = scene_info["scene"]
    name = dataset_name(scene)
    config = {
        "inherit_from": scene_info["source_config"],
        "Dataset": {"name": name, "max_pts_num": args.max_points},
        "Testset": {"name": name, "max_pts_num": args.max_points},
        "Results": {
            "save_dir": str(args.save_root),
            "save_gt": False,
            "save_exr": False,
            "save_mesh": False,
            "skip_eval": False,
        },
        "Mapper": {
            "use_multi_reso": False,
            "initialization_frames": 4,
            "optimization_iters": 10,
            "initialization_iters": 10,
            "post_refinement": {"max_steps": 100, "opt_cam": False},
            "KFGraph": {"kf_interval": 1, "global_window_size": 4},
            "CameraOptimizer": {"pose_refine_init_steps": 0, "pose_opt_steps": 4},
        },
        "Model": {
            "sh_degree": 3,
            "extra_pts_num": args.extra_points,
            "err_threshold": 0.05,
            "camera_scale_rescalar": 0.25,
            "scene_scale": 1.0,
        },
        "HashBlock": {"use_hash": False},
        "FrontViewSampling": {
            "enabled": True,
            "selection_mode": "depth_stratified",
            "pool_multiplier": 2,
            "evidence_fraction": 0.5,
            "reference_frames": 2,
            "photo_sigma": 0.08,
            "photo_mode": "consistency",
            "parallax_reference_deg": 2.0,
            "parallax_floor": 0.25,
            "confidence_power": 1.0,
            "shuffle_evidence": False,
            "shuffle_seed": 42,
            "depth_edges_m": [20.0, 50.0],
            "depth_fractions": [0.25, 0.45, 0.30],
        },
        "FrontViewFarField": {
            "enabled": True,
            "depth_m": 80.0,
            "projective_cell_px": 12,
            "depth_bin_ratio": 1.10,
            "shuffle_responsibility": False,
            "shuffle_seed": 42,
        },
        "FrontViewScaleCover": {
            "enabled": True,
            "radius_multiplier": 0.3,
            "scale_compatibility": 1.0,
            "neighbors": 32,
            "rebuild_rows": 8192,
            "color_distance_threshold": 0.15,
            "shuffle_occupancy": False,
            "shuffle_seed": 42,
        },
        "StreamingAppearanceLOD": {
            "enabled": True,
            "birth_degree": 2,
            "target_degree": 3,
            "min_views": 2,
            "max_target_fraction": 0.75,
            "promotion_interval": 10,
            "selection_mode": "gradient_agreement",
            "utility_ema_decay": 0.9,
            "shuffle_seed": 43,
        },
    }
    path = args.config_dir / f"360DVO_{scene}_tgbr75.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def process_env(gpu: str, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["CUDA_HOME"] = str(args.cuda_home)
    env["PATH"] = f"{args.cuda_home / 'bin'}:{env.get('PATH', '')}"
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env["TORCH_EXTENSIONS_DIR"] = str(
        Path.home() / ".cache" / "torch_extensions" / f"online3dgs_gpu{gpu}"
    )
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    return env


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        return subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def find_run_dir(save_root: Path, name: str, exp_name: str) -> Path:
    parent = save_root / name
    matches = sorted(parent.glob(f"*_{exp_name}"), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No output matching {exp_name} under {parent}")
    return matches[-1]


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def video_is_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return False
    try:
        streams = json.loads(probe.stdout).get("streams", [])
        return bool(streams and int(streams[0]["width"]) > 0 and int(streams[0]["height"]) > 0)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def collect_metrics(run_dir: Path) -> dict[str, Any]:
    results = read_json(run_dir / "results.json")
    final = read_json(run_dir / "eval" / "final_result.json")
    render = read_json(run_dir / "videos_full" / "render_metrics.json")
    source_metrics = results.get("eval_res", {})
    final_metrics = final.get("mean", {})
    render_metrics = render.get("mean", {})
    for label, metrics in (
        ("results.eval_res", source_metrics),
        ("eval/final_result.mean", final_metrics),
        ("videos_full/render_metrics.mean", render_metrics),
    ):
        for key in ("psnr", "ssim", "lpips"):
            if not finite_number(metrics.get(key)):
                raise RuntimeError(f"Invalid {label}.{key} in {run_dir}")

    scale_cover = results.get("frontview_scale_cover") or {}
    appearance = results.get("streaming_appearance_lod") or {}
    return {
        "psnr": float(source_metrics["psnr"]),
        "ssim": float(source_metrics["ssim"]),
        "lpips": float(source_metrics["lpips"]),
        "final_psnr": float(final_metrics["psnr"]),
        "final_ssim": float(final_metrics["ssim"]),
        "final_lpips": float(final_metrics["lpips"]),
        "render_psnr": float(render_metrics["psnr"]),
        "render_ssim": float(render_metrics["ssim"]),
        "render_lpips": float(render_metrics["lpips"]),
        "online_recon_time": results.get("online_recon_time"),
        "eval_time": results.get("eval_time"),
        "num_processed_frames": results.get("num_processed_frames"),
        "num_keyframes": results.get("num_keyframes"),
        "num_gaussians": results.get("num_gaussians"),
        "hash_query_rows": scale_cover.get("hash_query_rows"),
        "hash_set_rows": scale_cover.get("hash_set_rows"),
        "scale_cover_backend": scale_cover.get("query_backend"),
        "sh3_rows": appearance.get("target_degree_rows"),
        "sh3_fraction": appearance.get("target_degree_fraction"),
        "render_frame_count": render.get("frame_count"),
    }


def validate_completed_run(run_dir: Path) -> dict[str, Any]:
    required = (
        run_dir / "point_cloud.ply",
        run_dir / "results.json",
        run_dir / "eval" / "final_result.json",
        run_dir / "videos_full" / "render_metrics.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing required artifacts: " + ", ".join(missing))
    videos = (
        run_dir / "videos_full" / "render_vs_gt.mp4",
        run_dir / "videos_full" / "render_depth.mp4",
    )
    invalid = [str(path) for path in videos if not video_is_valid(path)]
    if invalid:
        raise RuntimeError("Missing or invalid videos: " + ", ".join(invalid))
    return {
        **collect_metrics(run_dir),
        "render_vs_gt_video": str(videos[0]),
        "render_depth_video": str(videos[1]),
    }


def latest_success(save_root: Path, scene: str) -> tuple[Path, dict[str, Any]] | None:
    parent = save_root / dataset_name(scene)
    if not parent.is_dir():
        return None
    for run_dir in sorted(parent.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        try:
            return run_dir, validate_completed_run(run_dir)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
    return None


CSV_FIELDS = [
    "scene",
    "status",
    "gpu",
    "frame_count",
    "psnr",
    "ssim",
    "lpips",
    "render_psnr",
    "render_ssim",
    "render_lpips",
    "online_recon_time",
    "num_processed_frames",
    "num_keyframes",
    "num_gaussians",
    "hash_query_rows",
    "hash_set_rows",
    "scale_cover_backend",
    "run_dir",
    "render_vs_gt_video",
    "render_depth_video",
    "slam_log",
    "render_log",
    "error",
]


class BatchState:
    def __init__(self, payload: dict[str, Any], summary_path: Path, csv_path: Path):
        self.payload = payload
        self.summary_path = summary_path
        self.csv_path = csv_path
        self.latest_path = summary_path.parent / "batch_status_latest.json"
        self.lock = threading.Lock()

    def update(self, scene: str, **values: Any) -> None:
        with self.lock:
            self.payload["records"][scene].update(values)
            self.payload["updated_at"] = iso_now()
            self._write_locked()

    def finish(self) -> None:
        with self.lock:
            self.payload["finished_at"] = iso_now()
            self.payload["updated_at"] = iso_now()
            self._write_locked()

    def write(self) -> None:
        with self.lock:
            self._write_locked()

    def _write_locked(self) -> None:
        for path in (self.summary_path, self.latest_path):
            temp = path.with_suffix(path.suffix + ".tmp")
            with temp.open("w", encoding="utf-8") as handle:
                json.dump(self.payload, handle, indent=2, sort_keys=True)
            os.replace(temp, path)

        temp_csv = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        with temp_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for record in self.payload["records"].values():
                writer.writerow({key: record.get(key, "") for key in CSV_FIELDS})
        os.replace(temp_csv, self.csv_path)


def run_scene(
    scene_info: dict[str, Any],
    gpu: str,
    config_path: Path,
    args: argparse.Namespace,
    batch_tag: str,
    state: BatchState,
) -> None:
    scene = scene_info["scene"]
    started = time.monotonic()
    exp_name = f"all360dvo_tgbr75_{scene}_seed{args.seed}_gpu{gpu}_{batch_tag}"
    slam_log = args.save_root / "batch_logs" / f"{scene}_gpu{gpu}_{batch_tag}_slam.log"
    render_log = args.save_root / "batch_logs" / f"{scene}_gpu{gpu}_{batch_tag}_render.log"
    state.update(
        scene,
        status="running_slam",
        gpu=gpu,
        started_at=iso_now(),
        config=str(config_path),
        slam_log=str(slam_log),
        render_log=str(render_log),
    )
    print(f"[GPU {gpu}] START {scene} ({scene_info['frame_count']} frames)", flush=True)

    env = process_env(gpu, args)
    slam_command = [
        str(args.python),
        "slam_new.py",
        "--config",
        str(config_path),
        "--exp_name",
        exp_name,
        "--seed",
        str(args.seed),
    ]
    return_code = run_logged(slam_command, slam_log, env)
    if return_code != 0:
        raise RuntimeError(f"slam_new.py exited with code {return_code}; see {slam_log}")

    run_dir = find_run_dir(args.save_root, dataset_name(scene), exp_name)
    state.update(scene, status="running_render", run_dir=str(run_dir), slam_return_code=return_code)
    render_command = [
        str(args.python),
        "render.py",
        "--run_dir",
        str(run_dir),
        "--output_dir",
        str(run_dir / "videos_full"),
        "--fps",
        str(args.fps),
        "--max_frames",
        "-1",
        "--device",
        "cuda:0",
        "--far_gs_depth_threshold",
        "80",
        "--skip_novel",
        "--skip_primitives",
    ]
    render_code = run_logged(render_command, render_log, env)
    if render_code != 0:
        raise RuntimeError(f"render.py exited with code {render_code}; see {render_log}")

    metrics = validate_completed_run(run_dir)
    elapsed = time.monotonic() - started
    state.update(
        scene,
        status="success",
        finished_at=iso_now(),
        wall_time=elapsed,
        render_return_code=render_code,
        **metrics,
    )
    print(
        f"[GPU {gpu}] DONE  {scene}: PSNR={metrics['psnr']:.5f}, "
        f"SSIM={metrics['ssim']:.6f}, LPIPS={metrics['lpips']:.6f}",
        flush=True,
    )


def worker(
    gpu: str,
    jobs: queue.Queue[tuple[dict[str, Any], Path]],
    args: argparse.Namespace,
    batch_tag: str,
    state: BatchState,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            scene_info, config_path = jobs.get_nowait()
        except queue.Empty:
            return
        scene = scene_info["scene"]
        try:
            run_scene(scene_info, gpu, config_path, args, batch_tag, state)
        except Exception as exc:  # noqa: BLE001 - record scene failure and preserve the queue.
            state.update(scene, status="failed", finished_at=iso_now(), error=str(exc))
            print(f"[GPU {gpu}] FAIL  {scene}: {exc}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                stop_event.set()
        finally:
            jobs.task_done()


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.save_root = args.save_root.resolve()
    args.config_dir = args.config_dir.resolve()
    gpu_ids = parse_csv_names(args.gpu_ids)
    assert gpu_ids is not None
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"Duplicate GPU IDs: {gpu_ids}")
    if not args.python.is_file():
        raise FileNotFoundError(f"Python interpreter does not exist: {args.python}")
    if not (args.cuda_home / "bin" / "nvcc").is_file():
        raise FileNotFoundError(f"CUDA toolkit does not contain nvcc: {args.cuda_home}")

    scenes = discover_scenes(args.data_root, args.scenes)
    args.save_root.mkdir(parents=True, exist_ok=True)
    args.config_dir.mkdir(parents=True, exist_ok=True)
    batch_tag = now_tag()
    summary_path = args.save_root / f"batch_summary_{batch_tag}.json"
    csv_path = args.save_root / f"batch_metrics_{batch_tag}.csv"
    payload: dict[str, Any] = {
        "batch_tag": batch_tag,
        "started_at": iso_now(),
        "updated_at": iso_now(),
        "repo_root": str(ROOT),
        "data_root": str(args.data_root),
        "save_root": str(args.save_root),
        "gpu_ids": gpu_ids,
        "seed": args.seed,
        "method": "hash-free PBSD + TSC + FPR-80 + TGBR-75 SH2-to-SH3",
        "max_points": args.max_points,
        "extra_points": args.extra_points,
        "records": {
            scene["scene"]: {
                "scene": scene["scene"],
                "status": "pending",
                "frame_count": scene["frame_count"],
                "scene_dir": scene["scene_dir"],
            }
            for scene in scenes
        },
    }
    state = BatchState(payload, summary_path, csv_path)

    jobs: queue.Queue[tuple[dict[str, Any], Path]] = queue.Queue()
    runnable_count = 0
    for scene_info in scenes:
        scene = scene_info["scene"]
        config_path = write_runtime_config(scene_info, args)
        if args.skip_existing_success:
            existing = latest_success(args.save_root, scene)
            if existing is not None:
                run_dir, metrics = existing
                state.update(
                    scene,
                    status="skipped_existing_success",
                    run_dir=str(run_dir),
                    config=str(config_path),
                    **metrics,
                )
                continue
        if args.dry_run:
            state.update(
                scene,
                status="dry_run",
                config=str(config_path),
                command=(
                    f"CUDA_VISIBLE_DEVICES=<worker> {args.python} slam_new.py "
                    f"--config {config_path} --seed {args.seed}"
                ),
            )
            continue
        jobs.put((scene_info, config_path))
        runnable_count += 1

    state.write()
    print(f"Discovered {len(scenes)} complete scenes; queued {runnable_count}", flush=True)
    print(f"GPUs: {','.join(gpu_ids)}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    if args.dry_run or runnable_count == 0:
        state.finish()
        return 0

    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=worker,
            name=f"gpu-{gpu}",
            args=(gpu, jobs, args, batch_tag, state, stop_event),
        )
        for gpu in gpu_ids
    ]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        print("Interrupted: active children received SIGINT; queued scenes remain pending", flush=True)
        for thread in threads:
            thread.join()

    if stop_event.is_set():
        while True:
            try:
                scene_info, _ = jobs.get_nowait()
            except queue.Empty:
                break
            state.update(scene_info["scene"], status="cancelled_after_failure")
            jobs.task_done()
    state.finish()

    failed = [
        record for record in state.payload["records"].values() if record["status"] == "failed"
    ]
    print(f"Batch finished: {len(scenes) - len(failed)}/{len(scenes)} without failure", flush=True)
    print(f"Metrics CSV: {csv_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
