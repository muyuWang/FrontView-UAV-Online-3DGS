#!/usr/bin/env python3
"""Download one AirVLN/AerialVLN simulator environment from Kaggle.

Kaggle hosts the simulator dataset as nested files under paths such as:
env_10/env_10/LinuxNoEditor/AirVLN/Content/Paks/AirVLN-LinuxNoEditor.pak

The public download API supports fetching one file by passing file_name. Some
large files are returned as zip archives that contain the real binary, so this
script downloads to a cache directory and then installs/unpacks files into the
expected ENVs/env_x/env_x/LinuxNoEditor layout.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AIRVLN_ROOT = REPO_ROOT / "data" / "airvln"
DATASET_REF = "shuboliu/aerialvln-simulators"
DOWNLOAD_API = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", type=int, required=True, help="Environment id, e.g. 10.")
    parser.add_argument(
        "--airvln-root",
        type=Path,
        default=DEFAULT_AIRVLN_ROOT,
        help=f"AirVLN data root. Default: {DEFAULT_AIRVLN_ROOT}",
    )
    parser.add_argument("--force", action="store_true", help="Redownload cached files.")
    return parser.parse_args()


def kaggle_url(dataset_path: str) -> str:
    return f"{DOWNLOAD_API}?file_name={quote(dataset_path)}"


def run_curl(url: str, output_path: Path, force: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        print(f"Using cached: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if force and part_path.exists():
        part_path.unlink()
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "-C",
        "-",
        "-o",
        str(part_path),
        url,
    ]
    print(f"Downloading: {url}")
    subprocess.run(cmd, check=True)
    part_path.replace(output_path)


def read_manifest(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rel_paths: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        rel_paths.append(line.split("\t", 1)[0])
    return rel_paths


def cache_name(env_name: str, rel_path: str) -> str:
    dataset_path = f"{env_name}/{env_name}/LinuxNoEditor/{rel_path}"
    return dataset_path.replace("/", "__") + ".download"


def install_cached_file(cache_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(cache_path):
        with zipfile.ZipFile(cache_path) as zf:
            names = [name for name in zf.namelist() if not name.endswith("/")]
            if len(names) != 1:
                raise RuntimeError(f"Expected one file in {cache_path}, got {names}")
            member = names[0]
            tmp_dir = target_path.parent / ".extract_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            zf.extract(member, tmp_dir)
            extracted = tmp_dir / member
            shutil.move(str(extracted), str(target_path))
            try:
                extracted.parent.rmdir()
                tmp_dir.rmdir()
            except OSError:
                pass
    else:
        shutil.copy2(cache_path, target_path)


def main() -> int:
    args = parse_args()
    env_name = f"env_{args.env_id}"
    root = args.airvln_root.resolve()
    target_root = root / "ENVs" / env_name / env_name / "LinuxNoEditor"
    cache_root = root / "downloads" / f"{env_name}_single_file_zips"

    manifest_rel = "Manifest_NonUFSFiles_Linux.txt"
    manifest_dataset_path = f"{env_name}/{env_name}/LinuxNoEditor/{manifest_rel}"
    manifest_cache = cache_root / cache_name(env_name, manifest_rel)
    run_curl(kaggle_url(manifest_dataset_path), manifest_cache, args.force)
    target_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_cache, target_root / manifest_rel)

    rel_paths = read_manifest(manifest_cache)
    # The manifest lists Non-UFS runtime files but not the cooked scene pak.
    # The pak is the actual environment content and is required to launch.
    pak_rel = "AirVLN/Content/Paks/AirVLN-LinuxNoEditor.pak"
    if pak_rel not in rel_paths:
        rel_paths.append(pak_rel)
    if manifest_rel not in rel_paths:
        rel_paths.append(manifest_rel)

    for idx, rel_path in enumerate(rel_paths, start=1):
        dataset_path = f"{env_name}/{env_name}/LinuxNoEditor/{rel_path}"
        cache_path = cache_root / cache_name(env_name, rel_path)
        target_path = target_root / rel_path
        print(f"[{idx}/{len(rel_paths)}] {rel_path}")
        run_curl(kaggle_url(dataset_path), cache_path, args.force)
        install_cached_file(cache_path, target_path)

    for executable in [
        target_root / "AirVLN.sh",
        target_root / "AirVLN" / "Binaries" / "Linux" / "AirVLN-Linux-Shipping",
    ]:
        if executable.exists():
            mode = executable.stat().st_mode
            executable.chmod(mode | 0o111)

    print(f"Installed: {target_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        raise SystemExit(exc.returncode)
