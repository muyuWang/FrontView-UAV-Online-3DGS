#!/usr/bin/env python3
"""Create MP4 previews for HorizonGS synthetic aerial/street RGB folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_horizongs_real_previews import make_preview


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTHETIC_ROOT = REPO_ROOT / "data" / "HorizonGS" / "synthetic"


def find_sequences(synthetic_root: Path) -> list[tuple[Path, Path]]:
    sequences: list[tuple[Path, Path]] = []
    for scene_dir in sorted([p for p in synthetic_root.iterdir() if p.is_dir()]):
        for name in ("aerial", "street"):
            rgb_dir = scene_dir / name / "rgb"
            if rgb_dir.is_dir():
                sequences.append((rgb_dir, scene_dir / f"{name}.mp4"))
    return sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    synthetic_root = args.synthetic_root.resolve()
    if args.downsample <= 0:
        raise ValueError("--downsample must be > 0")
    if not synthetic_root.is_dir():
        raise FileNotFoundError(synthetic_root)

    sequences = find_sequences(synthetic_root)
    if not sequences:
        raise RuntimeError(f"No aerial/street RGB folders found under {synthetic_root}")

    print(f"Found {len(sequences)} RGB folders under {synthetic_root}")
    for rgb_dir, output_path in sequences:
        make_preview(
            image_dir=rgb_dir,
            fps=args.fps,
            downsample=args.downsample,
            overwrite=not args.no_overwrite,
            output_path=output_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
