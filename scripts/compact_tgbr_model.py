#!/usr/bin/env python3
"""Convert a dense final TGBR PLY into its sparse high-band representation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils_new.tgbr_sparse_model import convert_dense_tgbr_ply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-degree", type=int, default=2)
    parser.add_argument("--target-degree", type=int, default=3)
    parser.add_argument("--stats", type=Path)
    args = parser.parse_args()

    if args.output.resolve() == args.input.resolve():
        raise ValueError("Refusing to overwrite the source dense PLY")
    stats = convert_dense_tgbr_ply(
        args.input,
        args.output,
        base_degree=args.base_degree,
        target_degree=args.target_degree,
    )
    stats_path = args.stats or args.output.with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
