#!/usr/bin/env python3
"""Fit a robust far-field sky sidecar from frames already observed by a run."""

import argparse
from pathlib import Path
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils_new.background_model import fit_and_save_sky_background


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--render-top-fraction", type=float, default=0.35)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-support-frames", type=int, default=20)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    model, path = fit_and_save_sky_background(
        config,
        run_dir,
        options={
            "render_top_fraction": args.render_top_fraction,
            "frame_stride": args.frame_stride,
            "min_support_frames": args.min_support_frames,
        },
    )
    print(path)
    print("RGB: {}".format(list(model.rgb)))


if __name__ == "__main__":
    main()
