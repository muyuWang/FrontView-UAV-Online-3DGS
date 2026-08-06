#!/usr/bin/env python3
"""Convert the 10x-denser AirVLN scene10 capture to Online-3DGS-Monocular format.

Input:
  data/airvln_extracted/aerialvln_s_scene10_10x

Output:
  data/Online3DGS_AirVLN/aerialvln_s_scene10_10x

The output layout matches this repo's Aria/ORB dataset loader:
  rectified/aria_XXXX.png
  trajectory_orb.json
  trajectory.json
  orb_point_clouds/point_cloud_<idx>.txt
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARGS = [
    "--input-dir",
    str(REPO_ROOT / "data" / "airvln_extracted" / "aerialvln_s_scene10_10x"),
    "--output-dir",
    str(REPO_ROOT / "data" / "Online3DGS_AirVLN" / "aerialvln_s_scene10_10x"),
    "--config-dir",
    str(REPO_ROOT / "configs" / "airvln"),
    "--name",
    "AerialVLNS-scene10-10x",
    "--config-prefix",
    "AerialVLNS_scene10_10x",
    "--smoke-frames",
    "20",
    "--pair-gaps",
    "5,10,20,40",
]


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from convert_airvln_to_online3dgs import main as convert_main

    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    return convert_main()


if __name__ == "__main__":
    raise SystemExit(main())
