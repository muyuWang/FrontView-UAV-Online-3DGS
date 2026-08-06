#!/usr/bin/env python3
"""Run slam_new.py with the worldvln conda bin directory on PATH."""

from __future__ import annotations

import os
import runpy
import sys


WORLDVLN_BIN = "/home/wmy/miniconda3/envs/worldvln/bin"


def main() -> None:
    os.environ["PATH"] = WORLDVLN_BIN + os.pathsep + os.environ.get("PATH", "")
    sys.argv = ["slam_new.py", *sys.argv[1:]]
    runpy.run_path("slam_new.py", run_name="__main__")


if __name__ == "__main__":
    main()
