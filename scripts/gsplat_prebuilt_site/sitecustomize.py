"""Register a prebuilt gsplat CUDA extension before gsplat is imported."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import torch  # noqa: F401 - loads libc10/libtorch before the CUDA extension.


path = os.environ.get("GSPLAT_PREBUILT_EXTENSION")
if path:
    library = Path(path).expanduser().resolve()
    if not library.is_file():
        raise RuntimeError(f"Missing prebuilt gsplat extension: {library}")
    spec = importlib.util.spec_from_file_location("gsplat_cuda", library)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load prebuilt gsplat extension: {library}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gsplat_cuda"] = module
    spec.loader.exec_module(module)
    sys.modules["gsplat.csrc"] = module
