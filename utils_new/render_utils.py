"""Dependency-free helpers shared by offline rendering and its tests."""

from pathlib import Path
from typing import Any, Mapping


def select_gaussian_ply(run_dir: Path, config: Mapping[str, Any]) -> Path:
    """Select the complete progressive export when the run provides one."""
    baseline_path = run_dir / "point_cloud.ply"
    aerocommit_path = run_dir / "point_cloud_aerocommit_full.ply"
    progressive_path = run_dir / "point_cloud_progressive_full.ply"
    aerocommit_enabled = bool(config.get("AeroCommit", {}).get("enabled", False))
    progressive_enabled = bool(
        config.get("ProgressiveMapping", {}).get("enabled", False)
    )
    if progressive_enabled and progressive_path.exists():
        return progressive_path
    if aerocommit_enabled and aerocommit_path.exists():
        return aerocommit_path
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing Gaussian PLY: {baseline_path}")
    return baseline_path
