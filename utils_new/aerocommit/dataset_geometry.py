"""Dataset geometry checks for sparse fast-path admission."""

import json
from pathlib import Path


def inspect_sparse_world_geometry(dataset_config, default_rmse_threshold=0.25):
    """Return whether sparse points share one persistent world frame, if known."""
    dataset_path = Path(dataset_config.get("dataset_path", ""))
    stats_path = dataset_path / "conversion_stats.json"
    if not stats_path.is_file():
        return None, "conversion metadata unavailable"

    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)

    geometry = stats.get("sparse_world_geometry")
    if geometry == "persistent":
        return True, "conversion metadata marks sparse geometry persistent"
    if geometry == "frame_local_reprojected":
        return False, "conversion metadata marks sparse geometry frame-local"

    if stats.get("pose_source") != "gt":
        return None, "legacy conversion metadata has no geometry classification"

    alignment = stats.get("rtk_alignment") or {}
    rmse = alignment.get("rmse_all_m")
    if rmse is None:
        return None, "legacy GT conversion has no alignment RMSE"
    threshold = float(
        stats.get("max_gt_alignment_rmse_m", default_rmse_threshold)
    )
    persistent = float(rmse) <= threshold
    return persistent, "legacy GT conversion RMSE {:.3f}m {} {:.3f}m".format(
        float(rmse), "<=" if persistent else ">", threshold
    )


def guard_sparse_fast_path(aerocommit_config, dataset_config):
    """Disable irreversible sparse admission for known frame-local geometry."""
    persistent, reason = inspect_sparse_world_geometry(dataset_config)
    changed = False
    if persistent is False:
        if bool(aerocommit_config.get("diagnostic_allow_unsafe_hybrid", False)):
            return {
                "persistent": False,
                "changed": False,
                "diagnostic_override": True,
                "reason": reason,
            }
        admission = aerocommit_config["admission"]
        if admission["trusted_sparse_fast_path"]:
            admission["trusted_sparse_fast_path"] = False
            changed = True
        bootstrap_frames = int(aerocommit_config["bootstrap_frames"])
        if bootstrap_frames > 1:
            aerocommit_config["bootstrap_frames"] = 1
            changed = True
    return {
        "persistent": persistent,
        "changed": changed,
        "diagnostic_override": False,
        "reason": reason,
    }
