"""Diagnostic-only switches for causal decomposition of late map corruption."""

from __future__ import annotations


DEFAULT_CAUSAL_DEPTH_AUDIT_CONFIG = {
    "enabled": False,
    "start_frame": -1,
    "stop_birth": False,
    "stop_opacity_pruning": False,
    "freeze_existing_geometry": False,
    "isolate_future_births": False,
    "audit_opacity_pruning": False,
}


def validate_causal_depth_audit_config(config=None):
    result = dict(DEFAULT_CAUSAL_DEPTH_AUDIT_CONFIG)
    if config is not None:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(
                "Unknown CausalDepthAudit options: {}".format(sorted(unknown))
            )
        result.update(config)

    result["enabled"] = bool(result["enabled"])
    result["start_frame"] = int(result["start_frame"])
    for key in (
        "stop_birth",
        "stop_opacity_pruning",
        "freeze_existing_geometry",
        "isolate_future_births",
        "audit_opacity_pruning",
    ):
        result[key] = bool(result[key])

    if result["enabled"] and result["start_frame"] < 0:
        raise ValueError("CausalDepthAudit.start_frame must be nonnegative")
    if result["isolate_future_births"] and not result["freeze_existing_geometry"]:
        raise ValueError(
            "CausalDepthAudit.isolate_future_births requires "
            "freeze_existing_geometry"
        )
    return result
