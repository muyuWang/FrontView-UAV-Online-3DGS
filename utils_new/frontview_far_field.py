"""Range-selective projective responsibility for forward-view UAV mapping."""

from copy import deepcopy
import math

import numpy as np


DEFAULT_FRONT_VIEW_FAR_FIELD_CONFIG = {
    "enabled": False,
    "depth_m": 50.0,
    "projective_cell_px": 12,
    "depth_bin_ratio": 1.10,
    "shuffle_responsibility": False,
    "shuffle_seed": 42,
}


def validate_front_view_far_field_config(config=None):
    merged = deepcopy(DEFAULT_FRONT_VIEW_FAR_FIELD_CONFIG)
    if config is not None:
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError(
                "Unknown FrontViewFarField options: {}".format(sorted(unknown))
            )
        merged.update(config)
    if not isinstance(merged["enabled"], bool):
        raise TypeError("FrontViewFarField.enabled must be boolean")
    if not isinstance(merged["shuffle_responsibility"], bool):
        raise TypeError("FrontViewFarField.shuffle_responsibility must be boolean")
    if not isinstance(merged["shuffle_seed"], int):
        raise TypeError("FrontViewFarField.shuffle_seed must be an integer")
    if float(merged["depth_m"]) <= 0.0:
        raise ValueError("FrontViewFarField.depth_m must be positive")
    if int(merged["projective_cell_px"]) <= 0:
        raise ValueError("FrontViewFarField.projective_cell_px must be positive")
    if float(merged["depth_bin_ratio"]) <= 1.0:
        raise ValueError("FrontViewFarField.depth_bin_ratio must be greater than one")
    return merged


def projective_survivor_mask(uv, depths, scores, config):
    """Keep the strongest row in each image/log-depth responsibility cell."""

    uv = np.asarray(uv, dtype=np.float32)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if uv.shape != (len(depths), 2) or scores.shape != depths.shape:
        raise ValueError("Projective far-field arrays must align")
    keep = np.zeros((len(depths),), dtype=np.bool_)
    if len(depths) == 0:
        return keep
    xy = np.floor(uv / float(config["projective_cell_px"])).astype(np.int64)
    depth_bin = np.floor(
        np.log(np.maximum(depths, 1.0e-8))
        / math.log(float(config["depth_bin_ratio"]))
    ).astype(np.int64)
    order = np.argsort(-scores, kind="stable")
    occupied = set()
    for index in order.tolist():
        key = (int(xy[index, 0]), int(xy[index, 1]), int(depth_bin[index]))
        if key in occupied:
            continue
        occupied.add(key)
        keep[index] = True
    return keep
