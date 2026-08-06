"""Collision-free log-structured multiscale occupancy for UAV mapping."""

from copy import deepcopy

import numpy as np


DEFAULT_FRONT_VIEW_SPARSE_SCALE_MAP_CONFIG = {
    "enabled": False,
    "start_scale": 20.0,
    "levels": 8,
    "coordinate_bits": 21,
}


def validate_front_view_sparse_scale_map_config(config=None):
    merged = deepcopy(DEFAULT_FRONT_VIEW_SPARSE_SCALE_MAP_CONFIG)
    if config is not None:
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError(
                "Unknown FrontViewSparseScaleMap options: {}".format(
                    sorted(unknown)
                )
            )
        merged.update(config)
    if not isinstance(merged["enabled"], bool):
        raise TypeError("FrontViewSparseScaleMap.enabled must be boolean")
    if float(merged["start_scale"]) <= 0.0:
        raise ValueError("FrontViewSparseScaleMap.start_scale must be positive")
    if not isinstance(merged["levels"], int) or merged["levels"] < 1:
        raise ValueError("FrontViewSparseScaleMap.levels must be positive")
    bits = merged["coordinate_bits"]
    if not isinstance(bits, int) or bits < 2 or 3 * bits > 63:
        raise ValueError(
            "FrontViewSparseScaleMap.coordinate_bits must fit three axes in int64"
        )
    return merged


class FrontViewSparseScaleMap:
    """Exact multiscale occupancy stored as binary log-structured sorted runs."""

    def __init__(self, config=None):
        self.config = validate_front_view_sparse_scale_map_config(config)
        self.scales = np.asarray(
            [
                float(self.config["start_scale"]) * (2**level)
                for level in range(int(self.config["levels"]))
            ],
            dtype=np.float64,
        )
        self._runs = [[] for _ in range(len(self.scales))]
        self.stats = {
            "query_calls": 0,
            "query_rows": 0,
            "occupied_rows": 0,
            "set_calls": 0,
            "set_rows": 0,
            "inserted_unique_keys": 0,
            "run_merges": 0,
            "hash_query_rows": 0,
            "hash_set_rows": 0,
        }

    @property
    def enabled(self):
        return bool(self.config["enabled"])

    def _points(self, coords):
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        if len(coords) and not np.isfinite(coords).all():
            raise ValueError("Sparse scale-map coordinates must be finite")
        return coords

    def _active_levels(self, target_size=None):
        if target_size is None:
            return np.arange(len(self.scales), dtype=np.int64)
        target_size = float(target_size)
        if not np.isfinite(target_size) or target_size <= 0.0:
            raise ValueError("Sparse scale-map target size must be positive")
        maximum_scale = 1.0 / target_size
        levels = np.flatnonzero(self.scales <= maximum_scale)
        if len(levels) == 0:
            return np.asarray([0], dtype=np.int64)
        return levels

    def _encode(self, coords, scale):
        quantized = np.trunc(coords * float(scale)).astype(np.int64)
        bits = int(self.config["coordinate_bits"])
        bias = 1 << (bits - 1)
        if np.any(quantized < -bias) or np.any(quantized >= bias):
            maximum_world = (bias - 1) / float(scale)
            raise OverflowError(
                "Sparse scale-map coordinate exceeds +/-{:.3f} m at scale {}".format(
                    maximum_world, float(scale)
                )
            )
        encoded = quantized + bias
        return (
            (encoded[:, 0] << (2 * bits))
            | (encoded[:, 1] << bits)
            | encoded[:, 2]
        ).astype(np.int64, copy=False)

    @staticmethod
    def _run_contains(run, keys):
        if len(run) == 0 or len(keys) == 0:
            return np.zeros((len(keys),), dtype=np.bool_)
        positions = np.searchsorted(run, keys)
        valid = positions < len(run)
        safe = np.minimum(positions, len(run) - 1)
        return valid & (run[safe] == keys)

    def occupied(self, coords, target_size=None):
        coords = self._points(coords)
        result = np.ones((len(coords),), dtype=np.bool_)
        for level in self._active_levels(target_size).tolist():
            keys = self._encode(coords, self.scales[level])
            level_occupied = np.zeros((len(coords),), dtype=np.bool_)
            for run in self._runs[level]:
                if run is not None:
                    level_occupied |= self._run_contains(run, keys)
            result &= level_occupied
            if not np.any(result):
                break
        self.stats["query_calls"] += 1
        self.stats["query_rows"] += len(coords)
        self.stats["occupied_rows"] += int(np.sum(result))
        return result

    def _insert_run(self, level, keys):
        carry = np.unique(np.asarray(keys, dtype=np.int64).reshape(-1))
        if len(carry) == 0:
            return
        slot = 0
        while True:
            if slot == len(self._runs[level]):
                self._runs[level].append(carry)
                break
            existing = self._runs[level][slot]
            if existing is None:
                self._runs[level][slot] = carry
                break
            carry = np.union1d(existing, carry)
            self._runs[level][slot] = None
            self.stats["run_merges"] += 1
            slot += 1

    def register(self, coords, target_size=None):
        coords = self._points(coords)
        if len(coords) == 0:
            return
        inserted = 0
        for level in self._active_levels(target_size).tolist():
            keys = np.unique(self._encode(coords, self.scales[level]))
            inserted += len(keys)
            self._insert_run(level, keys)
        self.stats["set_calls"] += 1
        self.stats["set_rows"] += len(coords)
        self.stats["inserted_unique_keys"] += inserted

    def summary(self):
        result = dict(self.stats)
        run_counts = []
        key_counts = []
        for level_runs in self._runs:
            active = [run for run in level_runs if run is not None]
            run_counts.append(len(active))
            key_counts.append(sum(len(run) for run in active))
        result.update(
            {
                "enabled": self.enabled,
                "active_run_counts": run_counts,
                "stored_key_counts": key_counts,
                "stored_keys": int(sum(key_counts)),
                "hash_calls_zero": (
                    result["hash_query_rows"] == 0
                    and result["hash_set_rows"] == 0
                ),
            }
        )
        return result
