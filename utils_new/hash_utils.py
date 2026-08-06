# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


class HashBlock:
    def __init__(self, config):
        self.config = config
        self.use_hash = config["use_hash"]

        if self.use_hash:
            self.use_color_hash = config["use_color_hash"]
            self.remove_conflict = config["remove_conflict"]
            self.hash_size = config["hash_size"]
            self.color_hash_size = config["color_hash_size"]
            start_scale = config["start_scale"]
            hash_level = config["hash_level"]

            if self.use_color_hash:
                self.hash_maps = np.zeros(
                    (hash_level, self.hash_size**3 * self.color_hash_size),
                    dtype=np.bool_,
                )
            else:
                self.hash_maps = np.zeros(
                    (hash_level, self.hash_size**3), dtype=np.bool_
                )

            self.conflict_hash = np.zeros((self.hash_size**3), dtype=np.int_)
            self.hash_scales = [start_scale * (2**l) for l in range(hash_level)]
            self.color_scale = 255.0 / self.hash_scales[0] / 64.0

    def hash(self, x, T):
        return np.bitwise_xor(x, np.bitwise_xor(x * 2654435761, x * 805459861)) % T

    def _active_levels(self, target_size=None):
        """Return represented hash levels, falling back to the coarsest level."""
        if target_size is None:
            return list(enumerate(self.hash_scales))
        maximum_scale = 1.0 / max(float(target_size), np.finfo(np.float32).tiny)
        levels = [
            (level, scale)
            for level, scale in enumerate(self.hash_scales)
            if scale <= maximum_scale
        ]
        # A footprint coarser than level zero is still representable at the
        # coarsest available resolution. Returning no levels makes an empty
        # hash look fully occupied and prevents any point from being inserted.
        return levels if levels else [(0, self.hash_scales[0])]

    def getOccupy(self, coords, colors, target_size=None):
        if self.use_hash:
            if self.use_color_hash:
                return self.getHashOccupy_colored(coords, colors, target_size)
            else:
                return self.getHashOccupy(coords, target_size)
        else:
            return np.zeros((coords.shape[0],), dtype=np.bool_)

    def setOccupy(self, coords, colors, target_size=None):
        if self.use_hash:
            if self.use_color_hash:
                return self.setHashOccupy_colored(coords, colors, target_size)
            else:
                return self.setHashOccupy(coords, target_size)

    def getHashOccupy(self, coords, target_size=None):
        res = np.ones((coords.shape[0],), dtype=np.bool_)
        for l, hash_scale in self._active_levels(target_size):
            x, y, z = (
                (coords[:, 0] * hash_scale).astype(np.int_),
                (coords[:, 1] * hash_scale).astype(np.int_),
                (coords[:, 2] * hash_scale).astype(np.int_),
            )
            x, y, z = (
                self.hash(x, self.hash_size),
                self.hash(y, self.hash_size),
                self.hash(z, self.hash_size),
            )
            encoded_idx = x * self.hash_size * self.hash_size + y * self.hash_size + z
            res = np.logical_and(res, self.hash_maps[l][encoded_idx])
        return res

    def getHashOccupy_colored(self, coords, colors, target_size=None):
        scaled_grey_color = np.mean(colors, axis=1, keepdims=True) * self.color_scale
        coords_color = np.concatenate([coords, scaled_grey_color], axis=1)
        res = np.ones((coords_color.shape[0],), dtype=np.bool_)
        for l, hash_scale in self._active_levels(target_size):
            x, y, z, c = (
                (coords_color[:, 0] * hash_scale).astype(np.int_),
                (coords_color[:, 1] * hash_scale).astype(np.int_),
                (coords_color[:, 2] * hash_scale).astype(np.int_),
                (coords_color[:, 3] * hash_scale).astype(np.int_),
            )
            x, y, z, c = (
                self.hash(x, self.hash_size),
                self.hash(y, self.hash_size),
                self.hash(z, self.hash_size),
                self.hash(c, self.color_hash_size),
            )
            encoded_idx = (
                x * self.hash_size * self.hash_size * self.color_hash_size
                + y * self.hash_size * self.color_hash_size
                + z * self.color_hash_size
                + c
            )
            res = np.logical_and(res, self.hash_maps[l][encoded_idx])
        return res

    def setHashOccupy(self, coords, target_size=None):
        for l, hash_scale in self._active_levels(target_size):
            x, y, z = (
                (coords[:, 0] * hash_scale).astype(np.int_),
                (coords[:, 1] * hash_scale).astype(np.int_),
                (coords[:, 2] * hash_scale).astype(np.int_),
            )
            x, y, z = (
                self.hash(x, self.hash_size),
                self.hash(y, self.hash_size),
                self.hash(z, self.hash_size),
            )
            encoded_idx = x * self.hash_size * self.hash_size + y * self.hash_size + z
            self.hash_maps[l][encoded_idx] = True

    def setHashOccupy_colored(self, coords, colors, target_size=None):
        scaled_grey_color = np.mean(colors, axis=1, keepdims=True) * self.color_scale
        coords_color = np.concatenate([coords, scaled_grey_color], axis=1)
        for l, hash_scale in self._active_levels(target_size):
            x, y, z, c = (
                (coords_color[:, 0] * hash_scale).astype(np.int_),
                (coords_color[:, 1] * hash_scale).astype(np.int_),
                (coords_color[:, 2] * hash_scale).astype(np.int_),
                (coords_color[:, 3] * hash_scale).astype(np.int_),
            )
            x, y, z, c = (
                self.hash(x, self.hash_size),
                self.hash(y, self.hash_size),
                self.hash(z, self.hash_size),
                self.hash(c, self.color_hash_size),
            )
            encoded_idx = (
                x * self.hash_size * self.hash_size * self.color_hash_size
                + y * self.hash_size * self.color_hash_size
                + z * self.color_hash_size
                + c
            )
            self.hash_maps[l][encoded_idx] = True

    def get_no_conflict_index(self, coords, target_size=None):
        if (
            self.use_hash and self.remove_conflict and target_size is None
        ):  # if target_size is valid, there is no conflict
            unique_idx = np.arange(coords.shape[0])
            x, y, z = (
                (coords[:, 0] * self.hash_scales[-1]).astype(np.int_),
                (coords[:, 1] * self.hash_scales[-1]).astype(np.int_),
                (coords[:, 2] * self.hash_scales[-1]).astype(np.int_),
            )
            x, y, z = (
                self.hash(x, self.hash_size),
                self.hash(y, self.hash_size),
                self.hash(z, self.hash_size),
            )
            encoded_idx = x * self.hash_size * self.hash_size + y * self.hash_size + z
            self.conflict_hash[encoded_idx] = unique_idx

            return unique_idx == self.conflict_hash[encoded_idx]
        else:
            return np.ones((coords.shape[0],), dtype=np.bool_)

    def get_sequential_no_conflict_index(self, coords, colors, target_size=None):
        """Match one-row-at-a-time occupancy while deferring the actual hash write."""
        count = int(coords.shape[0])
        if not self.use_hash or count == 0:
            return np.ones((count,), dtype=np.bool_)

        level_keys = []
        scaled_grey = (
            np.mean(colors, axis=1) * self.color_scale
            if self.use_color_hash
            else None
        )
        for level, hash_scale in self._active_levels(target_size):
            x = self.hash((coords[:, 0] * hash_scale).astype(np.int_), self.hash_size)
            y = self.hash((coords[:, 1] * hash_scale).astype(np.int_), self.hash_size)
            z = self.hash((coords[:, 2] * hash_scale).astype(np.int_), self.hash_size)
            if self.use_color_hash:
                color = self.hash(
                    (scaled_grey * hash_scale).astype(np.int_), self.color_hash_size
                )
                keys = (
                    x * self.hash_size * self.hash_size * self.color_hash_size
                    + y * self.hash_size * self.color_hash_size
                    + z * self.color_hash_size
                    + color
                )
            else:
                keys = x * self.hash_size * self.hash_size + y * self.hash_size + z
            level_keys.append((level, keys))

        accepted = np.zeros((count,), dtype=np.bool_)
        pending = [set() for _ in level_keys]
        for row in range(count):
            occupied = all(
                bool(self.hash_maps[level][keys[row]]) or int(keys[row]) in pending[index]
                for index, (level, keys) in enumerate(level_keys)
            )
            if occupied:
                continue
            accepted[row] = True
            for index, (_, keys) in enumerate(level_keys):
                pending[index].add(int(keys[row]))
        return accepted

    def save_hash_table(self, path):
        saved_hash = {}
        saved_hash["color_hash"] = self.use_color_hash
        saved_hash["hash_size"] = self.hash_size
        saved_hash["color_hash_size"] = self.color_hash_size
        saved_hash["hash_maps"] = self.hash_maps
        saved_hash["hash_scales"] = self.hash_scales
        np.savez(path, **saved_hash)
