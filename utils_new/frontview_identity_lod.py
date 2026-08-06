"""Persistent projective identity and bounded LOD births for front-view mapping."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Dict, Mapping, Tuple

import numpy as np
import torch


DEFAULT_FRONT_VIEW_IDENTITY_LOD_CONFIG = {
    "enabled": False,
    "mode": "identity_lod",
    "projective_cell_px": 12.0,
    "min_image_gate_px": 2.0,
    "max_image_gate_px": 12.0,
    "radius_gate_scale": 0.75,
    "depth_ratio": 1.08,
    "uncertainty_scale": 1.0,
    "use_world_gate": False,
    "world_gate_scale": 1.5,
    "world_score_weight": 0.5,
    "min_lod_radius_px": 5.0,
    "min_lod_residual": 0.10,
    "max_lod_level": 1,
    "slot_mode": "quadrant",
    "children_per_sector": 1,
    "lod_cell_px": 3.0,
    "lod_capacity_radius_px": 3.0,
    "max_children_per_node": 16,
    "shuffle_identity": False,
    "shuffle_seed": 42,
    "refill_depthcov_budget": True,
    "render_handoff_enabled": False,
    "handoff_radius_start_px": 4.0,
    "handoff_radius_end_px": 8.0,
    "handoff_full_children": 4,
    "handoff_parent_floor": 0.25,
    "handoff_batch_reduce": "max",
    "shuffle_handoff": False,
}


def validate_front_view_identity_lod_config(config=None):
    merged = deepcopy(DEFAULT_FRONT_VIEW_IDENTITY_LOD_CONFIG)
    if config is not None:
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError(
                "Unknown FrontViewIdentityLOD options: {}".format(sorted(unknown))
            )
        merged.update(config)
    for key in (
        "enabled",
        "shuffle_identity",
        "refill_depthcov_budget",
        "use_world_gate",
        "render_handoff_enabled",
        "shuffle_handoff",
    ):
        if not isinstance(merged[key], bool):
            raise TypeError("FrontViewIdentityLOD.{} must be boolean".format(key))
    if merged["mode"] not in (
        "identity_only",
        "identity_lod",
        "frustum_lattice",
    ):
        raise ValueError(
            "FrontViewIdentityLOD.mode must be identity_only, identity_lod, or frustum_lattice"
        )
    if merged["slot_mode"] not in ("quadrant", "footprint"):
        raise ValueError("FrontViewIdentityLOD.slot_mode must be quadrant or footprint")
    for key in (
        "projective_cell_px",
        "min_image_gate_px",
        "max_image_gate_px",
        "radius_gate_scale",
        "depth_ratio",
        "uncertainty_scale",
        "world_gate_scale",
        "world_score_weight",
        "min_lod_radius_px",
        "lod_cell_px",
        "lod_capacity_radius_px",
        "handoff_radius_start_px",
        "handoff_radius_end_px",
    ):
        if float(merged[key]) <= 0.0:
            raise ValueError("FrontViewIdentityLOD.{} must be positive".format(key))
    if float(merged["depth_ratio"]) <= 1.0:
        raise ValueError("FrontViewIdentityLOD.depth_ratio must be greater than one")
    if float(merged["max_image_gate_px"]) < float(merged["min_image_gate_px"]):
        raise ValueError("FrontViewIdentityLOD image gates are reversed")
    if float(merged["min_lod_residual"]) < 0.0:
        raise ValueError("FrontViewIdentityLOD.min_lod_residual cannot be negative")
    if not isinstance(merged["max_lod_level"], int) or merged["max_lod_level"] < 0:
        raise ValueError("FrontViewIdentityLOD.max_lod_level must be nonnegative")
    if (
        not isinstance(merged["children_per_sector"], int)
        or merged["children_per_sector"] <= 0
        or merged["children_per_sector"] > 8
    ):
        raise ValueError(
            "FrontViewIdentityLOD.children_per_sector must be in [1, 8]"
        )
    if (
        not isinstance(merged["max_children_per_node"], int)
        or merged["max_children_per_node"] <= 0
    ):
        raise ValueError(
            "FrontViewIdentityLOD.max_children_per_node must be positive"
        )
    if not isinstance(merged["shuffle_seed"], int):
        raise TypeError("FrontViewIdentityLOD.shuffle_seed must be an integer")
    if float(merged["handoff_radius_end_px"]) <= float(
        merged["handoff_radius_start_px"]
    ):
        raise ValueError("FrontViewIdentityLOD handoff radii must be increasing")
    if (
        not isinstance(merged["handoff_full_children"], int)
        or merged["handoff_full_children"] <= 0
    ):
        raise ValueError(
            "FrontViewIdentityLOD.handoff_full_children must be positive"
        )
    if not 0.0 <= float(merged["handoff_parent_floor"]) <= 1.0:
        raise ValueError(
            "FrontViewIdentityLOD.handoff_parent_floor must be in [0, 1]"
        )
    if merged["handoff_batch_reduce"] not in ("max", "mean"):
        raise ValueError(
            "FrontViewIdentityLOD.handoff_batch_reduce must be max or mean"
        )
    return merged


@dataclass
class ResponsibilityNode:
    uid: int
    root_uid: int
    parent_uid: int = -1
    level: int = 0
    sector: int = -1
    track_id: int = -1
    support_count: int = 1
    last_support_frame: int = -1
    active: bool = True


class FrontViewIdentityLOD:
    """Associate births to visible persistent GS IDs without a 3D spatial index."""

    def __init__(self, config=None):
        self.config = validate_front_view_identity_lod_config(config)
        self.next_uid = 0
        self.nodes: Dict[int, ResponsibilityNode] = {}
        self.children: Dict[Tuple[int, int], int] = {}
        self.child_counts: Dict[int, int] = {}
        self.track_to_uid: Dict[int, int] = {}
        self._child_count_cache_cpu = None
        self._child_count_cache_by_device = {}
        self.stats = {
            "calls": 0,
            "candidate_rows": 0,
            "visible_rows": 0,
            "sparse_track_rejected": 0,
            "depthcov_budget": 0,
            "depthcov_committed": 0,
            "new_roots": 0,
            "repeat_rejected": 0,
            "same_frame_rejected": 0,
            "lod_children": 0,
            "lod_slot_rejected": 0,
            "lattice_births": 0,
            "lattice_capacity_rejected": 0,
            "pruned_uids": 0,
            "hash_query_rows": 0,
            "hash_set_rows": 0,
        }

    @property
    def enabled(self):
        return bool(self.config["enabled"])

    def allocate_uids(self, count):
        count = int(count)
        start = self.next_uid
        self.next_uid += count
        return np.arange(start, start + count, dtype=np.int64)

    def register_existing_roots(self, uids):
        for uid in np.asarray(uids, dtype=np.int64).reshape(-1).tolist():
            if uid < 0 or uid in self.nodes:
                continue
            self.nodes[uid] = ResponsibilityNode(uid=uid, root_uid=uid)
            self.next_uid = max(self.next_uid, uid + 1)

    @staticmethod
    def _encode_cells(x_cell, y_cell, depth_bin):
        shift = 1 << 19
        valid = (
            (x_cell >= -shift)
            & (x_cell < shift)
            & (y_cell >= -shift)
            & (y_cell < shift)
            & (depth_bin >= -shift)
            & (depth_bin < shift)
        )
        x_code = x_cell + shift
        y_code = y_cell + shift
        depth_code = depth_bin + shift
        keys = (x_code << 40) | (y_code << 20) | depth_code
        return torch.where(valid, keys, torch.full_like(keys, -1))

    def _projected_matches(
        self,
        uv,
        depths,
        depth_confidences,
        projection_info,
        global_uids,
        global_means,
        global_scales,
        world_points,
        log_scales,
        frame_id,
    ):
        count = len(depths)
        empty_uid = np.full((count,), -1, dtype=np.int64)
        empty_float = np.zeros((count,), dtype=np.float32)
        if projection_info is None:
            return (
                empty_uid,
                empty_float,
                empty_float.copy(),
                empty_float.copy(),
                np.zeros((count,), dtype=np.int32),
                0,
            )
        required = ("gaussian_ids", "means2d", "depths", "radii")
        if any(projection_info.get(name) is None for name in required):
            raise ValueError("FrontViewIdentityLOD requires packed gsplat projections")

        gaussian_ids = projection_info["gaussian_ids"].detach().reshape(-1).long()
        means2d = projection_info["means2d"].detach().reshape(-1, 2)
        projected_depths = projection_info["depths"].detach().reshape(-1)
        radii = projection_info["radii"].detach()
        if radii.ndim > 1:
            radii = radii.amax(dim=-1)
        radii = radii.reshape(-1).to(means2d.dtype)
        device = gaussian_ids.device
        if not torch.is_tensor(global_uids):
            global_uids = torch.as_tensor(global_uids, device=device)
        global_uids = global_uids.reshape(-1).long().to(device)
        valid = (
            (gaussian_ids >= 0)
            & (gaussian_ids < global_uids.numel())
            & torch.isfinite(means2d).all(dim=1)
            & torch.isfinite(projected_depths)
            & (projected_depths > 0.0)
            & torch.isfinite(radii)
            & (radii > 0.0)
        )
        gaussian_ids = gaussian_ids[valid]
        means2d = means2d[valid]
        projected_depths = projected_depths[valid]
        radii = radii[valid]
        visible_uids = global_uids[gaussian_ids]
        valid_uid = visible_uids >= 0
        means2d = means2d[valid_uid]
        projected_depths = projected_depths[valid_uid]
        radii = radii[valid_uid]
        visible_uids = visible_uids[valid_uid]
        visible_gaussian_ids = gaussian_ids[valid_uid]
        visible_count = int(visible_uids.numel())
        if visible_count == 0 or count == 0:
            return (
                empty_uid,
                empty_float,
                empty_float.copy(),
                empty_float.copy(),
                np.zeros((count,), dtype=np.int32),
                visible_count,
            )

        if self.config["shuffle_identity"] and visible_count > 1:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self.config["shuffle_seed"]) + int(frame_id))
            visible_uids = visible_uids[
                torch.randperm(visible_count, generator=generator, device=device)
            ]

        cell = float(self.config["projective_cell_px"])
        log_ratio = math.log(float(self.config["depth_ratio"]))
        p_cell = torch.floor(means2d / cell).long()
        p_depth_bin = torch.floor(torch.log(projected_depths) / log_ratio).long()
        p_keys = self._encode_cells(p_cell[:, 0], p_cell[:, 1], p_depth_bin)
        valid_key = p_keys >= 0
        p_keys = p_keys[valid_key]
        means2d = means2d[valid_key]
        projected_depths = projected_depths[valid_key]
        radii = radii[valid_key]
        visible_uids = visible_uids[valid_key]
        visible_gaussian_ids = visible_gaussian_ids[valid_key]
        if p_keys.numel() == 0:
            return (
                empty_uid,
                empty_float,
                empty_float.copy(),
                empty_float.copy(),
                np.zeros((count,), dtype=np.int32),
                visible_count,
            )

        radius_order = torch.argsort(radii, descending=True, stable=True)
        key_order = torch.argsort(p_keys[radius_order], stable=True)
        representative_order = radius_order[key_order]
        sorted_keys = p_keys[representative_order]

        candidate_uv = torch.as_tensor(uv, device=device, dtype=means2d.dtype)
        candidate_depths = torch.as_tensor(
            depths, device=device, dtype=projected_depths.dtype
        )
        confidence = torch.as_tensor(
            depth_confidences, device=device, dtype=projected_depths.dtype
        ).clamp(0.0, 1.0)
        c_cell = torch.floor(candidate_uv / cell).long()
        c_depth_bin = torch.floor(
            torch.log(torch.clamp(candidate_depths, min=1.0e-8)) / log_ratio
        ).long()
        offsets = torch.tensor(
            [
                (dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-2, -1, 0, 1, 2)
            ],
            device=device,
            dtype=torch.long,
        )
        neighbor_keys = self._encode_cells(
            c_cell[:, None, 0] + offsets[None, :, 0],
            c_cell[:, None, 1] + offsets[None, :, 1],
            c_depth_bin[:, None] + offsets[None, :, 2],
        )
        positions = torch.searchsorted(sorted_keys, neighbor_keys)
        right_positions = torch.searchsorted(sorted_keys, neighbor_keys, right=True)
        in_range = positions < sorted_keys.numel()
        safe_positions = positions.clamp(max=sorted_keys.numel() - 1)
        key_match = in_range & (sorted_keys[safe_positions] == neighbor_keys)
        projected_rows = representative_order[safe_positions]
        neighbor_uv = means2d[projected_rows]
        neighbor_depth = projected_depths[projected_rows]
        neighbor_radius = radii[projected_rows]
        image_distance = torch.linalg.vector_norm(
            candidate_uv[:, None, :] - neighbor_uv, dim=-1
        )
        image_gate = torch.clamp(
            neighbor_radius * float(self.config["radius_gate_scale"]),
            min=float(self.config["min_image_gate_px"]),
            max=float(self.config["max_image_gate_px"]),
        )
        log_distance = torch.abs(
            torch.log(
                torch.clamp(candidate_depths[:, None], min=1.0e-8)
                / torch.clamp(neighbor_depth, min=1.0e-8)
            )
        )
        log_gate = log_ratio * (
            1.0
            + float(self.config["uncertainty_scale"])
            * (1.0 - confidence[:, None])
        )
        eligible = (
            key_match
            & torch.isfinite(candidate_uv).all(dim=1, keepdim=True)
            & torch.isfinite(candidate_depths[:, None])
            & (candidate_depths[:, None] > 0.0)
            & (image_distance <= image_gate)
            & (log_distance <= log_gate)
        )
        score = image_distance / image_gate + log_distance / torch.clamp(
            log_gate, min=1.0e-8
        )
        if self.config["use_world_gate"]:
            if global_means is None or global_scales is None:
                raise ValueError("World-gated frustum association requires Gaussian geometry")
            global_means = torch.as_tensor(
                global_means, device=device, dtype=means2d.dtype
            ).reshape(-1, 3)
            global_scales = torch.as_tensor(
                global_scales, device=device, dtype=means2d.dtype
            ).reshape(-1, 3)
            candidate_world = torch.as_tensor(
                world_points, device=device, dtype=means2d.dtype
            ).reshape(-1, 3)
            candidate_scale = torch.exp(
                torch.as_tensor(log_scales, device=device, dtype=means2d.dtype)
                .reshape(count, -1)
                .mean(dim=1)
            )
            neighbor_gaussian_ids = visible_gaussian_ids[projected_rows]
            neighbor_world = global_means[neighbor_gaussian_ids]
            neighbor_scale = global_scales[neighbor_gaussian_ids].amax(dim=-1)
            world_distance = torch.linalg.vector_norm(
                candidate_world[:, None, :] - neighbor_world, dim=-1
            )
            world_gate = float(self.config["world_gate_scale"]) * torch.maximum(
                neighbor_scale, candidate_scale[:, None]
            )
            world_eligible = (
                torch.isfinite(candidate_world).all(dim=1, keepdim=True)
                & torch.isfinite(neighbor_world).all(dim=-1)
                & torch.isfinite(world_gate)
                & (world_gate > 0.0)
                & (world_distance <= world_gate)
            )
            eligible &= world_eligible
            score = score + float(self.config["world_score_weight"]) * (
                world_distance / torch.clamp(world_gate, min=1.0e-8)
            )
        score = torch.where(eligible, score, torch.full_like(score, torch.inf))
        best_score, best_neighbor = torch.min(score, dim=1)
        matched = torch.isfinite(best_score)
        chosen_rows = projected_rows[
            torch.arange(count, device=device), best_neighbor
        ]
        matched_uids = torch.full((count,), -1, device=device, dtype=torch.long)
        matched_uids[matched] = visible_uids[chosen_rows[matched]]
        matched_x = torch.zeros((count,), device=device, dtype=means2d.dtype)
        matched_y = torch.zeros_like(matched_x)
        matched_radius = torch.zeros_like(matched_x)
        matched_count = torch.zeros((count,), device=device, dtype=torch.int32)
        matched_x[matched] = means2d[chosen_rows[matched], 0]
        matched_y[matched] = means2d[chosen_rows[matched], 1]
        matched_radius[matched] = radii[chosen_rows[matched]]
        neighbor_counts = (right_positions - positions).to(torch.int32)
        matched_count[matched] = neighbor_counts[
            torch.arange(count, device=device)[matched], best_neighbor[matched]
        ]
        return (
            matched_uids.cpu().numpy(),
            matched_x.cpu().numpy(),
            matched_y.cpu().numpy(),
            matched_radius.cpu().numpy(),
            matched_count.cpu().numpy(),
            visible_count,
        )

    def _observe(self, uid, frame_id):
        node = self.nodes.get(int(uid))
        if node is None or not node.active or node.last_support_frame == int(frame_id):
            return
        node.support_count += 1
        node.last_support_frame = int(frame_id)

    def filter_candidates(
        self,
        *,
        frame_id,
        uv,
        depths,
        residual_scores,
        depth_confidences,
        sparse_valid,
        track_ids,
        projection_info,
        global_uids,
        depthcov_budget,
        global_means=None,
        global_scales=None,
        world_points=None,
        log_scales=None,
    ):
        """Return accepted row indices and graph assignments in accepted-row order."""

        uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        residual_scores = np.asarray(residual_scores, dtype=np.float32).reshape(-1)
        depth_confidences = np.asarray(depth_confidences, dtype=np.float32).reshape(-1)
        sparse_valid = np.asarray(sparse_valid, dtype=np.bool_).reshape(-1)
        track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        count = len(depths)
        if any(len(value) != count for value in (uv, residual_scores, depth_confidences, sparse_valid, track_ids)):
            raise ValueError("FrontViewIdentityLOD candidate arrays must align")

        cell = float(self.config["projective_cell_px"])
        (
            matched_uids,
            matched_x,
            matched_y,
            matched_radii,
            matched_counts,
            visible_count,
        ) = (
            self._projected_matches(
                uv,
                depths,
                depth_confidences,
                projection_info,
                global_uids,
                global_means,
                global_scales,
                world_points,
                log_scales,
                frame_id,
            )
        )

        accepted = []
        parents = []
        levels = []
        sectors = []
        reserved_slots = set()
        pending_tracks = set()
        pending_cells = set()
        pending_cell_counts = {}
        depthcov_count = 0
        log_ratio = math.log(float(self.config["depth_ratio"]))

        for index in range(count):
            if sparse_valid[index]:
                track_id = int(track_ids[index])
                if track_id >= 0 and (
                    track_id in self.track_to_uid or track_id in pending_tracks
                ):
                    self.stats["sparse_track_rejected"] += 1
                    continue
                accepted.append(index)
                parents.append(-1)
                levels.append(0)
                sectors.append(-1)
                if track_id >= 0:
                    pending_tracks.add(track_id)
                continue

            if depthcov_count >= int(depthcov_budget):
                continue
            x, y, z = float(uv[index, 0]), float(uv[index, 1]), float(depths[index])
            if not np.isfinite((x, y, z)).all() or z <= 0.0:
                continue
            candidate_cell = (
                int(math.floor(x / cell)),
                int(math.floor(y / cell)),
                int(math.floor(math.log(z) / log_ratio)),
            )
            parent_uid = -1
            child_level = 0
            child_sector = -1
            matched_uid = int(matched_uids[index])
            if matched_uid >= 0:
                nx = float(matched_x[index])
                ny = float(matched_y[index])
                radius = float(matched_radii[index])
                if self.config["mode"] == "frustum_lattice":
                    capacity = min(
                        int(self.config["max_children_per_node"]),
                        max(
                            1,
                            int(
                                math.ceil(
                                    (
                                        radius
                                        / float(
                                            self.config["lod_capacity_radius_px"]
                                        )
                                    )
                                    ** 2
                                )
                            ),
                        ),
                    )
                    occupied = int(matched_counts[index]) + pending_cell_counts.get(
                        candidate_cell, 0
                    )
                    if (
                        radius < float(self.config["min_lod_radius_px"])
                        or float(residual_scores[index])
                        < float(self.config["min_lod_residual"])
                        or occupied >= capacity
                    ):
                        self.stats["lattice_capacity_rejected"] += 1
                        continue
                    self.stats["lattice_births"] += 1
                else:
                    self._observe(matched_uid, frame_id)
                    parent = self.nodes.get(int(matched_uid))
                    can_split = (
                        self.config["mode"] == "identity_lod"
                        and parent is not None
                        and parent.active
                        and parent.level < int(self.config["max_lod_level"])
                        and radius >= float(self.config["min_lod_radius_px"])
                        and float(residual_scores[index])
                        >= float(self.config["min_lod_residual"])
                    )
                    if not can_split:
                        self.stats["repeat_rejected"] += 1
                        continue
                    child_sector = -1
                    slot = None
                    if self.config["slot_mode"] == "footprint":
                        capacity = min(
                            int(self.config["max_children_per_node"]),
                            max(
                                1,
                                int(
                                    math.ceil(
                                        (
                                            radius
                                            / float(
                                                self.config[
                                                    "lod_capacity_radius_px"
                                                ]
                                            )
                                        )
                                        ** 2
                                    )
                                ),
                            ),
                        )
                        reserved_count = sum(
                            reserved_parent == int(matched_uid)
                            for reserved_parent, _ in reserved_slots
                        )
                        if (
                            self.child_counts.get(int(matched_uid), 0)
                            + reserved_count
                            < capacity
                        ):
                            qx = int(
                                math.floor(
                                    (x - nx) / float(self.config["lod_cell_px"])
                                )
                            )
                            qy = int(
                                math.floor(
                                    (y - ny) / float(self.config["lod_cell_px"])
                                )
                            )
                            qx = int(np.clip(qx, -31, 31))
                            qy = int(np.clip(qy, -31, 31))
                            child_sector = (qx + 31) * 63 + (qy + 31)
                            candidate_slot = (
                                int(matched_uid),
                                int(child_sector),
                            )
                            existing_child = self.children.get(candidate_slot)
                            if (
                                existing_child is None
                                or self.nodes.get(existing_child, None) is None
                                or not self.nodes[existing_child].active
                            ) and candidate_slot not in reserved_slots:
                                slot = candidate_slot
                    else:
                        quadrant = (1 if x >= nx else 0) + (
                            2 if y >= ny else 0
                        )
                        for subslot in range(
                            int(self.config["children_per_sector"])
                        ):
                            candidate_sector = (
                                quadrant
                                * int(self.config["children_per_sector"])
                                + subslot
                            )
                            candidate_slot = (
                                int(matched_uid),
                                int(candidate_sector),
                            )
                            existing_child = self.children.get(candidate_slot)
                            if (
                                existing_child is None
                                or self.nodes.get(existing_child, None) is None
                                or not self.nodes[existing_child].active
                            ) and candidate_slot not in reserved_slots:
                                child_sector = candidate_sector
                                slot = candidate_slot
                                break
                    if slot is None:
                        self.stats["lod_slot_rejected"] += 1
                        continue
                    reserved_slots.add(slot)
                    parent_uid = int(matched_uid)
                    child_level = int(parent.level) + 1
            elif candidate_cell in pending_cells:
                self.stats["same_frame_rejected"] += 1
                continue

            accepted.append(index)
            parents.append(parent_uid)
            levels.append(child_level)
            sectors.append(child_sector)
            depthcov_count += 1
            pending_cells.add(candidate_cell)
            pending_cell_counts[candidate_cell] = (
                pending_cell_counts.get(candidate_cell, 0) + 1
            )

        self.stats["calls"] += 1
        self.stats["candidate_rows"] += count
        self.stats["visible_rows"] += visible_count
        self.stats["depthcov_budget"] += int(depthcov_budget)
        self.stats["depthcov_committed"] += depthcov_count
        return (
            np.asarray(accepted, dtype=np.int64),
            np.asarray(parents, dtype=np.int64),
            np.asarray(levels, dtype=np.int16),
            np.asarray(sectors, dtype=np.int16),
        )

    def prepare_commit(self, proposals):
        count = len(proposals)
        parents = np.asarray(proposals.responsibility_parent_uids, dtype=np.int64)
        levels = np.asarray(proposals.responsibility_levels, dtype=np.int16)
        sectors = np.asarray(proposals.responsibility_sectors, dtype=np.int16)
        keep = []
        reserved_slots = set()
        pending_tracks = set()
        for index in range(count):
            track_id = int(proposals.track_ids[index])
            if track_id >= 0 and (
                track_id in self.track_to_uid or track_id in pending_tracks
            ):
                continue
            parent_uid = int(parents[index])
            if parent_uid >= 0:
                parent = self.nodes.get(parent_uid)
                slot = (parent_uid, int(sectors[index]))
                if parent is None or not parent.active or slot in self.children or slot in reserved_slots:
                    continue
                reserved_slots.add(slot)
            keep.append(index)
            if track_id >= 0:
                pending_tracks.add(track_id)
        keep = np.asarray(keep, dtype=np.int64)
        return keep, self.allocate_uids(len(keep))

    def mark_committed(self, uids, proposals):
        uids = np.asarray(uids, dtype=np.int64).reshape(-1)
        if len(uids) != len(proposals):
            raise ValueError("Committed responsibility UIDs must align with proposals")
        for index, uid in enumerate(uids.tolist()):
            parent_uid = int(proposals.responsibility_parent_uids[index])
            level = int(proposals.responsibility_levels[index])
            sector = int(proposals.responsibility_sectors[index])
            track_id = int(proposals.track_ids[index])
            if parent_uid >= 0:
                parent = self.nodes[parent_uid]
                root_uid = parent.root_uid
                self.children[(parent_uid, sector)] = uid
                self.child_counts[parent_uid] = self.child_counts.get(parent_uid, 0) + 1
                self.stats["lod_children"] += 1
            else:
                root_uid = uid
                self.stats["new_roots"] += 1
            self.nodes[uid] = ResponsibilityNode(
                uid=uid,
                root_uid=root_uid,
                parent_uid=parent_uid,
                level=level,
                sector=sector,
                track_id=track_id,
                last_support_frame=int(proposals.source_frame_id),
            )
            if track_id >= 0:
                self.track_to_uid[track_id] = uid
        self._child_count_cache_cpu = None
        self._child_count_cache_by_device.clear()

    def release(self, uids):
        released = 0
        for uid in np.asarray(uids, dtype=np.int64).reshape(-1).tolist():
            node = self.nodes.get(uid)
            if node is None or not node.active:
                continue
            node.active = False
            released += 1
            if node.track_id >= 0 and self.track_to_uid.get(node.track_id) == uid:
                del self.track_to_uid[node.track_id]
            if node.parent_uid >= 0:
                removed = self.children.pop((node.parent_uid, node.sector), None)
                if removed is not None:
                    self.child_counts[node.parent_uid] = max(
                        0, self.child_counts.get(node.parent_uid, 1) - 1
                    )
            for slot, child_uid in list(self.children.items()):
                if slot[0] == uid or child_uid == uid:
                    self.children.pop(slot, None)
                    self.child_counts[slot[0]] = max(
                        0, self.child_counts.get(slot[0], 1) - 1
                    )
        self.stats["pruned_uids"] += released
        if released:
            self._child_count_cache_cpu = None
            self._child_count_cache_by_device.clear()
        return released

    def render_handoff_multipliers(
        self,
        uids,
        means,
        scales,
        poses,
        focal_pixels,
        *,
        frame_id=0,
    ):
        """Return detached per-Gaussian coarse-to-fine opacity responsibility."""

        count = int(means.shape[0])
        if not self.config["render_handoff_enabled"] or count == 0:
            return means.new_ones((count,))
        uids = torch.as_tensor(uids, device=means.device, dtype=torch.long).reshape(-1)
        if uids.shape != (count,):
            raise ValueError("Handoff UIDs must align with Gaussian geometry")
        poses = torch.as_tensor(poses, device=means.device, dtype=means.dtype)
        if poses.ndim == 2:
            poses = poses.unsqueeze(0)
        focal_pixels = torch.as_tensor(
            focal_pixels, device=means.device, dtype=means.dtype
        ).reshape(-1)
        if poses.shape[0] != focal_pixels.numel():
            raise ValueError("Handoff poses and focal lengths must align")

        if self._child_count_cache_cpu is None:
            child_table = np.zeros((self.next_uid,), dtype=np.float32)
            for uid, value in self.child_counts.items():
                node = self.nodes.get(int(uid))
                if (
                    0 <= int(uid) < self.next_uid
                    and node is not None
                    and node.active
                    and value > 0
                ):
                    child_table[int(uid)] = float(value)
            self._child_count_cache_cpu = child_table
            self._child_count_cache_by_device.clear()
        device_key = str(means.device)
        child_table = self._child_count_cache_by_device.get(device_key)
        if child_table is None:
            child_table = torch.from_numpy(self._child_count_cache_cpu).to(
                device=means.device, dtype=means.dtype
            )
            self._child_count_cache_by_device[device_key] = child_table
        child_counts = torch.zeros((count,), device=means.device, dtype=means.dtype)
        valid_uid = (uids >= 0) & (uids < child_table.numel())
        child_counts[valid_uid] = child_table[uids[valid_uid]]
        parent_rows = torch.nonzero(child_counts > 0.0, as_tuple=False).reshape(-1)
        if self.config["shuffle_handoff"] and parent_rows.numel() > 1:
            generator = torch.Generator(device=means.device)
            generator.manual_seed(int(self.config["shuffle_seed"]) + int(frame_id))
            shuffled = child_counts[parent_rows][
                torch.randperm(
                    parent_rows.numel(), generator=generator, device=means.device
                )
            ]
            child_counts = child_counts.clone()
            child_counts[parent_rows] = shuffled

        camera_points = torch.einsum(
            "nd,cdk->cnk", means, poses[:, :3, :3].transpose(1, 2)
        ) + poses[:, None, :3, 3]
        depths = camera_points[..., 2]
        projected_radius = (
            3.0
            * focal_pixels[:, None]
            * scales.amax(dim=1)[None, :]
            / torch.clamp(depths, min=1.0e-6)
        )
        visible = torch.isfinite(projected_radius) & (depths > 0.0)
        projected_radius = torch.where(
            visible, projected_radius, torch.zeros_like(projected_radius)
        )
        if self.config["handoff_batch_reduce"] == "mean":
            denominator = torch.clamp(visible.sum(dim=0), min=1)
            radius = projected_radius.sum(dim=0) / denominator
        else:
            radius = projected_radius.amax(dim=0)
        start = float(self.config["handoff_radius_start_px"])
        end = float(self.config["handoff_radius_end_px"])
        transition = torch.clamp((radius - start) / (end - start), 0.0, 1.0)
        transition = transition * transition * (3.0 - 2.0 * transition)
        maturity = torch.clamp(
            child_counts / float(self.config["handoff_full_children"]), 0.0, 1.0
        )
        floor = float(self.config["handoff_parent_floor"])
        multipliers = 1.0 - (1.0 - floor) * transition * maturity
        return multipliers.detach()

    def summary(self):
        result = dict(self.stats)
        active = [node for node in self.nodes.values() if node.active]
        result.update(
            {
                "active_nodes": len(active),
                "active_roots": sum(node.parent_uid < 0 for node in active),
                "active_child_slots": len(self.children),
                "tracked_roots": len(self.track_to_uid),
                "max_observed_level": max((node.level for node in active), default=0),
                "render_handoff_enabled": bool(
                    self.config["render_handoff_enabled"]
                ),
                "hash_calls_zero": (
                    result["hash_query_rows"] == 0 and result["hash_set_rows"] == 0
                ),
            }
        )
        return result
