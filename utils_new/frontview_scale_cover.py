"""Continuous scale-cover ownership for forward-view UAV Gaussian births."""

import time
from copy import deepcopy

import numpy as np


DEFAULT_FRONT_VIEW_SCALE_COVER_CONFIG = {
    "enabled": False,
    "query_backend": "scipy_kdtree",
    "reuse_same_epoch_admission": False,
    "radius_multiplier": 1.0,
    "scale_compatibility": 1.0,
    "color_distance_threshold": -1.0,
    "neighbors": 32,
    "rebuild_rows": 8192,
    "shuffle_occupancy": False,
    "shuffle_seed": 42,
    "shuffle_depth_edges_m": [20.0, 50.0, 80.0],
    "evidence_quota_routing": False,
    "shuffle_evidence_quota": False,
    "quota_residual_weight": 1.0,
    "quota_coverage_weight": 0.5,
    "quota_confidence_weight": 0.25,
    "quota_unoccupied_bonus": 0.25,
    "quota_multiview_support_weight": 0.0,
    "shuffle_multiview_support": False,
    "quota_min_depth_m": 0.0,
    "quota_route_sparse": False,
    "quota_projective_cell_px": 0.0,
    "projective_revisit_gate": "disabled",
    "revisit_min_frame_gap": 60,
    "revisit_max_position_distance_m": 2.0,
    "revisit_max_view_angle_deg": 15.0,
    "revisit_min_cumulative_turn_deg": 90.0,
    "revisit_fixed_start_frame": -1,
    "dynamic_handoff_enabled": False,
    "active_occupancy_enabled": False,
    "shuffle_lifecycle_release": False,
    "source_priority_enabled": False,
    "shuffle_source_priority": False,
    "appearance_certificate_enabled": False,
    "appearance_min_residual": 0.0,
    "appearance_min_depth_confidence": 0.0,
    "appearance_allow_sparse": False,
    "shuffle_appearance_certificates": False,
    "per_candidate_footprints": False,
    "footprint_ratio_min": 0.5,
    "footprint_ratio_max": 2.0,
    "shuffle_footprints": False,
    "projected_handoff_enabled": False,
    "projected_handoff_radius_px": 0.5,
    "shuffle_projected_handoff": False,
    "sparse_track_identity_enabled": False,
    "shuffle_sparse_track_identity": False,
    "directional_ownership_enabled": False,
    "directional_max_angle_deg": 45.0,
    "shuffle_directional_ownership": False,
    "frustum_conflict_budget_enabled": False,
    "frustum_conflict_min_fraction": 0.05,
    "frustum_conflict_min_rows": 16,
    "frustum_conflict_budget_rows": 64,
    "shuffle_frustum_conflict_budget": False,
    "handoff_radius_start_px": 4.0,
    "handoff_radius_end_px": 8.0,
    "handoff_parent_floor": 0.5,
    "handoff_area_full": 1.0,
    "handoff_batch_reduce": "max",
    "shuffle_handoff": False,
}


def validate_front_view_scale_cover_config(config=None):
    merged = deepcopy(DEFAULT_FRONT_VIEW_SCALE_COVER_CONFIG)
    if config is not None:
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError(
                "Unknown FrontViewScaleCover options: {}".format(sorted(unknown))
            )
        merged.update(config)
    if merged["query_backend"] not in ("scipy_kdtree", "pytorch3d_knn"):
        raise ValueError(
            "FrontViewScaleCover.query_backend must be scipy_kdtree or "
            "pytorch3d_knn"
        )
    for key in (
        "enabled",
        "shuffle_occupancy",
        "evidence_quota_routing",
        "shuffle_evidence_quota",
        "shuffle_multiview_support",
        "quota_route_sparse",
        "dynamic_handoff_enabled",
        "active_occupancy_enabled",
        "shuffle_lifecycle_release",
        "source_priority_enabled",
        "shuffle_source_priority",
        "appearance_certificate_enabled",
        "appearance_allow_sparse",
        "shuffle_appearance_certificates",
        "per_candidate_footprints",
        "shuffle_footprints",
        "projected_handoff_enabled",
        "shuffle_projected_handoff",
        "sparse_track_identity_enabled",
        "shuffle_sparse_track_identity",
        "directional_ownership_enabled",
        "shuffle_directional_ownership",
        "frustum_conflict_budget_enabled",
        "shuffle_frustum_conflict_budget",
        "shuffle_handoff",
        "reuse_same_epoch_admission",
    ):
        if not isinstance(merged[key], bool):
            raise TypeError("FrontViewScaleCover.{} must be boolean".format(key))
    for key in ("radius_multiplier", "scale_compatibility"):
        if float(merged[key]) <= 0.0:
            raise ValueError("FrontViewScaleCover.{} must be positive".format(key))
    for key in ("footprint_ratio_min", "footprint_ratio_max"):
        if float(merged[key]) <= 0.0:
            raise ValueError("FrontViewScaleCover.{} must be positive".format(key))
    if float(merged["projected_handoff_radius_px"]) <= 0.0:
        raise ValueError(
            "FrontViewScaleCover.projected_handoff_radius_px must be positive"
        )
    if not 0.0 < float(merged["directional_max_angle_deg"]) < 180.0:
        raise ValueError(
            "FrontViewScaleCover.directional_max_angle_deg must be in (0, 180)"
        )
    if not 0.0 <= float(merged["frustum_conflict_min_fraction"]) <= 1.0:
        raise ValueError(
            "FrontViewScaleCover.frustum_conflict_min_fraction must be in [0, 1]"
        )
    for key in ("frustum_conflict_min_rows", "frustum_conflict_budget_rows"):
        if not isinstance(merged[key], int) or merged[key] < 1:
            raise ValueError("FrontViewScaleCover.{} must be positive".format(key))
    if merged["frustum_conflict_budget_enabled"] and not merged[
        "directional_ownership_enabled"
    ]:
        raise ValueError(
            "Frustum-conflict budgeting requires directional ownership"
        )
    if merged["shuffle_frustum_conflict_budget"] and not merged[
        "frustum_conflict_budget_enabled"
    ]:
        raise ValueError(
            "Frustum-conflict shuffling requires frustum-conflict budgeting"
        )
    if merged["shuffle_evidence_quota"] and not merged[
        "evidence_quota_routing"
    ]:
        raise ValueError(
            "Evidence-quota shuffling requires evidence-quota routing"
        )
    if merged["shuffle_multiview_support"] and (
        not merged["evidence_quota_routing"]
        or float(merged["quota_multiview_support_weight"]) <= 0.0
    ):
        raise ValueError(
            "Multiview-support shuffling requires support-weighted evidence routing"
        )
    if merged["shuffle_occupancy"] and merged["evidence_quota_routing"]:
        raise ValueError(
            "Occupancy shuffling and evidence-quota routing are mutually exclusive"
        )
    if float(merged["footprint_ratio_max"]) < float(
        merged["footprint_ratio_min"]
    ):
        raise ValueError("FrontViewScaleCover footprint ratios must increase")
    for key in (
        "handoff_radius_start_px",
        "handoff_radius_end_px",
        "handoff_area_full",
    ):
        if float(merged[key]) <= 0.0:
            raise ValueError("FrontViewScaleCover.{} must be positive".format(key))
    if float(merged["handoff_radius_end_px"]) <= float(
        merged["handoff_radius_start_px"]
    ):
        raise ValueError("FrontViewScaleCover handoff radii must increase")
    if not 0.0 <= float(merged["handoff_parent_floor"]) <= 1.0:
        raise ValueError("FrontViewScaleCover handoff_parent_floor must be in [0, 1]")
    if merged["handoff_batch_reduce"] not in ("max", "mean"):
        raise ValueError("FrontViewScaleCover handoff_batch_reduce must be max or mean")
    if float(merged["color_distance_threshold"]) < -1.0:
        raise ValueError(
            "FrontViewScaleCover.color_distance_threshold must be -1 or nonnegative"
        )
    for key in ("appearance_min_residual", "appearance_min_depth_confidence"):
        if float(merged[key]) < 0.0:
            raise ValueError("FrontViewScaleCover.{} must be non-negative".format(key))
    for key in ("neighbors", "rebuild_rows"):
        if not isinstance(merged[key], int) or merged[key] < 1:
            raise ValueError("FrontViewScaleCover.{} must be positive".format(key))
    quota_weights = (
        "quota_residual_weight",
        "quota_coverage_weight",
        "quota_confidence_weight",
        "quota_unoccupied_bonus",
        "quota_multiview_support_weight",
    )
    for key in quota_weights:
        if float(merged[key]) < 0.0:
            raise ValueError("FrontViewScaleCover.{} must be non-negative".format(key))
    if float(merged["quota_min_depth_m"]) < 0.0:
        raise ValueError("FrontViewScaleCover.quota_min_depth_m must be non-negative")
    if merged["evidence_quota_routing"] and not any(
        float(merged[key]) > 0.0 for key in quota_weights
    ):
        raise ValueError("Evidence-quota routing requires a positive score weight")
    if float(merged["quota_projective_cell_px"]) < 0.0:
        raise ValueError(
            "FrontViewScaleCover.quota_projective_cell_px must be non-negative"
        )
    if merged["projective_revisit_gate"] not in (
        "disabled",
        "pose",
        "fixed_frame",
    ):
        raise ValueError(
            "FrontViewScaleCover.projective_revisit_gate must be disabled, pose, "
            "or fixed_frame"
        )
    if (
        merged["projective_revisit_gate"] != "disabled"
        and float(merged["quota_projective_cell_px"]) <= 0.0
    ):
        raise ValueError("Projective revisit gating requires projective quota routing")
    if not isinstance(merged["revisit_min_frame_gap"], int) or merged[
        "revisit_min_frame_gap"
    ] < 1:
        raise ValueError("FrontViewScaleCover.revisit_min_frame_gap must be positive")
    if float(merged["revisit_max_position_distance_m"]) <= 0.0:
        raise ValueError(
            "FrontViewScaleCover.revisit_max_position_distance_m must be positive"
        )
    if not 0.0 < float(merged["revisit_max_view_angle_deg"]) < 180.0:
        raise ValueError(
            "FrontViewScaleCover.revisit_max_view_angle_deg must be in (0, 180)"
        )
    if float(merged["revisit_min_cumulative_turn_deg"]) < 0.0:
        raise ValueError(
            "FrontViewScaleCover.revisit_min_cumulative_turn_deg must be non-negative"
        )
    if not isinstance(merged["revisit_fixed_start_frame"], int):
        raise TypeError(
            "FrontViewScaleCover.revisit_fixed_start_frame must be an integer"
        )
    if (
        merged["projective_revisit_gate"] == "fixed_frame"
        and merged["revisit_fixed_start_frame"] < 0
    ):
        raise ValueError("Fixed-frame revisit gating requires a non-negative start frame")
    if not isinstance(merged["shuffle_seed"], int):
        raise TypeError("FrontViewScaleCover.shuffle_seed must be an integer")
    edges = np.asarray(merged["shuffle_depth_edges_m"], dtype=np.float32)
    if edges.ndim != 1 or len(edges) == 0 or np.any(edges <= 0.0):
        raise ValueError("FrontViewScaleCover shuffle depth edges must be positive")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("FrontViewScaleCover shuffle depth edges must increase")
    merged["shuffle_depth_edges_m"] = edges.tolist()
    return merged


class FrontViewScaleCover:
    """Persistent continuous cover whose resolution follows birth view scale."""

    def __init__(self, config=None):
        self.config = validate_front_view_scale_cover_config(config)
        self._base_points = np.empty((0, 3), dtype=np.float32)
        self._base_scales = np.empty((0,), dtype=np.float32)
        self._base_colors = np.empty((0, 3), dtype=np.float32)
        self._base_uids = np.empty((0,), dtype=np.int64)
        self._base_source_ranks = np.empty((0,), dtype=np.int8)
        self._base_view_directions = np.empty((0, 3), dtype=np.float32)
        self._base_tree = None
        self._base_points_gpu_by_device = {}
        self._pending_points = []
        self._pending_scales = []
        self._pending_colors = []
        self._pending_uids = []
        self._pending_source_ranks = []
        self._pending_view_directions = []
        self._pending_rows = 0
        self._far_points = np.empty((0, 3), dtype=np.float32)
        self._far_scales = np.empty((0,), dtype=np.float32)
        self._far_colors = np.empty((0, 3), dtype=np.float32)
        self._far_uids = np.empty((0,), dtype=np.int64)
        self._far_source_ranks = np.empty((0,), dtype=np.int8)
        self._far_view_directions = np.empty((0, 3), dtype=np.float32)
        self._projected_handoff_last_frame = None
        self._committed_sparse_tracks = set()
        self.admission_epoch = 0
        self.next_uid = 0
        self._node_parent = {}
        self._node_scale = {}
        self._node_active = {}
        self._coverage_cache_cpu = None
        self._coverage_cache_by_device = {}
        self._active_uid_table = np.zeros((0,), dtype=np.bool_)
        self._revisit_frame_ids = []
        self._revisit_centers = []
        self._revisit_forwards = []
        self._revisit_cumulative_turns = []
        self._revisit_certificates = {}
        self.stats = {
            "query_calls": 0,
            "query_rows": 0,
            "occupied_rows": 0,
            "coarse_refinement_bypass_rows": 0,
            "appearance_packet_bypass_rows": 0,
            "registered_rows": 0,
            "tree_rebuilds": 0,
            "shuffled_rows": 0,
            "evidence_quota_calls": 0,
            "evidence_quota_rows": 0,
            "evidence_quota_selected_rows": 0,
            "evidence_quota_reassigned_rows": 0,
            "evidence_quota_routed_band_rows": [
                0 for _ in range(len(self.config["shuffle_depth_edges_m"]) + 1)
            ],
            "evidence_quota_reassigned_band_rows": [
                0 for _ in range(len(self.config["shuffle_depth_edges_m"]) + 1)
            ],
            "shuffled_evidence_quota_rows": 0,
            "projective_quota_rows": 0,
            "multiview_support_rows": 0,
            "multiview_support_positive_rows": 0,
            "multiview_support_score_sum": 0.0,
            "shuffled_multiview_support_rows": 0,
            "revisit_certificate_checked_frames": 0,
            "revisit_certificate_certified_frames": 0,
            "revisit_certificate_first_frame": -1,
            "revisit_certificate_cache_hits": 0,
            "revisit_gate_skipped_calls": 0,
            "revisit_gate_skipped_rows": 0,
            "revisit_support_distance_sum_m": 0.0,
            "revisit_support_distance_min_m": None,
            "revisit_support_distance_max_m": None,
            "revisit_support_angle_sum_deg": 0.0,
            "revisit_support_turn_sum_deg": 0.0,
            "hash_query_rows": 0,
            "hash_set_rows": 0,
            "parent_candidate_rows": 0,
            "linked_child_rows": 0,
            "shuffled_parent_rows": 0,
            "released_uids": 0,
            "handoff_render_calls": 0,
            "handoff_faded_rows": 0,
            "released_cover_rows": 0,
            "lifecycle_rebuilds": 0,
            "shuffled_release_rows": 0,
            "release_calls": 0,
            "priority_bypass_rows": 0,
            "shuffled_priority_rows": 0,
            "appearance_certificate_calls": 0,
            "appearance_certificate_rows": 0,
            "appearance_certificate_eligible_rows": 0,
            "shuffled_appearance_certificate_rows": 0,
            "footprint_calls": 0,
            "footprint_rows": 0,
            "footprint_min_clamped_rows": 0,
            "footprint_max_clamped_rows": 0,
            "shuffled_footprint_rows": 0,
            "scipy_query_calls": 0,
            "gpu_query_calls": 0,
            "query_time_ms": 0.0,
            "tree_build_time_ms": 0.0,
            "same_epoch_commit_reuses": 0,
            "stale_commit_requeries": 0,
            "projected_handoff_calls": 0,
            "projected_handoff_staged_rows": 0,
            "projected_handoff_projected_rows": 0,
            "projected_handoff_eligible_rows": 0,
            "projected_handoff_activated_rows": 0,
            "shuffled_projected_handoff_rows": 0,
            "sparse_track_identity_calls": 0,
            "sparse_track_identity_rows": 0,
            "sparse_track_repeat_rows": 0,
            "sparse_track_spatial_bypass_rows": 0,
            "sparse_track_registered_rows": 0,
            "sparse_track_released_rows": 0,
            "shuffled_sparse_track_rows": 0,
            "directional_query_rows": 0,
            "directional_bypass_rows": 0,
            "shuffled_directional_rows": 0,
            "frustum_conflict_calls": 0,
            "frustum_conflict_rows": 0,
            "frustum_conflict_depthcov_rows": 0,
            "frustum_conflict_trigger_calls": 0,
            "frustum_conflict_selected_rows": 0,
            "frustum_conflict_sparse_suppressed_rows": 0,
            "shuffled_frustum_conflict_rows": 0,
        }

    @property
    def enabled(self):
        return bool(self.config["enabled"])

    @staticmethod
    def _points(values):
        result = np.asarray(values, dtype=np.float32).reshape(-1, 3)
        if len(result) and not np.isfinite(result).all():
            raise ValueError("Scale-cover points must be finite")
        return result

    def _view_directions(self, values, count):
        if values is None:
            if self.directional_ownership_enabled and count:
                raise ValueError("Directional ownership requires view directions")
            return np.zeros((count, 3), dtype=np.float32)
        result = np.asarray(values, dtype=np.float32).reshape(-1, 3)
        if result.shape != (count, 3) or np.any(~np.isfinite(result)):
            raise ValueError("Scale-cover view directions must align and be finite")
        norms = np.linalg.norm(result, axis=1)
        if np.any(norms <= 1.0e-8):
            if self.directional_ownership_enabled:
                raise ValueError("Directional ownership requires nonzero view directions")
            return result
        return (result / norms[:, None]).astype(np.float32, copy=False)

    def observe_raw_pose(self, world_to_camera, frame_id):
        """Cache a causal VI-pose revisit certificate for one input frame."""

        mode = self.config["projective_revisit_gate"]
        if mode == "disabled":
            return True
        frame_id = int(frame_id)
        if frame_id in self._revisit_certificates:
            self.stats["revisit_certificate_cache_hits"] += 1
            return self._revisit_certificates[frame_id]
        if self._revisit_frame_ids and frame_id <= self._revisit_frame_ids[-1]:
            raise ValueError("Revisit poses must be observed in increasing frame order")

        pose = np.asarray(world_to_camera, dtype=np.float32)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Revisit certification requires a finite 4x4 pose")
        rotation = pose[:3, :3]
        center = -rotation.T @ pose[:3, 3]
        forward = rotation.T @ np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        forward_norm = float(np.linalg.norm(forward))
        if not np.isfinite(center).all() or forward_norm <= 1.0e-8:
            raise ValueError("Revisit certification requires a valid camera frame")
        forward = (forward / forward_norm).astype(np.float32, copy=False)

        cumulative_turn = 0.0
        if self._revisit_forwards:
            cosine = float(
                np.clip(np.dot(self._revisit_forwards[-1], forward), -1.0, 1.0)
            )
            cumulative_turn = self._revisit_cumulative_turns[-1] + float(
                np.rad2deg(np.arccos(cosine))
            )

        certified = False
        support = None
        if mode == "fixed_frame":
            certified = frame_id >= int(self.config["revisit_fixed_start_frame"])
        elif self._revisit_frame_ids:
            prior_frames = np.asarray(self._revisit_frame_ids, dtype=np.int64)
            temporal = (
                frame_id - prior_frames
                >= int(self.config["revisit_min_frame_gap"])
            )
            if np.any(temporal):
                centers = np.asarray(self._revisit_centers, dtype=np.float32)
                forwards = np.asarray(self._revisit_forwards, dtype=np.float32)
                prior_turns = np.asarray(
                    self._revisit_cumulative_turns, dtype=np.float32
                )
                distances = np.linalg.norm(centers - center[None, :], axis=1)
                angles = np.rad2deg(
                    np.arccos(
                        np.clip(np.sum(forwards * forward[None, :], axis=1), -1.0, 1.0)
                    )
                )
                turns = cumulative_turn - prior_turns
                eligible = (
                    temporal
                    & (
                        distances
                        <= float(self.config["revisit_max_position_distance_m"])
                    )
                    & (
                        angles <= float(self.config["revisit_max_view_angle_deg"])
                    )
                    & (
                        turns
                        >= float(self.config["revisit_min_cumulative_turn_deg"])
                    )
                )
                rows = np.flatnonzero(eligible)
                if len(rows):
                    order = np.lexsort((prior_frames[rows], angles[rows], distances[rows]))
                    row = int(rows[order[0]])
                    certified = True
                    support = (
                        float(distances[row]),
                        float(angles[row]),
                        float(turns[row]),
                    )

        self._revisit_frame_ids.append(frame_id)
        self._revisit_centers.append(center.astype(np.float32, copy=False))
        self._revisit_forwards.append(forward)
        self._revisit_cumulative_turns.append(float(cumulative_turn))
        self._revisit_certificates[frame_id] = bool(certified)
        self.stats["revisit_certificate_checked_frames"] += 1
        if certified:
            self.stats["revisit_certificate_certified_frames"] += 1
            if self.stats["revisit_certificate_first_frame"] < 0:
                self.stats["revisit_certificate_first_frame"] = frame_id
        if support is not None:
            distance, angle, turn = support
            self.stats["revisit_support_distance_sum_m"] += distance
            self.stats["revisit_support_angle_sum_deg"] += angle
            self.stats["revisit_support_turn_sum_deg"] += turn
            current_min = self.stats["revisit_support_distance_min_m"]
            current_max = self.stats["revisit_support_distance_max_m"]
            self.stats["revisit_support_distance_min_m"] = (
                distance if current_min is None else min(current_min, distance)
            )
            self.stats["revisit_support_distance_max_m"] = (
                distance if current_max is None else max(current_max, distance)
            )
        return bool(certified)

    @staticmethod
    def _target_scale(value):
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError("Scale-cover target size must be finite and positive")
        return result

    @classmethod
    def _target_scales(cls, values, count):
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 0:
            array = np.full((count,), cls._target_scale(array.item()), dtype=np.float32)
        else:
            array = array.reshape(-1)
            if array.shape != (count,):
                raise ValueError("Scale-cover target sizes must align with points")
            if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
                raise ValueError("Scale-cover target sizes must be finite and positive")
        return array

    def candidate_target_sizes(
        self,
        depths,
        focal_pixels,
        camera_scale_rescalar,
        view_scale_size,
        sparse_valid,
        frame_id,
    ):
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        sparse_valid = np.asarray(sparse_valid, dtype=np.bool_).reshape(-1)
        if depths.shape != sparse_valid.shape:
            raise ValueError("Footprint depths and source masks must align")
        scalar_size = self._target_scale(view_scale_size)
        if not self.config["per_candidate_footprints"]:
            return np.full(depths.shape, scalar_size, dtype=np.float32)
        focal_pixels = float(focal_pixels)
        camera_scale_rescalar = float(camera_scale_rescalar)
        if focal_pixels <= 0.0 or camera_scale_rescalar <= 0.0:
            raise ValueError("Footprint camera scale must be positive")
        raw = 0.5 * depths / focal_pixels * camera_scale_rescalar
        minimum = scalar_size * float(self.config["footprint_ratio_min"])
        maximum = scalar_size * float(self.config["footprint_ratio_max"])
        self.stats["footprint_calls"] += 1
        self.stats["footprint_rows"] += len(raw)
        self.stats["footprint_min_clamped_rows"] += int(np.sum(raw < minimum))
        self.stats["footprint_max_clamped_rows"] += int(np.sum(raw > maximum))
        sizes = np.clip(raw, minimum, maximum).astype(np.float32, copy=False)
        if self.config["shuffle_footprints"] and len(sizes) > 1:
            rng = np.random.default_rng(
                int(self.config["shuffle_seed"]) + int(frame_id)
            )
            sizes = sizes.copy()
            for sparse in (False, True):
                rows = np.flatnonzero(sparse_valid == sparse)
                if len(rows) > 1:
                    sizes[rows] = sizes[rows[rng.permutation(len(rows))]]
                    self.stats["shuffled_footprint_rows"] += len(rows)
        return sizes

    @property
    def dynamic_handoff_enabled(self):
        return bool(self.enabled and self.config["dynamic_handoff_enabled"])

    @property
    def tracks_uids(self):
        return bool(
            self.enabled
            and (
                self.config["dynamic_handoff_enabled"]
                or self.config["active_occupancy_enabled"]
                or self.config["projected_handoff_enabled"]
            )
        )

    @property
    def projected_handoff_enabled(self):
        return bool(self.enabled and self.config["projected_handoff_enabled"])

    @property
    def sparse_track_identity_enabled(self):
        return bool(self.enabled and self.config["sparse_track_identity_enabled"])

    @property
    def directional_ownership_enabled(self):
        return bool(self.enabled and self.config["directional_ownership_enabled"])

    def candidate_view_directions(
        self,
        points,
        camera_center,
        depths,
        sparse_valid,
        frame_id,
    ):
        points = self._points(points)
        center = np.asarray(camera_center, dtype=np.float32).reshape(3)
        directions = self._view_directions(points - center[None, :], len(points))
        if (
            not self.directional_ownership_enabled
            or not self.config["shuffle_directional_ownership"]
            or len(points) <= 1
        ):
            return directions
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        sparse_valid = np.asarray(sparse_valid, dtype=np.bool_).reshape(-1)
        if depths.shape != (len(points),) or sparse_valid.shape != (len(points),):
            raise ValueError("Directional shuffle arrays must align")
        bands = np.digitize(
            depths,
            np.asarray(self.config["shuffle_depth_edges_m"], dtype=np.float32),
        )
        result = directions.copy()
        rng = np.random.default_rng(
            int(self.config["shuffle_seed"]) + int(frame_id)
        )
        for sparse in (False, True):
            for band in range(len(self.config["shuffle_depth_edges_m"]) + 1):
                rows = np.flatnonzero((sparse_valid == sparse) & (bands == band))
                if len(rows) > 1:
                    result[rows] = directions[rows[rng.permutation(len(rows))]]
                    self.stats["shuffled_directional_rows"] += len(rows)
        return result

    def _frustum_conflict_selection(
        self,
        occupied,
        directional_bypass,
        residual_scores,
        depth_confidences,
        sparse_valid,
        depths,
        frame_id,
        *,
        preselected=False,
    ):
        conflict = np.asarray(directional_bypass, dtype=np.bool_) & ~np.asarray(
            occupied, dtype=np.bool_
        )
        if preselected or not self.config["frustum_conflict_budget_enabled"]:
            return conflict

        count = len(conflict)
        residual_scores = np.asarray(residual_scores, dtype=np.float32).reshape(-1)
        depth_confidences = np.asarray(
            depth_confidences, dtype=np.float32
        ).reshape(-1)
        sparse_valid = np.asarray(sparse_valid, dtype=np.bool_).reshape(-1)
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        if not (
            residual_scores.shape
            == depth_confidences.shape
            == sparse_valid.shape
            == depths.shape
            == (count,)
        ):
            raise ValueError("Frustum-conflict evidence arrays must align")

        self.stats["frustum_conflict_calls"] += 1
        self.stats["frustum_conflict_rows"] += int(np.sum(conflict))
        self.stats["frustum_conflict_sparse_suppressed_rows"] += int(
            np.sum(conflict & sparse_valid)
        )
        depthcov_conflict = conflict & ~sparse_valid
        conflict_count = int(np.sum(depthcov_conflict))
        self.stats["frustum_conflict_depthcov_rows"] += conflict_count
        spatially_owned_depthcov = (~sparse_valid) & (occupied | conflict)
        conflict_fraction = conflict_count / max(
            int(np.sum(spatially_owned_depthcov)), 1
        )
        if (
            conflict_count < int(self.config["frustum_conflict_min_rows"])
            or conflict_fraction
            < float(self.config["frustum_conflict_min_fraction"])
        ):
            return np.zeros((count,), dtype=np.bool_)

        self.stats["frustum_conflict_trigger_calls"] += 1
        candidates = np.flatnonzero(depthcov_conflict)
        budget = min(int(self.config["frustum_conflict_budget_rows"]), len(candidates))
        scores = np.nan_to_num(
            residual_scores * depth_confidences,
            nan=-np.inf,
            posinf=np.finfo(np.float32).max,
            neginf=-np.inf,
        )
        ranked = candidates[
            np.lexsort((candidates, -scores[candidates]))[:budget]
        ]
        selected = np.zeros((count,), dtype=np.bool_)
        selected[ranked] = True

        if self.config["shuffle_frustum_conflict_budget"] and budget:
            bands = np.digitize(
                depths,
                np.asarray(self.config["shuffle_depth_edges_m"], dtype=np.float32),
            )
            shuffled = np.zeros((count,), dtype=np.bool_)
            rng = np.random.default_rng(
                int(self.config["shuffle_seed"]) + int(frame_id)
            )
            for band in np.unique(bands[ranked]):
                band_budget = int(np.sum(bands[ranked] == band))
                pool = np.flatnonzero(depthcov_conflict & (bands == band))
                shuffled[
                    rng.choice(pool, size=band_budget, replace=False)
                ] = True
            selected = shuffled
            self.stats["shuffled_frustum_conflict_rows"] += budget

        self.stats["frustum_conflict_selected_rows"] += int(np.sum(selected))
        return selected

    def apply_sparse_track_identity(
        self,
        occupied,
        track_ids,
        sparse_valid,
        depths,
        frame_id,
    ):
        """Let persistent sparse identities supersede metric proximity."""

        occupied = np.asarray(occupied, dtype=np.bool_).reshape(-1)
        if not self.sparse_track_identity_enabled or len(occupied) == 0:
            return occupied
        track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        sparse_valid = np.asarray(sparse_valid, dtype=np.bool_).reshape(-1)
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        if not (
            occupied.shape == track_ids.shape == sparse_valid.shape == depths.shape
        ):
            raise ValueError("Sparse-track identity arrays must align")

        tracked = sparse_valid & (track_ids >= 0)
        rows = np.flatnonzero(tracked)
        if len(rows) == 0:
            return occupied
        seen = set(self._committed_sparse_tracks)
        repeated = np.zeros(len(rows), dtype=np.bool_)
        for position, row in enumerate(rows):
            track_id = int(track_ids[row])
            repeated[position] = track_id in seen
            seen.add(track_id)
        identity_occupied = repeated
        if self.config["shuffle_sparse_track_identity"] and len(rows) > 1:
            bands = np.digitize(
                depths,
                np.asarray(self.config["shuffle_depth_edges_m"], dtype=np.float32),
            )
            shuffled = identity_occupied.copy()
            rng = np.random.default_rng(
                int(self.config["shuffle_seed"]) + int(frame_id)
            )
            row_bands = bands[rows]
            for band in range(len(self.config["shuffle_depth_edges_m"]) + 1):
                positions = np.flatnonzero(row_bands == band)
                if len(positions) > 1:
                    shuffled[positions] = identity_occupied[
                        positions[rng.permutation(len(positions))]
                    ]
                    self.stats["shuffled_sparse_track_rows"] += len(positions)
            identity_occupied = shuffled

        result = occupied.copy()
        self.stats["sparse_track_identity_calls"] += 1
        self.stats["sparse_track_identity_rows"] += len(rows)
        self.stats["sparse_track_repeat_rows"] += int(np.sum(repeated))
        self.stats["sparse_track_spatial_bypass_rows"] += int(
            np.sum(occupied[rows] & ~identity_occupied)
        )
        result[rows] = identity_occupied
        return result

    def register_sparse_tracks(self, track_ids, sparse_valid):
        if not self.sparse_track_identity_enabled:
            return 0
        track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        sparse_valid = np.asarray(sparse_valid, dtype=np.bool_).reshape(-1)
        if track_ids.shape != sparse_valid.shape:
            raise ValueError("Sparse-track registration arrays must align")
        tracks = np.unique(track_ids[sparse_valid & (track_ids >= 0)])
        before = len(self._committed_sparse_tracks)
        self._committed_sparse_tracks.update(int(track_id) for track_id in tracks)
        added = len(self._committed_sparse_tracks) - before
        self.stats["sparse_track_registered_rows"] += added
        return added

    def release_sparse_tracks(self, track_ids):
        if not self.sparse_track_identity_enabled:
            return 0
        released = 0
        for track_id in np.unique(np.asarray(track_ids, dtype=np.int64).reshape(-1)):
            if track_id >= 0 and int(track_id) in self._committed_sparse_tracks:
                self._committed_sparse_tracks.remove(int(track_id))
                released += 1
        self.stats["sparse_track_released_rows"] += released
        return released

    def stage_projected_handoff(
        self,
        points,
        target_size,
        colors=None,
        uids=None,
        source_ranks=None,
        view_directions=None,
    ):
        """Keep far births outside the cover until they become resolvable."""

        if not self.projected_handoff_enabled:
            return
        points = self._points(points)
        if len(points) == 0:
            return
        scales = self._target_scales(target_size, len(points))
        if colors is None:
            colors = np.zeros((len(points), 3), dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        if len(colors) != len(points) or not np.isfinite(colors).all():
            raise ValueError("Projected-handoff colors must align and be finite")
        if uids is None:
            raise ValueError("Projected handoff requires stable Gaussian UIDs")
        uids = np.asarray(uids, dtype=np.int64).reshape(-1)
        if len(uids) != len(points) or np.any(uids < 0):
            raise ValueError("Projected-handoff UIDs must align and be nonnegative")
        if source_ranks is None:
            source_ranks = np.ones((len(points),), dtype=np.int8)
        source_ranks = np.asarray(source_ranks, dtype=np.int8).reshape(-1)
        if len(source_ranks) != len(points) or np.any(source_ranks < 0):
            raise ValueError("Projected-handoff source ranks must align")
        view_directions = self._view_directions(view_directions, len(points))

        self.register_uids(uids)
        self._far_points = np.concatenate((self._far_points, points), axis=0)
        self._far_scales = np.concatenate((self._far_scales, scales), axis=0)
        self._far_colors = np.concatenate((self._far_colors, colors), axis=0)
        self._far_uids = np.concatenate((self._far_uids, uids), axis=0)
        self._far_source_ranks = np.concatenate(
            (self._far_source_ranks, source_ranks), axis=0
        )
        self._far_view_directions = np.concatenate(
            (self._far_view_directions, view_directions), axis=0
        )
        self.stats["projected_handoff_staged_rows"] += len(points)

    def activate_projected_handoff(
        self,
        world_to_camera,
        focal_pixels,
        image_width,
        image_height,
        principal_x,
        principal_y,
        near,
        far,
        frame_id,
    ):
        """Move newly resolvable far births into persistent scale ownership."""

        if not self.projected_handoff_enabled:
            return 0
        frame_id = int(frame_id)
        if self._projected_handoff_last_frame == frame_id:
            return 0
        self._projected_handoff_last_frame = frame_id
        self.stats["projected_handoff_calls"] += 1
        if len(self._far_points) == 0:
            return 0

        pose = np.asarray(world_to_camera, dtype=np.float32)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(
                "Projected handoff requires a finite 4x4 world-to-camera pose"
            )
        focal_pixels = float(focal_pixels)
        image_width = float(image_width)
        image_height = float(image_height)
        principal_x = float(principal_x)
        principal_y = float(principal_y)
        if focal_pixels <= 0.0 or image_width <= 0.0 or image_height <= 0.0:
            raise ValueError("Projected handoff camera dimensions must be positive")
        if not np.isfinite(principal_x) or not np.isfinite(principal_y):
            raise ValueError("Projected handoff principal point must be finite")

        camera_points = self._far_points @ pose[:3, :3].T + pose[:3, 3]
        depths = camera_points[:, 2]
        safe_depths = np.maximum(depths, 1.0e-8)
        u = focal_pixels * camera_points[:, 0] / safe_depths + principal_x
        v = focal_pixels * camera_points[:, 1] / safe_depths + principal_y
        radii = focal_pixels * self._far_scales / safe_depths
        valid_uid = (
            (self._far_uids >= 0)
            & (self._far_uids < len(self._active_uid_table))
        )
        active = np.zeros(len(self._far_uids), dtype=np.bool_)
        active[valid_uid] = self._active_uid_table[self._far_uids[valid_uid]]
        visible = (
            active
            & np.isfinite(depths)
            & np.isfinite(radii)
            & (depths > float(near))
            & (depths < float(far))
            & (u >= 0.0)
            & (u < image_width)
            & (v >= 0.0)
            & (v < image_height)
        )
        eligible = visible & (
            radii >= float(self.config["projected_handoff_radius_px"])
        )
        self.stats["projected_handoff_projected_rows"] += int(np.sum(visible))
        self.stats["projected_handoff_eligible_rows"] += int(np.sum(eligible))

        activate = np.flatnonzero(eligible)
        if self.config["shuffle_projected_handoff"] and len(activate):
            pool = np.flatnonzero(active)
            rng = np.random.default_rng(
                int(self.config["shuffle_seed"]) + frame_id
            )
            activate = rng.choice(
                pool, size=min(len(activate), len(pool)), replace=False
            )
            self.stats["shuffled_projected_handoff_rows"] += len(activate)
        if len(activate) == 0:
            return 0

        self.register(
            self._far_points[activate],
            self._far_scales[activate],
            self._far_colors[activate],
            uids=self._far_uids[activate],
            source_ranks=self._far_source_ranks[activate],
            view_directions=self._far_view_directions[activate],
        )
        keep = np.ones(len(self._far_points), dtype=np.bool_)
        keep[activate] = False
        self._far_points = self._far_points[keep]
        self._far_scales = self._far_scales[keep]
        self._far_colors = self._far_colors[keep]
        self._far_uids = self._far_uids[keep]
        self._far_source_ranks = self._far_source_ranks[keep]
        self._far_view_directions = self._far_view_directions[keep]
        self.stats["projected_handoff_activated_rows"] += len(activate)
        return int(len(activate))

    def allocate_uids(self, count):
        count = int(count)
        if count < 0:
            raise ValueError("UID count must be nonnegative")
        start = self.next_uid
        self.next_uid += count
        uids = np.arange(start, start + count, dtype=np.int64)
        self.register_uids(uids)
        return uids

    def register_uids(self, uids):
        changed = False
        uids = np.asarray(uids, dtype=np.int64).reshape(-1)
        maximum = int(uids.max(initial=-1))
        if maximum >= len(self._active_uid_table):
            self._active_uid_table = np.pad(
                self._active_uid_table,
                (0, maximum + 1 - len(self._active_uid_table)),
                constant_values=False,
            )
        for uid in uids.tolist():
            if uid < 0:
                continue
            self.next_uid = max(self.next_uid, uid + 1)
            if uid not in self._node_active:
                self._node_parent[uid] = -1
                self._node_scale[uid] = float("nan")
                self._node_active[uid] = True
                changed = True
            self._active_uid_table[uid] = True
        if changed:
            self._invalidate_coverage_cache()

    def _invalidate_coverage_cache(self):
        self._coverage_cache_cpu = None
        self._coverage_cache_by_device.clear()

    def _flush(self):
        if self._pending_rows == 0:
            return

        pending_points = np.concatenate(self._pending_points, axis=0)
        pending_scales = np.concatenate(self._pending_scales, axis=0)
        pending_colors = np.concatenate(self._pending_colors, axis=0)
        pending_view_directions = np.concatenate(
            self._pending_view_directions, axis=0
        )
        self._base_points = np.concatenate(
            (self._base_points, pending_points), axis=0
        )
        self._base_scales = np.concatenate(
            (self._base_scales, pending_scales), axis=0
        )
        self._base_colors = np.concatenate(
            (self._base_colors, pending_colors), axis=0
        )
        self._base_uids = np.concatenate(
            (self._base_uids, np.concatenate(self._pending_uids, axis=0)), axis=0
        )
        self._base_source_ranks = np.concatenate(
            (
                self._base_source_ranks,
                np.concatenate(self._pending_source_ranks, axis=0),
            ),
            axis=0,
        )
        self._base_view_directions = np.concatenate(
            (self._base_view_directions, pending_view_directions), axis=0
        )
        self._base_points_gpu_by_device.clear()
        if self.config["query_backend"] == "scipy_kdtree":
            from scipy.spatial import cKDTree

            build_start = time.perf_counter()
            self._base_tree = cKDTree(self._base_points)
            self.stats["tree_build_time_ms"] += (
                time.perf_counter() - build_start
            ) * 1000.0
        else:
            self._base_tree = None
        self._pending_points.clear()
        self._pending_scales.clear()
        self._pending_colors.clear()
        self._pending_uids.clear()
        self._pending_source_ranks.clear()
        self._pending_view_directions.clear()
        self._pending_rows = 0
        self.stats["tree_rebuilds"] += 1

    def register(
        self,
        points,
        target_size,
        colors=None,
        uids=None,
        parent_uids=None,
        source_ranks=None,
        view_directions=None,
    ):
        points = self._points(points)
        if len(points) == 0:
            return
        scales = self._target_scales(target_size, len(points))
        if colors is None:
            colors = np.zeros((len(points), 3), dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        if len(colors) != len(points) or not np.isfinite(colors).all():
            raise ValueError("Scale-cover colors must align and be finite")
        if uids is None:
            uids = np.full((len(points),), -1, dtype=np.int64)
        uids = np.asarray(uids, dtype=np.int64).reshape(-1)
        if len(uids) != len(points):
            raise ValueError("Scale-cover UIDs must align")
        if self.tracks_uids and np.any(uids < 0):
            raise ValueError("Lifecycle-aware scale cover requires stable UIDs")
        if parent_uids is None:
            parent_uids = np.full((len(points),), -1, dtype=np.int64)
        parent_uids = np.asarray(parent_uids, dtype=np.int64).reshape(-1)
        if len(parent_uids) != len(points):
            raise ValueError("Scale-cover parent UIDs must align")
        if source_ranks is None:
            source_ranks = np.ones((len(points),), dtype=np.int8)
        source_ranks = np.asarray(source_ranks, dtype=np.int8).reshape(-1)
        if len(source_ranks) != len(points) or np.any(source_ranks < 0):
            raise ValueError("Scale-cover source ranks must align and be nonnegative")
        view_directions = self._view_directions(view_directions, len(points))
        self._pending_points.append(points.copy())
        self._pending_scales.append(scales.copy())
        self._pending_colors.append(colors.copy())
        self._pending_uids.append(uids.copy())
        self._pending_source_ranks.append(source_ranks.copy())
        self._pending_view_directions.append(view_directions.copy())
        self._pending_rows += len(points)
        self.admission_epoch += 1
        self.stats["registered_rows"] += len(points)
        if self.tracks_uids:
            self.register_uids(uids)
        if self.dynamic_handoff_enabled:
            linked = 0
            for row, (uid, parent_uid) in enumerate(
                zip(uids.tolist(), parent_uids.tolist())
            ):
                scale = float(scales[row])
                self._node_scale[uid] = scale
                parent_scale = self._node_scale.get(parent_uid, float("nan"))
                parent_active = bool(self._node_active.get(parent_uid, False))
                if (
                    parent_uid >= 0
                    and parent_active
                    and np.isfinite(parent_scale)
                    and parent_scale > scale
                ):
                    self._node_parent[uid] = int(parent_uid)
                    linked += 1
            self.stats["linked_child_rows"] += linked
            self._invalidate_coverage_cache()
        if self._pending_rows >= int(self.config["rebuild_rows"]):
            self._flush()

    def _query_neighbors(
        self,
        tree,
        tree_points,
        points,
        neighbors,
        max_radius,
        *,
        cache_base=False,
    ):
        query_start = time.perf_counter()
        if self.config["query_backend"] == "scipy_kdtree":
            distances, indices = tree.query(
                points,
                k=neighbors,
                distance_upper_bound=max_radius,
                workers=1,
            )
            self.stats["scipy_query_calls"] += 1
        else:
            import torch

            try:
                from pytorch3d.ops import knn_points
            except ImportError as exc:
                raise RuntimeError(
                    "pytorch3d_knn requires pytorch3d in the active environment"
                ) from exc
            if not torch.cuda.is_available():
                raise RuntimeError("pytorch3d_knn requires a CUDA device")

            device = torch.device("cuda", torch.cuda.current_device())
            device_key = str(device)
            reference = None
            if cache_base:
                reference = self._base_points_gpu_by_device.get(device_key)
            if reference is None:
                reference = torch.as_tensor(
                    tree_points, device=device, dtype=torch.float32
                ).contiguous()
                if cache_base:
                    self._base_points_gpu_by_device[device_key] = reference
            queries = torch.as_tensor(
                points, device=device, dtype=torch.float32
            ).contiguous()
            with torch.no_grad():
                result = knn_points(
                    queries.unsqueeze(0),
                    reference.unsqueeze(0),
                    K=neighbors,
                    return_nn=False,
                )
                distances = torch.sqrt(torch.clamp_min(result.dists[0], 0.0))
            distances = distances.cpu().numpy()
            indices = result.idx[0].cpu().numpy()
            self.stats["gpu_query_calls"] += 1
        self.stats["query_time_ms"] += (
            time.perf_counter() - query_start
        ) * 1000.0
        return distances, indices

    def _query_one_tree(
        self,
        tree,
        tree_points,
        tree_scales,
        tree_colors,
        tree_uids,
        tree_source_ranks,
        tree_view_directions,
        points,
        colors,
        source_ranks,
        view_directions,
        appearance_eligible,
        target_sizes,
        radii,
        *,
        cache_base=False,
    ):
        count = len(points)
        occupied = np.zeros((count,), dtype=np.bool_)
        nearby = np.zeros((count,), dtype=np.bool_)
        if len(tree_scales) == 0 or count == 0 or (
            self.config["query_backend"] == "scipy_kdtree" and tree is None
        ):
            return (
                occupied,
                nearby,
                np.zeros((count,), dtype=np.bool_),
                np.full((count,), -1, dtype=np.int64),
                np.full((count,), np.inf, dtype=np.float32),
                np.zeros((count,), dtype=np.bool_),
                np.zeros((count,), dtype=np.bool_),
            )
        neighbors = min(int(self.config["neighbors"]), len(tree_scales))
        distances, indices = self._query_neighbors(
            tree,
            tree_points,
            points,
            neighbors,
            float(np.max(radii)),
            cache_base=cache_base,
        )
        distances = np.asarray(distances)
        indices = np.asarray(indices)
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        if distances.ndim != 2 or indices.shape != distances.shape:
            raise RuntimeError("Scale-cover neighbor results must have shape [Q, K]")
        valid = (
            np.isfinite(distances)
            & (indices < len(tree_scales))
            & (distances <= radii[:, None])
        )
        if self.tracks_uids:
            safe_uid_indices = np.minimum(indices, max(len(tree_uids) - 1, 0))
            neighbor_uids = tree_uids[safe_uid_indices]
            uid_in_range = (
                (neighbor_uids >= 0)
                & (neighbor_uids < len(self._active_uid_table))
            )
            active = np.zeros(uid_in_range.shape, dtype=np.bool_)
            active[uid_in_range] = self._active_uid_table[
                neighbor_uids[uid_in_range]
            ]
            valid &= active
        nearby = np.any(valid, axis=1)
        safe_indices = np.minimum(indices, max(len(tree_scales) - 1, 0))
        scale_compatible = tree_scales[safe_indices] <= (
            target_sizes[:, None] * float(self.config["scale_compatibility"])
        )
        threshold = float(self.config["color_distance_threshold"])
        if threshold >= 0.0:
            color_distance = np.linalg.norm(
                tree_colors[safe_indices] - colors[:, None, :], axis=-1
            )
            color_compatible = color_distance <= threshold
            color_compatible |= ~appearance_eligible[:, None]
        else:
            color_compatible = np.ones(valid.shape, dtype=np.bool_)
        if self.config["source_priority_enabled"]:
            source_compatible = tree_source_ranks[safe_indices] >= source_ranks[
                :, None
            ]
        else:
            source_compatible = np.ones(valid.shape, dtype=np.bool_)
        if self.directional_ownership_enabled:
            direction_cosine = np.sum(
                tree_view_directions[safe_indices]
                * view_directions[:, None, :],
                axis=-1,
            )
            direction_compatible = direction_cosine >= np.cos(
                np.deg2rad(float(self.config["directional_max_angle_deg"]))
            )
        else:
            direction_compatible = np.ones(valid.shape, dtype=np.bool_)
        ownership_eligible = (
            valid & scale_compatible & color_compatible & source_compatible
        )
        occupied = np.any(ownership_eligible & direction_compatible, axis=1)
        directional_bypass = (
            np.any(ownership_eligible & ~direction_compatible, axis=1)
            & ~occupied
        )
        appearance_bypass = (
            appearance_eligible
            & np.any(valid & scale_compatible, axis=1)
            & ~occupied
        )
        priority_bypass = (
            np.any(
                valid
                & scale_compatible
                & color_compatible
                & ~source_compatible,
                axis=1,
            )
            & ~occupied
        )
        coarse_eligible = (
            valid
            & ~scale_compatible
            & color_compatible
            & source_compatible
            & direction_compatible
            & (tree_uids[safe_indices] >= 0)
        )
        parent_scores = np.where(coarse_eligible, distances, np.inf)
        best_parent = np.argmin(parent_scores, axis=1)
        best_scores = parent_scores[np.arange(count), best_parent].astype(np.float32)
        parent_uids = np.full((count,), -1, dtype=np.int64)
        has_parent = np.isfinite(best_scores)
        parent_uids[has_parent] = tree_uids[
            safe_indices[np.arange(count), best_parent]
        ][has_parent]
        return (
            occupied,
            nearby,
            appearance_bypass,
            parent_uids,
            best_scores,
            priority_bypass,
            directional_bypass,
        )

    def occupied_with_parents(
        self,
        points,
        target_size,
        colors=None,
        source_ranks=None,
        appearance_eligible=None,
        view_directions=None,
        residual_scores=None,
        depth_confidences=None,
        sparse_valid=None,
        depths=None,
        frame_id=0,
        directional_preselected=False,
    ):
        points = self._points(points)
        if colors is None:
            colors = np.zeros((len(points), 3), dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        if len(colors) != len(points) or not np.isfinite(colors).all():
            raise ValueError("Scale-cover query colors must align and be finite")
        if source_ranks is None:
            source_ranks = np.ones((len(points),), dtype=np.int8)
        source_ranks = np.asarray(source_ranks, dtype=np.int8).reshape(-1)
        if len(source_ranks) != len(points) or np.any(source_ranks < 0):
            raise ValueError("Scale-cover query ranks must align and be nonnegative")
        if appearance_eligible is None:
            appearance_eligible = np.ones((len(points),), dtype=np.bool_)
        appearance_eligible = np.asarray(
            appearance_eligible, dtype=np.bool_
        ).reshape(-1)
        if len(appearance_eligible) != len(points):
            raise ValueError("Scale-cover appearance eligibility must align")
        view_directions = self._view_directions(view_directions, len(points))
        target_sizes = self._target_scales(target_size, len(points))
        radii = target_sizes * float(self.config["radius_multiplier"])
        (
            occupied,
            nearby,
            appearance_bypass,
            parent_uids,
            parent_scores,
            priority_bypass,
            directional_bypass,
        ) = self._query_one_tree(
            self._base_tree,
            self._base_points,
            self._base_scales,
            self._base_colors,
            self._base_uids,
            self._base_source_ranks,
            self._base_view_directions,
            points,
            colors,
            source_ranks,
            view_directions,
            appearance_eligible,
            target_sizes,
            radii,
            cache_base=True,
        )
        if self._pending_rows:
            pending_points = np.concatenate(self._pending_points, axis=0)
            pending_scales = np.concatenate(self._pending_scales, axis=0)
            pending_colors = np.concatenate(self._pending_colors, axis=0)
            pending_tree = None
            if self.config["query_backend"] == "scipy_kdtree":
                from scipy.spatial import cKDTree

                pending_tree = cKDTree(pending_points)
            pending_uids = np.concatenate(self._pending_uids, axis=0)
            pending_source_ranks = np.concatenate(
                self._pending_source_ranks, axis=0
            )
            pending_view_directions = np.concatenate(
                self._pending_view_directions, axis=0
            )
            (
                pending_occupied,
                pending_nearby,
                pending_appearance_bypass,
                pending_parent_uids,
                pending_parent_scores,
                pending_priority_bypass,
                pending_directional_bypass,
            ) = (
                    self._query_one_tree(
                        pending_tree,
                        pending_points,
                        pending_scales,
                        pending_colors,
                        pending_uids,
                        pending_source_ranks,
                        pending_view_directions,
                        points,
                        colors,
                        source_ranks,
                        view_directions,
                        appearance_eligible,
                        target_sizes,
                        radii,
                    )
                )
            occupied |= pending_occupied
            nearby |= pending_nearby
            appearance_bypass |= pending_appearance_bypass
            priority_bypass |= pending_priority_bypass
            directional_bypass |= pending_directional_bypass
            pending_better = pending_parent_scores < parent_scores
            parent_uids[pending_better] = pending_parent_uids[pending_better]
            parent_scores[pending_better] = pending_parent_scores[pending_better]
        raw_directional_conflict = directional_bypass & ~occupied
        directional_bypass = self._frustum_conflict_selection(
            occupied,
            raw_directional_conflict,
            residual_scores,
            depth_confidences,
            sparse_valid,
            depths,
            frame_id,
            preselected=directional_preselected,
        )
        if self.config["frustum_conflict_budget_enabled"]:
            # Rows not selected by the fixed budget retain strict spatial ownership.
            occupied |= raw_directional_conflict & ~directional_bypass
        self.stats["query_calls"] += 1
        self.stats["query_rows"] += len(points)
        self.stats["occupied_rows"] += int(np.sum(occupied))
        self.stats["coarse_refinement_bypass_rows"] += int(
            np.sum(nearby & ~occupied)
        )
        self.stats["appearance_packet_bypass_rows"] += int(
            np.sum(appearance_bypass & ~occupied)
        )
        self.stats["priority_bypass_rows"] += int(np.sum(priority_bypass))
        if self.directional_ownership_enabled:
            self.stats["directional_query_rows"] += len(points)
            self.stats["directional_bypass_rows"] += int(
                np.sum(directional_bypass)
            )
        parent_uids[occupied] = -1
        self.stats["parent_candidate_rows"] += int(np.sum(parent_uids >= 0))
        return occupied, parent_uids

    def occupied(
        self,
        points,
        target_size,
        colors=None,
        source_ranks=None,
        appearance_eligible=None,
        view_directions=None,
        residual_scores=None,
        depth_confidences=None,
        sparse_valid=None,
        depths=None,
        frame_id=0,
        directional_preselected=False,
    ):
        occupied, _ = self.occupied_with_parents(
            points,
            target_size,
            colors,
            source_ranks,
            appearance_eligible,
            view_directions,
            residual_scores,
            depth_confidences,
            sparse_valid,
            depths,
            frame_id,
            directional_preselected,
        )
        return occupied

    def appearance_certificates(
        self,
        residual_scores,
        depth_confidences,
        sparse_valid,
        depths,
        frame_id,
    ):
        residual_scores = np.asarray(residual_scores, dtype=np.float32).reshape(-1)
        depth_confidences = np.asarray(
            depth_confidences, dtype=np.float32
        ).reshape(-1)
        sparse_valid = np.asarray(sparse_valid, dtype=np.bool_).reshape(-1)
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        if not (
            residual_scores.shape
            == depth_confidences.shape
            == sparse_valid.shape
            == depths.shape
        ):
            raise ValueError("Scale-cover appearance certificate arrays must align")
        if not self.config["appearance_certificate_enabled"]:
            return np.ones(residual_scores.shape, dtype=np.bool_)

        source_eligible = ~sparse_valid
        if self.config["appearance_allow_sparse"]:
            source_eligible = np.ones(sparse_valid.shape, dtype=np.bool_)
        eligible = (
            source_eligible
            & np.isfinite(residual_scores)
            & (
                residual_scores
                >= float(self.config["appearance_min_residual"])
            )
            & np.isfinite(depth_confidences)
            & (
                depth_confidences
                >= float(self.config["appearance_min_depth_confidence"])
            )
        )
        self.stats["appearance_certificate_calls"] += 1
        self.stats["appearance_certificate_rows"] += len(eligible)
        self.stats["appearance_certificate_eligible_rows"] += int(
            np.sum(eligible)
        )
        if self.config["shuffle_appearance_certificates"] and len(eligible) > 1:
            bands = np.digitize(
                depths,
                np.asarray(self.config["shuffle_depth_edges_m"], dtype=np.float32),
            )
            rng = np.random.default_rng(
                int(self.config["shuffle_seed"]) + int(frame_id)
            )
            shuffled = eligible.copy()
            for sparse in (False, True):
                if sparse and not self.config["appearance_allow_sparse"]:
                    continue
                for band in range(len(self.config["shuffle_depth_edges_m"]) + 1):
                    rows = np.flatnonzero(
                        (sparse_valid == sparse) & (bands == band)
                    )
                    if len(rows) > 1:
                        shuffled[rows] = eligible[
                            rows[rng.permutation(len(rows))]
                        ]
                        self.stats["shuffled_appearance_certificate_rows"] += len(
                            rows
                        )
            eligible = shuffled
        return eligible

    def candidate_source_ranks(self, sparse_valid, depths, frame_id):
        sparse_valid = np.asarray(sparse_valid, dtype=np.bool_).reshape(-1)
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        if sparse_valid.shape != depths.shape:
            raise ValueError("Scale-cover source-rank arrays must align")
        ranks = np.where(sparse_valid, 2, 1).astype(np.int8)
        if not self.config["shuffle_source_priority"] or len(ranks) <= 1:
            return ranks
        bands = np.digitize(
            depths,
            np.asarray(self.config["shuffle_depth_edges_m"], dtype=np.float32),
        )
        rng = np.random.default_rng(
            int(self.config["shuffle_seed"]) + int(frame_id)
        )
        for band in range(len(self.config["shuffle_depth_edges_m"]) + 1):
            rows = np.flatnonzero(bands == band)
            if len(rows) > 1:
                ranks[rows] = ranks[rows[rng.permutation(len(rows))]]
                self.stats["shuffled_priority_rows"] += len(rows)
        return ranks

    def shuffle_parents(self, parent_uids, depths, sparse_valid, frame_id):
        parent_uids = np.asarray(parent_uids, dtype=np.int64).reshape(-1)
        if not self.config["shuffle_handoff"] or len(parent_uids) <= 1:
            return parent_uids
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        sparse_valid = np.asarray(sparse_valid, dtype=np.bool_).reshape(-1)
        if not (parent_uids.shape == depths.shape == sparse_valid.shape):
            raise ValueError("Scale-cover parent shuffle arrays must align")
        bands = np.digitize(
            depths,
            np.asarray(self.config["shuffle_depth_edges_m"], dtype=np.float32),
        )
        rng = np.random.default_rng(
            int(self.config["shuffle_seed"]) + int(frame_id)
        )
        result = parent_uids.copy()
        for sparse in (False, True):
            for band in range(len(self.config["shuffle_depth_edges_m"]) + 1):
                rows = np.flatnonzero(
                    (parent_uids >= 0)
                    & (sparse_valid == sparse)
                    & (bands == band)
                )
                if len(rows) > 1:
                    result[rows] = parent_uids[rows[rng.permutation(len(rows))]]
                    self.stats["shuffled_parent_rows"] += len(rows)
        return result

    def release(self, uids):
        uids = np.asarray(uids, dtype=np.int64).reshape(-1)
        self.stats["release_calls"] += 1
        if self.config["shuffle_lifecycle_release"] and len(self._base_uids):
            cover_uids = np.unique(self._base_uids[self._base_uids >= 0])
            active_cover_uids = cover_uids[
                cover_uids < len(self._active_uid_table)
            ]
            active_cover_uids = active_cover_uids[
                self._active_uid_table[active_cover_uids]
            ]
            target_count = int(np.sum(np.isin(uids, active_cover_uids)))
            if target_count and len(active_cover_uids):
                rng = np.random.default_rng(
                    int(self.config["shuffle_seed"])
                    + int(self.stats["release_calls"])
                )
                uids = rng.choice(
                    active_cover_uids,
                    size=min(target_count, len(active_cover_uids)),
                    replace=False,
                )
                self.stats["shuffled_release_rows"] += len(uids)
        released = 0
        for uid in uids.tolist():
            if uid >= 0 and self._node_active.get(uid, False):
                self._node_active[uid] = False
                if uid < len(self._active_uid_table):
                    self._active_uid_table[uid] = False
                released += 1
        self.stats["released_uids"] += released
        if released:
            self.admission_epoch += 1
            self._invalidate_coverage_cache()
            if self.tracks_uids:
                self._compact_active_cover()
        return released

    def _compact_active_cover(self):
        self._flush()
        if len(self._base_uids) == 0:
            return
        valid_uid = (
            (self._base_uids >= 0)
            & (self._base_uids < len(self._active_uid_table))
        )
        keep = np.zeros((len(self._base_uids),), dtype=np.bool_)
        keep[valid_uid] = self._active_uid_table[self._base_uids[valid_uid]]
        removed = int(np.sum(~keep))
        if removed == 0:
            return
        self._base_points = self._base_points[keep]
        self._base_scales = self._base_scales[keep]
        self._base_colors = self._base_colors[keep]
        self._base_uids = self._base_uids[keep]
        self._base_source_ranks = self._base_source_ranks[keep]
        self._base_view_directions = self._base_view_directions[keep]
        self._base_points_gpu_by_device.clear()
        if (
            len(self._base_points)
            and self.config["query_backend"] == "scipy_kdtree"
        ):
            from scipy.spatial import cKDTree

            build_start = time.perf_counter()
            self._base_tree = cKDTree(self._base_points)
            self.stats["tree_build_time_ms"] += (
                time.perf_counter() - build_start
            ) * 1000.0
        else:
            self._base_tree = None
        self.stats["released_cover_rows"] += removed
        self.stats["lifecycle_rebuilds"] += 1

    def _coverage_table(self):
        if self._coverage_cache_cpu is not None:
            return self._coverage_cache_cpu
        coverage = np.zeros((self.next_uid,), dtype=np.float32)
        for uid, parent_uid in self._node_parent.items():
            if parent_uid < 0:
                continue
            if not self._node_active.get(uid, False):
                continue
            if not self._node_active.get(parent_uid, False):
                continue
            child_scale = self._node_scale.get(uid, float("nan"))
            parent_scale = self._node_scale.get(parent_uid, float("nan"))
            if not (
                np.isfinite(child_scale)
                and np.isfinite(parent_scale)
                and 0.0 < child_scale < parent_scale
                and 0 <= parent_uid < self.next_uid
            ):
                continue
            coverage[parent_uid] += float((child_scale / parent_scale) ** 2)
        self._coverage_cache_cpu = coverage
        self._coverage_cache_by_device.clear()
        return coverage

    def render_handoff_multipliers(
        self, uids, means, scales, poses, focal_pixels, *, frame_id=0
    ):
        """Return detached scale-conservative coarse-to-fine responsibility."""

        import torch

        count = int(means.shape[0])
        if not self.dynamic_handoff_enabled or count == 0:
            return means.new_ones((count,))
        uids = torch.as_tensor(uids, device=means.device, dtype=torch.long).reshape(-1)
        if uids.shape != (count,):
            raise ValueError("Scale-cover handoff UIDs must align with geometry")
        poses = torch.as_tensor(poses, device=means.device, dtype=means.dtype)
        if poses.ndim == 2:
            poses = poses.unsqueeze(0)
        focal_pixels = torch.as_tensor(
            focal_pixels, device=means.device, dtype=means.dtype
        ).reshape(-1)
        if poses.shape[0] != focal_pixels.numel():
            raise ValueError("Scale-cover poses and focal lengths must align")

        device_key = str(means.device)
        coverage_table = self._coverage_cache_by_device.get(device_key)
        if coverage_table is None:
            coverage_table = torch.from_numpy(self._coverage_table()).to(
                device=means.device, dtype=means.dtype
            )
            self._coverage_cache_by_device[device_key] = coverage_table
        coverage = torch.zeros((count,), device=means.device, dtype=means.dtype)
        valid_uid = (uids >= 0) & (uids < coverage_table.numel())
        coverage[valid_uid] = coverage_table[uids[valid_uid]]

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
            radius = projected_radius.sum(dim=0) / torch.clamp(
                visible.sum(dim=0), min=1
            )
        else:
            radius = projected_radius.amax(dim=0)
        start = float(self.config["handoff_radius_start_px"])
        end = float(self.config["handoff_radius_end_px"])
        transition = torch.clamp((radius - start) / (end - start), 0.0, 1.0)
        transition = transition * transition * (3.0 - 2.0 * transition)
        maturity = torch.clamp(
            coverage / float(self.config["handoff_area_full"]), 0.0, 1.0
        )
        floor = float(self.config["handoff_parent_floor"])
        multipliers = 1.0 - (1.0 - floor) * transition * maturity
        self.stats["handoff_render_calls"] += 1
        self.stats["handoff_faded_rows"] += int(
            torch.count_nonzero(multipliers < 1.0).item()
        )
        return multipliers.detach()

    def shuffle(self, occupied, depths, sparse_valid, frame_id, eligible=None):
        occupied = np.asarray(occupied, dtype=np.bool_).reshape(-1)
        if not self.config["shuffle_occupancy"] or len(occupied) <= 1:
            return occupied
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        sparse_valid = np.asarray(sparse_valid, dtype=np.bool_).reshape(-1)
        if not (occupied.shape == depths.shape == sparse_valid.shape):
            raise ValueError("Scale-cover shuffle arrays must align")
        if eligible is None:
            eligible = np.ones(occupied.shape, dtype=np.bool_)
        else:
            eligible = np.asarray(eligible, dtype=np.bool_).reshape(-1)
            if eligible.shape != occupied.shape:
                raise ValueError("Scale-cover shuffle eligibility must align")
        bands = np.digitize(
            depths,
            np.asarray(self.config["shuffle_depth_edges_m"], dtype=np.float32),
        )
        rng = np.random.default_rng(
            int(self.config["shuffle_seed"]) + int(frame_id)
        )
        result = occupied.copy()
        for sparse in (False, True):
            for band in range(len(self.config["shuffle_depth_edges_m"]) + 1):
                rows = np.flatnonzero(
                    eligible & (sparse_valid == sparse) & (bands == band)
                )
                if len(rows) > 1:
                    result[rows] = occupied[rows[rng.permutation(len(rows))]]
                    self.stats["shuffled_rows"] += len(rows)
        return result

    def route_evidence_quota(
        self,
        occupied,
        depths,
        sparse_valid,
        residual_scores,
        coverage_scores,
        depth_confidences,
        frame_id,
        eligible=None,
        uv=None,
        multiview_support_scores=None,
    ):
        """Route each depth band's existing birth quota to its best evidence."""

        occupied = np.asarray(occupied, dtype=np.bool_).reshape(-1)
        if not self.config["evidence_quota_routing"] or len(occupied) == 0:
            return occupied
        count = len(occupied)
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        sparse_valid = np.asarray(sparse_valid, dtype=np.bool_).reshape(-1)
        residual_scores = np.asarray(
            residual_scores, dtype=np.float32
        ).reshape(-1)
        coverage_scores = np.asarray(
            coverage_scores, dtype=np.float32
        ).reshape(-1)
        depth_confidences = np.asarray(
            depth_confidences, dtype=np.float32
        ).reshape(-1)
        if multiview_support_scores is None:
            multiview_support_scores = np.zeros((count,), dtype=np.float32)
        else:
            multiview_support_scores = np.asarray(
                multiview_support_scores, dtype=np.float32
            ).reshape(-1)
        if not (
            depths.shape
            == sparse_valid.shape
            == residual_scores.shape
            == coverage_scores.shape
            == depth_confidences.shape
            == multiview_support_scores.shape
            == (count,)
        ):
            raise ValueError("Evidence-quota routing arrays must align")
        if eligible is None:
            eligible = np.ones((count,), dtype=np.bool_)
        else:
            eligible = np.asarray(eligible, dtype=np.bool_).reshape(-1)
            if eligible.shape != (count,):
                raise ValueError("Evidence-quota eligibility must align")
        projective_cell_px = float(self.config["quota_projective_cell_px"])
        if projective_cell_px > 0.0:
            uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
            if uv.shape != (count, 2) or not np.isfinite(uv).all():
                raise ValueError(
                    "Projective evidence-quota routing requires finite aligned UV"
                )

        support_scores = np.nan_to_num(
            multiview_support_scores, nan=0.0, posinf=1.0, neginf=0.0
        )
        finite_scores = (
            float(self.config["quota_residual_weight"])
            * np.nan_to_num(residual_scores, nan=0.0, posinf=1.0, neginf=0.0)
            + float(self.config["quota_coverage_weight"])
            * np.nan_to_num(coverage_scores, nan=0.0, posinf=1.0, neginf=0.0)
            + float(self.config["quota_confidence_weight"])
            * np.nan_to_num(depth_confidences, nan=0.0, posinf=1.0, neginf=0.0)
            + float(self.config["quota_unoccupied_bonus"]) * (~occupied)
            + float(self.config["quota_multiview_support_weight"])
            * support_scores
        )
        bands = np.digitize(
            depths,
            np.asarray(self.config["shuffle_depth_edges_m"], dtype=np.float32),
        )
        result = occupied.copy()
        routed_rows = eligible & (
            depths >= float(self.config["quota_min_depth_m"])
        )
        if not self.config["quota_route_sparse"]:
            routed_rows &= ~sparse_valid
        if self.config["projective_revisit_gate"] != "disabled":
            certified = self._revisit_certificates.get(int(frame_id))
            if certified is None:
                raise RuntimeError(
                    "Revisit-gated routing requires observing the frame's raw pose"
                )
            if not certified:
                self.stats["revisit_gate_skipped_calls"] += 1
                self.stats["revisit_gate_skipped_rows"] += int(np.sum(routed_rows))
                return occupied
        rng = np.random.default_rng(
            int(self.config["shuffle_seed"]) + int(frame_id)
        )
        self.stats["evidence_quota_calls"] += 1
        self.stats["evidence_quota_rows"] += int(np.sum(routed_rows))
        support_weight = float(self.config["quota_multiview_support_weight"])
        if support_weight > 0.0:
            support_rows = routed_rows & ~sparse_valid
            self.stats["multiview_support_rows"] += int(np.sum(support_rows))
            self.stats["multiview_support_positive_rows"] += int(
                np.sum(support_rows & (support_scores > 0.0))
            )
            self.stats["multiview_support_score_sum"] += float(
                np.sum(support_scores[support_rows])
            )
        routed_sources = (False, True) if self.config["quota_route_sparse"] else (False,)
        for sparse in routed_sources:
            for band in range(len(self.config["shuffle_depth_edges_m"]) + 1):
                rows = np.flatnonzero(
                    routed_rows & (sparse_valid == sparse) & (bands == band)
                )
                if len(rows) == 0:
                    continue
                self.stats["evidence_quota_routed_band_rows"][band] += len(rows)
                original = occupied[rows].copy()
                if (
                    support_weight > 0.0
                    and self.config["shuffle_multiview_support"]
                    and not sparse
                    and len(rows) > 1
                ):
                    shuffled = support_scores[rows[rng.permutation(len(rows))]]
                    finite_scores[rows] += support_weight * (
                        shuffled - support_scores[rows]
                    )
                    self.stats["shuffled_multiview_support_rows"] += len(rows)
                keep_count = int(np.sum(~occupied[rows]))
                result[rows] = True
                if keep_count == 0:
                    continue
                if keep_count == len(rows):
                    selected = rows
                elif self.config["shuffle_evidence_quota"]:
                    selected = rng.choice(rows, size=keep_count, replace=False)
                    self.stats["shuffled_evidence_quota_rows"] += len(rows)
                elif projective_cell_px > 0.0:
                    cells = np.floor(uv[rows] / projective_cell_px).astype(np.int64)
                    cell_keys = list(map(tuple, cells.tolist()))
                    score_order = np.lexsort((rows, -finite_scores[rows]))
                    cell_counts = {}
                    cell_rank = np.empty((len(rows),), dtype=np.int64)
                    for position in score_order.tolist():
                        key = cell_keys[position]
                        cell_rank[position] = cell_counts.get(key, 0)
                        cell_counts[key] = int(cell_rank[position]) + 1
                    selected = rows[
                        np.lexsort(
                            (rows, -finite_scores[rows], cell_rank)
                        )[:keep_count]
                    ]
                    self.stats["projective_quota_rows"] += len(rows)
                else:
                    selected = rows[
                        np.lexsort((rows, -finite_scores[rows]))[:keep_count]
                    ]
                result[selected] = False
                self.stats["evidence_quota_selected_rows"] += keep_count
                self.stats["evidence_quota_reassigned_band_rows"][band] += int(
                    np.sum(result[rows] != original)
                )
        self.stats["evidence_quota_reassigned_rows"] += int(
            np.sum(result[routed_rows] != occupied[routed_rows])
        )
        return result

    def summary(self):
        result = deepcopy(self.stats)
        result["enabled"] = self.enabled
        result["query_backend"] = self.config["query_backend"]
        result["admission_epoch"] = int(self.admission_epoch)
        result["shuffle_occupancy"] = bool(self.config["shuffle_occupancy"])
        result["evidence_quota_routing"] = bool(
            self.config["evidence_quota_routing"]
        )
        result["shuffle_evidence_quota"] = bool(
            self.config["shuffle_evidence_quota"]
        )
        result["quota_multiview_support_weight"] = float(
            self.config["quota_multiview_support_weight"]
        )
        result["shuffle_multiview_support"] = bool(
            self.config["shuffle_multiview_support"]
        )
        support_rows = int(result["multiview_support_rows"])
        result["multiview_support_score_mean"] = (
            float(result["multiview_support_score_sum"]) / support_rows
            if support_rows
            else None
        )
        result["quota_route_sparse"] = bool(self.config["quota_route_sparse"])
        result["quota_projective_cell_px"] = float(
            self.config["quota_projective_cell_px"]
        )
        result["quota_min_depth_m"] = float(self.config["quota_min_depth_m"])
        result["projective_revisit_gate"] = self.config[
            "projective_revisit_gate"
        ]
        result["revisit_min_frame_gap"] = int(
            self.config["revisit_min_frame_gap"]
        )
        result["revisit_max_position_distance_m"] = float(
            self.config["revisit_max_position_distance_m"]
        )
        result["revisit_max_view_angle_deg"] = float(
            self.config["revisit_max_view_angle_deg"]
        )
        result["revisit_min_cumulative_turn_deg"] = float(
            self.config["revisit_min_cumulative_turn_deg"]
        )
        result["revisit_fixed_start_frame"] = int(
            self.config["revisit_fixed_start_frame"]
        )
        result["revisit_pose_history_frames"] = len(self._revisit_frame_ids)
        result["revisit_cumulative_turn_deg"] = (
            float(self._revisit_cumulative_turns[-1])
            if self._revisit_cumulative_turns
            else 0.0
        )
        certified = int(result["revisit_certificate_certified_frames"])
        result["revisit_support_distance_mean_m"] = (
            float(result["revisit_support_distance_sum_m"]) / certified
            if certified and self.config["projective_revisit_gate"] == "pose"
            else None
        )
        result["revisit_support_angle_mean_deg"] = (
            float(result["revisit_support_angle_sum_deg"]) / certified
            if certified and self.config["projective_revisit_gate"] == "pose"
            else None
        )
        result["revisit_support_turn_mean_deg"] = (
            float(result["revisit_support_turn_sum_deg"]) / certified
            if certified and self.config["projective_revisit_gate"] == "pose"
            else None
        )
        result["dynamic_handoff_enabled"] = self.dynamic_handoff_enabled
        result["active_occupancy_enabled"] = bool(
            self.config["active_occupancy_enabled"]
        )
        result["shuffle_lifecycle_release"] = bool(
            self.config["shuffle_lifecycle_release"]
        )
        result["source_priority_enabled"] = bool(
            self.config["source_priority_enabled"]
        )
        result["appearance_certificate_enabled"] = bool(
            self.config["appearance_certificate_enabled"]
        )
        result["shuffle_appearance_certificates"] = bool(
            self.config["shuffle_appearance_certificates"]
        )
        result["shuffle_source_priority"] = bool(
            self.config["shuffle_source_priority"]
        )
        result["shuffle_handoff"] = bool(self.config["shuffle_handoff"])
        result["projected_handoff_enabled"] = self.projected_handoff_enabled
        result["shuffle_projected_handoff"] = bool(
            self.config["shuffle_projected_handoff"]
        )
        result["projected_handoff_pending_rows"] = int(len(self._far_points))
        result["sparse_track_identity_enabled"] = (
            self.sparse_track_identity_enabled
        )
        result["shuffle_sparse_track_identity"] = bool(
            self.config["shuffle_sparse_track_identity"]
        )
        result["active_sparse_tracks"] = int(len(self._committed_sparse_tracks))
        result["directional_ownership_enabled"] = (
            self.directional_ownership_enabled
        )
        result["directional_max_angle_deg"] = float(
            self.config["directional_max_angle_deg"]
        )
        result["shuffle_directional_ownership"] = bool(
            self.config["shuffle_directional_ownership"]
        )
        result["active_cover_rows"] = int(
            len(self._base_points) + self._pending_rows
        )
        result["hash_calls_zero"] = (
            result["hash_query_rows"] == 0 and result["hash_set_rows"] == 0
        )
        coverage = self._coverage_table() if self.dynamic_handoff_enabled else np.zeros(0)
        result["active_uids"] = int(sum(self._node_active.values()))
        result["handoff_parent_rows"] = int(np.count_nonzero(coverage > 0.0))
        result["handoff_area_sum"] = float(np.sum(coverage))
        return result
