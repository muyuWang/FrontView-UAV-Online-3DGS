"""Fail-closed canonical world-frame contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorldFrameContract:
    dataset_path: str
    world_frame_id: str
    geometry_mode: str
    calibration_version: str
    pose_source: str
    depth_source: str
    sparse_world_geometry: str
    permanent_birth_valid: bool
    failure_reason: str = ""

    @classmethod
    def from_dataset(cls, dataset_path, calibration_version="panoair_v1"):
        root = Path(dataset_path).resolve()
        stats_path = root / "conversion_stats.json"
        if not stats_path.is_file():
            return cls(
                str(root),
                "invalid_unknown_world",
                "invalid",
                calibration_version,
                "unknown",
                "unknown",
                "unknown",
                False,
                "conversion_stats.json is missing",
            )
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        pose_source = str(stats.get("pose_source", "unknown"))
        pose_source_kind = str(stats.get("pose_source_kind", pose_source)).lower()
        geometry = str(stats.get("sparse_world_geometry", "unknown"))
        method = str(stats.get("method", ""))
        if pose_source == "colmap" and geometry == "persistent":
            return cls(
                str(root),
                "panoair_seq1_colmap_sim3_v1",
                "colmap_canonical",
                calibration_version,
                pose_source,
                "colmap_track_world_point",
                geometry,
                True,
            )
        if pose_source == "gt" and geometry == "persistent" and "triangulation" in method:
            return cls(
                str(root),
                "panoair_seq1_rtk_triangulated_v1",
                "rtk_canonical",
                calibration_version,
                pose_source,
                "rtk_pose_orb_triangulation",
                geometry,
                True,
            )
        coordinate_contract = str(stats.get("coordinate_contract", "")).lower()
        if (
            pose_source == "orbslam3_vi"
            and geometry == "persistent"
            and "one sim(3)" in coordinate_contract
            and "poses" in coordinate_contract
            and "points" in coordinate_contract
        ):
            return cls(
                str(root),
                "panoair_seq1_orbslam3_vi_sim3_v1",
                "visual_inertial_canonical",
                calibration_version,
                pose_source,
                "orbslam3_vi_track_world_point",
                geometry,
                True,
            )
        if (
            pose_source_kind == "fast_livo2"
            and geometry == "persistent"
            and stats.get("point_coordinate_system") == "normalized world"
        ):
            scene = str(stats.get("scene", root.name)).lower().replace(" ", "_")
            return cls(
                str(root),
                f"lvba_{scene}_fast_livo2_v1",
                "lidar_canonical",
                calibration_version,
                pose_source,
                "fast_livo2_lidar_world_point",
                geometry,
                True,
            )
        return cls(
            str(root),
            "invalid_frame_local_reprojection",
            "hybrid_frame_local",
            calibration_version,
            pose_source,
            "colmap_camera_depth_reprojected_by_rtk_pose",
            geometry,
            False,
            "pose and sparse depth do not share a persistent world identity",
        )

    @property
    def fingerprint(self):
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]

    def require_permanent_birth(self, allow_invalid_stress=False):
        if self.permanent_birth_valid:
            return
        if allow_invalid_stress:
            return
        raise RuntimeError(
            "World-frame contract rejects permanent birth: {}".format(
                self.failure_reason
            )
        )
