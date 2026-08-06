"""Online all-path shadow admission controller and falsification controls."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from utils_new.aerocommit.types import AeroCommitFrameStats, GaussianProposalBatch

from .evidence import WorldIdentityEvidence
from .nuisance import NuisanceState, SharedNuisanceSolver
from .shadow import ShadowGroup, ShadowObservation


class OfflineSparseTrackCache:
    """Cache the same world test for sparse tracks already reconstructed offline."""

    def __init__(self, dataset_path, config):
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.entries = {}
        self.poses = []
        self.intrinsics = []
        self._build()

    def _build(self):
        trajectory_path = self.dataset_path / "trajectory.json"
        sidecar_dir = self.dataset_path / "orb_point_ids"
        if not trajectory_path.is_file() or not sidecar_dir.is_dir():
            return
        cameras = json.loads(trajectory_path.read_text(encoding="utf-8"))["cameras"]
        count = min(int(self.config["offline_cache_frames"]), len(cameras))
        centers = []
        for frame_id, camera in enumerate(cameras[:count]):
            pose = np.asarray(camera["T_camera_world"], dtype=np.float64)
            intrinsics = np.asarray(
                [
                    [camera["intrinsic"]["fx"], 0.0, camera["intrinsic"]["cx"]],
                    [0.0, camera["intrinsic"]["fy"], camera["intrinsic"]["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            self.poses.append(pose)
            self.intrinsics.append(intrinsics)
            centers.append(np.linalg.inv(pose)[:3, 3])
            ids_path = sidecar_dir / f"point_ids_{frame_id}.npy"
            points_path = (
                self.dataset_path
                / "orb_point_clouds"
                / f"point_cloud_{frame_id}.npy"
            )
            if not ids_path.is_file() or not points_path.is_file():
                continue
            identities = np.load(ids_path).astype(np.int64)
            points = np.load(points_path).astype(np.float32)
            if identities.shape != (len(points),):
                raise ValueError("Offline sparse identity cache has mismatched rows")
            center = centers[-1]
            for track_id, point in zip(identities.tolist(), points):
                entry = self.entries.get(track_id)
                item = (frame_id, point.copy())
                if entry is None:
                    self.entries[track_id] = {
                        "first": item,
                        "first_center": center.copy(),
                        "farthest": [],
                    }
                    continue
                distance = float(np.linalg.norm(center - entry["first_center"]))
                candidates = entry["farthest"] + [(distance, item)]
                candidates.sort(key=lambda candidate: candidate[0], reverse=True)
                unique = []
                seen_frames = set()
                for candidate in candidates:
                    if candidate[1][0] in seen_frames:
                        continue
                    seen_frames.add(candidate[1][0])
                    unique.append(candidate)
                    if len(unique) == 2:
                        break
                entry["farthest"] = unique

    def observations(self, track_id, source_kind):
        entry = self.entries.get(int(track_id))
        if entry is None or len(entry["farthest"]) < 2:
            return []
        selected = [entry["first"]] + [item for _, item in entry["farthest"]]
        selected.sort(key=lambda item: item[0])
        observations = []
        for frame_id, point in selected:
            pose = self.poses[frame_id]
            intrinsics = self.intrinsics[frame_id]
            camera = pose @ np.append(point, 1.0)
            if not np.isfinite(camera).all() or camera[2] <= 1.0e-8:
                return []
            screen = intrinsics @ camera[:3]
            observations.append(
                ShadowObservation(
                    frame_id=int(frame_id),
                    uv=(screen[:2] / screen[2]).astype(np.float64),
                    inverse_depth=float(1.0 / camera[2]),
                    inverse_depth_variance=max(
                        float(self.config["inverse_depth_sigma_floor"]) ** 2,
                        float(0.01 / camera[2]) ** 2,
                    ),
                    pose_id=int(frame_id),
                    world_to_camera=pose,
                    pose_covariance=np.diag(
                        [float(self.config["pose_translation_sigma_m"]) ** 2] * 3
                        + [
                            math.radians(
                                float(self.config["pose_rotation_sigma_deg"])
                            )
                            ** 2
                        ]
                        * 3
                    ),
                    source_kind=str(source_kind),
                    track_confidence=1.0,
                    rgb=np.zeros((3,), dtype=np.float64),
                    world_point=np.asarray(point, dtype=np.float64),
                    intrinsics=intrinsics,
                )
            )
        return observations


class UntrackedFlowAssociator:
    def __init__(self, cycle_threshold_px=1.5):
        self.cycle_threshold_px = float(cycle_threshold_px)
        self.previous_gray = None
        self.previous_uv = np.empty((0, 2), dtype=np.float32)
        self.previous_ids = np.empty((0,), dtype=np.int64)
        self.next_id = -2

    def assign(self, image, uv):
        gray = np.asarray(image, dtype=np.float32).mean(axis=2)
        gray = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
        uv = np.asarray(uv, dtype=np.float32)
        assigned = np.full((len(uv),), -1, dtype=np.int64)
        if self.previous_gray is not None and len(self.previous_uv) and len(uv):
            forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                self.previous_uv.reshape(-1, 1, 2),
                None,
                winSize=(21, 21),
                maxLevel=3,
            )
            backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                self.previous_gray,
                forward,
                None,
                winSize=(21, 21),
                maxLevel=3,
            )
            cycle = np.linalg.norm(
                backward.reshape(-1, 2) - self.previous_uv, axis=1
            )
            valid = (
                status_forward.reshape(-1).astype(bool)
                & status_backward.reshape(-1).astype(bool)
                & np.isfinite(forward.reshape(-1, 2)).all(axis=1)
                & (cycle <= self.cycle_threshold_px)
            )
            if np.any(valid):
                from scipy.spatial import cKDTree

                predicted = forward.reshape(-1, 2)[valid]
                distances, indices = cKDTree(uv).query(predicted, k=1)
                order = np.argsort(distances)
                claimed = set()
                valid_ids = self.previous_ids[valid]
                for position in order:
                    current_index = int(indices[position])
                    if distances[position] > 2.0 or current_index in claimed:
                        continue
                    claimed.add(current_index)
                    assigned[current_index] = int(valid_ids[position])
        for index in np.flatnonzero(assigned == -1):
            assigned[index] = self.next_id
            self.next_id -= 1
        self.previous_gray = gray
        self.previous_uv = uv.copy()
        self.previous_ids = assigned.copy()
        return assigned


class WorldTestController:
    def __init__(self, config, contract, authority, gaussian_model, output_dir):
        self.config = config
        self.contract = contract
        self.authority = authority
        self.gaussian_model = gaussian_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.groups = {}
        self.next_group_id = 0
        self.committed_tracks = set()
        self.rejected_tracks = set()
        self.committed_records = {}
        self.solver = SharedNuisanceSolver(config)
        self.evaluator = WorldIdentityEvidence(config)
        self.flow = UntrackedFlowAssociator(config["flow_cycle_threshold_px"])
        self.rng = np.random.default_rng(int(config["random_seed"]))
        self.frame_records = []
        self.commit_schedule = {"schema_version": 1, "frames": {}}
        self.reference_schedule = self._load_schedule(config.get("schedule_path"))
        self.telemetry_path = self.output_dir / "worldtest_telemetry.jsonl"
        self.cached_nuisance = None
        self.offline_nuisance = NuisanceState(
            True,
            "",
            0,
            int(config["nuisance_knot_stride"]),
            np.empty((0, 6), dtype=np.float64),
            0.0,
            0.0,
            np.diag(
                [
                    float(config["pose_translation_sigma_m"]) ** 2,
                    math.radians(float(config["pose_rotation_sigma_deg"])) ** 2,
                    float(config["inverse_depth_sigma_floor"]) ** 2,
                ]
            ),
            1.0,
            0,
        )
        self.offline_cache = (
            OfflineSparseTrackCache(contract.dataset_path, config)
            if bool(config["offline_sparse_track_cache"])
            else None
        )

    @staticmethod
    def _load_schedule(path):
        if path is None:
            return None
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "frames" not in payload:
            raise ValueError("WorldTest schedule has no frames mapping")
        return payload

    def _provenance(self, source_frame_id, track_id, source_kind):
        return {
            "world_frame_id": self.contract.world_frame_id,
            "geometry_mode": self.contract.geometry_mode,
            "calibration_version": self.contract.calibration_version,
            "source_frame_id": int(source_frame_id),
            "track_id": int(track_id),
            "pose_source": self.contract.pose_source,
            "depth_source": self.contract.depth_source,
            "source_kind": str(source_kind),
        }

    def _assign_missing_tracks(self, proposals, cam):
        missing = proposals.track_ids < 0
        if not np.any(missing):
            return
        image = cam.get_gt_image(proposals.level).detach().cpu().numpy()
        proposals.track_ids[missing] = self.flow.assign(image, proposals.uv[missing])

    def _observation(self, proposals, index, cam):
        inverse_depth = float(proposals.inverse_depths[index])
        inverse_depth_variance = max(
            float(self.config["inverse_depth_sigma_floor"]) ** 2,
            (0.01 * inverse_depth) ** 2,
        )
        pose_sigma_t = float(self.config["pose_translation_sigma_m"])
        pose_sigma_r = math.radians(float(self.config["pose_rotation_sigma_deg"]))
        pose_covariance = np.diag(
            [pose_sigma_t**2] * 3 + [pose_sigma_r**2] * 3
        )
        return ShadowObservation(
            frame_id=int(cam.cam_idx),
            uv=np.asarray(proposals.uv[index], dtype=np.float64),
            inverse_depth=inverse_depth,
            inverse_depth_variance=inverse_depth_variance,
            pose_id=int(cam.cam_idx),
            world_to_camera=cam.get_raw_pose().detach().cpu().numpy().astype(np.float64),
            pose_covariance=pose_covariance,
            source_kind=str(proposals.source_kinds[index]),
            track_confidence=float(proposals.depth_confidences[index]),
            rgb=np.asarray(proposals.colors[index], dtype=np.float64),
            world_point=np.asarray(proposals.world_points[index], dtype=np.float64),
            intrinsics=cam.get_int_mat(proposals.level).detach().cpu().numpy().astype(np.float64),
        )

    def _observe(self, proposals, cam):
        self._assign_missing_tracks(proposals, cam)
        proposed = 0
        for index in range(len(proposals)):
            track_id = int(proposals.track_ids[index])
            source_kind = str(proposals.source_kinds[index])
            key = (source_kind, track_id)
            if key in self.committed_tracks:
                record = self.committed_records[key]
                pose = cam.get_raw_pose().detach().cpu().numpy()
                intrinsics = cam.get_int_mat(proposals.level).detach().cpu().numpy()
                camera_point = pose @ np.append(record["world_point"], 1.0)
                if np.isfinite(camera_point).all() and camera_point[2] > 1.0e-8:
                    screen = intrinsics @ camera_point[:3]
                    predicted_uv = screen[:2] / screen[2]
                    pixel_residual = np.linalg.norm(
                        predicted_uv - proposals.uv[index]
                    ) / float(self.config["pixel_sigma"])
                    predicted_rho = 1.0 / camera_point[2]
                    rho_sigma = max(
                        float(self.config["inverse_depth_sigma_floor"]),
                        0.01 * float(proposals.inverse_depths[index]),
                    )
                    depth_residual = abs(
                        predicted_rho - float(proposals.inverse_depths[index])
                    ) / rho_sigma
                    record["future_residuals"].append(
                        float(max(pixel_residual, depth_residual))
                    )
                continue
            if key in self.rejected_tracks:
                continue
            batch = proposals.select([index])
            batch.metadata.update(self._provenance(batch.source_frame_id, track_id, source_kind))
            group = self.groups.get(key)
            if group is None:
                if len(self.groups) >= int(self.config["max_groups"]):
                    continue
                group = ShadowGroup(
                    group_id=self.next_group_id,
                    track_id=track_id,
                    source_kind=source_kind,
                    created_frame=int(cam.cam_idx),
                    proposal_batch=batch,
                )
                self.next_group_id += 1
                self.groups[key] = group
                if self.offline_cache is not None and source_kind == "sparse":
                    cached_observations = self.offline_cache.observations(
                        track_id, source_kind
                    )
                    for cached in cached_observations:
                        group.add(cached, batch, max_views=self.config["max_views"])
                    group.offline_cached = len(cached_observations) >= int(
                        self.config["min_views"]
                    )
            if group.offline_cached:
                group.proposal_batch = batch
                group.last_frame = int(cam.cam_idx)
            else:
                group.add(
                    self._observation(proposals, index, cam),
                    batch,
                    max_views=self.config["max_views"],
                )
            proposed += 1
        return proposed

    def observe_external_geometry(
        self,
        cam,
        world_points,
        colors,
        log_scales,
        source_kind,
        track_ids=None,
    ):
        """Route detail/flow/surface births into shadow state without mutation."""
        points = np.asarray(world_points, dtype=np.float32)
        if points.size == 0:
            return 0
        colors = np.asarray(colors, dtype=np.float32)
        log_scales = np.asarray(log_scales, dtype=np.float32)
        pose = cam.get_raw_pose().detach().cpu().numpy()
        intrinsics = cam.get_int_mat(0).detach().cpu().numpy()
        homogeneous = np.concatenate(
            [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
        )
        camera = homogeneous @ pose.T
        screen = camera[:, :3] @ intrinsics.T
        uv = screen[:, :2] / np.maximum(screen[:, 2:3], 1.0e-8)
        depths = camera[:, 2]
        valid = (
            np.isfinite(uv).all(axis=1)
            & np.isfinite(depths)
            & (depths > float(cam.near))
            & (depths < float(cam.far))
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < cam.get_width(0))
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < cam.get_height(0))
        )
        points, colors, log_scales = points[valid], colors[valid], log_scales[valid]
        uv, depths = uv[valid], depths[valid]
        if not len(points):
            return 0
        if track_ids is None:
            selected_track_ids = np.full((len(points),), -1, dtype=np.int64)
        else:
            selected_track_ids = np.asarray(track_ids, dtype=np.int64)[valid]
        half_patch = 4.0
        batch = GaussianProposalBatch(
            source_frame_id=int(cam.cam_idx),
            level=0,
            uv=uv.astype(np.float32),
            patch_bboxes=np.stack(
                [
                    uv[:, 0] - half_patch,
                    uv[:, 1] - half_patch,
                    uv[:, 0] + half_patch,
                    uv[:, 1] + half_patch,
                ],
                axis=1,
            ).astype(np.float32),
            depths=depths.astype(np.float32),
            inverse_depths=(1.0 / depths).astype(np.float32),
            world_points=points,
            log_scales=log_scales,
            colors=colors,
            residual_scores=np.zeros((len(points),), dtype=np.float32),
            coverage_scores=np.ones((len(points),), dtype=np.float32),
            sparse_depth_valid=np.zeros((len(points),), dtype=np.bool_),
            track_ids=selected_track_ids,
            source_kinds=np.full((len(points),), str(source_kind), dtype="U32"),
            view_scale_size=float(cam.get_view_size(0) * self.gaussian_model.camera_scale_rescalar),
        )
        return self._observe(batch, cam)

    def _expire(self, frame_id):
        expired = []
        maximum_age = int(self.config["candidate_max_age"])
        for key, group in self.groups.items():
            if int(frame_id) - group.last_frame > maximum_age:
                expired.append(key)
        for key in expired:
            del self.groups[key]
        return len(expired)

    def shadow_external_splats(self, max_splats=2000):
        """Build detached renderer-only splats from shadows visible this frame."""
        groups = sorted(
            self.groups.values(), key=lambda group: group.last_frame, reverse=True
        )[: int(max_splats)]
        batches = [group.proposal_batch for group in groups if group.proposal_batch is not None]
        if not batches:
            return None
        means = np.concatenate([batch.world_points for batch in batches], axis=0)
        log_scales = np.concatenate([batch.log_scales for batch in batches], axis=0)
        colors = np.concatenate([batch.colors for batch in batches], axis=0)
        if log_scales.shape[1] == 1:
            log_scales = np.repeat(log_scales, 3, axis=1)
        sh_count = (int(self.gaussian_model.max_sh_degree) + 1) ** 2
        sh = np.zeros((len(means), sh_count, 3), dtype=np.float32)
        sh[:, 0, :] = (colors - 0.5) / 0.28209479177387814
        quaternions = np.zeros((len(means), 4), dtype=np.float32)
        quaternions[:, 0] = 1.0
        return {
            "means": torch.from_numpy(means).detach(),
            "scales": torch.from_numpy(np.exp(log_scales)).detach(),
            "quats": torch.from_numpy(quaternions).detach(),
            "opacities": torch.full((len(means),), 0.02).detach(),
            "shs": torch.from_numpy(sh).detach(),
        }

    @staticmethod
    def _write_rgb(path, tensor):
        rgb = np.clip(tensor.detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(
            np.uint8
        )
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def save_shadow_debug(self, cam):
        if not bool(self.config["debug"]):
            return
        frame_id = int(cam.cam_idx)
        if frame_id % 10 != 0:
            return
        external = self.shadow_external_splats()
        if external is None:
            return
        debug_dir = self.output_dir / "worldtest_debug"
        debug_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            permanent = self.gaussian_model.render(cam, level=0)
            combined = self.gaussian_model.render(
                cam, level=0, external_splats=external
            )
            shadow_alpha = torch.clamp(
                combined["opacity"] - permanent["opacity"], min=0.0
            )
            cap = float(self.config["shadow_alpha_cap"])
            scale = torch.clamp(cap / torch.clamp(shadow_alpha, min=1.0e-8), max=1.0)
            assisted = permanent["render"] + (
                combined["render"] - permanent["render"]
            ) * scale
            assisted = torch.clamp(assisted, 0.0, 1.0)
            mask = torch.clamp(shadow_alpha, max=cap) / cap
        self._write_rgb(
            debug_dir / f"{frame_id:06d}_permanent.png", permanent["render"]
        )
        self._write_rgb(
            debug_dir / f"{frame_id:06d}_shadow_assisted.png", assisted
        )
        mask_image = np.rint(
            np.clip(mask.squeeze(-1).detach().cpu().numpy(), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        cv2.imwrite(str(debug_dir / f"{frame_id:06d}_shadow_mask.png"), mask_image)

    def _ready(self):
        minimum = int(self.config["min_views"])
        true_qg = self.config["admission_mode"] == "true_qg"
        ready = [
            group for group in self.groups.values()
            if group.distinct_view_count >= minimum
            and (
                not true_qg
                or not group.offline_cached
                or group.cached_result is None
            )
        ]
        ready.sort(key=lambda group: (-group.age, group.group_id))
        return ready[: int(self.config["max_evaluations_per_frame"])]

    def _target_items(self, frame_id):
        if self.reference_schedule is None:
            raise RuntimeError(
                "Control mode requires a true-q_g schedule via WorldTestGS.schedule_path"
            )
        return list(self.reference_schedule["frames"].get(str(int(frame_id)), []))

    @staticmethod
    def _bucket(group):
        return (int(group.age), str(group.source_kind))

    def _select_control(self, ready, evaluated, frame_id):
        mode = self.config["admission_mode"]
        targets = self._target_items(frame_id)
        count = len(targets)
        if count == 0:
            return []
        if len(ready) < count:
            raise RuntimeError(
                "Control cannot match true q_g count at frame {}: {} ready < {} required".format(
                    frame_id, len(ready), count
                )
            )
        if mode == "equal_count_random":
            indices = self.rng.choice(len(ready), size=count, replace=False)
            return [ready[int(index)] for index in indices]
        if mode == "npo_lite":
            return sorted(ready, key=self.evaluator.npo_lite_score)[:count]
        remaining = list(ready)
        selected = []
        for target in targets:
            age = int(target["age"])
            source_kind = str(target["source_kind"])
            pool = [group for group in remaining if group.source_kind == source_kind]
            if not pool:
                pool = remaining
            if mode == "matched_delay":
                chosen = min(pool, key=lambda group: (abs(group.age - age), group.group_id))
            elif mode == "shuffled_qg":
                same_bucket = [group for group in pool if group.age == age]
                if not same_bucket:
                    same_bucket = pool
                scores = np.asarray([evaluated[group.group_id].q_g for group in same_bucket])
                shuffled = self.rng.permutation(scores)
                chosen = same_bucket[int(np.argmax(shuffled))]
            else:
                raise ValueError("Unknown control mode")
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    def process(self, cam, proposals, is_key_frame=True, proposal_ms=0.0):
        start = time.perf_counter()
        frame_id = int(cam.cam_idx)
        stats = AeroCommitFrameStats(frame_id=frame_id)
        stats.proposal_ms = float(proposal_ms)
        stats.num_raw_proposals = len(proposals)
        if not is_key_frame:
            stats.num_expired = self._expire(frame_id)
            return stats
        stats.num_new_candidates = self._observe(proposals, cam)
        ready = self._ready()
        stats.num_waiting_candidates = len(self.groups)
        nuisance_start = time.perf_counter()
        update_interval = int(self.config["nuisance_update_interval"])
        if self.cached_nuisance is None or frame_id % update_interval == 0:
            self.cached_nuisance = self.solver.solve(
                [
                    group
                    for group in self.groups.values()
                    if not group.offline_cached
                ],
                frame_id,
            )
        nuisance = self.cached_nuisance
        evaluated = {}
        for group in ready:
            if group.offline_cached and group.cached_result is not None:
                result = group.cached_result
            else:
                result = self.evaluator.evaluate(
                    group,
                    self.offline_nuisance if group.offline_cached else nuisance,
                )
            if group.offline_cached and group.cached_result is None:
                group.cached_result = result
            group.q_g = result.q_g
            group.evidence = {
                "worst_heldout_frame": result.worst_heldout_frame,
                "worst_prior_scale": result.worst_prior_scale,
                "rank_ratio": result.rank_ratio,
                "failure_reason": result.failure_reason,
            }
            evaluated[group.group_id] = result
        stats.num_risk_evaluations = len(evaluated)
        stats.risk_gate_ms = (time.perf_counter() - nuisance_start) * 1000.0
        mode = self.config["admission_mode"]
        if mode == "true_qg":
            selected = [group for group in ready if evaluated[group.group_id].passed]
            selected.sort(key=lambda group: group.q_g, reverse=True)
            selected = selected[: int(self.config["max_commits_per_frame"])]
        else:
            selected = self._select_control(ready, evaluated, frame_id)
        rejected_offline_keys = []
        if mode == "true_qg":
            selected_ids = {group.group_id for group in selected}
            for group in ready:
                if (
                    group.offline_cached
                    and group.group_id not in selected_ids
                    and not evaluated[group.group_id].passed
                ):
                    key = (group.source_kind, group.track_id)
                    rejected_offline_keys.append(key)
                    self.rejected_tracks.add(key)
        frame_schedule = []
        committed_keys = []
        q_values = []
        certificates = []
        for group in selected:
            batch = group.proposal_batch
            result = evaluated.get(group.group_id)
            q_g = result.q_g if result is not None else float("-inf")
            certificate = self.authority.issue(
                source_frame_id=batch.source_frame_id,
                track_id=group.track_id,
                source_kind=group.source_kind,
                issued_frame_id=frame_id,
                observation_frame_ids=[item.frame_id for item in group.observations],
                q_g=q_g,
                evidence_mode=mode,
            )
            certificates.append(certificate)
        if selected:
            commit = self.gaussian_model.commit_certified_proposals(
                [group.proposal_batch for group in selected],
                certificates,
                initial_opacity=0.25,
            )
            if (
                commit.group_id is not None
                and bool(self.config["freeze_committed_means"])
            ):
                self.gaussian_model.freeze_group_positions(commit.group_id)
            elif (
                commit.group_id is not None
                and self.config["committed_mean_trust_radius_m"] is not None
            ):
                self.gaussian_model.bound_group_positions(
                    commit.group_id,
                    float(self.config["committed_mean_trust_radius_m"]),
                )
            committed_positions = set(int(index) for index in commit.committed_indices)
        else:
            committed_positions = set()
        for position, group in enumerate(selected):
            if position in committed_positions:
                batch = group.proposal_batch
                result = evaluated.get(group.group_id)
                q_g = result.q_g if result is not None else float("-inf")
                key = (group.source_kind, group.track_id)
                committed_keys.append(key)
                self.committed_tracks.add(key)
                self.committed_records[key] = {
                    "world_point": np.asarray(
                        batch.world_points[0],
                        dtype=np.float64,
                    ),
                    "future_residuals": [],
                    "committed_frame": frame_id,
                }
                stats.num_committed_candidates += 1
                stats.num_committed_gaussians += 1
                q_values.append(q_g)
                frame_schedule.append(
                    {
                        "track_id": int(group.track_id),
                        "source_kind": group.source_kind,
                        "age": int(group.age),
                        "q_g": float(q_g),
                    }
                )
        for key in committed_keys:
            del self.groups[key]
        for key in rejected_offline_keys:
            self.groups.pop(key, None)
        stats.num_expired = self._expire(frame_id)
        stats.num_active_gaussians = int(self.gaussian_model.get_num_gaussians)
        stats.num_trainable_gaussians = stats.num_active_gaussians
        if q_values:
            finite = np.asarray([value for value in q_values if math.isfinite(value)])
            if finite.size:
                stats.risk_min = float(finite.min())
                stats.risk_median = float(np.median(finite))
                stats.risk_p95 = float(np.percentile(finite, 95))
                stats.risk_max = float(finite.max())
        stats.frame_total_ms = (time.perf_counter() - start) * 1000.0
        self.commit_schedule["frames"][str(frame_id)] = frame_schedule
        ready_q = np.asarray(
            [group.q_g for group in ready if math.isfinite(group.q_g)],
            dtype=np.float64,
        )
        q_summary = {
            "finite_count": int(ready_q.size),
            "min": float(ready_q.min()) if ready_q.size else None,
            "median": float(np.median(ready_q)) if ready_q.size else None,
            "p95": float(np.percentile(ready_q, 95)) if ready_q.size else None,
            "max": float(ready_q.max()) if ready_q.size else None,
        }
        failure_counts = {}
        source_counts = {}
        for group in ready:
            source_counts[group.source_kind] = source_counts.get(group.source_kind, 0) + 1
            reason = str(group.evidence.get("failure_reason", ""))
            if reason:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
        record = {
            "frame_id": frame_id,
            "mode": mode,
            "proposed": stats.num_raw_proposals,
            "shadow": len(self.groups),
            "evaluated": stats.num_risk_evaluations,
            "committed": stats.num_committed_gaussians,
            "expired": stats.num_expired,
            "frame_total_ms": stats.frame_total_ms,
            "q_g_summary": q_summary,
            "evaluated_by_source": source_counts,
            "failure_reasons": failure_counts,
            "nuisance_valid": nuisance.valid,
            "nuisance_rank_ratio": nuisance.undamped_rank_ratio,
            "nuisance_failure": nuisance.failure_reason,
            "offline_cached_evaluations": int(
                sum(group.offline_cached for group in ready)
            ),
            "authority_bypass_count": self.authority.bypass_count,
            "commit_api_calls": int(bool(selected)),
        }
        self.frame_records.append(record)
        with self.telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return stats

    def finalize(self):
        schedule_path = self.output_dir / "worldtest_commit_schedule.json"
        schedule_path.write_text(
            json.dumps(self.commit_schedule, indent=2), encoding="utf-8"
        )
        committed = sum(
            len(items) for items in self.commit_schedule["frames"].values()
        )
        future_records = [
            record
            for record in self.committed_records.values()
            if len(record["future_residuals"]) >= 2
        ]
        invalid = [
            record
            for record in future_records
            if float(np.median(record["future_residuals"])) > 3.0
        ]
        latencies = np.asarray(
            [record.get("frame_total_ms", 0.0) for record in self.frame_records],
            dtype=np.float64,
        )
        summary = {
            "mode": self.config["admission_mode"],
            "frames": len(self.frame_records),
            "committed_groups": committed,
            "remaining_shadow_groups": len(self.groups),
            "shadow_bytes": sum(group.byte_size() for group in self.groups.values()),
            "schedule_path": str(schedule_path),
            "future_view_evaluated_commits": len(future_records),
            "future_view_invalid_commits": len(invalid),
            "future_view_invalid_commit_rate": (
                float(len(invalid) / len(future_records)) if future_records else None
            ),
            "keyframe_latency_ms": {
                "mean": float(latencies.mean()) if latencies.size else 0.0,
                "p95": float(np.percentile(latencies, 95)) if latencies.size else 0.0,
                "max": float(latencies.max()) if latencies.size else 0.0,
            },
            "certificate_authority": self.authority.summary(),
        }
        (self.output_dir / "worldtest_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
