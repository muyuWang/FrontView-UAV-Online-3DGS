import types

import numpy as np
import torch

from utils_new.aerocommit.types import GaussianProposalBatch
from utils_new.gaussian_models import GaussianModel, Gaussians
from utils_new.frontview_birth import (
    TrackResponsibilityLedger,
    validate_front_view_birth_config,
)
from utils_new.frontview_identity_lod import (
    FrontViewIdentityLOD,
    validate_front_view_identity_lod_config,
)
from utils_new.frontview_residual_cover import (
    FrontViewResidualCover,
    validate_front_view_residual_cover_config,
)
from utils_new.frontview_scale_cover import (
    FrontViewScaleCover,
    validate_front_view_scale_cover_config,
)
from utils_new.hash_utils import HashBlock
from utils_new.worldtest_gs.certificate import CertificateAuthority
from utils_new.worldtest_gs.contract import WorldFrameContract


class RecordingHashBlock:
    def __init__(self):
        self.query_calls = 0
        self.set_calls = 0
        self.occupied_points = []

    def getOccupy(self, coords, colors, target_size=None):
        self.query_calls += 1
        return np.zeros((len(coords),), dtype=np.bool_)

    def get_no_conflict_index(self, coords, target_size=None):
        return np.ones((len(coords),), dtype=np.bool_)

    def get_sequential_no_conflict_index(self, coords, colors, target_size=None):
        return np.ones((len(coords),), dtype=np.bool_)

    def setOccupy(self, coords, colors, target_size=None):
        self.set_calls += 1
        self.occupied_points.append(np.asarray(coords).copy())


def proposal_batch(count=2):
    uv = np.stack(
        [np.linspace(10.5, 20.5, count), np.linspace(15.5, 25.5, count)], axis=1
    ).astype(np.float32)
    depth = np.linspace(2.0, 3.0, count, dtype=np.float32)
    means = np.stack(
        [np.linspace(0.0, 0.1, count), np.zeros(count), depth], axis=1
    ).astype(np.float32)
    colors = np.stack(
        [
            np.linspace(0.2, 0.8, count),
            np.linspace(0.3, 0.7, count),
            np.linspace(0.4, 0.6, count),
        ],
        axis=1,
    ).astype(np.float32)
    return GaussianProposalBatch(
        source_frame_id=7,
        level=0,
        uv=uv,
        patch_bboxes=np.concatenate([uv - 4.0, uv + 4.0], axis=1),
        depths=depth,
        inverse_depths=1.0 / depth,
        world_points=means,
        log_scales=np.full((count, 1), -4.0, dtype=np.float32),
        colors=colors,
        residual_scores=np.full((count,), 0.2, dtype=np.float32),
        coverage_scores=np.full((count,), 0.8, dtype=np.float32),
        sparse_depth_valid=np.ones((count,), dtype=np.bool_),
        view_scale_size=0.1,
    )


def minimal_model():
    model = GaussianModel.__new__(GaussianModel)
    model.device = "cpu"
    model.BS = 1
    model.scene_scale = 1.0
    model.max_sh_degree = 0
    model.MAX_LEVEL = 1
    model.gaussian_pos_schedule_steps = 0
    model.init_gaussian_config = None
    model.current_gaussian_group = {0: 0}
    model.active_gaussian_groups = {0: [0]}
    model.valid_groups = [0]
    model.gaussian_groups = [Gaussians(BS=1, scene_scale=1.0, max_sh_degree=0)]
    model.gaussian_groups[0].to_device("cpu")
    model.hash_block = RecordingHashBlock()
    model.worldtest_certificate_authority = None
    model.worldtest_group_certificates = {}
    model.frozen_geometry_group_ids = set()
    model.frozen_position_group_ids = set()
    model.progressive_group_ids = set()
    return model


def certificate_authority():
    contract = WorldFrameContract(
        "dataset",
        "world",
        "colmap_canonical",
        "v1",
        "colmap",
        "colmap_track_world_point",
        "persistent",
        True,
    )
    return CertificateAuthority(contract, 1.0)


def initialize_adam(group):
    loss = sum(torch.square(value).sum() for value in group.splats.values())
    loss.backward()
    group.update()


def optimizer_snapshot(group):
    result = {}
    for name, optimizer in group.optimizers.items():
        parameter = optimizer.param_groups[0]["params"][0]
        state = optimizer.state[parameter]
        result[name] = {
            "exp_avg": state["exp_avg"].detach().clone(),
            "exp_avg_sq": state["exp_avg_sq"].detach().clone(),
        }
    return result


def test_waiting_proposals_do_not_mutate_permanent_map():
    model = minimal_model()
    before_num = model.gaussian_groups[0].get_num
    before_parameters = {
        name: value.detach().clone()
        for name, value in model.gaussian_groups[0].splats.items()
    }

    waiting = proposal_batch()

    assert len(waiting) == 2
    assert model.hash_block.set_calls == 0
    assert model.gaussian_groups[0].get_num == before_num
    for name, value in before_parameters.items():
        assert torch.equal(model.gaussian_groups[0].splats[name], value)


def test_commit_appends_parameters_and_zero_extends_adam_moments():
    model = minimal_model()
    group = model.gaussian_groups[0]
    initialize_adam(group)
    before_num = group.get_num
    before_state = optimizer_snapshot(group)

    result = model.commit_proposals(proposal_batch())

    assert result.committed == 2
    assert group.get_num == before_num + 2
    assert model.hash_block.set_calls == 1
    for name, optimizer in group.optimizers.items():
        parameter = optimizer.param_groups[0]["params"][0]
        state = optimizer.state[parameter]
        assert torch.equal(
            state["exp_avg"][:before_num], before_state[name]["exp_avg"]
        )
        assert torch.equal(
            state["exp_avg_sq"][:before_num], before_state[name]["exp_avg_sq"]
        )
        assert torch.count_nonzero(state["exp_avg"][before_num:]) == 0
        assert torch.count_nonzero(state["exp_avg_sq"][before_num:]) == 0


def test_frontview_birth_uses_track_responsibility_without_hash_reads_or_writes():
    model = minimal_model()
    model.frontview_birth_config = validate_front_view_birth_config(
        {"enabled": True}
    )
    model.frontview_track_ledger = TrackResponsibilityLedger()
    model.frontview_birth_stats = {"scale_capped_rows": 0}
    batch = proposal_batch(3)
    batch.track_ids[:] = np.asarray([11, 11, -1])
    batch.source_kinds[:] = np.asarray(["sparse", "sparse", "depthcov"])

    first = model.commit_proposals(batch)
    second = model.commit_proposals(batch)

    assert first.committed == 2
    assert second.committed == 1
    assert model.hash_block.query_calls == 0
    assert model.hash_block.set_calls == 0
    assert model.frontview_track_ledger.summary()["committed_tracks"] == 1


def test_frontview_identity_commit_never_reads_or_writes_hashblock():
    model = minimal_model()
    model.frontview_identity_lod_config = validate_front_view_identity_lod_config(
        {"enabled": True, "mode": "identity_only"}
    )
    model.frontview_identity_lod = FrontViewIdentityLOD(
        model.frontview_identity_lod_config
    )
    batch = proposal_batch(3)
    batch.track_ids[:] = np.asarray([11, 11, -1])
    batch.source_kinds[:] = np.asarray(["sparse", "sparse", "depthcov"])

    first = model.commit_proposals(batch)
    second = model.commit_proposals(batch)

    assert first.committed == 2
    assert second.committed == 1
    assert model.hash_block.query_calls == 0
    assert model.hash_block.set_calls == 0
    assert model.frontview_identity_lod.summary()["hash_calls_zero"] is True


def test_frontview_residual_cover_commit_never_reads_or_writes_hashblock():
    model = minimal_model()
    model.frontview_residual_cover_config = (
        validate_front_view_residual_cover_config({"enabled": True})
    )
    model.frontview_residual_cover = FrontViewResidualCover(
        model.frontview_residual_cover_config
    )
    batch = proposal_batch(3)
    batch.track_ids[:] = np.asarray([11, 11, -1])
    batch.source_kinds[:] = np.asarray(["sparse", "sparse", "depthcov"])

    first = model.commit_proposals(batch)
    second = model.commit_proposals(batch)

    assert first.committed == 2
    assert second.committed == 1
    assert model.hash_block.query_calls == 0
    assert model.hash_block.set_calls == 0
    assert model.frontview_residual_cover.summary()["hash_calls_zero"] is True


def test_unified_scale_cover_commit_never_falls_back_to_hashblock():
    model = minimal_model()
    model.frontview_scale_cover_config = validate_front_view_scale_cover_config(
        {"enabled": True, "rebuild_rows": 1}
    )
    model.frontview_scale_cover = FrontViewScaleCover(
        model.frontview_scale_cover_config
    )
    model.frontview_far_field_config = {"enabled": False}
    batch = proposal_batch(2)
    batch.source_kinds[:] = np.asarray(["sparse", "depthcov"])

    first = model.commit_proposals(batch)
    second = model.commit_proposals(batch)

    assert first.committed == 2
    assert second.committed == 0
    assert model.hash_block.query_calls == 0
    assert model.hash_block.set_calls == 0
    summary = model.frontview_scale_cover.summary()
    assert summary["registered_rows"] == 2
    assert summary["hash_calls_zero"] is True


def test_directional_scale_cover_sparse_proposal_carries_view_directions():
    model = minimal_model()
    model.camera_scale_rescalar = 1.0
    model.init_scale_offset = 0.0
    model.err_threshold = 0.1
    model.frequency_sampling_config = {"enabled": False}
    model.frontview_scale_cover_config = validate_front_view_scale_cover_config(
        {
            "enabled": True,
            "directional_ownership_enabled": True,
            "directional_max_angle_deg": 45.0,
        }
    )
    model.frontview_scale_cover = FrontViewScaleCover(
        model.frontview_scale_cover_config
    )
    model.frontview_residual_cover_config = {"enabled": False}
    model.frontview_identity_lod_config = {"enabled": False}
    model.frontview_birth_config = {"enabled": False}
    model.frontview_sparse_scale_map_config = {"enabled": False}

    cam = types.SimpleNamespace(
        cam_idx=1,
        get_color_pts_depth=lambda: np.asarray(
            [[0.0, 0.0, 2.0, 0.2, 0.3, 0.4, 2.0]], dtype=np.float32
        ),
        get_point_ids=lambda: np.asarray([17], dtype=np.int64),
        get_raw_pose=lambda: torch.eye(4),
        get_pose=lambda: torch.eye(4),
        get_int_mat=lambda _level: torch.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        ),
        get_fx=lambda _level: 100.0,
        get_fy=lambda _level: 100.0,
        get_view_size=lambda _level: 0.1,
    )

    proposals = model.propose_new_gaussians_pts_only(cam)

    assert len(proposals) == 1
    assert proposals.view_directions.shape == (1, 3)
    assert np.allclose(proposals.view_directions[0], [0.0, 0.0, 1.0])


def test_scale_cover_reuses_only_unchanged_admission_epoch():
    model = minimal_model()
    model.frontview_scale_cover_config = validate_front_view_scale_cover_config(
        {
            "enabled": True,
            "rebuild_rows": 1,
            "reuse_same_epoch_admission": True,
        }
    )
    model.frontview_scale_cover = FrontViewScaleCover(
        model.frontview_scale_cover_config
    )
    model.frontview_far_field_config = {"enabled": False}
    batch = proposal_batch(2)
    batch.metadata["scale_cover_epoch"] = model.frontview_scale_cover.admission_epoch

    first = model.commit_proposals(batch)
    stale = model.commit_proposals(batch)

    assert first.committed == 2
    assert stale.committed == 0
    summary = model.frontview_scale_cover.summary()
    assert summary["same_epoch_commit_reuses"] == 1
    assert summary["stale_commit_requeries"] == 1
    assert summary["query_calls"] == 1


def test_frontview_retirement_respects_sparse_tracks_and_capacity():
    model = minimal_model()
    model.opacity_prune_threshold = 0.01
    model.frontview_residual_cover_config = validate_front_view_residual_cover_config(
        {
            "enabled": True,
            "retirement_enabled": True,
            "retirement_capacity_base": 1,
            "retirement_capacity_per_frame": 0.0,
            "retirement_min_age_frames": 0,
        }
    )
    model.frontview_residual_cover = FrontViewResidualCover(
        model.frontview_residual_cover_config
    )
    batch = proposal_batch(2)
    batch.track_ids[:] = np.asarray([23, -1])
    batch.source_kinds[:] = np.asarray(["sparse", "depthcov"])
    model.commit_proposals(batch)

    model.prune_w_opacity(current_frame_id=200, processed_frames=200)

    assert model.gaussian_groups[0].get_num == 2
    assert 23 in model.gaussian_groups[0].non_trainable_params["track_ids"]
    assert model.frontview_residual_cover.summary()["retirement_rows"] == 1
    assert model.hash_block.query_calls == 0
    assert model.hash_block.set_calls == 0


def test_certified_batch_commit_validates_each_track_and_mutates_hash_once():
    model = minimal_model()
    authority = certificate_authority()
    model.worldtest_certificate_authority = authority
    batches = [proposal_batch(2).select([index]) for index in range(2)]
    certificates = []
    for track_id, batch in enumerate(batches, start=11):
        batch.track_ids[:] = track_id
        batch.source_kinds[:] = "sparse"
        certificates.append(
            authority.issue(
                source_frame_id=batch.source_frame_id,
                track_id=track_id,
                source_kind="sparse",
                issued_frame_id=9,
                observation_frame_ids=(1, 4, 7),
                q_g=2.0,
                evidence_mode="true_qg",
            )
        )

    result = model.commit_certified_proposals(batches, certificates)

    assert result.committed == 2
    assert model.hash_block.query_calls == 1
    assert model.hash_block.set_calls == 1
    assert authority.validation_count == 2
    assert authority.bypass_count == 0
    assert model.worldtest_group_certificates[result.group_id] == {
        certificate.certificate_id for certificate in certificates
    }


def test_deferred_hash_filter_matches_sequential_occupancy_for_duplicate_rows():
    block = HashBlock(
        {
            "use_hash": True,
            "start_scale": 20,
            "hash_size": 8,
            "hash_level": 3,
            "use_color_hash": False,
            "color_hash_size": 2,
            "remove_conflict": True,
        }
    )
    points = np.asarray([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], dtype=np.float32)
    colors = np.ones((2, 3), dtype=np.float32)

    accepted = block.get_sequential_no_conflict_index(points, colors, 0.005)

    assert accepted.tolist() == [True, False]
    assert not block.getOccupy(points[:1], colors[:1], 0.005)[0]
    block.setOccupy(points[:1], colors[:1], 0.005)
    assert block.getOccupy(points[1:], colors[1:], 0.005)[0]


def test_hash_filter_uses_coarsest_level_for_large_metric_footprint():
    block = HashBlock(
        {
            "use_hash": True,
            "start_scale": 20,
            "hash_size": 8,
            "hash_level": 3,
            "use_color_hash": False,
            "color_hash_size": 2,
            "remove_conflict": True,
        }
    )
    points = np.asarray([[1.0, 2.0, 40.0], [1.0, 2.0, 40.0]], dtype=np.float32)
    colors = np.ones((2, 3), dtype=np.float32)
    target_size = 0.12  # 1 / target_size is below the coarsest scale of 20.

    assert block.get_sequential_no_conflict_index(
        points, colors, target_size
    ).tolist() == [True, False]
    assert not block.getOccupy(points[:1], colors[:1], target_size)[0]
    block.setOccupy(points[:1], colors[:1], target_size)
    assert block.getOccupy(points[1:], colors[1:], target_size)[0]


def test_freeze_positions_leaves_appearance_and_footprint_trainable():
    model = minimal_model()

    model.freeze_group_positions(0)

    group = model.gaussian_groups[0]
    assert group.optimizers["means"].param_groups[0]["lr"] == 0.0
    assert group.optimizers["scales"].param_groups[0]["lr"] > 0.0
    assert group.optimizers["quats"].param_groups[0]["lr"] > 0.0
    assert group.optimizers["opacities"].param_groups[0]["lr"] > 0.0
    assert group.optimizers["sh0"].param_groups[0]["lr"] > 0.0


def test_bounded_positions_allow_local_motion_without_cumulative_anchor_drift():
    model = minimal_model()
    group = model.gaussian_groups[0]
    anchor = group.splats["means"].detach().clone()

    model.bound_group_positions(0, 0.05)
    group.splats["means"].data.add_(torch.tensor([[0.12, 0.0, 0.0]]))
    group.constrain_mean_displacement()

    displacement = torch.linalg.norm(group.splats["means"] - anchor, dim=1)
    assert torch.allclose(displacement, torch.full_like(displacement, 0.05))
    model.bound_group_positions(0, 0.05)
    assert torch.allclose(group.non_trainable_params["mean_anchors"], anchor)


def test_baseline_wrapper_immediately_commits_complete_host_batch():
    model = minimal_model()
    batch = proposal_batch(3)

    def fake_propose(self, cam, create_new_group=False, render_pkg=None, level=0):
        return batch

    model.propose_new_gaussians = types.MethodType(fake_propose, model)
    result = model.add_new_gaussians(types.SimpleNamespace(cam_idx=7))

    assert result.proposed == 3
    assert result.selected == 3
    assert result.committed == 3
    assert model.hash_block.set_calls == 1


def test_force_new_group_registers_integer_group_ids():
    model = minimal_model()
    model.commit_proposals(proposal_batch(), force_new_group=True)

    assert model.current_gaussian_group[0] == 1
    assert model.active_gaussian_groups[0] == [0, 1]
    assert model.valid_groups == [0, 1]
    assert all(isinstance(group_id, int) for group_id in model.valid_groups)
