import json
import math

import numpy as np
import torch

from utils_new.aerocommit.archive_store import ArchiveStore
from utils_new.aerocommit.candidate_bank import CandidateBank
from utils_new.aerocommit.commit_refiner import CommitRefiner
from utils_new.aerocommit.config import validate_aerocommit_config
from utils_new.aerocommit.detail_refiner import DetailRefiner
from utils_new.aerocommit.manager import (
    AeroCommitManager,
    frequency_probation_opacities,
    select_budgeted_bootstrap_indices,
    select_budgeted_fast_path_indices,
    split_candidate_masks,
    split_fast_path_masks,
)
from utils_new.aerocommit.dataset_geometry import guard_sparse_fast_path
from utils_new.aerocommit.npo_lite import NPOLiteEvaluator
from utils_new.aerocommit.pose_uncertainty import PoseUncertaintyProvider
from utils_new.aerocommit.types import (
    CandidateRecord,
    CandidateStatus,
    GaussianProposalBatch,
    SupportEdge,
)


def proposal(uv=(32.0, 32.0), depth=5.0, residual=0.1, count=2):
    uv = np.tile(np.asarray(uv, dtype=np.float32), (count, 1))
    uv[:, 0] += np.linspace(-0.5, 0.5, count, dtype=np.float32)
    points = np.stack(
        ((uv[:, 0] - 32.0) * depth / 100.0, np.zeros(count), np.full(count, depth)),
        axis=1,
    ).astype(np.float32)
    return GaussianProposalBatch(
        source_frame_id=0,
        level=0,
        uv=uv,
        patch_bboxes=np.concatenate((uv - 4.0, uv + 4.0), axis=1),
        depths=np.full((count,), depth, dtype=np.float32),
        inverse_depths=np.full((count,), 1.0 / depth, dtype=np.float32),
        world_points=points,
        log_scales=np.full((count, 1), math.log(0.25), dtype=np.float32),
        colors=np.full((count, 3), 0.5, dtype=np.float32),
        residual_scores=np.full((count,), residual, dtype=np.float32),
        coverage_scores=np.full((count,), 0.8, dtype=np.float32),
        sparse_depth_valid=np.ones((count,), dtype=np.bool_),
        view_scale_size=0.1,
    )


def camera_pose(center_x=0.0):
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = -center_x
    return pose


K = np.asarray([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def descriptor_and_patch():
    patch = np.linspace(0.1, 0.9, 16, dtype=np.float32).reshape(4, 4)
    descriptor = np.tile(patch.reshape(-1, 1), (1, 3)).reshape(-1)
    descriptor -= descriptor.mean()
    descriptor /= np.linalg.norm(descriptor)
    return descriptor.astype(np.float32), patch


def synthetic_candidate(baseline, covariance_scale=1.0):
    descriptor, patch = descriptor_and_patch()
    batch = proposal()
    candidate = CandidateRecord(
        candidate_id=1,
        reference_frame_id=0,
        reference_pose=camera_pose(0.0),
        reference_K=K.copy(),
        reference_uv=np.asarray([32.0, 32.0], dtype=np.float32),
        patch_bbox=np.asarray([28.0, 28.0, 36.0, 36.0], dtype=np.float32),
        reference_gray_patch=patch,
        reference_descriptor=descriptor,
        mean_color=np.full((3,), 0.5, dtype=np.float32),
        rho_mean=0.2,
        rho_variance=1.0e-4,
        depth_prior=5.0,
        created_frame=0,
        last_seen_frame=2,
        proposal_batch=batch,
        status=CandidateStatus.READY_FOR_RISK,
        stable_residual_ema=0.1,
    )
    edges = []
    for frame_id, center in enumerate((0.0, baseline, 2.0 * baseline)):
        pose = camera_pose(center)
        observed_u = 32.0 - 100.0 * center / 5.0
        edge_patch = patch + np.linspace(0.0, 0.01 * frame_id, 16, dtype=np.float32).reshape(4, 4)
        parallax = math.atan2(abs(center), 5.0)
        edges.append(
            SupportEdge(
                frame_id=frame_id,
                world_to_camera=pose,
                intrinsics=K.copy(),
                uv=np.asarray([observed_u, 32.0], dtype=np.float32),
                descriptor=descriptor.copy(),
                gray_patch=edge_patch,
                world_point=np.asarray([0.0, 0.0, 5.0], dtype=np.float32),
                depth=5.0,
                log_scale=math.log(0.25),
                color=np.full((3,), 0.5, dtype=np.float32),
                association_error=0.01,
                photometric_residual=0.1,
                parallax_rad=parallax,
                pose_covariance=np.diag([1.0e-4] * 3 + [1.0e-5] * 3).astype(np.float32)
                * covariance_scale,
            )
        )
    candidate.support_edges = edges
    candidate.parallax_max_rad = edges[-1].parallax_rad
    candidate.association_error_ema = 0.01
    return candidate


def admission_config(**updates):
    config = validate_aerocommit_config()["admission"]
    config.update(updates)
    return config


def test_candidate_association_reuses_track_and_keeps_four_supports():
    config = admission_config(min_support=2, max_support_edges=4)
    bank = CandidateBank(config, PoseUncertaintyProvider(config))
    descriptor, patch = descriptor_and_patch()
    first = proposal()
    groups = [(first, 1.0, 0.0)]
    _, new = bank.associate_and_update(
        groups,
        descriptor[None],
        patch[None],
        np.full((1, 3), 0.5, dtype=np.float32),
        camera_pose(),
        K,
        0,
        64,
        64,
    )
    assert len(new) == 1
    candidate_id = new[0]
    for frame_id in range(1, 7):
        center = 0.02 * frame_id
        current_uv = (32.0 - 100.0 * center / 5.0, 32.0)
        matched, new = bank.associate_and_update(
            [(proposal(uv=current_uv), 1.0, 0.0)],
            descriptor[None],
            patch[None],
            np.full((1, 3), 0.5, dtype=np.float32),
            camera_pose(center),
            K,
            frame_id,
            64,
            64,
        )
        assert matched == [candidate_id]
        assert new == []
    assert len(bank.candidates) == 1
    assert bank.candidates[candidate_id].support_count == 4


def test_candidate_fuses_distinct_support_proposals_with_a_fixed_budget():
    config = admission_config(
        min_support=2,
        fuse_support_proposals=True,
        max_fused_proposals_per_candidate=3,
        side_fusion_capacity_multiplier=1.0,
        fusion_voxel_scale_ratio=0.1,
    )
    bank = CandidateBank(config, PoseUncertaintyProvider(config))
    descriptor, patch = descriptor_and_patch()
    first = proposal(count=2)
    _, new = bank.associate_and_update(
        [(first, 1.0, 0.0)],
        descriptor[None],
        patch[None],
        np.full((1, 3), 0.5, dtype=np.float32),
        camera_pose(),
        K,
        0,
        64,
        64,
    )
    support = proposal(count=2)
    support.world_points[:, 1] += np.asarray([0.1, 0.2], dtype=np.float32)
    matched, _ = bank.associate_and_update(
        [(support, 1.0, 0.0)],
        descriptor[None],
        patch[None],
        np.full((1, 3), 0.5, dtype=np.float32),
        camera_pose(),
        K,
        1,
        64,
        64,
    )
    candidate = bank.candidates[new[0]]
    assert matched == new
    assert len(candidate.proposal_batch) == 3
    assert candidate.fused_proposal_count == 1
    assert candidate.representative_world_point is not None


def test_latest_consistent_snapshot_is_committed_but_reference_is_retained():
    config = admission_config(
        min_support=2,
        fuse_support_proposals=False,
        commit_snapshot_policy="latest_consistent",
    )
    bank = CandidateBank(config, PoseUncertaintyProvider(config))
    descriptor, patch = descriptor_and_patch()
    first = proposal(count=2)
    _, new = bank.associate_and_update(
        [(first, 1.0, 0.0)],
        descriptor[None],
        patch[None],
        np.full((1, 3), 0.5, dtype=np.float32),
        camera_pose(),
        K,
        0,
        64,
        64,
    )
    latest = proposal(count=3)
    bank.associate_and_update(
        [(latest, 1.0, 0.0)],
        descriptor[None],
        patch[None],
        np.full((1, 3), 0.5, dtype=np.float32),
        camera_pose(),
        K,
        1,
        64,
        64,
    )
    candidate = bank.candidates[new[0]]
    assert len(candidate.original_proposal_batch) == 2
    assert len(candidate.proposal_batch) == 3


def test_frequency_candidate_becomes_ready_after_two_consistent_views():
    config = admission_config(
        min_support=3,
        frequency_candidate_enabled=True,
        frequency_candidate_score_threshold=0.65,
        frequency_candidate_min_support=2,
    )
    bank = CandidateBank(config, PoseUncertaintyProvider(config))
    descriptor, patch = descriptor_and_patch()
    first = proposal()
    first.frequency_scores[:] = 0.9
    _, new = bank.associate_and_update(
        [(first, 1.0, 0.0)],
        descriptor[None],
        patch[None],
        np.full((1, 3), 0.5, dtype=np.float32),
        camera_pose(),
        K,
        0,
        64,
        64,
    )
    support = proposal(uv=(31.6, 32.0))
    support.frequency_scores[:] = 0.9
    matched, _ = bank.associate_and_update(
        [(support, 1.0, 0.0)],
        descriptor[None],
        patch[None],
        np.full((1, 3), 0.5, dtype=np.float32),
        camera_pose(0.02),
        K,
        1,
        64,
        64,
    )

    assert matched == new
    assert bank.candidates[new[0]].support_count == 2
    assert bank.candidates[new[0]].status == CandidateStatus.READY_FOR_RISK


def test_npo_risk_is_higher_for_weak_parallax():
    evaluator = NPOLiteEvaluator(admission_config(), device="cpu")
    weak = synthetic_candidate(0.002)
    strong = synthetic_candidate(0.2)
    result = evaluator.evaluate([weak, strong])
    assert np.isfinite(result.commitment_risk).all()
    assert result.commitment_risk[0] > result.commitment_risk[1]
    assert result.information[0] < result.information[1]


def test_larger_pose_covariance_does_not_reduce_npo_risk():
    evaluator = NPOLiteEvaluator(admission_config(), device="cpu")
    certain = synthetic_candidate(0.1, covariance_scale=1.0)
    uncertain = synthetic_candidate(0.1, covariance_scale=100.0)
    result = evaluator.evaluate([certain, uncertain])
    assert result.commitment_risk[1] >= result.commitment_risk[0]


def test_commit_refinement_reduces_reprojection_loss():
    candidate = synthetic_candidate(0.2)
    candidate.rho_mean = 0.16
    refiner = CommitRefiner(
        {
            "enabled": True,
            "iterations": 20,
            "learning_rate": 0.05,
        },
        device="cpu",
    )
    batches, before, after = refiner.refine([candidate])
    assert len(batches) == 1
    assert after < before
    assert candidate.refined_rho is not None


def test_commit_refinement_preserves_within_group_color_detail():
    candidate = synthetic_candidate(0.2)
    candidate.proposal_batch.colors = np.asarray(
        [[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]], dtype=np.float32
    )
    before_delta = np.diff(candidate.proposal_batch.colors, axis=0)
    refiner = CommitRefiner(
        {
            "enabled": True,
            "iterations": 1,
            "learning_rate": 0.01,
            "max_depth_correction_ratio": 0.2,
            "color_fusion_strength": 0.35,
        },
        device="cpu",
    )
    batches, _, _ = refiner.refine([candidate])
    after_delta = np.diff(batches[0].colors, axis=0)
    assert np.allclose(after_delta, before_delta, atol=1.0e-6)


def test_detail_split_replaces_each_parent_with_four_children():
    candidate = synthetic_candidate(0.2)
    refiner = DetailRefiner(
        {
            "enabled": True,
            "refine_min_views": 3,
            "refine_min_projected_radius_px": 0.0,
            "refine_min_stable_residual": 0.01,
            "child_count": 4,
            "child_scale_ratio": 0.55,
            "max_splits_per_keyframe": 1,
            "side_score_boost": 2.0,
        }
    )
    output, split_count, added = refiner.refine([candidate], [candidate.proposal_batch])
    assert split_count == 1
    assert len(output[0]) == 4 * len(candidate.proposal_batch)
    assert added == 3 * len(candidate.proposal_batch)


def test_frequency_probation_keeps_high_frequency_depthcov_out_of_hard_commit():
    batch = proposal(count=4)
    batch.sparse_depth_valid = np.asarray([True, False, False, False])
    batch.depth_confidences = np.asarray([1.0, 0.80, 0.55, 0.20], dtype=np.float32)
    batch.frequency_scores = np.asarray([0.9, 0.9, 0.9, 0.9], dtype=np.float32)
    config = admission_config(
        trusted_sparse_fast_path=True,
        trusted_depthcov_fast_path=True,
        trusted_depth_confidence_threshold=0.35,
        frequency_gate_score_threshold=0.65,
        trusted_frequency_depth_confidence_threshold=0.75,
        frequency_probation_enabled=True,
    )

    trusted, depth_trusted, probation, deferred = split_fast_path_masks(
        batch, config, True, True
    )

    assert trusted.tolist() == [True, True, False, False]
    assert depth_trusted.tolist() == [False, True, False, False]
    assert probation.tolist() == [False, False, True, False]
    assert not deferred.any()


def test_frequency_probation_is_disabled_by_default():
    batch = proposal(count=2)
    batch.sparse_depth_valid[:] = False
    batch.depth_confidences = np.asarray([0.55, 0.20], dtype=np.float32)
    batch.frequency_scores[:] = 0.9
    config = admission_config(
        trusted_depthcov_fast_path=True,
        trusted_depth_confidence_threshold=0.35,
        frequency_gate_score_threshold=0.65,
        trusted_frequency_depth_confidence_threshold=0.75,
    )

    trusted, _, probation, deferred = split_fast_path_masks(
        batch, config, False, True
    )

    assert trusted.tolist() == [True, False]
    assert not probation.any()
    assert not deferred.any()


def test_frequency_probation_opacity_tracks_depth_confidence():
    batch = proposal(count=3)
    batch.depth_confidences = np.asarray([0.35, 0.55, 0.75], dtype=np.float32)
    config = admission_config(
        trusted_depth_confidence_threshold=0.35,
        trusted_frequency_depth_confidence_threshold=0.75,
        frequency_probation_initial_opacity=0.15,
        frequency_probation_max_opacity=0.45,
    )

    opacity = frequency_probation_opacities(batch, config)

    assert np.allclose(opacity, [0.15, 0.30, 0.45])


def test_fast_path_budget_keeps_frequency_priority_and_uniform_coverage():
    batch = proposal(count=10)
    batch.frequency_scores = np.asarray(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 2.0, 3.0],
        dtype=np.float32,
    )
    batch.residual_scores[:] = 0.0

    selected = select_budgeted_fast_path_indices(batch, 4, 0.5)

    assert selected.tolist() == [0, 7, 8, 9]
    assert select_budgeted_fast_path_indices(batch, 0, 0.5).tolist() == list(
        range(10)
    )


def test_bootstrap_budget_prefers_measured_sparse_geometry():
    batch = proposal(count=10)
    batch.sparse_depth_valid = np.asarray(
        [False, True, False, True, False, True, False, True, False, True],
        dtype=np.bool_,
    )
    batch.frequency_scores = np.arange(10, dtype=np.float32)

    selected = select_budgeted_bootstrap_indices(batch, 4, 0.5)

    assert selected.tolist() == [1, 5, 7, 9]
    assert batch.sparse_depth_valid[selected].all()


def test_frame_local_sparse_geometry_disables_irreversible_fast_path(tmp_path):
    (tmp_path / "conversion_stats.json").write_text(
        json.dumps(
            {
                "pose_source": "gt",
                "sparse_world_geometry": "frame_local_reprojected",
            }
        ),
        encoding="utf-8",
    )
    config = {
        "bootstrap_frames": 5,
        "admission": {"trusted_sparse_fast_path": True},
    }

    result = guard_sparse_fast_path(config, {"dataset_path": str(tmp_path)})

    assert result["persistent"] is False
    assert result["changed"] is True
    assert config["bootstrap_frames"] == 1
    assert config["admission"]["trusted_sparse_fast_path"] is False


def test_depthcov_candidate_filter_preserves_only_unclaimed_sparse_proposals():
    batch = proposal(count=5)
    batch.sparse_depth_valid = np.asarray(
        [True, False, True, False, False], dtype=np.bool_
    )
    excluded = np.asarray([True, False, False, False, True], dtype=np.bool_)

    candidate, filtered = split_candidate_masks(batch, excluded, False)

    assert candidate.tolist() == [False, False, True, False, False]
    assert filtered.tolist() == [False, True, False, True, False]


def test_depthcov_candidate_filter_rejects_only_stable_depth_disagreement():
    batch = proposal(count=5)
    batch.sparse_depth_valid = np.asarray(
        [True, False, False, False, False], dtype=np.bool_
    )
    batch.depths = np.asarray([5.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    batch.stable_depths = np.asarray(
        [20.0, 11.0, 14.0, -1.0, np.nan], dtype=np.float32
    )

    candidate, filtered = split_candidate_masks(
        batch, np.zeros((5,), dtype=np.bool_), True, 0.25
    )

    assert candidate.tolist() == [True, True, False, True, True]
    assert filtered.tolist() == [False, False, True, False, False]


def test_frequency_gate_does_not_defer_mid_frequency_depthcov():
    batch = proposal(count=2)
    batch.sparse_depth_valid[:] = False
    batch.depth_confidences[:] = 0.50
    batch.frequency_scores = np.asarray([0.50, 0.90], dtype=np.float32)
    config = admission_config(
        trusted_depthcov_fast_path=True,
        trusted_depth_confidence_threshold=0.35,
        frequency_gate_enabled=True,
        frequency_gate_score_threshold=0.65,
        trusted_frequency_depth_confidence_threshold=0.75,
    )

    trusted, _, _, deferred = split_fast_path_masks(batch, config, False, True)

    assert trusted.tolist() == [True, False]
    assert deferred.tolist() == [False, True]


def test_archive_round_trip_uses_fp16_and_restores_float32(tmp_path):
    store = ArchiveStore(str(tmp_path))
    params = {
        "means": torch.randn(5, 3),
        "scales": torch.randn(5, 3),
        "quats": torch.randn(5, 4),
        "opacities": torch.randn(5),
        "sh0": torch.randn(5, 1, 3),
        "shN": torch.empty(5, 0, 3),
    }
    archive_id = store.archive(7, 0, params, 12)
    assert all(value.dtype == torch.float16 for value in store.groups[archive_id].params.values())
    restored = store.restore_params(archive_id, "cpu", torch.float32)
    assert all(value.dtype == torch.float32 for value in restored.values())
    assert torch.allclose(restored["means"], params["means"], atol=1.0e-3)


def test_full_map_refinement_restores_archives_and_freezes_geometry():
    class FakeGaussianModel:
        device = "cpu"

        def __init__(self):
            self.restored = []
            self.frozen = []

        def restore_gaussian_group(
            self, params, level=0, optimize=True, admission_certificate=None
        ):
            assert optimize is True
            assert admission_certificate is None
            assert all(value.dtype == torch.float32 for value in params.values())
            group_id = 10 + len(self.restored)
            self.restored.append((group_id, level, params))
            return group_id

        def freeze_group_geometry(self, group_id):
            self.frozen.append(group_id)

    params = {
        "means": torch.randn(7, 3),
        "scales": torch.randn(7, 3),
        "quats": torch.randn(7, 4),
        "opacities": torch.randn(7),
        "sh0": torch.randn(7, 1, 3),
        "shN": torch.empty(7, 0, 3),
    }
    manager = AeroCommitManager.__new__(AeroCommitManager)
    manager.gaussian_model = FakeGaussianModel()
    manager.archive_store = ArchiveStore(None)
    manager.active_group_metadata = {}
    manager.archive_store.archive(
        3, 1, params, 12, metadata={"created_frame": 4}
    )

    result = manager.restore_all_archives_for_refinement(freeze_geometry=True)

    assert result["groups"] == 1
    assert result["gaussians"] == 7
    assert manager.archive_store.gaussian_count == 0
    assert manager.gaussian_model.frozen == [10]
    assert manager.active_group_metadata[10]["created_frame"] == 4
    assert manager.active_group_metadata[10]["last_seen_frame"] == 12
