import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from utils_new.aerocommit.types import GaussianProposalBatch
from utils_new.gaussian_models import GaussianModel
from utils_new.worldtest_gs.certificate import CertificateAuthority
from utils_new.worldtest_gs.config import validate_worldtest_config
from utils_new.worldtest_gs.contract import WorldFrameContract
from utils_new.worldtest_gs.controller import WorldTestController
from utils_new.worldtest_gs.evidence import (
    WorldIdentityEvidence,
    linear_gaussian_predictive,
    log_gaussian,
    monte_carlo_predictive_log_density,
)
from utils_new.worldtest_gs.nuisance import SharedNuisanceSolver
from utils_new.worldtest_gs.shadow import (
    ShadowGroup,
    ShadowObservation,
    cap_shadow_alpha,
)


def project(point, pose, intrinsics):
    camera = pose @ np.append(point, 1.0)
    screen = intrinsics @ camera[:3]
    return screen[:2] / screen[2], 1.0 / camera[2]


def test_fixed_sim3_preserves_projection():
    point = np.asarray([1.2, -0.4, 5.0])
    pose = np.eye(4)
    pose[:3, 3] = [0.2, -0.1, 0.3]
    intrinsics = np.asarray([[300.0, 0.0, 480.0], [0.0, 300.0, 320.0], [0.0, 0.0, 1.0]])
    angle = 0.4
    world_rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    scale = 3.7
    translation = np.asarray([2.0, -3.0, 1.0])
    transformed_point = scale * (world_rotation @ point) + translation
    old_from_new = np.eye(4)
    old_from_new[:3, :3] = world_rotation.T / scale
    old_from_new[:3, 3] = -(world_rotation.T @ translation) / scale
    transformed_pose = pose @ old_from_new
    uv_before, _ = project(point, pose, intrinsics)
    uv_after, _ = project(transformed_point, transformed_pose, intrinsics)
    np.testing.assert_allclose(uv_after, uv_before, atol=1.0e-10)


def test_contract_accepts_only_canonical_modes(tmp_path):
    (tmp_path / "conversion_stats.json").write_text(
        json.dumps(
            {
                "pose_source": "gt",
                "sparse_world_geometry": "frame_local_reprojected",
                "method": "per-frame COLMAP depths",
            }
        )
    )
    contract = WorldFrameContract.from_dataset(tmp_path)
    assert not contract.permanent_birth_valid
    with pytest.raises(RuntimeError, match="rejects permanent birth"):
        contract.require_permanent_birth()


def test_contract_accepts_fast_livo2_persistent_lidar(tmp_path):
    (tmp_path / "conversion_stats.json").write_text(
        json.dumps(
            {
                "scene": "Red_Sculpture",
                "pose_source": "FAST-LIVO2 causal LiDAR-inertial odometry",
                "pose_source_kind": "fast_livo2",
                "sparse_world_geometry": "persistent",
                "point_coordinate_system": "normalized world",
            }
        )
    )
    contract = WorldFrameContract.from_dataset(
        tmp_path, calibration_version="lvba_red_sculpture_lio_v1"
    )
    assert contract.permanent_birth_valid
    assert contract.geometry_mode == "lidar_canonical"
    assert contract.depth_source == "fast_livo2_lidar_world_point"
    contract.require_permanent_birth()


def test_contract_accepts_jointly_aligned_orbslam3_vi_world(tmp_path):
    stats_path = tmp_path / "conversion_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "pose_source": "orbslam3_vi",
                "sparse_world_geometry": "persistent",
                "coordinate_contract": (
                    "one Sim(3) is applied jointly to all ORB body poses and all "
                    "ORB map points"
                ),
            }
        )
    )
    contract = WorldFrameContract.from_dataset(tmp_path)
    assert contract.permanent_birth_valid
    assert contract.geometry_mode == "visual_inertial_canonical"
    assert contract.depth_source == "orbslam3_vi_track_world_point"

    stats_path.write_text(
        json.dumps(
            {
                "pose_source": "orbslam3_vi",
                "sparse_world_geometry": "persistent",
                "coordinate_contract": "poses aligned without a point guarantee",
            }
        )
    )
    assert not WorldFrameContract.from_dataset(tmp_path).permanent_birth_valid


def make_groups(incompatible=False, pure_rotation=False, nonfinite=False):
    config = validate_worldtest_config({"enabled": True})
    intrinsics = np.asarray([[300.0, 0.0, 480.0], [0.0, 300.0, 320.0], [0.0, 0.0, 1.0]])
    groups = []
    for track_id, depth in enumerate((4.0, 6.0, 8.0)):
        point = np.asarray([0.2 * track_id, 0.1, depth])
        group = ShadowGroup(track_id, track_id, "sparse", 0, None)
        for frame_id in range(3):
            center_x = 0.0 if pure_rotation else 0.12 * frame_id
            pose = np.eye(4)
            pose[0, 3] = -center_x
            measurement_pose = pose.copy()
            if incompatible:
                measurement_pose[0, 3] -= (0.0, 0.35, -0.45)[frame_id]
            uv, rho = project(point, measurement_pose, intrinsics)
            if incompatible:
                rho *= (1.0, 1.18, 0.82)[frame_id]
            if nonfinite and frame_id == 2:
                rho = float("nan")
            ray = np.linalg.solve(intrinsics, np.asarray([uv[0], uv[1], 1.0])) / rho
            per_frame_world = (np.linalg.inv(pose) @ np.append(ray, 1.0))[:3]
            observation = ShadowObservation(
                frame_id=frame_id,
                uv=np.asarray(uv),
                inverse_depth=float(rho),
                inverse_depth_variance=1.0e-6,
                pose_id=frame_id,
                world_to_camera=pose,
                pose_covariance=np.eye(6) * 1.0e-6,
                source_kind="sparse",
                track_confidence=1.0,
                rgb=np.asarray([0.2, 0.5, 0.1]),
                world_point=per_frame_world,
                intrinsics=intrinsics,
            )
            group.add(observation, None, max_views=8)
        groups.append(group)
    nuisance = SharedNuisanceSolver(config).solve(groups, 2)
    return config, groups, nuisance


def test_consistent_three_view_qg_passes_and_identity_is_single_world_point():
    config, groups, nuisance = make_groups()
    assert nuisance.valid
    result = WorldIdentityEvidence(config).evaluate(groups[0], nuisance)
    assert result.passed
    assert result.q_g > math.log(19.0)
    points = np.asarray([item.world_point for item in groups[0].observations])
    np.testing.assert_allclose(points - points[:1], 0.0, atol=1.0e-6)


def test_pose_depth_incompatibility_lowers_qg():
    config, groups_good, nuisance_good = make_groups()
    _, groups_bad, nuisance_bad = make_groups(incompatible=True)
    good = WorldIdentityEvidence(config).evaluate(groups_good[0], nuisance_good)
    bad = WorldIdentityEvidence(config).evaluate(groups_bad[0], nuisance_bad)
    assert good.q_g > bad.q_g + 5.0
    assert not bad.passed


def test_weak_parallax_and_nonfinite_abstain():
    config, groups_rotation, nuisance_rotation = make_groups(pure_rotation=True)
    rotation = WorldIdentityEvidence(config).evaluate(groups_rotation[0], nuisance_rotation)
    assert not rotation.passed
    assert rotation.q_g == float("-inf")
    _, groups_nan, nuisance_nan = make_groups(nonfinite=True)
    nonfinite = WorldIdentityEvidence(config).evaluate(groups_nan[0], nuisance_nan)
    assert not nonfinite.passed
    assert nonfinite.q_g == float("-inf")


def test_laplace_linear_predictive_matches_monte_carlo():
    latent_mean = np.asarray([0.2, -0.1])
    latent_covariance = np.asarray([[0.3, 0.05], [0.05, 0.2]])
    matrix = np.asarray([[1.0, 0.5], [-0.3, 0.8]])
    noise = np.diag([0.4, 0.25])
    value = np.asarray([0.1, -0.2])
    mean, covariance = linear_gaussian_predictive(
        latent_mean, latent_covariance, matrix, noise
    )
    analytic, _ = log_gaussian(value, mean, covariance)
    monte_carlo = monte_carlo_predictive_log_density(
        value,
        latent_mean,
        latent_covariance,
        matrix,
        noise,
        samples=200000,
        seed=4,
    )
    assert abs(analytic - monte_carlo) < 0.03


def proposal(source_kind="sparse"):
    return GaussianProposalBatch(
        source_frame_id=2,
        level=0,
        uv=np.asarray([[10.0, 20.0]], dtype=np.float32),
        patch_bboxes=np.asarray([[6.0, 16.0, 14.0, 24.0]], dtype=np.float32),
        depths=np.asarray([5.0], dtype=np.float32),
        inverse_depths=np.asarray([0.2], dtype=np.float32),
        world_points=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32),
        log_scales=np.asarray([[-4.0]], dtype=np.float32),
        colors=np.asarray([[0.2, 0.4, 0.1]], dtype=np.float32),
        residual_scores=np.zeros(1, dtype=np.float32),
        coverage_scores=np.ones(1, dtype=np.float32),
        sparse_depth_valid=np.asarray([source_kind == "sparse"]),
        track_ids=np.asarray([7]),
        source_kinds=np.asarray([source_kind]),
        view_scale_size=0.01,
    )


def authority():
    contract = WorldFrameContract(
        "dataset", "world", "colmap_canonical", "v1", "colmap",
        "colmap_track_world_point", "persistent", True
    )
    return CertificateAuthority(contract, math.log(19.0))


@pytest.mark.parametrize("source_kind", ["sparse", "depthcov", "residual"])
def test_proposal_paths_cannot_commit_without_certificate(source_kind):
    model = GaussianModel.__new__(GaussianModel)
    model.worldtest_certificate_authority = authority()
    with pytest.raises(RuntimeError, match="no AdmissionCertificate"):
        GaussianModel.commit_proposals(model, proposal(source_kind))


def test_detail_and_archive_paths_cannot_bypass_certificate():
    model = GaussianModel.__new__(GaussianModel)
    model.worldtest_certificate_authority = authority()
    with pytest.raises(RuntimeError, match="no AdmissionCertificate"):
        GaussianModel.add_track_detail_gaussians(
            model,
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.zeros((1, 1)),
            0.2,
            1.2,
        )
    with pytest.raises(RuntimeError, match="no AdmissionCertificate"):
        GaussianModel.restore_gaussian_group(model, {})


def test_shadow_has_no_parameters_and_alpha_cap_is_exact():
    _, groups, _ = make_groups()
    group = groups[0]
    assert not hasattr(group, "parameters")
    assert all(not hasattr(observation, "requires_grad") for observation in group.observations)
    alpha = np.asarray([[[0.08, 0.02]], [[0.07, 0.01]]], dtype=np.float32)
    capped = cap_shadow_alpha(alpha, 0.1)
    assert np.max(np.sum(capped, axis=0)) <= 0.1000001
    np.testing.assert_allclose(capped[0, 0, 0] / capped[1, 0, 0], 8.0 / 7.0)


@pytest.mark.parametrize("mode", ["equal_count_random", "matched_delay", "shuffled_qg", "npo_lite"])
def test_controls_exactly_match_true_commit_count(mode):
    controller = WorldTestController.__new__(WorldTestController)
    controller.config = {"admission_mode": mode}
    controller.reference_schedule = {
        "frames": {"4": [{"age": 2, "source_kind": "sparse"}] * 3}
    }
    controller.rng = np.random.default_rng(3)
    controller.evaluator = WorldIdentityEvidence(validate_worldtest_config())
    ready = [
        SimpleNamespace(
            group_id=index,
            age=2 + index % 2,
            source_kind="sparse",
            observations=[
                SimpleNamespace(world_point=np.asarray([index, 0.0, 1.0]))
                for _ in range(3)
            ],
        )
        for index in range(8)
    ]
    evaluated = {
        group.group_id: SimpleNamespace(q_g=float(group.group_id)) for group in ready
    }
    selected = controller._select_control(ready, evaluated, 4)
    assert len(selected) == 3
    assert len({group.group_id for group in selected}) == 3


def test_control_keeps_unselected_cached_offline_groups_ready():
    cached = SimpleNamespace(q_g=4.0)
    group = SimpleNamespace(
        group_id=3,
        age=2,
        distinct_view_count=3,
        offline_cached=True,
        cached_result=cached,
    )
    controller = WorldTestController.__new__(WorldTestController)
    controller.groups = {("sparse", 7): group}
    controller.config = {
        "admission_mode": "equal_count_random",
        "min_views": 3,
        "max_evaluations_per_frame": 128,
    }
    assert controller._ready() == [group]
    controller.config["admission_mode"] = "true_qg"
    assert controller._ready() == []
