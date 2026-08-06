import numpy as np
import torch

from utils_new.frontview_identity_lod import FrontViewIdentityLOD


def _projection(uids, means, depths, radii):
    return {
        "gaussian_ids": torch.arange(len(uids)),
        "means2d": torch.tensor(means, dtype=torch.float32),
        "depths": torch.tensor(depths, dtype=torch.float32),
        "radii": torch.tensor(radii, dtype=torch.float32),
    }, torch.tensor(uids, dtype=torch.long)


def _filter(manager, uv, depths, residuals, budget, projection, uids):
    count = len(depths)
    return manager.filter_candidates(
        frame_id=5,
        uv=np.asarray(uv, dtype=np.float32),
        depths=np.asarray(depths, dtype=np.float32),
        residual_scores=np.asarray(residuals, dtype=np.float32),
        depth_confidences=np.ones(count, dtype=np.float32),
        sparse_valid=np.zeros(count, dtype=np.bool_),
        track_ids=np.full(count, -1, dtype=np.int64),
        projection_info=projection,
        global_uids=uids,
        depthcov_budget=budget,
    )


def test_visible_repeat_is_associated_and_rejected():
    manager = FrontViewIdentityLOD({"enabled": True, "mode": "identity_only"})
    projection, uids = _projection([7], [[100.0, 100.0]], [20.0], [8])
    manager.register_existing_roots([7])

    selected, parents, levels, sectors = _filter(
        manager,
        [[103.0, 101.0], [140.0, 100.0]],
        [20.2, 20.0],
        [0.5, 0.5],
        2,
        projection,
        uids,
    )

    assert selected.tolist() == [1]
    assert parents.tolist() == [-1]
    assert levels.tolist() == [0]
    assert sectors.tolist() == [-1]
    assert manager.summary()["repeat_rejected"] == 1


def test_lod_slots_are_bounded_to_one_child_per_sector():
    manager = FrontViewIdentityLOD(
        {
            "enabled": True,
            "mode": "identity_lod",
            "min_lod_radius_px": 4.0,
            "min_lod_residual": 0.1,
        }
    )
    projection, uids = _projection([9], [[100.0, 100.0]], [20.0], [10.0])
    manager.register_existing_roots([9])
    manager.children[(9, 3)] = 12
    manager.nodes[12] = manager.nodes[9].__class__(12, 9, 9, 1, 3)

    selected, parents, _, sectors = _filter(
        manager,
        [[103.0, 103.0], [97.0, 103.0]],
        [20.0, 20.0],
        [0.5, 0.5],
        2,
        projection,
        uids,
    )

    assert selected.tolist() == [1]
    assert parents.tolist() == [9]
    assert sectors.tolist() == [2]
    assert manager.summary()["lod_slot_rejected"] == 1


def test_same_frame_root_prevents_duplicate_birth():
    manager = FrontViewIdentityLOD({"enabled": True, "mode": "identity_only"})
    projection, uids = _projection([], [], [], [])
    selected, _, _, _ = _filter(
        manager,
        [[20.0, 20.0], [21.0, 20.0]],
        [30.0, 30.2],
        [0.5, 0.4],
        2,
        projection,
        uids,
    )
    assert selected.tolist() == [0]
    assert manager.summary()["same_frame_rejected"] == 1


def test_pruning_releases_child_slot():
    manager = FrontViewIdentityLOD({"enabled": True})
    manager.register_existing_roots([3])
    manager.nodes[4] = manager.nodes[3].__class__(4, 3, 3, 1, 2)
    manager.children[(3, 2)] = 4
    assert manager.release([4]) == 1
    assert (3, 2) not in manager.children
    assert manager.summary()["hash_calls_zero"] is True


def test_footprint_capacity_grows_with_projected_radius_but_stays_bounded():
    manager = FrontViewIdentityLOD(
        {
            "enabled": True,
            "mode": "identity_lod",
            "slot_mode": "footprint",
            "radius_gate_scale": 1.0,
            "min_lod_radius_px": 4.0,
            "min_lod_residual": 0.0,
            "lod_cell_px": 2.0,
            "lod_capacity_radius_px": 3.0,
            "max_children_per_node": 8,
        }
    )
    projection, uids = _projection([15], [[100.0, 100.0]], [20.0], [5.0])
    manager.register_existing_roots([15])
    selected, parents, _, sectors = _filter(
        manager,
        [[96.5, 100.0], [98.5, 100.0], [100.5, 100.0], [102.5, 100.0]],
        [20.0] * 4,
        [0.5] * 4,
        4,
        projection,
        uids,
    )
    assert selected.tolist() == [0, 1, 2]
    assert parents.tolist() == [15, 15, 15]
    assert len(np.unique(sectors)) == 3


def test_frustum_lattice_uses_visible_cell_count_without_persistent_parent():
    manager = FrontViewIdentityLOD(
        {
            "enabled": True,
            "mode": "frustum_lattice",
            "radius_gate_scale": 1.0,
            "min_lod_radius_px": 4.0,
            "min_lod_residual": 0.0,
            "lod_capacity_radius_px": 3.0,
            "max_children_per_node": 8,
        }
    )
    projection, uids = _projection(
        [21, 22],
        [[100.0, 100.0], [101.0, 100.0]],
        [20.0, 20.0],
        [5.0, 5.0],
    )
    manager.register_existing_roots([21, 22])
    selected, parents, levels, _ = _filter(
        manager,
        [[100.5, 100.0], [101.5, 100.0]],
        [20.0, 20.0],
        [0.5, 0.5],
        2,
        projection,
        uids,
    )
    assert selected.tolist() == [0]
    assert parents.tolist() == [-1]
    assert levels.tolist() == [0]
    assert manager.summary()["lattice_births"] == 1


def test_world_support_gate_prevents_false_projective_association():
    manager = FrontViewIdentityLOD(
        {
            "enabled": True,
            "mode": "identity_only",
            "use_world_gate": True,
            "world_gate_scale": 1.0,
        }
    )
    projection, uids = _projection([31], [[100.0, 100.0]], [20.0], [8.0])
    manager.register_existing_roots([31])
    selected, parents, _, _ = manager.filter_candidates(
        frame_id=7,
        uv=np.asarray([[100.5, 100.0]], dtype=np.float32),
        depths=np.asarray([20.0], dtype=np.float32),
        residual_scores=np.asarray([0.5], dtype=np.float32),
        depth_confidences=np.ones(1, dtype=np.float32),
        sparse_valid=np.zeros(1, dtype=np.bool_),
        track_ids=np.full(1, -1, dtype=np.int64),
        projection_info=projection,
        global_uids=uids,
        depthcov_budget=1,
        global_means=torch.zeros((1, 3)),
        global_scales=torch.full((1, 3), 0.1),
        world_points=np.asarray([[10.0, 0.0, 20.0]], dtype=np.float32),
        log_scales=np.asarray([[-2.3]], dtype=np.float32),
    )
    assert selected.tolist() == [0]
    assert parents.tolist() == [-1]


def test_render_handoff_fades_only_mature_coarse_parents_at_large_footprint():
    manager = FrontViewIdentityLOD(
        {
            "enabled": True,
            "render_handoff_enabled": True,
            "handoff_radius_start_px": 2.0,
            "handoff_radius_end_px": 4.0,
            "handoff_full_children": 1,
            "handoff_parent_floor": 0.0,
        }
    )
    manager.register_existing_roots([0])
    manager.nodes[1] = manager.nodes[0].__class__(1, 0, 0, 1, 0)
    manager.children[(0, 0)] = 1
    manager.child_counts[0] = 1
    manager.next_uid = 2

    multipliers = manager.render_handoff_multipliers(
        torch.tensor([0, 1]),
        torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]]),
        torch.full((2, 3), 0.1),
        torch.eye(4).unsqueeze(0),
        torch.tensor([100.0]),
    )

    assert torch.allclose(multipliers, torch.tensor([0.5, 1.0]), atol=1.0e-5)
