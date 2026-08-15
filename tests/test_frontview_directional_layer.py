import pytest
import torch

from utils_new.frontview_directional_layer import (
    FrontViewDirectionalLayer,
    anchor_pair_config,
    causal_anchor_crossfade_weight,
    directional_pose_score,
    rendered_inverse_depth_scale,
    maximin_streaming_subset,
    pose_distance_matrix,
    validate_front_view_directional_layer_config,
    warp_boundary_support,
)


class FakeCamera:
    def __init__(
        self, frame_id, image, pose=None, raw_pose=None, points=(), exposure_gain=1.0
    ):
        self.cam_idx = int(frame_id)
        self._image = image
        self._pose = torch.eye(4) if pose is None else pose
        self._raw_pose = self._pose if raw_pose is None else raw_pose
        self._points = list(points)
        self.exposure_gain = float(exposure_gain)
        height, width = image.shape[:2]
        self._intrinsics = torch.tensor(
            [
                [float(width), 0.0, 0.5 * float(width)],
                [0.0, float(width), 0.5 * float(height)],
                [0.0, 0.0, 1.0],
            ]
        )

    def get_pts(self):
        return self._points

    def get_gt_image(self, level=0):
        assert level == 0
        return self._image

    def get_raw_pose(self):
        return self._raw_pose

    def get_pose(self):
        return self._pose

    def get_int_mat(self, level=0):
        assert level == 0
        return self._intrinsics


def test_directional_layer_config_rejects_an_impossible_anchor_budget():
    with pytest.raises(ValueError, match="min_anchors"):
        validate_front_view_directional_layer_config(
            {"min_anchors": 3, "max_anchors": 2}
        )


def test_directional_layer_defaults_preserve_rotation_first_behavior():
    config = validate_front_view_directional_layer_config({"enabled": True})
    assert config["warp_mode"] == "rotation"
    assert config["source_fusion"] == "first"
    assert config["boundary_taper"] is False
    assert config["warp_depth_control"] == "aligned"
    assert config["pose_score_mode"] == "fixed_depth"
    assert config["uncertainty_bootstrap_enabled"] is False


def test_causal_anchor_crossfade_uses_the_minimal_discrete_transition():
    assert causal_anchor_crossfade_weight(20, 0, 20) == pytest.approx(0.0)
    assert causal_anchor_crossfade_weight(20, 0, 21) == pytest.approx(0.5)
    assert causal_anchor_crossfade_weight(20, 0, 22) == pytest.approx(1.0)
    assert causal_anchor_crossfade_weight(0, 20, 21) == pytest.approx(1.0)


def test_warp_boundary_support_tapers_only_unsupported_edges():
    height, width = 20, 30
    first = torch.zeros((height, width, 2))
    first[..., 0] = torch.arange(width).reshape(1, width)
    first[..., 1] = torch.arange(height).reshape(height, 1)
    second = first.clone()
    second[..., 0] += 4.0
    valid = [torch.ones((height, width), dtype=torch.bool)] * 2
    support = warp_boundary_support([first, second], valid, width, height)
    assert support[10, 10] == pytest.approx(1.0)
    assert support[10, 0] == pytest.approx(0.0)
    assert 0.0 < float(support[10, 2]) < 1.0
    cell_supported = warp_boundary_support(
        [first, second], valid, width, height, minimum_support_px=8.0
    )
    assert cell_supported[10, 2] < support[10, 2]

def test_anchor_pair_config_requires_a_shared_episode_profile():
    pair = [
        {"ownership_profile": {"blend_weight": 0.75}},
        {"ownership_profile": {"blend_weight": 0.75}},
    ]
    assert anchor_pair_config(pair, "blend_weight", 1.0) == pytest.approx(0.75)
    pair[1]["ownership_profile"]["blend_weight"] = 0.5
    assert anchor_pair_config(pair, "blend_weight", 1.0) == pytest.approx(1.0)


def test_directional_layer_rejects_unknown_depth_control():
    with pytest.raises(ValueError, match="warp_depth_control"):
        validate_front_view_directional_layer_config(
            {"warp_depth_control": "random_depth"}
        )


def test_adaptive_bridge_configuration_needs_no_frame_or_depth_threshold():
    config = validate_front_view_directional_layer_config(
        {
            "anchor_selection_mode": "episode_bridge_ward",
            "anchor_interval_frames": None,
            "pose_score_mode": "rendered_inverse_depth",
            "far_depth_m": None,
            "use_geometry_gate": False,
        }
    )
    assert config["anchor_interval_frames"] is None
    assert config["far_depth_m"] is None


def test_opacity_gate_needs_no_metric_far_threshold():
    config = validate_front_view_directional_layer_config(
        {
            "pose_score_mode": "rendered_inverse_depth",
            "far_depth_m": None,
            "geometry_gate_mode": "opacity",
        }
    )
    assert config["geometry_gate_mode"] == "opacity"


def test_legacy_geometry_gate_still_requires_metric_far_threshold():
    with pytest.raises(ValueError, match="requires far_depth_m"):
        validate_front_view_directional_layer_config(
            {
                "pose_score_mode": "rendered_inverse_depth",
                "far_depth_m": None,
                "use_geometry_gate": True,
            }
        )


def test_se3_warp_uses_target_depth_to_account_for_translation():
    height, width = 6, 8
    image = torch.arange(width, dtype=torch.float32).reshape(1, width, 1)
    image = image.expand(height, width, 3) / float(width - 1)
    source_pose = torch.eye(4)
    source_pose[0, 3] = -1.0
    layer = FrontViewDirectionalLayer({"enabled": True})
    layer.observe(FakeCamera(0, image, pose=source_pose))
    target = FakeCamera(20, image)
    pixels = layer._pixel_grid(height, width, torch.device("cpu"), torch.float32)
    inverse_intrinsics = torch.linalg.inv(target.get_int_mat(0))

    rotation, _ = layer._warp_anchor(
        layer.anchors[0],
        target.get_pose(),
        target.get_int_mat(0),
        height,
        width,
        target.exposure_gain,
        pixels=pixels,
        inverse_target_intrinsics=inverse_intrinsics,
    )
    se3, valid = layer._warp_anchor(
        layer.anchors[0],
        target.get_pose(),
        target.get_int_mat(0),
        height,
        width,
        target.exposure_gain,
        pixels=pixels,
        inverse_target_intrinsics=inverse_intrinsics,
        target_depth=torch.full((height, width, 1), 8.0),
    )

    assert torch.any(valid)
    assert not torch.allclose(rotation[valid], se3[valid])


def test_directional_pose_score_penalizes_translation_at_far_depth():
    target = torch.eye(4)
    translated = torch.eye(4)
    translated[0, 3] = -10.0
    assert directional_pose_score(target, target, 100.0) == pytest.approx(0.0)
    assert directional_pose_score(translated, target, 100.0) == pytest.approx(0.1)
    assert directional_pose_score(
        translated, target, inverse_depth_scale=0.01
    ) == pytest.approx(0.1)


def test_rendered_inverse_depth_scale_is_opacity_weighted():
    scale = rendered_inverse_depth_scale(
        torch.tensor([[[2.0], [4.0], [float("nan")]]]),
        torch.tensor([[[1.0], [0.5], [1.0]]]),
    )
    assert scale == pytest.approx((0.5 + 0.5 * 0.25) / 1.5)


def test_streaming_kcenter_keeps_the_maximin_pose_subset():
    poses = []
    for center in (0.0, 0.1, 2.0):
        pose = torch.eye(4)
        pose[0, 3] = -center
        poses.append(pose)
    distance = pose_distance_matrix(poses, translation_scale=1.0)
    assert distance[0, 1] == pytest.approx(0.1)
    remove, separation = maximin_streaming_subset(poses, 2, 1.0)
    assert remove == 1
    assert separation == pytest.approx(2.0)


def test_streaming_kcenter_observe_ignores_frame_interval():
    image = torch.zeros((4, 6, 3))
    layer = FrontViewDirectionalLayer(
        {
            "enabled": True,
            "anchor_selection_mode": "streaming_kcenter",
            "max_anchors": 2,
            "min_anchors": 2,
            "anchor_interval_frames": 100,
        }
    )
    for frame_id, center in enumerate((0.0, 0.1, 2.0)):
        pose = torch.eye(4)
        pose[0, 3] = -center
        layer.observe(FakeCamera(frame_id, image, pose=pose))
    assert [anchor["frame_id"] for anchor in layer.anchors] == [0, 2]


def test_ordered_ward_merges_only_adjacent_trajectory_segments():
    image = torch.zeros((4, 6, 3))
    layer = FrontViewDirectionalLayer(
        {
            "enabled": True,
            "anchor_selection_mode": "ordered_ward",
            "max_anchors": 2,
            "min_anchors": 2,
            "anchor_interval_frames": 100,
        }
    )
    for frame_id, center in enumerate((0.0, 0.1, 2.0)):
        pose = torch.eye(4)
        pose[0, 3] = -center
        layer.observe(FakeCamera(frame_id, image, pose=pose))
    assert [anchor["frame_id"] for anchor in layer.anchors] == [0, 2]
    assert layer.anchors[0]["segment_weight"] == 2
    assert layer.anchors[0]["segment_begin"] == 0
    assert layer.anchors[0]["segment_end"] == 1


def test_episode_ordered_ward_resets_when_sparse_geometry_recovers():
    image = torch.zeros((4, 6, 3))
    layer = FrontViewDirectionalLayer(
        {
            "enabled": True,
            "anchor_selection_mode": "episode_ordered_ward",
            "max_anchors": 2,
            "min_anchors": 2,
            "sparse_point_threshold": 2,
        }
    )
    layer.observe(FakeCamera(0, image))
    layer.observe(FakeCamera(1, image))
    assert len(layer.anchors) == 2
    layer.observe(FakeCamera(2, image, points=[1, 2]))
    assert layer.anchors == []
    layer.observe(FakeCamera(3, image))
    assert [anchor["frame_id"] for anchor in layer.anchors] == [3]


def test_episode_bridge_ward_preserves_previous_and_current_boundaries():
    image = torch.zeros((4, 6, 3))
    layer = FrontViewDirectionalLayer(
        {
            "enabled": True,
            "anchor_selection_mode": "episode_bridge_ward",
            "max_anchors": 3,
            "min_anchors": 2,
            "sparse_point_threshold": 2,
        }
    )
    layer.observe(FakeCamera(0, image))
    layer.observe(FakeCamera(1, image))
    layer.observe(FakeCamera(2, image, points=[1, 2]))
    assert [anchor["frame_id"] for anchor in layer.anchors] == [1]
    for frame_id in (3, 4, 5, 6):
        pose = torch.eye(4)
        pose[0, 3] = -float(frame_id)
        layer.observe(FakeCamera(frame_id, image, pose=pose))
    frame_ids = [anchor["frame_id"] for anchor in layer.anchors]
    assert 1 in frame_ids
    assert 3 in frame_ids
    assert len(frame_ids) == 3


def test_directional_layer_captures_only_sparse_interval_frames():
    image = torch.full((4, 6, 3), 0.25)
    layer = FrontViewDirectionalLayer(
        {
            "enabled": True,
            "sparse_point_threshold": 2,
            "anchor_interval_frames": 20,
        }
    )
    assert layer.observe(FakeCamera(0, image)) is True
    assert layer.observe(FakeCamera(10, image)) is False
    assert layer.observe(FakeCamera(20, image, points=[1, 2])) is False
    assert layer.observe(FakeCamera(20, image)) is True
    assert [anchor["frame_id"] for anchor in layer.anchors] == [0, 20]


def test_directional_layer_stores_the_pose_used_by_the_renderer():
    image = torch.full((4, 6, 3), 0.25)
    optimized = torch.eye(4)
    optimized[0, 3] = 2.0
    layer = FrontViewDirectionalLayer({"enabled": True})
    layer.observe(FakeCamera(0, image, pose=optimized, raw_pose=torch.eye(4)))
    assert torch.equal(layer.anchors[0]["pose"], optimized)


def test_directional_layer_preserves_rgb_precision_at_large_exposure_gain():
    image = torch.linspace(0.0, 1.0, 24).reshape(3, 8, 1).expand(3, 8, 3)
    layer = FrontViewDirectionalLayer(
        {"enabled": True, "causal_only": False, "blend_weight": 1.0}
    )
    layer.observe(FakeCamera(0, image, exposure_gain=20.0))
    layer.observe(FakeCamera(20, image, exposure_gain=20.0))
    layer.activate(True)
    result = layer.composite(
        FakeCamera(40, image, exposure_gain=20.0),
        torch.zeros_like(image),
        torch.full((3, 8, 1), 100.0),
        torch.ones((3, 8, 1)),
    )
    assert torch.max(torch.abs(result - image)) <= 1.0 / 255.0


def _identity_layer(second_image=None, use_geometry_gate=False):
    height, width = 6, 8
    x = torch.linspace(0.0, 1.0, width).reshape(1, width, 1)
    image = x.expand(height, width, 3).contiguous()
    layer = FrontViewDirectionalLayer(
        {
            "enabled": True,
            "far_depth_m": 10.0,
            "consistency_threshold": 0.08,
            "blend_weight": 1.0,
            "use_geometry_gate": use_geometry_gate,
        }
    )
    layer.observe(FakeCamera(0, image))
    layer.observe(FakeCamera(20, image if second_image is None else second_image))
    assert layer.activate(True) is True
    return layer, image


def test_directional_layer_replaces_only_consistent_far_pixels():
    layer, expected = _identity_layer()
    camera = FakeCamera(40, expected)
    colors = torch.zeros_like(expected)
    result = layer.composite(
        camera,
        colors,
        torch.full((6, 8, 1), 100.0),
        torch.ones((6, 8, 1)),
    )
    assert torch.allclose(result, expected, atol=1.0 / 255.0)


def test_directional_layer_rejects_inconsistent_or_near_pixels():
    layer, image = _identity_layer(second_image=torch.ones((6, 8, 3)))
    camera = FakeCamera(40, image)
    colors = torch.zeros_like(image)
    far_result = layer.composite(
        camera,
        colors,
        torch.full((6, 8, 1), 100.0),
        torch.ones((6, 8, 1)),
    )
    assert torch.count_nonzero(far_result[:, :-1]) == 0

    consistent, _ = _identity_layer(use_geometry_gate=True)
    near_result = consistent.composite(
        camera,
        colors,
        torch.ones((6, 8, 1)),
        torch.ones((6, 8, 1)),
    )
    assert torch.count_nonzero(near_result) == 0


def test_opacity_gate_does_not_claim_well_covered_pixels_without_depth():
    layer, image = _identity_layer()
    layer.config["geometry_gate_mode"] = "opacity"
    layer.config["pose_score_mode"] = "rendered_inverse_depth"
    layer.config["far_depth_m"] = None
    camera = FakeCamera(40, image)
    colors = torch.zeros_like(image)
    depth = torch.full((6, 8, 1), 100.0)
    covered = layer.composite(
        camera,
        colors,
        depth,
        torch.ones((6, 8, 1)),
    )
    missing = layer.composite(
        camera,
        colors,
        depth,
        torch.zeros((6, 8, 1)),
    )
    assert torch.count_nonzero(covered) == 0
    assert torch.allclose(missing, image, atol=1.0 / 255.0)


def test_metric_transmittance_continuously_allocates_residual_ownership():
    layer, image = _identity_layer()
    layer.config["geometry_gate_mode"] = "metric_transmittance"
    camera = FakeCamera(40, image)
    colors = torch.zeros_like(image)
    metric_opacity = torch.full((6, 8, 1), 0.25)
    result = layer.composite(
        camera,
        colors,
        torch.full((6, 8, 1), 100.0),
        torch.ones((6, 8, 1)),
        metric_opacity=metric_opacity,
    )
    assert torch.allclose(result, 0.75 * image, atol=1.0 / 255.0)


def test_metric_transmittance_requires_metric_certificate_render():
    layer, image = _identity_layer()
    layer.config["geometry_gate_mode"] = "metric_transmittance"
    with pytest.raises(RuntimeError, match="metric opacity"):
        layer.composite(
            FakeCamera(40, image),
            torch.zeros_like(image),
            torch.full((6, 8, 1), 100.0),
            torch.ones((6, 8, 1)),
        )


def test_uncertainty_mass_applies_boundary_support_to_ownership():
    height, width = 6, 8
    image = torch.ones((height, width, 3))
    layer = FrontViewDirectionalLayer(
        {
            "enabled": True,
            "geometry_gate_mode": "uncertainty_mass",
            "uncertainty_cell_px": 2.0,
            "boundary_taper": True,
            "blend_weight": 1.0,
        }
    )
    layer.observe(FakeCamera(0, image))
    layer.observe(FakeCamera(20, image))
    layer.activate(True)

    result = layer.composite(
        FakeCamera(40, image),
        torch.zeros_like(image),
        torch.full((height, width, 1), 100.0),
        torch.ones((height, width, 1)),
        uncertainty_opacity=torch.ones((height, width, 1)),
    )

    assert torch.count_nonzero(result[:, 0]) == 0
    assert torch.all(result[2:-2, width // 2] > 0.99)


def test_causal_crossfade_smooths_the_anchor_pair_certificate():
    height, width = 6, 8
    layer = FrontViewDirectionalLayer(
        {
            "enabled": True,
            "anchor_interval_frames": 1,
            "max_anchors": 3,
            "min_anchors": 2,
            "consistency_threshold": 0.1,
            "blend_weight": 1.0,
            "source_fusion": "causal_crossfade",
        }
    )
    anchors = (
        (0, 1.6, 0.2),
        (5, 0.0, 0.2),
        (10, 0.8, 0.8),
    )
    for frame_id, center, value in anchors:
        pose = torch.eye(4)
        pose[0, 3] = -center
        layer.observe(
            FakeCamera(frame_id, torch.full((height, width, 3), value), pose=pose)
        )
    layer.activate(True)
    colors = torch.zeros((height, width, 3))
    depth = torch.full((height, width, 1), 100.0)
    opacity = torch.ones((height, width, 1))

    transition = layer.composite(
        FakeCamera(11, colors), colors, depth, opacity
    )
    mature = layer.composite(FakeCamera(15, colors), colors, depth, opacity)

    expected = 0.2 * (1.0 - causal_anchor_crossfade_weight(10, 5, 11))
    assert torch.allclose(
        transition[1:-1, 0],
        torch.full_like(transition[1:-1, 0], expected),
        atol=0.01,
    )
    assert torch.all(transition[1:-1, width // 2] < transition[1:-1, 0])
    assert torch.count_nonzero(mature[:, :-1]) == 0


def test_directional_layer_causal_selection_rejects_future_anchors():
    layer, image = _identity_layer()
    selected = layer._select_anchors(FakeCamera(10, image))
    assert [anchor["frame_id"] for anchor in selected] == [0]


def test_directional_layer_rejects_anchors_outside_episode_support():
    layer, image = _identity_layer()
    for anchor in layer.anchors:
        anchor["support_begin_frame"] = 5
        anchor["support_end_frame"] = 30

    assert layer._select_anchors(FakeCamera(4, image)) == []
    assert [
        anchor["frame_id"] for anchor in layer._select_anchors(FakeCamera(25, image))
    ] == [0, 20]
    assert layer._select_anchors(FakeCamera(31, image)) == []


def test_episode_uncertainty_channel_is_requested_only_inside_support():
    layer, image = _identity_layer()
    for anchor in layer.anchors:
        anchor["support_begin_frame"] = 5
        anchor["support_end_frame"] = 30
        anchor["ownership_profile"] = {
            "geometry_gate_mode": "uncertainty_mass",
            "uncertainty_cell_px": 48.0,
        }

    assert layer.uncertainty_cell_px_for_camera(FakeCamera(4, image)) is None
    assert layer.uncertainty_cell_px_for_camera(FakeCamera(25, image)) == pytest.approx(
        48.0
    )
    assert layer.uncertainty_cell_px_for_camera(FakeCamera(31, image)) is None


def test_uncertainty_bootstrap_captures_before_uncertainty_is_observable():
    image = torch.ones((6, 8, 3))
    layer = FrontViewDirectionalLayer(
        {
            "enabled": True,
            "sparse_point_threshold": 1,
            "uncertainty_bootstrap_enabled": True,
            "uncertainty_bootstrap_max_anchors": 2,
            "uncertainty_bootstrap_cell_px": 48.0,
        }
    )

    assert layer.observe(FakeCamera(0, image, points=(0,)))
    assert layer.stats["uncertainty_bootstrap_begin_frame"] == 0
    assert [
        anchor["frame_id"] for anchor in layer.uncertainty_bootstrap_anchors
    ] == [0]


def test_uncertainty_bootstrap_uses_a_run_length_change_point():
    image = torch.ones((6, 8, 3))
    layer = FrontViewDirectionalLayer(
        {
            "enabled": True,
            "sparse_point_threshold": 1,
            "uncertainty_bootstrap_enabled": True,
            "uncertainty_bootstrap_max_anchors": 2,
            "uncertainty_bootstrap_cell_px": 48.0,
        }
    )
    for frame_id in range(2):
        assert layer.observe(
            FakeCamera(frame_id, image, points=(0,)),
            uncertainty_mass=torch.ones((6, 8, 1)),
        )
    assert len(layer.uncertainty_bootstrap_anchors) == 2
    assert layer.uncertainty_cell_px_for_camera(
        FakeCamera(3, image, points=(0,))
    ) == pytest.approx(48.0)

    assert layer.observe(
        FakeCamera(2, image, points=(0,)),
        uncertainty_mass=torch.zeros((6, 8, 1)),
    )
    assert layer.observe(
        FakeCamera(3, image, points=(0,)),
        uncertainty_mass=torch.ones((6, 8, 1)),
    )
    assert layer.stats["uncertainty_bootstrap_end_frame"] == -1

    for frame_id in (4, 5):
        assert layer.observe(
            FakeCamera(frame_id, image, points=(0,)),
            uncertainty_mass=torch.zeros((6, 8, 1)),
        )
        assert layer.stats["uncertainty_bootstrap_end_frame"] == -1
    assert layer.observe(
        FakeCamera(6, image, points=(0,)),
        uncertainty_mass=torch.zeros((6, 8, 1)),
    )

    assert layer.stats["uncertainty_bootstrap_max_uncertain_streak"] == 2
    assert layer.stats["uncertainty_bootstrap_end_frame"] == 3
    assert layer.stats["uncertainty_bootstrap_change_delay"] == 3
    assert all(
        anchor["support_end_frame"] == 3
        for anchor in layer.uncertainty_bootstrap_anchors
    )
    assert layer.uncertainty_cell_px_for_camera(
        FakeCamera(7, image, points=(0,))
    ) is None
