import pytest
import torch

from utils_new.frontview_directional_layer import (
    FrontViewDirectionalLayer,
    directional_pose_score,
    validate_front_view_directional_layer_config,
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


def test_directional_pose_score_penalizes_translation_at_far_depth():
    target = torch.eye(4)
    translated = torch.eye(4)
    translated[0, 3] = -10.0
    assert directional_pose_score(target, target, 100.0) == pytest.approx(0.0)
    assert directional_pose_score(translated, target, 100.0) == pytest.approx(0.1)


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


def test_directional_layer_causal_selection_rejects_future_anchors():
    layer, image = _identity_layer()
    selected = layer._select_anchors(FakeCamera(10, image))
    assert [anchor["frame_id"] for anchor in selected] == [0]
