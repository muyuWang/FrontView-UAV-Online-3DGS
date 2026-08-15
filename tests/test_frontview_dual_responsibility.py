import numpy as np
import pytest
import torch

from utils_new.frontview_dual_responsibility import (
    causal_finite_depth_certificates,
    geometry_decision_render,
    nearest_unique_replacement_positions,
    proposal_metric_confidences,
    validate_causal_dual_responsibility_config,
)


def test_geometry_decision_render_is_independent_of_directional_output():
    base = torch.tensor([1.0])
    directional = torch.tensor([2.0])

    assert geometry_decision_render(
        {"render": directional, "geometry_render": base}
    ) is base
    assert geometry_decision_render({"render": directional}) is directional


class _CertificateCamera:
    def __init__(self, image, pose, intrinsics):
        self.image = image
        self.pose = pose
        self.intrinsics = intrinsics
        self.exposure_gain = 1.0
        self.near = 0.01
        self.far = 1000.0

    def get_gt_image(self, level):
        assert level == 0
        return self.image

    def get_raw_pose(self):
        return self.pose

    def get_int_mat(self, level):
        assert level == 0
        return self.intrinsics

    def get_width(self, level):
        assert level == 0
        return self.image.shape[1]

    def get_height(self, level):
        assert level == 0
        return self.image.shape[0]


def _horizontal_image(width=64, height=32):
    columns = torch.linspace(0.0, 1.0, width).reshape(1, width, 1)
    return columns.expand(height, width, 3).contiguous()


def _camera_pair(reference_image):
    intrinsics = torch.tensor(
        [[40.0, 0.0, 32.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]]
    )
    current_pose = torch.eye(4)
    reference_pose = torch.eye(4)
    reference_pose[0, 3] = -1.0
    current = _CertificateCamera(_horizontal_image(), current_pose, intrinsics)
    reference = _CertificateCamera(reference_image, reference_pose, intrinsics)
    return current, reference


def test_finite_depth_certificate_accepts_se3_consistent_appearance():
    current_image = _horizontal_image()
    shifted = torch.zeros_like(current_image)
    shifted[:, :-4] = current_image[:, 4:]
    current, reference = _camera_pair(shifted)
    result = causal_finite_depth_certificates(
        current,
        [reference],
        torch.tensor([[32.5, 16.5]]),
        torch.tensor([10.0]),
        0,
    )
    assert result["observability"].item() > 0.99
    assert result["certificate"].item() > 0.9


def test_finite_depth_certificate_rejects_rotation_only_appearance():
    current, reference = _camera_pair(_horizontal_image())
    result = causal_finite_depth_certificates(
        current,
        [reference],
        torch.tensor([[32.5, 16.5]]),
        torch.tensor([10.0]),
        0,
    )
    assert result["observability"].item() > 0.99
    assert result["certificate"].item() < 0.1


def test_finite_depth_certificate_abstains_without_reference():
    current, _ = _camera_pair(_horizontal_image())
    result = causal_finite_depth_certificates(
        current,
        [],
        torch.tensor([[32.5, 16.5]]),
        torch.tensor([10.0]),
        0,
    )
    assert result["certificate"].item() == pytest.approx(1.0)


def test_dual_responsibility_is_default_disabled():
    config = validate_causal_dual_responsibility_config()
    assert config["enabled"] is False
    assert config["finite_depth_preserve_appearance_ownership"] is True
    assert config["geometry_use_metric_depth"] is True
    assert config["export_metric_depth"] is True


def test_metric_confidence_separates_coverage_fallback_from_geometry():
    result = proposal_metric_confidences(
        np.asarray(
            ["sparse", "tracked_metric", "depthcov", "depthcov_far", "depthcov"]
        ),
        np.asarray([0.2, 0.3, 0.75, 1.5, 0.9], dtype=np.float32),
        np.asarray([False, False, False, False, True]),
        {"depthcov_confidence_mode": "posterior"},
    )
    assert np.array_equal(
        result,
        np.asarray([1.0, 1.0, 0.75, 1.0, 0.0], dtype=np.float32),
    )


def test_binary_depthcov_mode_and_input_validation():
    result = proposal_metric_confidences(
        ["depthcov", "depthcov_far"],
        [0.1, 0.2],
        [False, True],
        {"depthcov_confidence_mode": "binary"},
    )
    assert np.array_equal(result, np.asarray([1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="must align"):
        proposal_metric_confidences(["sparse"], [1.0, 0.5], [False], {})


def test_tracking_replacements_are_unique_and_budget_preserving():
    positions = nearest_unique_replacement_positions(
        torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]),
        torch.tensor([1, 2, 3]),
        torch.tensor([[19.0, 0.0], [21.0, 0.0], [9.0, 0.0], [30.0, 0.0]]),
    )
    assert positions.tolist() == [2, 3, 1]
    assert len(torch.unique(positions)) == len(positions)


def test_tracking_replacement_fails_closed_without_proxy_slots():
    positions = nearest_unique_replacement_positions(
        torch.zeros((2, 2)), torch.empty(0, dtype=torch.long), torch.ones((3, 2))
    )
    assert positions.shape == (0,)
