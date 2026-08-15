import pytest
import torch

from utils_new.frontview_inverse_depth_certificate import (
    causal_frustum_inverse_depth_hypotheses,
    causal_inverse_depth_posterior,
    locally_track_supported_inverse_depth_hypotheses,
    track_supported_inverse_depth_hypotheses,
    validate_front_view_inverse_depth_certificate_config,
)


def test_cidec_config_defaults_disabled_and_threshold_free():
    config = validate_front_view_inverse_depth_certificate_config()
    assert not config["enabled"]
    assert "depth_m" not in config
    assert config["hypotheses"] % 2 == 1


def test_cidec_config_rejects_even_hypothesis_count():
    with pytest.raises(ValueError, match="hypotheses"):
        validate_front_view_inverse_depth_certificate_config({"hypotheses": 8})


def test_cidec_config_requires_history_to_cover_reference_budget():
    with pytest.raises(ValueError, match="history_frames"):
        validate_front_view_inverse_depth_certificate_config(
            {"reference_frames": 4, "history_frames": 3}
        )


def test_cidec_config_rejects_unknown_policy():
    with pytest.raises(ValueError, match="uncertified_policy"):
        validate_front_view_inverse_depth_certificate_config(
            {"uncertified_policy": "metric_anyway"}
        )


def test_cidec_config_validates_photometric_evidence_model():
    config = validate_front_view_inverse_depth_certificate_config(
        {"photometric_cost": "zncc", "view_aggregation": "median"}
    )
    assert config["photometric_cost"] == "zncc"
    assert config["view_aggregation"] == "median"
    with pytest.raises(ValueError, match="photometric_cost"):
        validate_front_view_inverse_depth_certificate_config(
            {"photometric_cost": "raw_rgb"}
        )


def test_cidec_abstention_has_stable_diagnostic_schema():
    camera = _Camera(torch.eye(4), torch.zeros((8, 8, 3)), 0)
    result = causal_inverse_depth_posterior(
        camera,
        [],
        torch.tensor([[4.5, 4.5]]),
        torch.tensor([10.0]),
        torch.tensor([0.1]),
        0,
        {"enabled": True},
    )
    assert set((
        "consensus_support",
        "consensus_pairwise_chi2",
        "mode_nll_margin",
        "leave_one_out_views",
        "leave_one_out_chi2",
    )).issubset(result)
    assert not result["certified"].item()


def test_track_supported_hypotheses_follow_metric_track_gauge():
    inverse = track_supported_inverse_depth_hypotheses(
        torch.tensor([48.0, 52.0, 60.0, 68.0]),
        torch.full((4,), 0.02),
        9,
    )
    depths = torch.reciprocal(inverse)
    assert len(depths) == 9
    assert 40.0 < float(depths.min()) < 60.0
    assert 55.0 < float(depths.max()) < 80.0


def test_causal_frustum_support_covers_farther_depth_without_metric_threshold():
    inverse = causal_frustum_inverse_depth_hypotheses(
        torch.tensor([8.0, 10.0, 12.0, 14.0]),
        torch.full((4,), 0.02),
        17,
        far_depth=1000.0,
    )
    depths = torch.reciprocal(inverse)
    assert float(depths.max()) == pytest.approx(1000.0, rel=1.0e-5)
    assert float(depths.min()) < 10.0
    assert bool(((depths >= 50.0) & (depths <= 100.0)).any().item())


def test_local_track_support_separates_two_image_regions():
    inverse = locally_track_supported_inverse_depth_hypotheses(
        torch.tensor([[1.0, 1.0], [99.0, 1.0]]),
        torch.tensor([[0.0, 0.0], [2.0, 0.0], [98.0, 0.0], [100.0, 0.0]]),
        torch.tensor([10.0, 12.0, 50.0, 60.0]),
        torch.full((4,), 0.01),
        9,
        neighbors=2,
    )
    depths = torch.reciprocal(inverse)
    assert float(depths[:, 0].max()) < 20.0
    assert float(depths[:, 1].min()) > 40.0


def test_track_supported_cidec_escapes_wrong_metric_fallback():
    torch.manual_seed(23)
    current_image = torch.rand((32, 64, 3))
    baseline = 1.0
    true_depth = 10.0
    disparity = int(round(40.0 * baseline / true_depth))
    current = _Camera(torch.eye(4), current_image, 3)
    references = []
    for index, multiplier in enumerate((1, 2), start=1):
        reference_pose = torch.eye(4)
        reference_pose[0, 3] = -baseline * multiplier
        references.append(
            _Camera(
                reference_pose,
                _shift_image(current_image, disparity * multiplier),
                index,
            )
        )
    result = causal_inverse_depth_posterior(
        current,
        references,
        torch.tensor([[30.5, 15.5]]),
        torch.tensor([100.0]),
        torch.tensor([0.02]),
        0,
        {
            "enabled": True,
            "hypothesis_source": "track_support",
            "reference_frames": 2,
            "history_frames": 2,
            "minimum_valid_views": 2,
            "hypotheses": 17,
            "patch_radius_px": 2,
            "photometric_cost": "zncc",
            "view_aggregation": "consensus",
            "photometric_temperature": 0.02,
            "information_gain_min": 0.01,
            "posterior_std_ratio_max": 0.9,
        },
        support_depths=torch.tensor([8.0, 10.0, 12.0, 14.0]),
        support_log_depth_stds=torch.full((4,), 0.02),
        support_uv=torch.tensor(
            [[28.0, 14.0], [30.0, 14.0], [30.0, 17.0], [33.0, 16.0]]
        ),
    )
    assert result["certified"].item()
    assert abs(torch.log(result["depths"] / true_depth).item()) < 0.25


def test_causal_frustum_cidec_recovers_farther_mode_than_tracks():
    torch.manual_seed(29)
    current_image = torch.rand((32, 64, 3))
    baseline = 8.0
    true_depth = 80.0
    disparity = int(round(40.0 * baseline / true_depth))
    current = _Camera(torch.eye(4), current_image, 4)
    references = []
    for index, multiplier in enumerate((1, 2, 3), start=1):
        reference_pose = torch.eye(4)
        reference_pose[0, 3] = -baseline * multiplier
        references.append(
            _Camera(
                reference_pose,
                _shift_image(current_image, disparity * multiplier),
                index,
            )
        )
    result = causal_inverse_depth_posterior(
        current,
        references,
        torch.tensor([[30.5, 15.5]]),
        torch.tensor([300.0]),
        torch.tensor([0.02]),
        0,
        {
            "enabled": True,
            "hypothesis_source": "causal_frustum",
            "reference_frames": 3,
            "history_frames": 3,
            "minimum_valid_views": 2,
            "hypotheses": 33,
            "patch_radius_px": 2,
            "photometric_cost": "zncc",
            "view_aggregation": "consensus",
            "photometric_temperature": 0.02,
            "information_gain_min": 0.01,
            "posterior_std_ratio_max": 0.9,
            "mode_nll_margin_min": 0.1,
            "leave_one_out_consistency": True,
        },
        support_depths=torch.tensor([8.0, 10.0, 12.0, 14.0]),
        support_log_depth_stds=torch.full((4,), 0.02),
        support_uv=torch.tensor(
            [[28.0, 14.0], [30.0, 14.0], [30.0, 17.0], [33.0, 16.0]]
        ),
    )
    assert result["certified"].item()
    assert result["leave_one_out_views"].item() == 3
    assert abs(torch.log(result["depths"] / true_depth).item()) < 0.25


class _Camera:
    def __init__(self, pose, image, cam_idx):
        self._pose = pose
        self._image = image
        self.cam_idx = cam_idx
        self.exposure_gain = 1.0
        self.near = 0.01
        self.far = 1000.0

    def get_raw_pose(self):
        return self._pose

    def get_int_mat(self, level=0):
        return torch.tensor(
            [[40.0, 0.0, 32.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]]
        )

    def get_gt_image(self, level=0):
        return self._image

    def get_width(self, level=0):
        return self._image.shape[1]

    def get_height(self, level=0):
        return self._image.shape[0]


def _shift_image(image, pixels):
    result = torch.zeros_like(image)
    result[:, :-pixels] = image[:, pixels:]
    return result


def test_cidec_recovers_depth_from_causal_disparity():
    torch.manual_seed(7)
    current_image = torch.rand((32, 64, 3))
    baseline = 1.0
    true_depth = 10.0
    disparity = int(round(40.0 * baseline / true_depth))
    reference_image = _shift_image(current_image, disparity)
    current = _Camera(torch.eye(4), current_image, 2)
    reference_pose = torch.eye(4)
    reference_pose[0, 3] = -baseline
    reference = _Camera(reference_pose, reference_image, 1)
    pixels = torch.tensor([[30.5, 15.5]])

    result = causal_inverse_depth_posterior(
        current,
        [reference],
        pixels,
        torch.tensor([5.0]),
        torch.tensor([0.02]),
        0,
        {
            "enabled": True,
            "hypotheses": 17,
            "minimum_log_depth_span": 1.0,
            "patch_radius_px": 2,
            "photometric_temperature": 0.02,
            "information_gain_min": 0.01,
            "posterior_std_ratio_max": 0.9,
        },
    )

    assert result["certified"].item()
    assert result["conflicted"].item()
    assert result["depths"].item() > 7.0
    assert abs(torch.log(result["depths"] / true_depth).item()) < 0.25


def test_zncc_cidec_is_invariant_to_affine_brightness():
    torch.manual_seed(11)
    current_image = 0.2 + 0.5 * torch.rand((32, 64, 3))
    baseline = 1.0
    true_depth = 10.0
    disparity = int(round(40.0 * baseline / true_depth))
    reference_image = 0.6 * _shift_image(current_image, disparity) + 0.15
    current = _Camera(torch.eye(4), current_image, 2)
    reference_pose = torch.eye(4)
    reference_pose[0, 3] = -baseline
    reference = _Camera(reference_pose, reference_image, 1)
    result = causal_inverse_depth_posterior(
        current,
        [reference],
        torch.tensor([[30.5, 15.5]]),
        torch.tensor([5.0]),
        torch.tensor([0.02]),
        0,
        {
            "enabled": True,
            "hypotheses": 17,
            "minimum_log_depth_span": 1.0,
            "patch_radius_px": 2,
            "photometric_cost": "zncc",
            "photometric_temperature": 0.02,
            "information_gain_min": 0.01,
            "posterior_std_ratio_max": 0.9,
        },
    )
    assert result["certified"].item()
    assert abs(torch.log(result["depths"] / true_depth).item()) < 0.25


def test_consensus_requires_two_compatible_views():
    torch.manual_seed(19)
    current_image = torch.rand((32, 64, 3))
    baseline = 1.0
    true_depth = 10.0
    disparity = int(round(40.0 * baseline / true_depth))
    current = _Camera(torch.eye(4), current_image, 3)
    references = []
    for index, multiplier in enumerate((1, 2), start=1):
        reference_pose = torch.eye(4)
        reference_pose[0, 3] = -baseline * multiplier
        references.append(
            _Camera(
                reference_pose,
                _shift_image(current_image, disparity * multiplier),
                index,
            )
        )
    result = causal_inverse_depth_posterior(
        current,
        references,
        torch.tensor([[30.5, 15.5]]),
        torch.tensor([5.0]),
        torch.tensor([0.02]),
        0,
        {
            "enabled": True,
            "reference_frames": 2,
            "history_frames": 2,
            "minimum_valid_views": 2,
            "hypotheses": 17,
            "minimum_log_depth_span": 1.0,
            "patch_radius_px": 2,
            "photometric_cost": "zncc",
            "view_aggregation": "consensus",
            "photometric_temperature": 0.02,
            "information_gain_min": 0.01,
            "posterior_std_ratio_max": 0.9,
        },
    )
    assert result["certified"].item()
    assert result["consensus_support"].item() == 2
    assert abs(torch.log(result["depths"] / true_depth).item()) < 0.25
