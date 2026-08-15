import numpy as np
import pytest
import torch

from utils_new.frontview_causal_landmark_memory import (
    CausalPersistentLandmarkMemory,
    information_gain_transport,
    shuffle_landmark_depths,
    validate_causal_landmark_memory_config,
)


class Camera:
    def __init__(self, frame, points, point_ids, pose=None):
        self.cam_idx = frame
        self._points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        self._ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
        self._pose = torch.as_tensor(
            np.eye(4) if pose is None else pose, dtype=torch.float32
        )
        self._intrinsics = torch.tensor(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        )

    def get_color_pts_depth(self):
        return np.column_stack(
            (
                self._points,
                np.zeros((len(self._points), 3), dtype=np.float32),
                self._points[:, 2] if len(self._points) else np.empty((0,)),
            )
        )

    def get_point_ids(self):
        return self._ids

    def get_raw_pose(self):
        return self._pose

    def get_int_mat(self, level=0):
        return self._intrinsics

    def get_width(self, level=0):
        return 100

    def get_height(self, level=0):
        return 100


def test_config_is_default_off_and_rejects_unknown_options():
    defaults = validate_causal_landmark_memory_config()
    assert not defaults["enabled"]
    assert not defaults["propagate_conditioned_uncertainty"]
    with pytest.raises(ValueError):
        validate_causal_landmark_memory_config({"future_points": True})
    assert (
        validate_causal_landmark_memory_config(
            {"conditioning_mode": "fallback_repair"}
        )["conditioning_mode"]
        == "fallback_repair"
    )
    assert (
        validate_causal_landmark_memory_config(
            {"conditioning_mode": "admitted_mean"}
        )["conditioning_mode"]
        == "admitted_mean"
    )
    with pytest.raises(ValueError):
        validate_causal_landmark_memory_config({"conditioning_mode": "future_map"})
    assert (
        validate_causal_landmark_memory_config({"transport_rule": "variance_gain"})[
            "transport_rule"
        ]
        == "variance_gain"
    )
    with pytest.raises(ValueError):
        validate_causal_landmark_memory_config({"transport_rule": "depth_threshold"})


def test_memory_is_strictly_causal_and_excludes_current_ids():
    memory = CausalPersistentLandmarkMemory({"enabled": True})
    first = Camera(0, [[0.0, 0.0, 10.0], [1.0, 0.0, 20.0]], [7, 9])
    memory.observe(first)
    assert len(memory.project(first)) == 0
    current = Camera(1, [[1.0, 0.0, 20.0]], [9])
    memory.observe(current)
    projected = memory.project(current, exclude_ids=current.get_point_ids())
    assert projected.point_ids.tolist() == [7]
    assert projected.depths.tolist() == pytest.approx([10.0])


def test_projection_zbuffers_and_respects_occupied_pixels():
    memory = CausalPersistentLandmarkMemory({"enabled": True})
    memory.observe(
        Camera(0, [[0.0, 0.0, 10.0], [0.0, 0.0, 20.0], [2.0, 0.0, 10.0]], [1, 2, 3])
    )
    current = Camera(1, [], [])
    projected = memory.project(current)
    assert projected.point_ids.tolist() == [1, 3]
    occupied = memory.project(current, occupied_pixel_indices=[50 * 100 + 50])
    assert occupied.point_ids.tolist() == [3]


def test_shuffled_control_preserves_locations_and_budget():
    memory = CausalPersistentLandmarkMemory({"enabled": True})
    memory.observe(Camera(0, [[-1.0, 0.0, 10.0], [1.0, 0.0, 20.0]], [1, 2]))
    batch = memory.project(Camera(1, [], []))
    shuffled = shuffle_landmark_depths(batch, 4)
    np.testing.assert_array_equal(shuffled.uv, batch.uv)
    np.testing.assert_array_equal(shuffled.point_ids, batch.point_ids)
    np.testing.assert_array_equal(np.sort(shuffled.depths), np.sort(batch.depths))
    assert len(shuffled) == len(batch)


def test_repair_statistics_separate_preserved_recovered_and_fallback_rows():
    memory = CausalPersistentLandmarkMemory({"enabled": True})
    memory.record_repair(
        valid_before=17,
        invalid_before=13,
        newly_valid=5,
        conditioning_landmarks=9,
    )
    memory.record_repair(
        valid_before=2,
        invalid_before=8,
        newly_valid=0,
        conditioning_landmarks=0,
    )
    summary = memory.summary()
    assert summary["repair_trigger_calls"] == 2
    assert summary["repair_conditioned_calls"] == 1
    assert summary["repair_skipped_no_landmarks"] == 1
    assert summary["repair_preserved_valid_rows"] == 19
    assert summary["repair_invalid_before_rows"] == 21
    assert summary["repair_newly_valid_rows"] == 5
    assert summary["repair_remaining_invalid_rows"] == 16


def test_admitted_mean_statistics_measure_log_depth_shift():
    memory = CausalPersistentLandmarkMemory({"enabled": True})
    memory.record_admitted_mean([10.0, 20.0], [20.0, 10.0], landmarks=3)
    summary = memory.summary()
    assert summary["admitted_mean_calls"] == 1
    assert summary["admitted_mean_conditioned_calls"] == 1
    assert summary["admitted_mean_rows"] == 2
    assert summary["mean_admitted_absolute_log_shift"] == pytest.approx(np.log(2.0))
    assert summary["mean_admitted_transport_weight"] == pytest.approx(1.0)


def test_admitted_mean_statistics_record_adaptive_transport_weights():
    memory = CausalPersistentLandmarkMemory(
        {"enabled": True, "transport_rule": "variance_gain"}
    )
    memory.record_admitted_mean(
        [10.0, 20.0],
        [11.0, 18.0],
        landmarks=3,
        transport_weights=[0.25, 0.75],
    )
    summary = memory.summary()
    assert summary["transport_rule"] == "variance_gain"
    assert summary["mean_admitted_transport_weight"] == pytest.approx(0.5)


def test_information_gain_transport_has_correct_limiting_behavior():
    original = torch.tensor([10.0, 10.0, 10.0])
    conditioned = torch.tensor([20.0, 20.0, 20.0])
    original_std = torch.tensor([0.1, 0.1, 0.1])
    conditioned_std = torch.tensor([0.2, 0.1, 0.0])
    transported, weight = information_gain_transport(
        original, conditioned, original_std, conditioned_std
    )
    torch.testing.assert_close(weight, torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(transported, torch.tensor([10.0, 10.0, 20.0]))


def test_original_posterior_responsibility_is_explicit_and_measured():
    defaults = validate_causal_landmark_memory_config()
    assert defaults["responsibility_coordinate"] == "metric"
    memory = CausalPersistentLandmarkMemory(
        {
            "enabled": True,
            "conditioning_mode": "admitted_mean",
            "responsibility_coordinate": "original_posterior",
        }
    )
    assert memory.uses_original_posterior_responsibility
    memory.record_responsibility_coordinates([20.0, 40.0], [10.0, 40.0])
    memory.record_responsibility_registration([20.0, 40.0], [10.0, 40.0])
    summary = memory.summary()
    assert summary["responsibility_coordinate_rows"] == 2
    assert summary["responsibility_shifted_rows"] == 1
    assert summary["responsibility_registered_rows"] == 1
    assert summary["mean_responsibility_absolute_log_shift"] == pytest.approx(
        np.log(2.0) / 2.0
    )


def test_original_responsibility_requires_admitted_mean_mode():
    memory = CausalPersistentLandmarkMemory(
        {
            "enabled": True,
            "conditioning_mode": "all_queries",
            "responsibility_coordinate": "original_posterior",
        }
    )
    assert not memory.uses_original_posterior_responsibility
    with pytest.raises(ValueError, match="responsibility_coordinate"):
        validate_causal_landmark_memory_config(
            {"responsibility_coordinate": "future_depth"}
        )
