import numpy as np
import pytest

from scripts.preprocess_360dvo_orbmono_fixed import (
    held_out_pose_accepted,
    plan_windows,
    source_fallback_selection,
    temporal_track_support_certificate,
)
from scripts.run_360dvo_orbmono_fixed_all_gpu4_7 import (
    capacity,
    pin_keyframes_on_gpu,
    prepared_scene_path,
    update_runtime_dataset_path,
)


def test_short_sequence_is_one_complete_window():
    assert plan_windows(180, 320, 220, 200) == [(0, 180)]


def test_small_tail_is_folded_into_overlapping_final_window():
    assert plan_windows(499, 120, 40, 100) == [
        (0, 120),
        (40, 120),
        (80, 120),
        (120, 120),
        (160, 120),
        (200, 120),
        (240, 120),
        (280, 120),
        (320, 120),
        (360, 139),
    ]


def test_long_sequence_keeps_a_substantial_final_window():
    windows = plan_windows(2250, 120, 40, 100)
    assert windows[0] == (0, 120)
    assert windows[-1] == (2120, 130)
    assert all(length >= 100 for _, length in windows)
    assert all(
        right_start < left_start + left_length
        for (left_start, left_length), (right_start, _) in zip(windows, windows[1:])
    )


def test_capacity_policy_protects_long_sequences():
    assert capacity(1000) == (5000, 3200)
    assert capacity(1001) == (1500, 1200)
    assert capacity(2000) == (1500, 1200)
    assert capacity(2001) == (800, 600)


def test_ultra_long_sequences_offload_inactive_keyframes():
    assert pin_keyframes_on_gpu(1000)
    assert not pin_keyframes_on_gpu(1001)


def test_runtime_config_tracks_the_certified_preprocess_output(tmp_path):
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "Dataset:\n  dataset_path: /old\nTestset:\n  dataset_path: /old\n",
        encoding="utf-8",
    )
    selected = tmp_path / "selected"
    selected.mkdir()

    update_runtime_dataset_path(config_path, selected)

    text = config_path.read_text(encoding="utf-8")
    assert text.count(str(selected.resolve())) == 2


def test_prepared_scene_prefers_image_certified_pose_contract(tmp_path):
    windowed = tmp_path / "grove_orbmono_windowed_gtcenter_tracks"
    constant = tmp_path / "grove_orbmono_epipolar_gtcenter_tracks"
    spline = tmp_path / "grove_orbmono_epipolar_spline_gtcenter_tracks"
    pose_contract = tmp_path / "grove_auto_pose_contract_tracks"
    for path in (windowed, constant, spline, pose_contract):
        path.mkdir()
        (path / "conversion_stats.json").write_text("{}\n", encoding="utf-8")

    assert prepared_scene_path(tmp_path, "grove") == pose_contract


def test_rejected_constant_pose_can_only_initialize_spline():
    statistics = {
        "epipolar_refinement_certificate": {
            "validation_inlier_fraction_gain": -0.001,
            "required_validation_gain": 0.0,
            "held_out_accepted": False,
        }
    }

    assert not held_out_pose_accepted(statistics)


def _write_temporal_points(root, counts):
    point_dir = root / "orb_point_clouds"
    point_dir.mkdir(parents=True)
    for frame, count in enumerate(counts):
        np.save(
            point_dir / f"point_cloud_{frame:05d}.npy",
            np.zeros((count, 3), dtype=np.float32),
        )


def _temporal_certificate(root, frame_count):
    return temporal_track_support_certificate(
        root,
        frame_count,
        support_points=32,
        min_supported_frame_fraction=0.5,
        temporal_bins=10,
        min_supported_frame_fraction_per_bin=0.25,
        min_passing_temporal_bin_fraction=0.8,
    )


def test_temporal_support_rejects_evidence_concentrated_in_two_bins(tmp_path):
    _write_temporal_points(tmp_path, [64] * 20 + [0] * 80)

    certificate = _temporal_certificate(tmp_path, 100)

    assert not certificate["accepted"]
    assert certificate["passing_temporal_bin_count"] == 2
    assert certificate["supported_frame_fraction"] == 0.2


def test_temporal_support_accepts_sequence_wide_persistent_geometry(tmp_path):
    _write_temporal_points(tmp_path, [64] * 8 + [0] * 2)

    certificate = _temporal_certificate(tmp_path, 10)

    assert certificate["accepted"]
    assert certificate["passing_temporal_bin_count"] == 8
    assert certificate["supported_frame_fraction"] == 0.8


def test_uncertified_source_geometry_fallback_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "conversion_stats.json").write_text(
        '{"frame_count": 12, "global_point_count": 345}\n', encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="Unsafe source fallback is disabled"):
        source_fallback_selection(
            source, 12, {"spline": "insufficient temporal support"}
        )
