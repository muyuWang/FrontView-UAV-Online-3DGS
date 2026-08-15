import numpy as np
import pytest

import render


def test_combine_render_gt_depth_makes_three_equal_width_panels():
    render_gt = np.zeros((8, 20, 3), dtype=np.uint8)
    render_gt[:, :10] = 31
    render_gt[:, 10:] = 127
    depth = np.full((8, 10, 3), 251, dtype=np.uint8)

    combined = render.combine_render_gt_depth(render_gt, depth)

    assert combined.shape == (8, 30, 3)
    np.testing.assert_array_equal(combined[:, :10], render_gt[:, :10])
    np.testing.assert_array_equal(combined[:, 10:20], render_gt[:, 10:])
    np.testing.assert_array_equal(combined[:, 20:], depth)


def test_combine_render_gt_depth_rejects_mismatched_height():
    with pytest.raises(ValueError, match="equal heights"):
        render.combine_render_gt_depth(
            np.zeros((8, 20, 3), dtype=np.uint8),
            np.zeros((6, 10, 3), dtype=np.uint8),
        )


def test_auto_encoder_falls_back_to_cpu_without_nvenc(monkeypatch):
    monkeypatch.setattr(render, "find_working_nvenc_ffmpeg", lambda: None)
    ffmpeg, encoder = render.resolve_h264_encoder("auto")
    assert ffmpeg == render.imageio_ffmpeg.get_ffmpeg_exe()
    assert encoder == "libx264"


def test_explicit_nvenc_fails_when_unavailable(monkeypatch):
    monkeypatch.setattr(render, "find_working_nvenc_ffmpeg", lambda: None)
    with pytest.raises(RuntimeError, match="NVENC was requested"):
        render.resolve_h264_encoder("nvenc")


def test_depth_colorizer_completes_unexplained_ray_mass_at_far_bound():
    colorizer = render.DepthColorizer(
        depth_min=2.0,
        depth_max=10.0,
        compositing_mode="transmittance_far",
    )
    depth = np.asarray([[2.0, 2.0, 2.0, np.nan]], dtype=np.float32)
    opacity = np.asarray([[1.0, 0.5, 0.0, 0.0]], dtype=np.float32)

    display_depth, valid = colorizer.depth_for_display(depth, opacity)

    np.testing.assert_allclose(display_depth, [2.0, 6.0, 10.0, 10.0])
    assert np.all(valid)


def test_depth_colorizer_far_completion_is_monotonic_in_metric_mass():
    colorizer = render.DepthColorizer(
        depth_min=1.0,
        depth_max=9.0,
        compositing_mode="transmittance_far",
    )
    opacity = np.asarray([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)

    display_depth, _ = colorizer.depth_for_display(
        np.full_like(opacity, 3.0), opacity
    )

    assert np.all(np.diff(display_depth.reshape(-1)) < 0.0)


def test_depth_colorizer_preserves_conditional_hit_diagnostic():
    colorizer = render.DepthColorizer(
        depth_min=2.0,
        depth_max=10.0,
        compositing_mode="conditional_hit",
    )
    depth = np.asarray([[4.0, 6.0]], dtype=np.float32)
    opacity = np.asarray([[0.5, 0.0]], dtype=np.float32)

    display_depth, valid = colorizer.depth_for_display(depth, opacity)

    np.testing.assert_allclose(display_depth, [4.0, 2.0])
    np.testing.assert_array_equal(valid, [True, False])
    assert (
        colorizer.metadata()["depth_definition"]
        == "conditional_expected_hit_depth"
    )
