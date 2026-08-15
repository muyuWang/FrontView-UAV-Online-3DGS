import math

import numpy as np
import pytest

from utils_new.frontview_track_depth_gauge import cross_fitted_track_depth_gauge


def _grid():
    return np.asarray(
        [[x, y] for x in (20.0, 80.0, 140.0, 200.0) for y in (30.0, 110.0)]
    )


def test_cross_fitted_gauge_recovers_metric_scale():
    pixels = _grid()
    relative = 20.0 + pixels[:, 0] / 20.0 + pixels[:, 1] / 40.0
    truth = 1.8 * relative
    result = cross_fitted_track_depth_gauge(
        pixels,
        relative,
        np.full(len(pixels), 0.03),
        truth,
        np.full(len(pixels), 0.02),
        [250.0, 300.0],
        fallback_log_std=0.05,
    )
    assert result.accepted_field
    assert math.exp(result.log_scale) == pytest.approx(1.8, rel=1.0e-5)
    assert min(result.fold_nll_gains) > 0.0


def test_cross_fitted_gauge_abstains_when_constant_prior_predicts_tracks():
    pixels = _grid()
    field = np.linspace(10.0, 80.0, len(pixels))
    tracks = np.full(len(pixels), 50.0)
    result = cross_fitted_track_depth_gauge(
        pixels,
        field,
        np.full(len(pixels), 0.03),
        tracks,
        np.full(len(pixels), 0.02),
        [50.0, 300.0],
        fallback_log_std=0.05,
    )
    assert not result.accepted_field
    assert result.selected_model == "fallback"
    assert result.selected_fallback_depth == pytest.approx(50.0)


def test_cross_fitted_gauge_requires_identifiable_folds():
    result = cross_fitted_track_depth_gauge(
        np.zeros((3, 2)),
        np.ones(3),
        np.full(3, 0.03),
        np.ones(3),
        np.full(3, 0.02),
        [1.0],
        fallback_log_std=0.05,
    )
    assert not result.accepted_field
    assert result.selected_model == "abstain"


def test_cross_fitted_gauge_rejects_invalid_fallbacks():
    pixels = _grid()
    with pytest.raises(ValueError):
        cross_fitted_track_depth_gauge(
            pixels,
            np.ones(len(pixels)),
            np.full(len(pixels), 0.03),
            np.ones(len(pixels)),
            np.full(len(pixels), 0.02),
            [float("nan")],
            fallback_log_std=0.05,
        )
