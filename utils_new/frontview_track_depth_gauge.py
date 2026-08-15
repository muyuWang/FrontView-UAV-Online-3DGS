"""Cross-fitted metric gauge selection for causal front-view depth recovery."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TrackDepthGauge:
    selected_model: str
    accepted_field: bool
    log_scale: float
    log_scale_variance: float
    selected_fallback_depth: float
    field_nll: float
    fallback_nll: float
    fold_nll_gains: tuple[float, float]
    sample_count: int


def _as_vector(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _fit_log_scale(residuals: np.ndarray, variances: np.ndarray) -> tuple[float, float]:
    precision = 1.0 / np.maximum(variances, np.finfo(np.float64).eps)
    information = float(np.sum(precision))
    return float(np.sum(precision * residuals) / information), 1.0 / information


def _gaussian_nll(residuals: np.ndarray, variances: np.ndarray) -> float:
    variances = np.maximum(variances, np.finfo(np.float64).eps)
    return float(
        0.5 * np.sum(np.log(2.0 * math.pi * variances) + residuals**2 / variances)
    )


def cross_fitted_track_depth_gauge(
    pixels,
    field_depths,
    field_log_stds,
    track_depths,
    track_log_stds,
    fallback_depths,
    *,
    fallback_log_std,
    shuffle_binding=False,
    seed=43,
) -> TrackDepthGauge:
    """Select a calibrated relative-depth field by held-out predictive risk.

    The field contributes per-pixel relative depth while certified tracks provide
    the metric gauge. Two deterministic spatial folds prevent the same track
    from both fitting and validating the gauge. The field is selected only when
    it beats the best constant fallback on both held-out folds.
    """

    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    field_depths = _as_vector(field_depths)
    field_log_stds = _as_vector(field_log_stds)
    track_depths = _as_vector(track_depths)
    track_log_stds = _as_vector(track_log_stds)
    fallback_depths = _as_vector(fallback_depths)
    count = len(field_depths)
    if not (
        len(pixels)
        == len(field_log_stds)
        == len(track_depths)
        == len(track_log_stds)
        == count
    ):
        raise ValueError("Track-depth gauge arrays must align")
    if count < 4:
        return TrackDepthGauge(
            "abstain",
            False,
            0.0,
            math.inf,
            math.nan,
            math.inf,
            math.inf,
            (0.0, 0.0),
            count,
        )
    if not len(fallback_depths):
        raise ValueError("At least one fallback depth hypothesis is required")
    if float(fallback_log_std) <= 0.0:
        raise ValueError("Fallback log-depth uncertainty must be positive")
    finite = (
        np.all(np.isfinite(pixels), axis=1)
        & np.isfinite(field_depths)
        & np.isfinite(field_log_stds)
        & np.isfinite(track_depths)
        & np.isfinite(track_log_stds)
        & (field_depths > 0.0)
        & (track_depths > 0.0)
        & (field_log_stds > 0.0)
        & (track_log_stds > 0.0)
    )
    pixels = pixels[finite]
    field_depths = field_depths[finite]
    field_log_stds = field_log_stds[finite]
    track_depths = track_depths[finite]
    track_log_stds = track_log_stds[finite]
    count = len(field_depths)
    if count < 4:
        return TrackDepthGauge(
            "abstain",
            False,
            0.0,
            math.inf,
            math.nan,
            math.inf,
            math.inf,
            (0.0, 0.0),
            count,
        )
    if bool(shuffle_binding):
        permutation = np.random.default_rng(int(seed)).permutation(count)
        track_depths = track_depths[permutation]
        track_log_stds = track_log_stds[permutation]

    # Alternating lexicographic ranks yield two spatially interleaved folds.
    order = np.lexsort((pixels[:, 1], pixels[:, 0]))
    fold = np.empty(count, dtype=np.int64)
    fold[order] = np.arange(count, dtype=np.int64) % 2
    log_field = np.log(field_depths)
    log_track = np.log(track_depths)
    residual = log_track - log_field
    observation_variance = field_log_stds**2 + track_log_stds**2
    fallback_variance = track_log_stds**2 + float(fallback_log_std) ** 2
    fallback_depths = fallback_depths[
        np.isfinite(fallback_depths) & (fallback_depths > 0.0)
    ]
    if not len(fallback_depths):
        raise ValueError("Fallback depth hypotheses must be finite and positive")

    field_fold_nlls = []
    fallback_fold_nlls = [[] for _ in fallback_depths]
    for validation_fold in (0, 1):
        training = fold != validation_fold
        validation = ~training
        shift, shift_variance = _fit_log_scale(
            residual[training], observation_variance[training]
        )
        local_field_nll = _gaussian_nll(
            residual[validation] - shift,
            observation_variance[validation] + shift_variance,
        )
        local_fallback_nlls = [
            _gaussian_nll(
                log_track[validation] - math.log(float(depth)),
                fallback_variance[validation],
            )
            for depth in fallback_depths
        ]
        field_fold_nlls.append(local_field_nll)
        for model_index, value in enumerate(local_fallback_nlls):
            fallback_fold_nlls[model_index].append(value)

    shift, shift_variance = _fit_log_scale(residual, observation_variance)
    selected_fallback = min(
        range(len(fallback_depths)),
        key=lambda index: sum(fallback_fold_nlls[index]),
    )
    selected_fallback_depth = float(fallback_depths[selected_fallback])
    selected_fallback_folds = fallback_fold_nlls[selected_fallback]
    gains = [
        selected_fallback_folds[index] - field_fold_nlls[index]
        for index in (0, 1)
    ]
    accepted = bool(gains[0] > 0.0 and gains[1] > 0.0)
    selected_model = "calibrated_field" if accepted else "fallback"
    return TrackDepthGauge(
        selected_model=selected_model,
        accepted_field=accepted,
        log_scale=shift if accepted else 0.0,
        log_scale_variance=shift_variance if accepted else math.inf,
        selected_fallback_depth=selected_fallback_depth,
        field_nll=float(sum(field_fold_nlls)),
        fallback_nll=float(sum(selected_fallback_folds)),
        fold_nll_gains=(float(gains[0]), float(gains[1])),
        sample_count=count,
    )
