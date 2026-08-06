"""Mutation-free diagnostics for assigning high-frequency residual responsibility."""

from dataclasses import dataclass
from itertools import combinations
from math import factorial
from typing import Callable, Dict, FrozenSet, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PLAYERS: Tuple[str, ...] = ("P", "G", "A")
Coalition = FrozenSet[str]
CoalitionLosses = Mapping[Coalition, Sequence[float]]


@dataclass(frozen=True)
class ResponsibilityDecision:
    shapley: Dict[str, float]
    leave_one_view_out_geometry: Tuple[float, ...]
    eligible: bool
    reason: str


def all_coalitions(players: Sequence[str] = PLAYERS) -> Tuple[Coalition, ...]:
    players = tuple(players)
    return tuple(
        frozenset(group)
        for size in range(len(players) + 1)
        for group in combinations(players, size)
    )


def _validated_loss_vectors(
    coalition_losses: CoalitionLosses,
    players: Sequence[str],
) -> Dict[Coalition, np.ndarray]:
    expected = set(all_coalitions(players))
    received = set(coalition_losses)
    if received != expected:
        missing = sorted(expected - received, key=lambda item: (len(item), sorted(item)))
        extra = sorted(received - expected, key=lambda item: (len(item), sorted(item)))
        raise ValueError(
            "Coalition losses must contain every coalition; missing={}, extra={}".format(
                missing, extra
            )
        )

    vectors = {
        coalition: np.atleast_1d(np.asarray(values, dtype=np.float64))
        for coalition, values in coalition_losses.items()
    }
    lengths = {values.size for values in vectors.values()}
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError("Every coalition must have the same non-zero view count")
    if any(values.ndim != 1 or not np.isfinite(values).all() for values in vectors.values()):
        raise ValueError("Coalition losses must be finite one-dimensional values")
    return vectors


def exact_shapley_values(
    coalition_losses: CoalitionLosses,
    players: Sequence[str] = PLAYERS,
) -> Dict[str, float]:
    """Return exact loss-reduction Shapley values; positive means lower loss."""

    players = tuple(players)
    if len(set(players)) != len(players) or not players:
        raise ValueError("Players must be unique and non-empty")
    vectors = _validated_loss_vectors(coalition_losses, players)
    losses = {coalition: float(values.mean()) for coalition, values in vectors.items()}
    count = len(players)
    denominator = factorial(count)
    values = {}
    for player in players:
        marginal = 0.0
        others = tuple(item for item in players if item != player)
        for coalition in all_coalitions(others):
            weight = (
                factorial(len(coalition))
                * factorial(count - len(coalition) - 1)
                / denominator
            )
            marginal += weight * (
                losses[coalition] - losses[coalition | frozenset((player,))]
            )
        values[player] = float(marginal)
    return values


def geometry_responsibility_decision(
    coalition_losses: CoalitionLosses,
    support_frames: int,
    min_support_frames: int = 3,
    margin: float = 0.0,
    require_dominance: bool = True,
    atol: float = 1.0e-9,
) -> ResponsibilityDecision:
    """Apply the conservative AeroCommit-F shadow gate without mutating a map."""

    vectors = _validated_loss_vectors(coalition_losses, PLAYERS)
    shapley = exact_shapley_values(vectors)
    view_count = next(iter(vectors.values())).size
    leave_one_out = []
    if view_count >= 2:
        for omitted in range(view_count):
            reduced = {
                coalition: np.delete(losses, omitted)
                for coalition, losses in vectors.items()
            }
            leave_one_out.append(exact_shapley_values(reduced)["G"])

    reason = "eligible"
    eligible = True
    if int(support_frames) < int(min_support_frames):
        eligible = False
        reason = "insufficient_support"
    elif shapley["G"] <= float(margin) + float(atol):
        eligible = False
        reason = "non_positive_geometry"
    elif require_dominance and shapley["G"] <= max(shapley["P"], shapley["A"]) + float(atol):
        eligible = False
        reason = "geometry_not_dominant"
    elif leave_one_out and min(leave_one_out) <= float(margin) + float(atol):
        eligible = False
        reason = "view_unstable_geometry"

    return ResponsibilityDecision(
        shapley=shapley,
        leave_one_view_out_geometry=tuple(float(value) for value in leave_one_out),
        eligible=eligible,
        reason=reason,
    )


def clip_whitened_update(
    frozen_base: np.ndarray,
    proposed: np.ndarray,
    uncertainty_scale: np.ndarray,
    radius: float,
    floor: float = 1.0e-8,
) -> np.ndarray:
    """Clip an absolute proposal relative to the frozen base in whitened units."""

    base = np.asarray(frozen_base, dtype=np.float64)
    proposal = np.asarray(proposed, dtype=np.float64)
    scale = np.asarray(uncertainty_scale, dtype=np.float64)
    if base.shape != proposal.shape:
        raise ValueError("Base and proposal must have identical shapes")
    try:
        np.broadcast_shapes(base.shape, scale.shape)
    except ValueError as exc:
        raise ValueError("Uncertainty scale must broadcast to the update") from exc
    if float(radius) < 0.0:
        raise ValueError("Trust radius must be non-negative")
    safe_scale = np.maximum(np.abs(scale), float(floor))
    delta = proposal - base
    norm = float(np.linalg.norm(delta / safe_scale))
    factor = 1.0 if norm <= float(radius) or norm == 0.0 else float(radius) / norm
    return (base + factor * delta).astype(np.result_type(frozen_base, proposed), copy=False)


def checkerboard_masks(height: int, width: int, phase: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("Checkerboard dimensions must be positive")
    rows, cols = np.indices((int(height), int(width)))
    fit = ((rows + cols + int(phase)) % 2) == 0
    return fit, ~fit


def two_level_laplacian_residual(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute full- and half-resolution Laplacian residual magnitudes."""

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have identical shapes")
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
    if prediction.ndim != 4:
        raise ValueError("Expected CHW or NCHW image tensors")

    def gray(image):
        if image.shape[1] == 1:
            return image
        if image.shape[1] != 3:
            raise ValueError("Expected one or three image channels")
        weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        return (image * weights).sum(dim=1, keepdim=True)

    kernel = prediction.new_tensor(
        ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
    ).view(1, 1, 3, 3)
    pred_gray = gray(prediction)
    target_gray = gray(target)
    fine = torch.abs(
        F.conv2d(pred_gray, kernel, padding=1)
        - F.conv2d(target_gray, kernel, padding=1)
    )
    pred_coarse = F.avg_pool2d(pred_gray, kernel_size=2, stride=2)
    target_coarse = F.avg_pool2d(target_gray, kernel_size=2, stride=2)
    coarse = torch.abs(
        F.conv2d(pred_coarse, kernel, padding=1)
        - F.conv2d(target_coarse, kernel, padding=1)
    )
    return fine, coarse


def frequency_weighted_pose_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor = None,
    gradient_threshold: float = 0.04,
    edge_weight: float = 1.0,
    side_start: float = 0.45,
    side_boost: float = 2.0,
    epsilon: float = 1.0e-3,
) -> torch.Tensor:
    """Pose-only RGB loss emphasizing causal side-view image gradients."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Pose loss expects matching NHWC image batches")
    if prediction.shape[-1] != 3:
        raise ValueError("Pose loss expects RGB images")
    gray_weights = target.new_tensor((0.299, 0.587, 0.114))
    gray = (target * gray_weights).sum(dim=-1)
    gradient = torch.zeros_like(gray)
    gradient[:, :, 1:] += torch.abs(gray[:, :, 1:] - gray[:, :, :-1])
    gradient[:, 1:, :] += torch.abs(gray[:, 1:, :] - gray[:, :-1, :])
    edge = torch.clamp(
        gradient / max(float(gradient_threshold), 1.0e-6), 0.0, 1.0
    )

    width = target.shape[2]
    columns = torch.linspace(
        -1.0, 1.0, width, device=target.device, dtype=target.dtype
    ).abs()
    side_start = min(max(float(side_start), 0.0), 1.0)
    side_ramp = torch.clamp(
        (columns - side_start) / max(1.0 - side_start, 1.0e-6), 0.0, 1.0
    ).view(1, 1, width)
    spatial = 1.0 + (max(float(side_boost), 1.0) - 1.0) * side_ramp
    weight = 1.0 + max(float(edge_weight), 0.0) * edge * spatial
    if valid_mask is not None:
        if valid_mask.ndim == 4 and valid_mask.shape[-1] == 1:
            valid_mask = valid_mask.squeeze(-1)
        if valid_mask.shape != weight.shape:
            raise ValueError("Pose valid mask has the wrong shape")
        weight = weight * valid_mask.to(weight.dtype)
    weight = weight.detach()

    residual = torch.sqrt(
        (prediction - target).square() + float(epsilon) ** 2
    ).mean(dim=-1) - float(epsilon)
    return (residual * weight).sum() / torch.clamp(weight.sum(), min=1.0)


def laplacian_pyramid_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    pixel_weight: torch.Tensor = None,
    fine_weight: float = 1.0,
    coarse_weight: float = 0.5,
) -> torch.Tensor:
    """Match RGB Laplacian bands under a detached spatial evidence mask.

    The function accepts NCHW/CHW or NHWC/HWC tensors. ``pixel_weight`` is a
    scalar mask per pixel and therefore cannot turn a color residual into a
    geometry-specific update by itself; the mapper controls which Gaussian
    parameter blocks are trainable when this loss is used.
    """

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have identical shapes")

    def as_nchw(image):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError("Expected HWC, CHW, NHWC, or NCHW image tensors")
        if image.shape[1] in (1, 3):
            return image
        if image.shape[-1] in (1, 3):
            return image.permute(0, 3, 1, 2)
        raise ValueError("Expected one or three image channels")

    pred = as_nchw(prediction)
    gt = as_nchw(target)
    if pred.shape[-2] < 2 or pred.shape[-1] < 2:
        raise ValueError("Laplacian pyramid requires at least a 2x2 image")

    kernel = pred.new_tensor(
        ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
    ).view(1, 1, 3, 3)
    kernel = kernel.expand(pred.shape[1], 1, 3, 3)

    fine = torch.abs(
        F.conv2d(pred, kernel, padding=1, groups=pred.shape[1])
        - F.conv2d(gt, kernel, padding=1, groups=gt.shape[1])
    )
    pred_coarse = F.avg_pool2d(pred, kernel_size=2, stride=2)
    gt_coarse = F.avg_pool2d(gt, kernel_size=2, stride=2)
    coarse = torch.abs(
        F.conv2d(pred_coarse, kernel, padding=1, groups=pred.shape[1])
        - F.conv2d(gt_coarse, kernel, padding=1, groups=gt.shape[1])
    )

    if pixel_weight is None:
        weight = pred.new_ones((pred.shape[0], 1, pred.shape[2], pred.shape[3]))
    else:
        weight = pixel_weight
        if weight.ndim == 2:
            weight = weight.unsqueeze(0).unsqueeze(0)
        elif weight.ndim == 3:
            weight = weight.unsqueeze(1)
        elif weight.ndim == 4 and weight.shape[-1] == 1:
            weight = weight.permute(0, 3, 1, 2)
        if weight.ndim != 4 or weight.shape[1] != 1:
            raise ValueError("Pixel weight must contain one scalar per pixel")
        if weight.shape[0] not in (1, pred.shape[0]) or weight.shape[-2:] != pred.shape[-2:]:
            raise ValueError("Pixel weight must broadcast over the image batch")
        weight = weight.to(device=pred.device, dtype=pred.dtype).detach()

    channel_count = pred.shape[1]
    fine_norm = torch.clamp(weight.sum() * channel_count, min=1.0)
    coarse_pixel_weight = F.avg_pool2d(weight, kernel_size=2, stride=2)
    coarse_norm = torch.clamp(coarse_pixel_weight.sum() * channel_count, min=1.0)
    return float(fine_weight) * (fine * weight).sum() / fine_norm + float(
        coarse_weight
    ) * (coarse * coarse_pixel_weight).sum() / coarse_norm


def evaluate_shadow_coalitions(
    evaluator: Callable[[Coalition, Dict[str, np.ndarray]], Sequence[float]],
    frozen_state: Mapping[str, np.ndarray],
    players: Sequence[str] = PLAYERS,
) -> Dict[Coalition, np.ndarray]:
    """Evaluate every coalition on independent array copies of frozen state."""

    snapshots = {
        name: np.asarray(value).copy() for name, value in frozen_state.items()
    }
    results = {}
    for coalition in all_coalitions(players):
        sandbox = {name: value.copy() for name, value in snapshots.items()}
        results[coalition] = np.atleast_1d(
            np.asarray(evaluator(coalition, sandbox), dtype=np.float64)
        )
    _validated_loss_vectors(results, tuple(players))
    for name, snapshot in snapshots.items():
        if not np.array_equal(np.asarray(frozen_state[name]), snapshot, equal_nan=True):
            raise RuntimeError("Shadow evaluator mutated live state: {}".format(name))
    return results


def injected_defect_game(
    defect_player: str,
    view_count: int = 3,
    defect_gain: float = 0.4,
    nuisance_gain: float = 0.02,
) -> Dict[Coalition, np.ndarray]:
    """Create a deterministic known-source defect game for attribution tests."""

    if defect_player not in PLAYERS:
        raise ValueError("Unknown defect player")
    invalid_gain = float(nuisance_gain) < 0.0 or float(defect_gain) <= float(
        nuisance_gain
    )
    if int(view_count) <= 0 or invalid_gain:
        raise ValueError("Defect gain must dominate a non-negative nuisance gain")
    base = np.linspace(1.0, 1.1, int(view_count), dtype=np.float64)
    game = {}
    for coalition in all_coalitions():
        reduction = sum(
            float(defect_gain) if player == defect_player else float(nuisance_gain)
            for player in coalition
        )
        game[coalition] = base - reduction
    return game
