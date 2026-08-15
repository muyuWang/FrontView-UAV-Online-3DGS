import pytest
import torch


def _gsplat_cuda_backend_available():
    if not torch.cuda.is_available():
        return False
    try:
        from gsplat.cuda._backend import _C
    except Exception:
        return False
    return _C is not None


pytestmark = pytest.mark.skipif(
    not _gsplat_cuda_backend_available(),
    reason="gsplat parity tests require a compiled CUDA backend",
)


def _inputs(device):
    torch.manual_seed(7)
    means = torch.tensor(
        [
            [-0.35, -0.20, 2.5],
            [0.25, -0.15, 2.8],
            [-0.15, 0.30, 3.0],
            [0.35, 0.25, 3.2],
        ],
        device=device,
    )
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4, device=device)
    scales = torch.full((4, 3), 0.16, device=device)
    opacities = torch.full((4,), 0.8, device=device)
    coefficients = torch.randn((4, 16, 3), device=device) * 0.05
    coefficients[[0, 2], 9:] = 0.0
    coefficients.requires_grad_()
    degrees = torch.tensor([2, 3, 2, 3], device=device, dtype=torch.uint8)
    viewmat = torch.eye(4, device=device)[None]
    K = torch.tensor(
        [[80.0, 0.0, 32.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]],
        device=device,
    )[None]
    return means, quats, scales, opacities, coefficients, degrees, viewmat, K


def _routed(
    inputs,
    probe=False,
    metric_confidences=None,
    appearance_confidences=None,
    uncertainty_confidences=None,
    uncertainty_cell_px=None,
):
    from utils_new.heterogeneous_sh_rasterizer import (
        heterogeneous_sh_rasterization,
    )

    means, quats, scales, opacities, coefficients, degrees, viewmat, K = inputs
    return heterogeneous_sh_rasterization(
        means,
        quats,
        scales,
        opacities,
        coefficients,
        degrees,
        viewmat,
        K,
        64,
        64,
        base_degree=2,
        target_degree=3,
        probe_inactive=probe,
        metric_confidences=metric_confidences,
        appearance_confidences=appearance_confidences,
        uncertainty_confidences=uncertainty_confidences,
        uncertainty_cell_px=uncertainty_cell_px,
        render_mode="RGB+ED",
    )


def test_mixed_degree_forward_matches_dense_zero_band_render():
    from gsplat.rendering import rasterization

    inputs = _inputs("cuda")
    means, quats, scales, opacities, coefficients, _, viewmat, K = inputs
    expected_color, expected_alpha, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        coefficients,
        viewmat,
        K,
        64,
        64,
        sh_degree=3,
        render_mode="RGB+ED",
    )
    actual_color, actual_alpha, meta = _routed(inputs)

    assert torch.allclose(actual_color, expected_color, atol=2.0e-6, rtol=2.0e-6)
    assert torch.allclose(actual_alpha, expected_alpha, atol=2.0e-6, rtol=2.0e-6)
    assert meta["heterogeneous_sh"]["packed_rows"] > 0
    assert meta["n_cameras"] == 1


def test_inactive_target_band_is_skipped_except_on_probe_steps():
    inputs = _inputs("cuda")
    coefficients = inputs[4]
    color, _, meta = _routed(inputs, probe=False)
    color[..., :3].square().mean().backward()
    inactive_gradient = coefficients.grad[[0, 2], 9:]
    assert torch.count_nonzero(inactive_gradient) == 0
    assert int(meta["heterogeneous_sh"]["probe_rows"].item()) == 0
    assert int(meta["heterogeneous_sh"]["skipped_target_band_rows"].item()) > 0

    inputs = _inputs("cuda")
    coefficients = inputs[4]
    color, _, meta = _routed(inputs, probe=True)
    color[..., :3].square().mean().backward()
    inactive_gradient = coefficients.grad[[0, 2], 9:]
    assert torch.count_nonzero(inactive_gradient) > 0
    assert int(meta["heterogeneous_sh"]["probe_rows"].item()) > 0
    assert int(meta["heterogeneous_sh"]["skipped_target_band_rows"].item()) == 0


def test_metric_depth_adds_channels_without_changing_rgb_or_full_depth():
    inputs = _inputs("cuda")
    expected_color, expected_alpha, _ = _routed(inputs)
    metric_confidences = torch.tensor(
        [1.0, 0.5, 0.0, 1.0], device="cuda", dtype=torch.float32
    )
    actual_color, actual_alpha, meta = _routed(
        inputs, metric_confidences=metric_confidences
    )

    assert actual_color.shape[-1] == 5
    assert torch.allclose(
        actual_color[..., :4], expected_color, atol=2.0e-6, rtol=2.0e-6
    )
    assert torch.allclose(actual_alpha, expected_alpha, atol=2.0e-6, rtol=2.0e-6)
    assert meta["metric_depth_mass"] is not None


def test_zero_confidence_proxies_do_not_contribute_metric_depth():
    inputs = _inputs("cuda")
    metric_confidences = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device="cuda", dtype=torch.float32
    )
    colors, _, meta = _routed(inputs, metric_confidences=metric_confidences)
    mass = meta["metric_depth_mass"]
    supported = mass[..., 0] > 1.0e-5

    assert torch.any(supported)
    assert torch.allclose(
        colors[..., 4][supported],
        torch.full_like(colors[..., 4][supported], 3.2),
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_uncertainty_mass_uses_uncertainty_confidence_not_metric_certificate():
    inputs = _inputs("cuda")
    metric_confidences = torch.zeros(4, device="cuda")
    uncertainty_confidences = torch.ones(4, device="cuda")
    _, _, meta = _routed(
        inputs,
        metric_confidences=metric_confidences,
        uncertainty_confidences=uncertainty_confidences,
        uncertainty_cell_px=0.1,
    )

    assert torch.count_nonzero(meta["metric_depth_mass"]) == 0
    assert torch.count_nonzero(meta["uncertainty_mass"]) == 0


def test_appearance_depth_preserves_pre_certificate_continuous_mass():
    inputs = _inputs("cuda")
    certificate_confidence = torch.zeros(4, device="cuda")
    appearance_confidence = torch.tensor(
        [1.0, 0.5, 0.0, 1.0], device="cuda", dtype=torch.float32
    )
    colors, alphas, meta = _routed(
        inputs,
        metric_confidences=certificate_confidence,
        appearance_confidences=appearance_confidence,
    )
    expected_colors, expected_alphas, expected_meta = _routed(
        inputs, metric_confidences=appearance_confidence
    )

    assert colors.shape[-1] == 5
    assert torch.count_nonzero(meta["metric_depth_mass"]) == 0
    assert torch.allclose(
        meta["appearance_depth_mass"],
        expected_meta["metric_depth_mass"],
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    supported = meta["appearance_depth_mass"][..., 0] > 1.0e-5
    assert torch.allclose(
        meta["appearance_depth"][..., 0][supported],
        expected_colors[..., 4][supported],
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert torch.allclose(alphas, expected_alphas, atol=2.0e-6, rtol=2.0e-6)
