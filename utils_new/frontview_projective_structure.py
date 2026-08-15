"""Image-conditioned covariance for weakly observable projective births."""

import numpy as np
import torch
import torch.nn.functional as F


def budget_normalized_information_radii(
    image,
    uv,
    eligible,
    base_radius_pixels,
    *,
    shuffle=False,
    seed=42,
):
    """Allocate a fixed projective birth budget by local image information.

    First-order RGB approximation error in a support cell is proportional to
    local gradient energy divided by sampling density. Minimizing its integral
    under a fixed density budget gives density proportional to the square root
    of gradient energy. Radius is inverse square-root density. Normalization
    enforces mean inverse support area equal to the isotropic budget exactly.
    """

    image = torch.as_tensor(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Information-support image must have shape [H, W, 3]")
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    eligible = np.asarray(eligible, dtype=np.bool_).reshape(-1)
    if len(uv) != len(eligible):
        raise ValueError("Information-support arrays must align")
    radius = float(base_radius_pixels)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Base support radius must be finite and positive")

    factors = np.ones((len(uv),), dtype=np.float32)
    rows = np.flatnonzero(eligible)
    if not len(rows):
        return factors, np.ones((len(uv),), dtype=np.float32)

    image = image.float()
    luminance = (
        image[..., 0] * 0.299
        + image[..., 1] * 0.587
        + image[..., 2] * 0.114
    )[None, None]
    kernels = luminance.new_tensor(
        [
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        ]
    )[:, None] / 8.0
    gradients = F.conv2d(luminance, kernels, padding=1)[0]
    energy = gradients.square().sum(dim=0)
    kernel_size = 2 * int(np.ceil(radius)) + 1
    local_energy = F.avg_pool2d(
        energy[None, None],
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )[0, 0]

    height, width = image.shape[:2]
    x = np.clip(np.floor(uv[rows, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.floor(uv[rows, 1]).astype(np.int64), 0, height - 1)
    x_tensor = torch.as_tensor(x, device=image.device, dtype=torch.long)
    y_tensor = torch.as_tensor(y, device=image.device, dtype=torch.long)
    sampled = local_energy[y_tensor, x_tensor]

    # The online frame mean is the isotropic prior. It regularizes textureless
    # locations without introducing a scene-specific threshold.
    prior = torch.clamp(
        torch.mean(local_energy), min=torch.finfo(image.dtype).eps
    )
    information_density = torch.sqrt(torch.clamp(sampled + prior, min=prior))
    information_density /= torch.mean(information_density)
    radius_factors = torch.rsqrt(information_density)
    radius_factors = radius_factors.detach().cpu().numpy().astype(np.float32)
    density = information_density.detach().cpu().numpy().astype(np.float32)

    if shuffle and len(rows) > 1:
        permutation = np.random.default_rng(int(seed)).permutation(len(rows))
        radius_factors = radius_factors[permutation]
        density = density[permutation]
    factors[rows] = radius_factors
    all_density = np.ones((len(uv),), dtype=np.float32)
    all_density[rows] = density
    return factors, all_density


def _rotation_matrices_to_quaternions(matrices):
    matrices = np.asarray(matrices, dtype=np.float64).reshape(-1, 3, 3)
    quaternions = np.empty((len(matrices), 4), dtype=np.float64)
    for row, matrix in enumerate(matrices):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quaternions[row] = (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        else:
            axis = int(np.argmax(np.diag(matrix)))
            other0 = (axis + 1) % 3
            other1 = (axis + 2) % 3
            scale = np.sqrt(
                1.0
                + matrix[axis, axis]
                - matrix[other0, other0]
                - matrix[other1, other1]
            ) * 2.0
            values = np.zeros((4,), dtype=np.float64)
            values[axis + 1] = 0.25 * scale
            values[0] = (matrix[other1, other0] - matrix[other0, other1]) / scale
            values[other0 + 1] = (
                matrix[other0, axis] + matrix[axis, other0]
            ) / scale
            values[other1 + 1] = (
                matrix[other1, axis] + matrix[axis, other1]
            ) / scale
            quaternions[row] = values
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions[quaternions[:, 0] < 0.0] *= -1.0
    return quaternions.astype(np.float32)


def structure_aligned_covariances(
    image,
    uv,
    view_directions,
    world_to_camera_rotation,
    eligible,
    *,
    support_radius_pixels=None,
    certificate_strength=None,
    shuffle=False,
    seed=42,
):
    """Return determinant-one tangent factors and ray-aligned rotations.

    The regularized image structure tensor is interpreted as tangent-plane
    precision.  Determinant normalization preserves projected Gaussian area,
    while its eigenvectors stretch support along edges and contract it across
    edges.  The isotropic regularizer is estimated from the current image.
    """

    image = torch.as_tensor(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Structure image must have shape [H, W, 3]")
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    directions = np.asarray(view_directions, dtype=np.float32).reshape(-1, 3)
    eligible = np.asarray(eligible, dtype=np.bool_).reshape(-1)
    if not (len(uv) == len(directions) == len(eligible)):
        raise ValueError("Structure covariance arrays must align")
    factors = np.ones((len(uv), 3), dtype=np.float32)
    quaternions = np.zeros((len(uv), 4), dtype=np.float32)
    quaternions[:, 0] = 1.0
    rows = np.flatnonzero(eligible)
    if len(rows) == 0:
        return factors, quaternions, np.ones((len(uv),), dtype=np.float32)

    image = image.float()
    luminance = (
        image[..., 0] * 0.299
        + image[..., 1] * 0.587
        + image[..., 2] * 0.114
    )[None, None]
    kernels = luminance.new_tensor(
        [
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        ]
    )[:, None] / 8.0
    gradients = F.conv2d(luminance, kernels, padding=1)[0]
    energy = gradients.square().sum(dim=0)

    height, width = image.shape[:2]
    x = np.clip(np.floor(uv[rows, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.floor(uv[rows, 1]).astype(np.int64), 0, height - 1)
    x_tensor = torch.as_tensor(x, device=image.device, dtype=torch.long)
    y_tensor = torch.as_tensor(y, device=image.device, dtype=torch.long)
    if support_radius_pixels is None:
        regularizer = torch.clamp(
            0.5 * torch.mean(energy), min=torch.finfo(image.dtype).eps
        )
        sampled = gradients[:, y_tensor, x_tensor].T.detach().cpu().numpy()
        sampled_energy = np.sum(sampled * sampled, axis=1)
        regularizer_value = float(regularizer.item())
        anisotropy = np.power(
            (sampled_energy + regularizer_value) / regularizer_value,
            0.25,
        ).astype(np.float32)
        tangent = np.stack((-sampled[:, 1], sampled[:, 0]), axis=1)
        tangent_norm = np.linalg.norm(tangent, axis=1)
        flat = tangent_norm <= np.finfo(np.float32).eps
        tangent[flat] = (1.0, 0.0)
        tangent_norm[flat] = 1.0
        tangent /= tangent_norm[:, None]
    else:
        radius = float(support_radius_pixels)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("Structure support radius must be finite and positive")
        kernel_size = 2 * int(np.ceil(radius)) + 1
        products = torch.stack(
            (
                gradients[0].square(),
                gradients[1].square(),
                gradients[0] * gradients[1],
            ),
            dim=0,
        )[None]
        tensor = F.avg_pool2d(
            products,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )[0]
        jxx = tensor[0, y_tensor, x_tensor]
        jyy = tensor[1, y_tensor, x_tensor]
        jxy = tensor[2, y_tensor, x_tensor]
        trace = jxx + jyy
        contrast = torch.sqrt(torch.clamp_min((jxx - jyy).square() + 4.0 * jxy.square(), 0.0))
        coherence = contrast / torch.clamp(trace, min=torch.finfo(image.dtype).eps)
        coherence = torch.clamp(coherence, 0.0, 1.0)
        angle = 0.5 * torch.atan2(2.0 * jxy, jxx - jyy)
        tangent = torch.stack((-torch.sin(angle), torch.cos(angle)), dim=1)

        if certificate_strength is None:
            strength = torch.ones_like(coherence)
        else:
            all_strength = np.asarray(certificate_strength, dtype=np.float32).reshape(-1)
            if len(all_strength) != len(eligible) or np.any(~np.isfinite(all_strength)):
                raise ValueError("Certificate strengths must align and be finite")
            if np.any((all_strength < 0.0) | (all_strength > 1.0)):
                raise ValueError("Certificate strengths must lie in [0, 1]")
            strength = torch.as_tensor(
                all_strength[rows], device=image.device, dtype=image.dtype
            )

        # Unit-determinant covariance with bounded log-anisotropy.  Coherence
        # supplies image evidence; certificate deficit suppresses it whenever
        # metric depth is close to becoming observable.
        anisotropy = torch.exp(0.5 * coherence * strength)
        anisotropy = anisotropy.detach().cpu().numpy().astype(np.float32)
        tangent = tangent.detach().cpu().numpy().astype(np.float32)

    if shuffle and len(rows) > 1:
        permutation = np.random.default_rng(int(seed)).permutation(len(rows))
        anisotropy = anisotropy[permutation]
        tangent = tangent[permutation]

    ray = directions[rows]
    ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1.0e-8)
    camera_to_world = np.asarray(
        world_to_camera_rotation, dtype=np.float32
    ).reshape(3, 3).T
    screen_x = np.broadcast_to(camera_to_world[:, 0], ray.shape).copy()
    screen_x -= np.sum(screen_x * ray, axis=1, keepdims=True) * ray
    screen_x /= np.maximum(
        np.linalg.norm(screen_x, axis=1, keepdims=True), 1.0e-8
    )
    screen_y = np.cross(ray, screen_x)
    camera_y = camera_to_world[:, 1]
    flip = np.sum(screen_y * camera_y[None, :], axis=1) < 0.0
    screen_y[flip] *= -1.0
    tangent_world = (
        tangent[:, :1] * screen_x + tangent[:, 1:] * screen_y
    )
    tangent_world /= np.maximum(
        np.linalg.norm(tangent_world, axis=1, keepdims=True), 1.0e-8
    )
    gradient_world = np.cross(ray, tangent_world)
    rotations = np.stack((tangent_world, gradient_world, ray), axis=2)

    factors[rows, 0] = anisotropy
    factors[rows, 1] = 1.0 / anisotropy
    quaternions[rows] = _rotation_matrices_to_quaternions(rotations)
    all_anisotropy = np.ones((len(uv),), dtype=np.float32)
    all_anisotropy[rows] = anisotropy
    return factors, quaternions, all_anisotropy
