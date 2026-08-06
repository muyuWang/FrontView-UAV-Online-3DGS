"""Camera and quaternion math shared by the progressive state machine."""

import math
from typing import Optional, Tuple

import torch


def camera_center(world_to_camera: torch.Tensor) -> torch.Tensor:
    """Return the world-space camera center for a world-to-camera matrix."""
    return torch.linalg.inv(world_to_camera)[:3, 3]


def unproject_pixel(
    uv: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor, world_to_camera: torch.Tensor
) -> torch.Tensor:
    """Unproject one pixel using the repository's world-to-camera pose convention."""
    uv1 = torch.cat((uv, torch.ones(1, device=uv.device, dtype=uv.dtype)))
    camera_point = torch.linalg.solve(intrinsics, uv1) * depth
    world_point = torch.linalg.inv(world_to_camera) @ torch.cat(
        (camera_point, torch.ones(1, device=uv.device, dtype=uv.dtype))
    )
    return world_point[:3]


def project_world(
    point: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: Optional[Tuple[int, int]] = None,
    near: float = 0.0,
    far: float = float("inf"),
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Project a world point and report positive-depth/image-bounds validity."""
    homogeneous = torch.cat((point, torch.ones(1, device=point.device, dtype=point.dtype)))
    camera_point = (world_to_camera @ homogeneous)[:3]
    z = camera_point[2]
    safe_z = torch.clamp(z, min=torch.finfo(z.dtype).eps)
    pixel = intrinsics @ camera_point
    uv = pixel[:2] / safe_z
    valid = bool(torch.isfinite(uv).all() and z > near and z < far)
    if valid and image_size is not None:
        height, width = image_size
        valid = bool(0 <= uv[0] < width and 0 <= uv[1] < height)
    return uv, z, valid


def project_world_batch(
    points: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: Optional[Tuple[int, int]] = None,
    near: float = 0.0,
    far: float = float("inf"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project N world points without introducing per-point GPU synchronizations."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape Nx3")
    if points.shape[0] == 0:
        return (
            points.new_empty((0, 2)),
            points.new_empty((0,)),
            torch.empty((0,), dtype=torch.bool, device=points.device),
        )
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    camera_points = (world_to_camera @ homogeneous.T).T[:, :3]
    depth = camera_points[:, 2]
    safe_depth = torch.clamp(depth, min=torch.finfo(depth.dtype).eps)
    pixels = (intrinsics @ camera_points.T).T
    uv = pixels[:, :2] / safe_depth[:, None]
    valid = torch.isfinite(uv).all(dim=1) & (depth > near) & (depth < far)
    if image_size is not None:
        height, width = image_size
        valid &= (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
    return uv, depth, valid


def parallax_angle(
    world_point: torch.Tensor, reference_pose: torch.Tensor, current_pose: torch.Tensor
) -> float:
    """Compute the angle between reference and current viewing rays."""
    ray_ref = world_point - camera_center(reference_pose)
    ray_cur = world_point - camera_center(current_pose)
    ray_ref = ray_ref / torch.clamp(torch.linalg.norm(ray_ref), min=1.0e-8)
    ray_cur = ray_cur / torch.clamp(torch.linalg.norm(ray_cur), min=1.0e-8)
    cosine = torch.clamp(torch.dot(ray_ref, ray_cur), -1.0, 1.0)
    return float(torch.acos(cosine).item())


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert a gsplat wxyz quaternion to a 3x3 rotation matrix."""
    q = quaternion / torch.clamp(torch.linalg.norm(quaternion), min=1.0e-8)
    w, x, y, z = q.unbind()
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        )
    ).reshape(3, 3)


def quaternion_from_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3 rotation matrix to a normalized gsplat wxyz quaternion."""
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape 3x3")
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0))
    qx = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)), m21 - m12
    )
    qy = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0)), m02 - m20
    )
    qz = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0)), m10 - m01
    )
    quaternion = torch.stack((qw, qx, qy, qz))
    return quaternion / torch.clamp(torch.linalg.norm(quaternion), min=1.0e-8)


def quaternion_from_normal(
    normal: torch.Tensor, tangent_hint: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Align local +Z to a normal while preserving a reference tangent axis."""
    normal = normal / torch.clamp(torch.linalg.norm(normal), min=1.0e-8)
    if tangent_hint is not None:
        tangent_x = tangent_hint - torch.dot(tangent_hint, normal) * normal
        tangent_x = tangent_x / torch.clamp(
            torch.linalg.norm(tangent_x), min=1.0e-8
        )
        tangent_y = torch.linalg.cross(normal, tangent_x)
        rotation = torch.stack((tangent_x, tangent_y, normal), dim=1)
        return quaternion_from_matrix(rotation)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=normal.device, dtype=normal.dtype)
    dot = torch.clamp(torch.dot(z_axis, normal), -1.0, 1.0)
    if float(dot) < -0.999999:
        return torch.tensor([0.0, 1.0, 0.0, 0.0], device=normal.device, dtype=normal.dtype)
    cross = torch.linalg.cross(z_axis, normal)
    quaternion = torch.cat(((1.0 + dot).reshape(1), cross))
    return quaternion / torch.clamp(torch.linalg.norm(quaternion), min=1.0e-8)


def fronto_parallel_quaternion(world_to_camera: torch.Tensor) -> torch.Tensor:
    """Build a plane orientation parallel to the reference image plane."""
    camera_to_world = torch.linalg.inv(world_to_camera)
    return quaternion_from_normal(camera_to_world[:3, 2])


def normalized_entropy(log_weights: torch.Tensor) -> float:
    probabilities = torch.softmax(log_weights, dim=0)
    entropy = -(probabilities * torch.log(torch.clamp(probabilities, min=1.0e-12))).sum()
    return float((entropy / math.log(max(2, probabilities.numel()))).item())
