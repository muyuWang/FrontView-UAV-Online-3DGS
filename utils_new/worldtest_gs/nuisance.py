"""Window-shared pose/depth nuisance MAP estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def skew(vector):
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def cubic_basis(frame_id, first_frame, stride, knot_count):
    coordinate = max(0.0, (float(frame_id) - float(first_frame)) / float(stride))
    base = int(math.floor(coordinate))
    t = coordinate - base
    weights = np.asarray(
        [
            (1.0 - t) ** 3 / 6.0,
            (3.0 * t**3 - 6.0 * t**2 + 4.0) / 6.0,
            (-3.0 * t**3 + 3.0 * t**2 + 3.0 * t + 1.0) / 6.0,
            t**3 / 6.0,
        ]
    )
    indices = np.clip(np.arange(base, base + 4), 0, knot_count - 1)
    combined = np.zeros((knot_count,), dtype=np.float64)
    np.add.at(combined, indices, weights)
    return combined


def left_pose_update(world_to_camera, delta):
    delta = np.asarray(delta, dtype=np.float64)
    updated = np.asarray(world_to_camera, dtype=np.float64).copy()
    rotation = np.eye(3) + skew(delta[3:])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = delta[:3]
    return transform @ updated


@dataclass
class NuisanceState:
    valid: bool
    failure_reason: str
    first_frame: int
    stride: int
    knot_values: np.ndarray
    depth_log_scale: float
    depth_bias: float
    covariance: np.ndarray
    undamped_rank_ratio: float
    observation_count: int

    def pose_delta(self, frame_id):
        if self.knot_values.size == 0:
            return np.zeros((6,), dtype=np.float64)
        basis = cubic_basis(
            frame_id, self.first_frame, self.stride, len(self.knot_values)
        )
        return basis @ self.knot_values

    def corrected_pose(self, observation):
        return left_pose_update(
            observation.world_to_camera, self.pose_delta(observation.frame_id)
        )

    def corrected_inverse_depth(self, inverse_depth):
        return math.exp(self.depth_log_scale) * float(inverse_depth) + self.depth_bias


class SharedNuisanceSolver:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _predict(point, observation, delta, log_scale, bias):
        pose = left_pose_update(observation.world_to_camera, delta)
        homogeneous = np.append(np.asarray(point, dtype=np.float64), 1.0)
        camera = (pose @ homogeneous)[:3]
        if not np.isfinite(camera).all() or camera[2] <= 1.0e-6:
            return None, None
        k = np.asarray(observation.intrinsics, dtype=np.float64)
        screen = k @ camera
        rho = math.exp(log_scale) / camera[2] + bias
        prediction = np.asarray(
            [screen[0] / screen[2], screen[1] / screen[2], rho]
        )
        fx, fy = k[0, 0], k[1, 1]
        x, y, z = camera
        projection_jacobian = np.asarray(
            [
                [fx / z, 0.0, -fx * x / (z * z)],
                [0.0, fy / z, -fy * y / (z * z)],
                [0.0, 0.0, -math.exp(log_scale) / (z * z)],
            ]
        )
        pose_jacobian = projection_jacobian @ np.concatenate(
            [np.eye(3), -skew(camera)], axis=1
        )
        return prediction, pose_jacobian

    def solve(self, groups, current_frame):
        minimum = int(self.config["min_views"])
        recent = []
        first_allowed = int(current_frame) - int(self.config["nuisance_window_frames"]) + 1
        for group in groups:
            observations = [
                observation
                for observation in group.observations
                if observation.frame_id >= first_allowed
            ]
            if len(observations) >= minimum:
                point = np.median(
                    np.asarray([observation.world_point for observation in observations]),
                    axis=0,
                )
                recent.append((point, observations))
        if not recent:
            return NuisanceState(
                False,
                "no valid shared shadow tracks",
                int(current_frame),
                4,
                np.empty((0, 6)),
                0.0,
                0.0,
                np.empty((0, 0)),
                0.0,
                0,
            )
        frame_ids = sorted(
            {observation.frame_id for _, observations in recent for observation in observations}
        )
        first_frame = frame_ids[0]
        stride = int(self.config["nuisance_knot_stride"])
        knot_count = 0
        if len(frame_ids) >= 4:
            knot_count = max(
                4,
                int(math.floor((frame_ids[-1] - first_frame) / stride)) + 4,
            )
        pose_parameter_count = max(0, knot_count - 1) * 6
        parameter_count = pose_parameter_count + 2
        theta = np.zeros((parameter_count,), dtype=np.float64)
        undamped_ratio = 0.0
        observation_count = 0
        covariance = np.empty((0, 0))
        for _ in range(3):
            rows = []
            residuals = []
            for point, observations in recent[:256]:
                for observation in observations:
                    basis = (
                        cubic_basis(observation.frame_id, first_frame, stride, knot_count)
                        if knot_count
                        else np.empty((0,))
                    )
                    delta = (
                        basis[1:] @ theta[:pose_parameter_count].reshape(-1, 6)
                        if pose_parameter_count
                        else np.zeros((6,))
                    )
                    prediction, pose_jacobian = self._predict(
                        point,
                        observation,
                        delta,
                        theta[-2],
                        theta[-1],
                    )
                    if prediction is None:
                        continue
                    observed = np.asarray(
                        [
                            observation.uv[0],
                            observation.uv[1],
                            observation.inverse_depth,
                        ]
                    )
                    sigma = np.asarray(
                        [
                            float(self.config["pixel_sigma"]),
                            float(self.config["pixel_sigma"]),
                            math.sqrt(max(observation.inverse_depth_variance, 1.0e-12)),
                        ]
                    )
                    residual = (observed - prediction) / sigma
                    jacobian = np.zeros((3, parameter_count), dtype=np.float64)
                    if pose_parameter_count:
                        for knot in range(1, knot_count):
                            block = slice((knot - 1) * 6, knot * 6)
                            jacobian[:, block] = pose_jacobian * basis[knot]
                    jacobian[2, -2] = math.exp(theta[-2]) / max(
                        (left_pose_update(observation.world_to_camera, delta) @ np.append(point, 1.0))[2],
                        1.0e-8,
                    )
                    jacobian[2, -1] = 1.0
                    jacobian /= sigma[:, None]
                    norm = float(np.linalg.norm(residual))
                    delta_huber = float(self.config["nuisance_huber_delta"])
                    weight = 1.0 if norm <= delta_huber else delta_huber / max(norm, 1.0e-8)
                    rows.append(math.sqrt(weight) * jacobian)
                    residuals.append(math.sqrt(weight) * residual)
            if not rows:
                break
            jacobian = np.concatenate(rows, axis=0)
            residual = np.concatenate(residuals)
            observation_count = len(rows)
            information = jacobian.T @ jacobian
            # Rank is assessed after column normalization so the ratio measures
            # identifiability rather than mixing metre/radian/inverse-depth units.
            column_norms = np.linalg.norm(jacobian, axis=0)
            if np.any(column_norms <= 1.0e-10):
                eigenvalues = np.zeros((parameter_count,), dtype=np.float64)
            else:
                normalized_jacobian = jacobian / column_norms[None, :]
                eigenvalues = np.linalg.eigvalsh(
                    normalized_jacobian.T @ normalized_jacobian
                )
            positive = eigenvalues[eigenvalues > 1.0e-10]
            undamped_ratio = (
                float(positive.min() / positive.max())
                if positive.size == parameter_count
                else 0.0
            )
            prior_precision = np.zeros((parameter_count,), dtype=np.float64)
            if pose_parameter_count:
                per_knot = np.asarray(
                    [
                        *([1.0 / float(self.config["pose_translation_sigma_m"]) ** 2] * 3),
                        *(
                            [
                                1.0
                                / math.radians(
                                    float(self.config["pose_rotation_sigma_deg"])
                                )
                                ** 2
                            ]
                            * 3
                        ),
                    ]
                )
                prior_precision[:pose_parameter_count] = np.tile(
                    per_knot, knot_count - 1
                )
            prior_precision[-2:] = np.asarray([400.0, 10000.0])
            hessian = information + np.diag(prior_precision) + float(
                self.config["nuisance_damping"]
            ) * np.eye(parameter_count)
            gradient = jacobian.T @ residual - prior_precision * theta
            if not np.isfinite(hessian).all() or not np.isfinite(gradient).all():
                break
            try:
                step = np.linalg.solve(hessian, gradient)
                covariance = np.linalg.inv(hessian)
            except np.linalg.LinAlgError:
                break
            theta += step
            if np.linalg.norm(step) < 1.0e-6:
                break
        rank_valid = undamped_ratio >= float(self.config["rank_ratio_threshold"])
        finite = np.isfinite(theta).all() and np.isfinite(covariance).all()
        valid = bool(observation_count > 0 and rank_valid and finite)
        knot_values = np.zeros((knot_count, 6), dtype=np.float64)
        if pose_parameter_count:
            knot_values[1:] = theta[:pose_parameter_count].reshape(-1, 6)
        return NuisanceState(
            valid,
            "" if valid else "rank deficient or non-finite shared nuisance information",
            first_frame,
            stride,
            knot_values,
            float(theta[-2]),
            float(theta[-1]),
            covariance,
            undamped_ratio,
            observation_count,
        )
