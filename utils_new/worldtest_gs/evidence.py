"""Held-out shared-world versus independent-view predictive evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def log_gaussian(value, mean, covariance):
    value = np.asarray(value, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0 or not np.isfinite(logdet):
        return float("-inf"), float("inf")
    delta = value - mean
    try:
        mahalanobis = float(delta @ np.linalg.solve(covariance, delta))
    except np.linalg.LinAlgError:
        return float("-inf"), float("inf")
    dimension = value.size
    return -0.5 * (dimension * math.log(2.0 * math.pi) + logdet + mahalanobis), mahalanobis


def linear_gaussian_predictive(latent_mean, latent_covariance, matrix, noise_covariance):
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix @ np.asarray(latent_mean, dtype=np.float64)
    covariance = (
        matrix @ np.asarray(latent_covariance, dtype=np.float64) @ matrix.T
        + np.asarray(noise_covariance, dtype=np.float64)
    )
    return mean, covariance


def monte_carlo_predictive_log_density(
    value,
    latent_mean,
    latent_covariance,
    matrix,
    noise_covariance,
    samples=100000,
    seed=0,
):
    rng = np.random.default_rng(seed)
    latents = rng.multivariate_normal(latent_mean, latent_covariance, size=int(samples))
    means = latents @ np.asarray(matrix, dtype=np.float64).T
    inverse = np.linalg.inv(noise_covariance)
    sign, logdet = np.linalg.slogdet(noise_covariance)
    if sign <= 0:
        return float("-inf")
    delta = np.asarray(value, dtype=np.float64)[None, :] - means
    log_values = -0.5 * (
        value.size * math.log(2.0 * math.pi)
        + logdet
        + np.einsum("ni,ij,nj->n", delta, inverse, delta)
    )
    maximum = float(np.max(log_values))
    return maximum + math.log(float(np.mean(np.exp(log_values - maximum))))


def project(point, pose, intrinsics):
    camera = np.asarray(pose, dtype=np.float64) @ np.append(point, 1.0)
    if not np.isfinite(camera).all() or camera[2] <= 1.0e-8:
        return None, None
    k = np.asarray(intrinsics, dtype=np.float64)
    screen = k @ camera[:3]
    prediction = np.asarray(
        [screen[0] / screen[2], screen[1] / screen[2], 1.0 / camera[2]]
    )
    x, y, z = camera[:3]
    jacobian_camera = np.asarray(
        [
            [k[0, 0] / z, 0.0, -k[0, 0] * x / z**2],
            [0.0, k[1, 1] / z, -k[1, 1] * y / z**2],
            [0.0, 0.0, -1.0 / z**2],
        ]
    )
    return prediction, jacobian_camera @ np.asarray(pose, dtype=np.float64)[:3, :3]


def unproject(observation, nuisance):
    rho = nuisance.corrected_inverse_depth(observation.inverse_depth)
    if not math.isfinite(rho) or rho <= 0.0:
        return None
    uv1 = np.asarray([observation.uv[0], observation.uv[1], 1.0])
    camera = np.linalg.solve(observation.intrinsics, uv1) / rho
    pose = nuisance.corrected_pose(observation)
    return (np.linalg.inv(pose) @ np.append(camera, 1.0))[:3]


def maximum_parallax_deg(points, observations, nuisance):
    center_point = np.median(np.asarray(points), axis=0)
    directions = []
    for observation in observations:
        center = np.linalg.inv(nuisance.corrected_pose(observation))[:3, 3]
        direction = center_point - center
        norm = np.linalg.norm(direction)
        if norm > 1.0e-8:
            directions.append(direction / norm)
    maximum = 0.0
    for index, first in enumerate(directions):
        for second in directions[index + 1 :]:
            maximum = max(
                maximum,
                math.degrees(math.acos(float(np.clip(first @ second, -1.0, 1.0)))),
            )
    return maximum


@dataclass
class EvidenceResult:
    q_g: float
    passed: bool
    worst_heldout_frame: int
    worst_prior_scale: float
    rank_ratio: float
    failure_reason: str
    heldout: list[dict]


class WorldIdentityEvidence:
    def __init__(self, config):
        self.config = config

    def evaluate(self, group, nuisance):
        observations = list(group.observations)
        if len({item.frame_id for item in observations}) < int(self.config["min_views"]):
            return EvidenceResult(
                float("-inf"), False, -1, 0.0, 0.0, "fewer than three views", []
            )
        if not nuisance.valid:
            return EvidenceResult(
                float("-inf"), False, -1, 0.0, nuisance.undamped_rank_ratio,
                nuisance.failure_reason, []
            )
        points = [unproject(observation, nuisance) for observation in observations]
        if any(point is None or not np.isfinite(point).all() for point in points):
            return EvidenceResult(
                float("-inf"), False, -1, 0.0, 0.0, "non-finite unprojection", []
            )
        parallax = maximum_parallax_deg(points, observations, nuisance)
        if parallax < float(self.config["minimum_parallax_deg"]):
            return EvidenceResult(
                float("-inf"), False, -1, 0.0, 0.0, "weak parallax rank failure", []
            )
        heldout_results = []
        width = max(float(2.0 * observations[0].intrinsics[0, 2]), 1.0)
        height = max(float(2.0 * observations[0].intrinsics[1, 2]), 1.0)
        near = float(self.config["scene_prior_near"])
        far = float(self.config["scene_prior_far"])
        for heldout_index, heldout in enumerate(observations):
            training_points = np.asarray(
                [point for index, point in enumerate(points) if index != heldout_index]
            )
            if len(training_points) < 2:
                continue
            point_mean = np.mean(training_points, axis=0)
            centered = training_points - point_mean
            empirical = centered.T @ centered / max(len(training_points) - 1, 1)
            base_variance = max(
                float(np.median([item.inverse_depth_variance for item in observations])),
                float(self.config["inverse_depth_sigma_floor"]) ** 2,
            )
            point_covariance = empirical / len(training_points) + np.eye(3) * base_variance
            pose = nuisance.corrected_pose(heldout)
            prediction, jacobian = project(point_mean, pose, heldout.intrinsics)
            if prediction is None:
                return EvidenceResult(
                    float("-inf"), False, heldout.frame_id, 0.0, 0.0,
                    "held-out projection is invalid", heldout_results
                )
            depth_scale = math.exp(nuisance.depth_log_scale)
            prediction[2] = nuisance.corrected_inverse_depth(prediction[2])
            jacobian[2] *= depth_scale
            observed_rho = nuisance.corrected_inverse_depth(heldout.inverse_depth)
            observed = np.asarray([heldout.uv[0], heldout.uv[1], observed_rho])
            noise = np.diag(
                [
                    float(self.config["pixel_sigma"]) ** 2,
                    float(self.config["pixel_sigma"]) ** 2,
                    max(
                        heldout.inverse_depth_variance,
                        float(self.config["inverse_depth_sigma_floor"]) ** 2,
                    ),
                ]
            )
            for prior_scale in self.config["prior_scales"]:
                covariance = (
                    jacobian @ (point_covariance * float(prior_scale) ** 2) @ jacobian.T
                    + noise
                )
                log_world, mahalanobis = log_gaussian(observed, prediction, covariance)
                if not math.isfinite(log_world):
                    evidence = float("-inf")
                    variance = float("inf")
                else:
                    rho = max(observed_rho, 1.0 / far)
                    log_independent = (
                        -math.log(width * height * (far - near)) - 2.0 * math.log(rho)
                    )
                    evidence = log_world - log_independent
                    variance = 0.25 + 0.5 * max(mahalanobis, 0.0)
                conservative = evidence - 1.645 * math.sqrt(variance)
                heldout_results.append(
                    {
                        "frame_id": int(heldout.frame_id),
                        "prior_scale": float(prior_scale),
                        "log_world": float(log_world),
                        "log_independent": float(log_independent)
                        if math.isfinite(log_world)
                        else float("-inf"),
                        "mahalanobis": float(mahalanobis),
                        "evidence": float(evidence),
                        "evidence_variance": float(variance),
                        "conservative_evidence": float(conservative),
                    }
                )
        if not heldout_results:
            return EvidenceResult(
                float("-inf"), False, -1, 0.0, 0.0, "no held-out result", []
            )
        worst = min(heldout_results, key=lambda item: item["conservative_evidence"])
        q_g = float(worst["conservative_evidence"])
        finite = math.isfinite(q_g)
        return EvidenceResult(
            q_g,
            finite and q_g > float(self.config["qg_threshold"]),
            int(worst["frame_id"]),
            float(worst["prior_scale"]),
            float(nuisance.undamped_rank_ratio),
            "" if finite else "non-finite evidence",
            heldout_results,
        )

    @staticmethod
    def npo_lite_score(group):
        points = np.asarray([observation.world_point for observation in group.observations])
        if len(points) < 3:
            return float("inf")
        center = np.median(points, axis=0)
        scale = max(float(np.median(np.linalg.norm(points, axis=1))), 1.0e-6)
        return float(np.median(np.linalg.norm(points - center, axis=1)) / scale)
