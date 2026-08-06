"""Pose covariance provider used by nuisance-aware risk evaluation."""

import math

import numpy as np


class PoseUncertaintyProvider:
    """Fixed diagonal fallback with an interface for future BA covariance."""

    def __init__(self, config):
        self.mode = config["pose_covariance_mode"]
        if self.mode != "fixed_diagonal":
            raise ValueError("Only fixed_diagonal pose covariance is available in the MVP")
        rotation_sigma = math.radians(float(config["pose_rotation_sigma_deg"]))
        translation_sigma = float(config["pose_translation_sigma_scene_units"])
        # Explicit convention: [tx, ty, tz, rx, ry, rz], left camera perturbation.
        self.covariance = np.diag(
            [translation_sigma**2] * 3 + [rotation_sigma**2] * 3
        ).astype(np.float32)

    def get_covariance(self, frame_id: int) -> np.ndarray:
        return self.covariance.copy()
