"""JSONL statistics and lightweight state overlays for progressive mapping."""

import json
import math
import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

from .projective_anchor_bank import promotion_thresholds_for_anchor
from .types import ProgressiveFrameStats, ProjectiveAnchor


class ProgressiveDebugWriter:
    def __init__(self, output_dir: Optional[str], enabled: bool, save_interval: int):
        self.enabled = bool(enabled and output_dir)
        self.save_interval = save_interval
        self.output_dir = output_dir
        self.stats_path = None
        self.p_histogram_path = None
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            self.stats_path = os.path.join(output_dir, "progressive_stats.jsonl")
            self.p_histogram_path = os.path.join(
                output_dir, "progressive_p_histograms.jsonl"
            )
        if self.enabled:
            os.makedirs(os.path.join(output_dir, "progressive_debug"), exist_ok=True)

    def write_stats(self, stats: ProgressiveFrameStats) -> None:
        if self.stats_path is None:
            return
        with open(self.stats_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(stats.to_dict(), sort_keys=True) + "\n")

    def should_write_histograms(self, frame_id: int) -> bool:
        return bool(
            self.p_histogram_path is not None
            and frame_id % self.save_interval == 0
        )

    @staticmethod
    def _histogram(values: Sequence[float], upper_bounds: Sequence[float]) -> Dict[str, object]:
        array = np.asarray(values, dtype=np.float64)
        counts = []
        lower = -np.inf
        for upper in upper_bounds:
            counts.append(int(np.count_nonzero((array > lower) & (array <= upper))))
            lower = upper
        counts.append(int(np.count_nonzero(array > lower)))
        return {
            "upper_bounds": list(upper_bounds),
            "counts": counts,
        }

    def write_p_histograms(
        self,
        frame_id: int,
        anchors: Iterable[ProjectiveAnchor],
        config: Dict[str, object],
        num_promoted: int,
        num_pruned: int,
        state_depth_bands: Optional[Dict[str, object]] = None,
    ) -> None:
        """Record P posterior distributions and per-condition promotion failures."""
        if (
            not self.should_write_histograms(frame_id)
        ):
            return
        anchors = list(anchors)
        best_weights = []
        if anchors:
            log_weights = torch.stack(
                [anchor.mode_log_weights.detach().float().cpu() for anchor in anchors]
            )
            best_weights = torch.softmax(log_weights, dim=1).amax(dim=1).tolist()
        observation_counts = [anchor.observation_count for anchor in anchors]
        entropies = [anchor.posterior_entropy for anchor in anchors]
        relative_stds = [
            math.sqrt(max(0.0, anchor.posterior_variance))
            / max(anchor.posterior_mean, 1.0e-8)
            for anchor in anchors
        ]
        parallaxes = [math.degrees(anchor.max_parallax_rad) for anchor in anchors]
        errors = [anchor.best_error_ema for anchor in anchors]
        ages = [max(0, frame_id - anchor.reference_frame_id) for anchor in anchors]
        unseen = [max(0, frame_id - anchor.last_seen_frame) for anchor in anchors]

        thresholds = {
            "min_observations": int(config["promotion_min_observations"]),
            "min_best_weight": float(config["promotion_min_best_weight"]),
            "max_normalized_entropy": float(
                config["promotion_max_normalized_entropy"]
            ),
            "max_relative_std": float(config["promotion_max_relative_std"]),
            "min_parallax_deg": float(config["promotion_min_parallax_deg"]),
            "max_match_error": float(config["promotion_max_match_error"]),
            "near": {
                "max_depth_m": float(config["near_promotion_max_depth_m"]),
                "min_observations": int(config["near_promotion_min_observations"]),
                "min_best_weight": float(config["near_promotion_min_best_weight"]),
                "max_normalized_entropy": float(
                    config["near_promotion_max_normalized_entropy"]
                ),
                "max_relative_std": float(config["near_promotion_max_relative_std"]),
                "min_parallax_deg": float(config["near_promotion_min_parallax_deg"]),
                "max_match_error": float(config["near_promotion_max_match_error"]),
            },
        }
        per_anchor_thresholds = [
            promotion_thresholds_for_anchor(anchor, config) for anchor in anchors
        ]
        failures = {
            "observations": sum(
                value < anchor_thresholds["min_observations"]
                for value, anchor_thresholds in zip(
                    observation_counts, per_anchor_thresholds
                )
            ),
            "best_weight": sum(
                value < anchor_thresholds["min_best_weight"]
                for value, anchor_thresholds in zip(best_weights, per_anchor_thresholds)
            ),
            "entropy": sum(
                value > anchor_thresholds["max_normalized_entropy"]
                for value, anchor_thresholds in zip(entropies, per_anchor_thresholds)
            ),
            "relative_std": sum(
                value > anchor_thresholds["max_relative_std"]
                for value, anchor_thresholds in zip(relative_stds, per_anchor_thresholds)
            ),
            "parallax": sum(
                value < anchor_thresholds["min_parallax_deg"]
                for value, anchor_thresholds in zip(parallaxes, per_anchor_thresholds)
            ),
            "match_error": sum(
                value > anchor_thresholds["max_match_error"]
                for value, anchor_thresholds in zip(errors, per_anchor_thresholds)
            ),
        }
        record = {
            "frame_id": frame_id,
            "num_anchors": len(anchors),
            "num_promoted_this_frame": num_promoted,
            "num_pruned_this_frame": num_pruned,
            "thresholds": thresholds,
            "promotion_failures": failures,
            "near_anchor_count": sum(
                anchor.reference_depth_valid
                and 0.0 < anchor.reference_depth_prior <= float(
                    config["near_promotion_max_depth_m"]
                )
                for anchor in anchors
            ),
            "state_depth_bands": state_depth_bands,
            "histograms": {
                "reference_depth_m": self._histogram(
                    [anchor.reference_depth_prior for anchor in anchors],
                    config["depth_histogram_edges_m"],
                ),
                "observation_count": self._histogram(
                    observation_counts, [1, 2, 3, 4, 6, 10, 20]
                ),
                "best_weight": self._histogram(
                    best_weights, [0.30, 0.40, 0.50, 0.55, 0.70, 0.85]
                ),
                "normalized_entropy": self._histogram(
                    entropies, [0.20, 0.40, 0.60, 0.80, 0.90, 0.98]
                ),
                "relative_std": self._histogram(
                    relative_stds, [0.10, 0.20, 0.30, 0.45, 0.60, 1.00]
                ),
                "parallax_deg": self._histogram(
                    parallaxes, [0.10, 0.25, 0.35, 0.50, 1.00, 2.00]
                ),
                "best_error_ema": self._histogram(
                    errors, [0.10, 0.20, 0.30, 0.40, 0.50, 0.75]
                ),
                "age_frames": self._histogram(
                    ages, [5, 10, 20, 30, 45, 60, 100]
                ),
                "unseen_frames": self._histogram(
                    unseen, [0, 1, 3, 5, 10, 20, 40]
                ),
            },
        }
        with open(self.p_histogram_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _rgb8(image: torch.Tensor) -> np.ndarray:
        return np.clip(image.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)

    def save_frame(
        self,
        frame_id: int,
        image: torch.Tensor,
        stable_render: torch.Tensor,
        opacity: torch.Tensor,
        candidate_mask: torch.Tensor,
        projective_points: Iterable[Tuple[float, float, float, float]],
        metric_points: Iterable[Tuple[float, float]],
        surface_points: Iterable[Tuple[float, float]],
        archive_points: Iterable[Tuple[float, float]],
        proxy_render: Optional[torch.Tensor] = None,
        state_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        if not self.enabled or frame_id % self.save_interval != 0:
            return
        import cv2

        debug_dir = os.path.join(self.output_dir, "progressive_debug")
        overlay = self._rgb8(image).copy()
        for x, y, entropy, best_weight in projective_points:
            color = (int(255 * entropy), 0, int(255 * (1.0 - entropy)))
            cv2.circle(overlay, (int(x), int(y)), max(2, int(2 + 4 * best_weight)), color, -1)
        for points, color in (
            (metric_points, (255, 80, 0)),
            (surface_points, (0, 220, 0)),
            (archive_points, (128, 128, 128)),
        ):
            for x, y in points:
                cv2.circle(overlay, (int(x), int(y)), 4, color, 1)
        display_counts = state_counts or {
            "P": len(list(projective_points)),
            "M": len(list(metric_points)),
            "S": len(list(surface_points)),
            "A": len(list(archive_points)),
        }
        count_text = "P:{P}  M:{M}  S:{S}  A:{A}".format(**display_counts)
        cv2.putText(
            overlay,
            count_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        stable = self._rgb8(stable_render).copy()
        text_size, baseline = cv2.getTextSize(
            count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2
        )
        cv2.rectangle(
            stable,
            (8, 7),
            (20 + text_size[0], 18 + text_size[1] + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            stable,
            count_text,
            (14, 14 + text_size[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        products = {
            "rgb": self._rgb8(image),
            "stable": stable,
            "opacity": np.repeat(self._rgb8(opacity.squeeze().unsqueeze(-1)), 3, axis=2),
            "candidate": np.repeat(candidate_mask.detach().cpu().numpy()[..., None].astype(np.uint8) * 255, 3, axis=2),
            "states": overlay,
        }
        if proxy_render is not None:
            products["proxy"] = self._rgb8(proxy_render)
        for name, value in products.items():
            cv2.imwrite(
                os.path.join(debug_dir, "{:06d}_{}.png".format(frame_id, name)),
                cv2.cvtColor(value, cv2.COLOR_RGB2BGR),
            )
