"""Parameter-domain proximal anchors for geometry-frozen appearance replay."""

from __future__ import annotations

import torch


APPEARANCE_PARAMETER_NAMES = ("sh0", "shN", "opacities")


def validate_appearance_anchor_config(config):
    config = dict(config or {})
    validated = {
        "sh0_weight": float(config.get("sh0_weight", 0.0)),
        "shN_weight": float(config.get("shN_weight", 0.0)),
        "opacity_weight": float(config.get("opacity_weight", 0.0)),
    }
    if any(value < 0.0 for value in validated.values()):
        raise ValueError("Appearance anchor weights must be non-negative")
    validated["enabled"] = any(value > 0.0 for value in validated.values())
    return validated


class AppearanceProximalAnchor:
    """Keep replay appearance near its causal online estimate without freezing it."""

    def __init__(self, gaussian_model, config):
        self.config = validate_appearance_anchor_config(config)
        self.weights = {
            "sh0": self.config["sh0_weight"],
            "shN": self.config["shN_weight"],
            "opacities": self.config["opacity_weight"],
        }
        self.anchors = {}
        if not self.config["enabled"]:
            return

        for group_id in gaussian_model.valid_groups:
            group = gaussian_model.gaussian_groups[group_id]
            if group.splats is None or group.get_num == 0:
                continue
            self.anchors[int(group_id)] = {
                name: group.splats[name].detach().clone()
                for name, weight in self.weights.items()
                if weight > 0.0
            }
        if not self.anchors:
            raise ValueError("Appearance anchoring requires active Gaussian parameters")

    def loss(self, gaussian_model):
        if not self.config["enabled"]:
            raise RuntimeError("Appearance anchor loss requested while disabled")

        component_sums = {}
        component_counts = {}
        for group_id, references in self.anchors.items():
            group = gaussian_model.gaussian_groups[group_id]
            if group.splats is None:
                raise RuntimeError("Anchored Gaussian group is no longer active")
            for name, reference in references.items():
                current = group.splats[name]
                if current.shape != reference.shape:
                    raise RuntimeError(
                        "Anchored parameter shape changed for group {} {}".format(
                            group_id, name
                        )
                    )
                squared_sum = torch.sum((current - reference) ** 2)
                component_sums[name] = component_sums.get(name, 0.0) + squared_sum
                component_counts[name] = component_counts.get(name, 0) + current.numel()

        components = {
            name: component_sums[name] / component_counts[name]
            for name in component_sums
        }
        total = sum(self.weights[name] * value for name, value in components.items())
        return total, components

    @torch.no_grad()
    def report(self, gaussian_model):
        total, components = self.loss(gaussian_model)
        return {
            "enabled": True,
            "weights": {
                "sh0": self.weights["sh0"],
                "shN": self.weights["shN"],
                "opacity": self.weights["opacities"],
            },
            "final_mse": {
                name: float(value.item()) for name, value in components.items()
            },
            "final_weighted_loss": float(total.item()),
            "group_count": len(self.anchors),
        }
