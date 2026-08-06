"""Byte-aware active-map accounting and archive pressure decisions."""

from dataclasses import dataclass


@dataclass
class MemoryBreakdown:
    parameter_bytes: int = 0
    gradient_bytes: int = 0
    optimizer_bytes: int = 0

    @property
    def active_trainable_bytes(self):
        return self.parameter_bytes + self.gradient_bytes + self.optimizer_bytes


class ActiveBudgetManager:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def measure(gaussian_model):
        result = MemoryBreakdown()
        for group_id in gaussian_model.valid_groups:
            group = gaussian_model.gaussian_groups[group_id]
            if group.splats is None:
                continue
            for parameter in group.splats.values():
                result.parameter_bytes += parameter.numel() * parameter.element_size()
                if parameter.grad is not None:
                    result.gradient_bytes += parameter.grad.numel() * parameter.grad.element_size()
            for optimizer in group.optimizers.values():
                for state in optimizer.state.values():
                    for value in state.values():
                        if hasattr(value, "numel"):
                            result.optimizer_bytes += value.numel() * value.element_size()
        for optimizer in gaussian_model.progressive_optimizers.values():
            for state in optimizer.state.values():
                for value in state.values():
                    if hasattr(value, "numel"):
                        result.optimizer_bytes += value.numel() * value.element_size()
        return result

    def over_budget(self, gaussian_count, memory):
        over_count = gaussian_count > int(self.config["max_active_trainable_gaussians"])
        byte_limit = self.config["max_active_trainable_bytes"]
        over_bytes = byte_limit is not None and memory.active_trainable_bytes > int(byte_limit)
        return over_count or over_bytes
