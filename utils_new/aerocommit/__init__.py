"""Causal Gaussian admission and active-map management for AeroCommit."""

from .config import validate_aerocommit_config
from .frequency_responsibility import (
    ResponsibilityDecision,
    exact_shapley_values,
    geometry_responsibility_decision,
)
from .manager import AeroCommitManager
from .types import CommitResult, GaussianProposalBatch

__all__ = [
    "AeroCommitManager",
    "CommitResult",
    "GaussianProposalBatch",
    "ResponsibilityDecision",
    "exact_shapley_values",
    "geometry_responsibility_decision",
    "validate_aerocommit_config",
]
