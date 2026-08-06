"""Pluggable candidate admission policies with a shared decision interface."""

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .types import CandidateRecord, RiskResult


@dataclass
class AdmissionDecision:
    candidate_id: int
    commit: bool
    score: float
    reason: str


class AdmissionPolicy:
    def __init__(self, config, npo_evaluator=None):
        self.config = config
        self.name = config["policy"]
        self.npo_evaluator = npo_evaluator

    def evaluate(self, candidates: Sequence[CandidateRecord]) -> List[AdmissionDecision]:
        if self.name == "npo_lite":
            if self.npo_evaluator is None:
                raise RuntimeError("npo_lite requires an NPOLiteEvaluator")
            result = self.npo_evaluator.evaluate(candidates)
            return self._npo_decisions(candidates, result)
        decisions = []
        minimum_parallax = math.radians(
            float(self.config["minimum_sanity_parallax_deg"])
        )
        for candidate in candidates:
            if self.name == "immediate":
                commit, score = True, 0.0
            elif self.name == "fixed_delay":
                score = float(candidate.age)
                commit = candidate.age >= int(self.config["fixed_delay_frames"])
            elif self.name == "parallax":
                score = candidate.parallax_max_rad
                commit = score >= minimum_parallax
            elif self.name == "posterior_variance":
                score = candidate.rho_variance / max(candidate.rho_mean**2, 1.0e-8)
                commit = score <= float(self.config["posterior_variance_threshold"])
            elif self.name == "residual":
                score = candidate.residual_mad_ema
                commit = score <= float(self.config["residual_threshold"])
            elif self.name == "depth_confidence":
                score = float(np.mean(candidate.proposal_batch.sparse_depth_valid))
                commit = score >= 0.5 and candidate.parallax_max_rad >= minimum_parallax
            elif self.name == "cheap_hessian_no_pose":
                score = candidate.parallax_max_rad / max(
                    math.sqrt(candidate.rho_variance + 1.0e-12), 1.0e-6
                )
                commit = candidate.parallax_max_rad >= minimum_parallax and score >= 1.0
            else:
                raise ValueError("Unsupported admission policy {}".format(self.name))
            decisions.append(
                AdmissionDecision(
                    candidate_id=candidate.candidate_id,
                    commit=bool(commit),
                    score=float(score),
                    reason=self.name,
                )
            )
        return decisions

    def _npo_decisions(
        self, candidates: Sequence[CandidateRecord], result: RiskResult
    ) -> List[AdmissionDecision]:
        by_id = {
            int(candidate_id): (float(risk), float(information))
            for candidate_id, risk, information in zip(
                result.candidate_ids, result.commitment_risk, result.information
            )
        }
        minimum_parallax = math.radians(
            float(self.config["minimum_sanity_parallax_deg"])
        )
        decisions = []
        for candidate in candidates:
            risk, information = by_id[candidate.candidate_id]
            candidate.last_risk = risk
            candidate.last_information = information
            frequency_candidate = (
                self.config["frequency_candidate_enabled"]
                and candidate.frequency_score
                >= float(self.config["frequency_candidate_score_threshold"])
            )
            required_support = (
                int(self.config["frequency_candidate_min_support"])
                if frequency_candidate
                else int(self.config["min_support"])
            )
            required_parallax = (
                math.radians(
                    float(self.config["frequency_candidate_min_parallax_deg"])
                )
                if frequency_candidate
                else minimum_parallax
            )
            risk_threshold = (
                float(self.config["frequency_candidate_risk_threshold"])
                if frequency_candidate
                else float(self.config["risk_threshold"])
            )
            commit = (
                candidate.support_count >= required_support
                and candidate.parallax_max_rad >= required_parallax
                and risk <= risk_threshold
                and candidate.association_error_ema
                <= float(self.config["association_residual_threshold"])
            )
            decisions.append(
                AdmissionDecision(
                    candidate_id=candidate.candidate_id,
                    commit=commit,
                    score=risk,
                    reason=(
                        "frequency_two_view_npo"
                        if frequency_candidate
                        else "npo_lite"
                    ),
                )
            )
        return decisions
