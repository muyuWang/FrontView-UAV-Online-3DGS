"""JSONL frame metrics and compact run summaries for AeroCommit."""

import json
import os
from dataclasses import asdict

import numpy as np


class MetricsRecorder:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.path = os.path.join(output_dir, "aerocommit_stats.jsonl")
        self.records = []

    def record(self, stats):
        payload = asdict(stats)
        self.records.append(payload)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def summary(self, extra=None):
        latency_keys = (
            "frame_total_ms",
            "proposal_ms",
            "candidate_association_ms",
            "risk_gate_ms",
            "commit_refinement_ms",
            "detail_refinement_ms",
            "archive_transfer_ms",
        )
        result = {"frames": len(self.records)}
        for key in latency_keys:
            values = np.asarray([record[key] for record in self.records], dtype=np.float64)
            if len(values):
                result[key] = {
                    "mean": float(np.mean(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                    "total": float(np.sum(values)),
                }
        if self.records:
            result["final"] = self.records[-1]
            result["total_committed_candidates"] = int(
                sum(record["num_committed_candidates"] for record in self.records)
            )
            result["total_committed_gaussians"] = int(
                sum(record["num_committed_gaussians"] for record in self.records)
            )
            result["total_detail_splits"] = int(
                sum(record["num_detail_splits"] for record in self.records)
            )
            result["total_side_detail_splits"] = int(
                sum(record["num_side_detail_splits"] for record in self.records)
            )
            result["total_fused_proposals"] = int(
                sum(record["num_fused_proposals"] for record in self.records)
            )
            result["total_frequency_probation_gaussians"] = int(
                sum(
                    record["num_frequency_probation_gaussians"]
                    for record in self.records
                )
            )
            result["total_filtered_depthcov_candidates"] = int(
                sum(
                    record["num_filtered_depthcov_candidates"]
                    for record in self.records
                )
            )
        result.update(extra or {})
        with open(
            os.path.join(self.output_dir, "aerocommit_summary.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(result, handle, indent=2)
        return result
