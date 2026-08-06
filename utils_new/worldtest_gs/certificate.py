"""Admission certificates and the single authority allowed to sign them."""

from __future__ import annotations

import math
import secrets
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class AdmissionCertificate:
    certificate_id: str
    issuer_nonce: str
    contract_fingerprint: str
    world_frame_id: str
    geometry_mode: str
    calibration_version: str
    source_frame_id: int
    track_id: int
    source_kind: str
    pose_source: str
    depth_source: str
    issued_frame_id: int
    observation_frame_ids: tuple[int, ...]
    q_g: float
    evidence_mode: str
    diagnostic_only: bool
    parent_certificate_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class CertificateAuthority:
    """Process-local signer; callers cannot construct a valid certificate by fields alone."""

    def __init__(self, contract, qg_threshold, allow_invalid_stress=False):
        self.contract = contract
        self.qg_threshold = float(qg_threshold)
        self.allow_invalid_stress = bool(allow_invalid_stress)
        self._nonce = secrets.token_hex(16)
        self._serial = 0
        self.issued = {}
        self.bypass_count = 0
        self.validation_count = 0

    def issue(
        self,
        *,
        source_frame_id,
        track_id,
        source_kind,
        issued_frame_id,
        observation_frame_ids: Sequence[int],
        q_g,
        evidence_mode,
        parent_certificate_id=None,
    ):
        self.contract.require_permanent_birth(self.allow_invalid_stress)
        frames = tuple(sorted(set(int(frame) for frame in observation_frame_ids)))
        if len(frames) < 3 and parent_certificate_id is None:
            raise ValueError("A new admission certificate requires three distinct views")
        if evidence_mode == "true_qg" and (
            not math.isfinite(float(q_g)) or float(q_g) <= self.qg_threshold
        ):
            raise ValueError("q_g does not satisfy the admission threshold")
        if parent_certificate_id is not None and parent_certificate_id not in self.issued:
            raise ValueError("Parent certificate is not issued by this authority")
        self._serial += 1
        certificate = AdmissionCertificate(
            certificate_id="{}-{:08d}".format(self.contract.fingerprint, self._serial),
            issuer_nonce=self._nonce,
            contract_fingerprint=self.contract.fingerprint,
            world_frame_id=self.contract.world_frame_id,
            geometry_mode=self.contract.geometry_mode,
            calibration_version=self.contract.calibration_version,
            source_frame_id=int(source_frame_id),
            track_id=int(track_id),
            source_kind=str(source_kind),
            pose_source=self.contract.pose_source,
            depth_source=self.contract.depth_source,
            issued_frame_id=int(issued_frame_id),
            observation_frame_ids=frames,
            q_g=float(q_g),
            evidence_mode=str(evidence_mode),
            diagnostic_only=not self.contract.permanent_birth_valid,
            parent_certificate_id=parent_certificate_id,
        )
        self.issued[certificate.certificate_id] = certificate
        return certificate

    def require(self, certificate, proposals=None, path="unknown"):
        if not isinstance(certificate, AdmissionCertificate):
            self.bypass_count += 1
            raise RuntimeError(
                "WorldTest permanent birth path '{}' has no AdmissionCertificate".format(
                    path
                )
            )
        canonical = self.issued.get(certificate.certificate_id)
        if canonical != certificate or certificate.issuer_nonce != self._nonce:
            self.bypass_count += 1
            raise RuntimeError("AdmissionCertificate was not signed by the active authority")
        if certificate.contract_fingerprint != self.contract.fingerprint:
            self.bypass_count += 1
            raise RuntimeError("AdmissionCertificate belongs to another world frame")
        if proposals is not None and len(proposals):
            track_ids = np.asarray(proposals.track_ids, dtype=np.int64)
            valid_ids = track_ids[track_ids >= 0]
            if valid_ids.size and np.any(valid_ids != int(certificate.track_id)):
                self.bypass_count += 1
                raise RuntimeError("AdmissionCertificate track ID does not cover the proposal")
            if int(proposals.source_frame_id) != int(certificate.source_frame_id):
                self.bypass_count += 1
                raise RuntimeError("AdmissionCertificate source frame does not match proposal")
        self.validation_count += 1
        return certificate

    def summary(self):
        return {
            "issued_count": len(self.issued),
            "validation_count": self.validation_count,
            "bypass_count": self.bypass_count,
            "contract": {
                "fingerprint": self.contract.fingerprint,
                "world_frame_id": self.contract.world_frame_id,
                "geometry_mode": self.contract.geometry_mode,
                "permanent_birth_valid": self.contract.permanent_birth_valid,
            },
        }
