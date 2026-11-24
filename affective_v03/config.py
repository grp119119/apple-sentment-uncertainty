"""Read and validate versioned policy configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .policy import PilotThresholds, UncertaintyAwarePolicy


@dataclass(frozen=True)
class PolicyConfiguration:
    policy_version: str
    calibration_status: str
    thresholds: PilotThresholds
    source_path: Path

    def build_policy(self) -> UncertaintyAwarePolicy:
        return UncertaintyAwarePolicy(
            self.thresholds,
            version=self.policy_version,
            calibration_status=self.calibration_status,
        )


def load_policy_configuration(path: Path) -> PolicyConfiguration:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_thresholds = data.get("thresholds")
    if not isinstance(raw_thresholds, dict):
        raise ValueError("policy config must include a thresholds object")
    thresholds = PilotThresholds(**raw_thresholds)
    for name, value in raw_thresholds.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"threshold {name} must be between 0 and 1")
    return PolicyConfiguration(
        policy_version=str(data.get("policy_version", "0.3-pilot-draft")),
        calibration_status=str(
            data.get("calibration_status", "PILOT_DRAFT_NOT_VALIDATED")
        ),
        thresholds=thresholds,
        source_path=path,
    )
