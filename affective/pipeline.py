"""End-to-end orchestration without online policy mutation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .fusion import FusionResult, fuse_observation
from .models import AffectLabel, InterventionDecision, PolicyContext
from .multimodal import MultimodalObservation
from .policy import InterventionPolicy
from .realizer import realize_response


@dataclass(frozen=True)
class PipelineResult:
    fusion: FusionResult
    decision: InterventionDecision
    response: str

    def to_dict(self) -> dict[str, object]:
        return {
            "affect": self.fusion.affect.to_dict(),
            "uncertainty": self.fusion.uncertainty.to_dict(),
            "decision": self.decision.to_dict(),
            "response": self.response,
            "fusion_notes": list(self.fusion.notes),
        }


class AffectiveInterventionPipeline:
    def __init__(self, policy: InterventionPolicy) -> None:
        self.policy = policy

    def run(
        self,
        observation: MultimodalObservation,
        context: PolicyContext,
        previous_distribution: Mapping[AffectLabel | str, float] | None = None,
    ) -> PipelineResult:
        fusion = fuse_observation(observation, previous_distribution)
        decision = self.policy.decide(fusion.affect, fusion.uncertainty, context)
        return PipelineResult(fusion, decision, realize_response(decision))
