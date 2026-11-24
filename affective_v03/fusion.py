"""Transparent late fusion that preserves probability distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .models import AffectEstimate, AffectLabel, UncertaintyProfile
from .multimodal import MultimodalObservation
from .uncertainty import build_uncertainty_profile


@dataclass(frozen=True)
class FusionResult:
    affect: AffectEstimate
    uncertainty: UncertaintyProfile
    notes: tuple[str, ...]


def _weighted_value(values: list[tuple[float, float]], fallback: float) -> float:
    weight = sum(item[1] for item in values)
    return sum(item[0] * item[1] for item in values) / weight if weight else fallback


def fuse_observation(
    observation: MultimodalObservation,
    previous_distribution: Mapping[AffectLabel | str, float] | None = None,
) -> FusionResult:
    categorical = [
        item
        for item in observation.evidence
        if item.available and item.quality > 0 and item.emotion_distribution is not None
    ]
    if not categorical:
        raise ValueError("at least one available categorical modality is required")
    total_weight = sum(item.quality for item in categorical)
    distribution = {
        label: sum(item.emotion_distribution[label] * item.quality for item in categorical)
        / total_weight
        for label in AffectLabel
    }
    active = [item for item in observation.evidence if item.available and item.quality > 0]
    valence = _weighted_value(
        [(float(item.valence), item.quality) for item in active if item.valence is not None],
        fallback=0.0,
    )
    arousal = _weighted_value(
        [(float(item.arousal), item.quality) for item in active if item.arousal is not None],
        fallback=0.5,
    )
    dominance_values = [
        (float(item.dominance), item.quality) for item in active if item.dominance is not None
    ]
    dominance = _weighted_value(dominance_values, fallback=0.5) if dominance_values else None
    affect = AffectEstimate(distribution, valence, arousal, dominance)
    uncertainty = build_uncertainty_profile(
        affect.distribution,
        observation.evidence,
        previous_distribution,
    )
    notes = (
        "turn-level label space: Ekman six emotions plus Neutral",
        "full probability distribution retained; top-1 is not treated as ground truth",
        "scenario-level ESConv self-report labels are stored separately",
    )
    return FusionResult(affect=affect, uncertainty=uncertainty, notes=notes)
