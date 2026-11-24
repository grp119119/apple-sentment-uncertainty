"""Input contracts shared by simulations and future real sensor adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from .models import AffectLabel, normalize_distribution


class Modality(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    FACE = "face"
    PHYSIOLOGY = "physiology"


@dataclass(frozen=True)
class ModalityEvidence:
    modality: Modality
    emotion_distribution: Mapping[AffectLabel | str, float] | None = None
    valence: float | None = None
    arousal: float | None = None
    dominance: float | None = None
    quality: float = 1.0
    available: bool = True
    prediction_samples: Sequence[Mapping[AffectLabel | str, float]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if isinstance(self.quality, bool) or not isinstance(self.quality, (int, float)):
            raise TypeError("quality must be numeric")
        if not 0 <= float(self.quality) <= 1:
            raise ValueError("quality must be between 0 and 1")
        object.__setattr__(self, "quality", float(self.quality))
        if self.available and self.emotion_distribution is not None:
            object.__setattr__(
                self,
                "emotion_distribution",
                normalize_distribution(self.emotion_distribution),
            )
        normalized_samples = tuple(normalize_distribution(item) for item in self.prediction_samples)
        object.__setattr__(self, "prediction_samples", normalized_samples)
        for name, value, low, high in (
            ("valence", self.valence, -1.0, 1.0),
            ("arousal", self.arousal, 0.0, 1.0),
            ("dominance", self.dominance, 0.0, 1.0),
        ):
            if value is not None and not low <= float(value) <= high:
                raise ValueError(f"{name} must be between {low} and {high}")


@dataclass(frozen=True)
class MultimodalObservation:
    evidence: tuple[ModalityEvidence, ...]

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError("at least one modality must be supplied")
        modalities = [item.modality for item in self.evidence]
        if len(modalities) != len(set(modalities)):
            raise ValueError("each modality may appear only once")
