"""Core contracts for the v0.3 intervention pipeline.

The label spaces and decision layers are intentionally separated:
- turn-level affect uses Ekman six emotions plus Neutral;
- scenario-level ESConv labels belong in study metadata, not this classifier;
- uncertainty controls epistemic stance and intervention mode;
- affect controls the concrete support action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class AffectLabel(str, Enum):
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    JOY = "joy"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


class ScenarioEmotion(str, Enum):
    """ESConv pre-chat self-report categories; never automatic diagnoses."""

    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SHAME = "shame"


class InterventionMode(str, Enum):
    CONFIRM = "CONFIRM"
    WAIT = "WAIT"
    TENTATIVE = "TENTATIVE"
    DIRECT = "DIRECT"
    DEFER = "DEFER"


class InterventionIntensity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SupportAction(str, Enum):
    """The eight ESConv support strategies, kept traceable to the dataset."""

    QUESTION = "Question"
    RESTATEMENT = "Restatement or Paraphrasing"
    REFLECTION = "Reflection of Feelings"
    SELF_DISCLOSURE = "Self-disclosure"
    AFFIRMATION = "Affirmation and Reassurance"
    SUGGESTION = "Providing Suggestions"
    INFORMATION = "Information"
    OTHERS = "Others"


class UserNeed(str, Enum):
    """Manually supplied need for the v4.1 policy prototype."""

    LISTEN = "LISTEN"
    COMFORT = "COMFORT"
    CLARIFY = "CLARIFY"
    GUIDANCE = "GUIDANCE"
    INFORMATION = "INFORMATION"
    PAUSE = "PAUSE"
    UNKNOWN = "UNKNOWN"


class HistoryEffect(str, Enum):
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    WORSENED = "worsened"
    UNKNOWN = "unknown"


class EmotionTrend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    UNKNOWN = "unknown"


class EpistemicStyle(str, Enum):
    CONFIRMATORY = "confirmatory"
    NON_COMMITTAL = "non_committal"
    TENTATIVE = "tentative"
    CERTAIN = "certain"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class HistoryRecord:
    """Deterministic within-session state; it never changes policy rules."""

    previous_mode: InterventionMode | None = None
    previous_action: SupportAction | None = None
    previous_effect: HistoryEffect = HistoryEffect.UNKNOWN
    repeat_count: int = 0
    rejected_actions: tuple[SupportAction, ...] = ()
    preferred_need: UserNeed | None = None
    corrected_emotion: AffectLabel | None = None
    emotion_trend: EmotionTrend = EmotionTrend.UNKNOWN

    def __post_init__(self) -> None:
        if isinstance(self.repeat_count, bool) or not isinstance(self.repeat_count, int):
            raise TypeError("repeat_count must be an integer")
        if self.repeat_count < 0:
            raise ValueError("repeat_count must be nonnegative")
        if len(self.rejected_actions) != len(set(self.rejected_actions)):
            raise ValueError("rejected_actions must not contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_mode": self.previous_mode.value if self.previous_mode else None,
            "previous_action": self.previous_action.value if self.previous_action else None,
            "previous_effect": self.previous_effect.value,
            "repeat_count": self.repeat_count,
            "rejected_actions": [item.value for item in self.rejected_actions],
            "preferred_need": self.preferred_need.value if self.preferred_need else None,
            "corrected_emotion": self.corrected_emotion.value if self.corrected_emotion else None,
            "emotion_trend": self.emotion_trend.value,
        }


class Privacy(str, Enum):
    PRIVATE = "private"
    PUBLIC = "public"


class TimePressure(str, Enum):
    LOW = "low"
    HIGH = "high"


class InteractionStage(str, Enum):
    OPENING = "opening"
    EXPLORING = "exploring"
    SUPPORTING = "supporting"
    CLOSING = "closing"


def _unit(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def normalize_distribution(
    distribution: Mapping[AffectLabel | str, float],
) -> dict[AffectLabel, float]:
    parsed = {label: 0.0 for label in AffectLabel}
    for raw_label, raw_value in distribution.items():
        label = raw_label if isinstance(raw_label, AffectLabel) else AffectLabel(raw_label)
        value = _unit(f"probability[{label.value}]", raw_value)
        parsed[label] += value
    total = sum(parsed.values())
    if total <= 0:
        raise ValueError("emotion distribution must contain positive probability mass")
    return {label: value / total for label, value in parsed.items()}


@dataclass(frozen=True)
class AffectEstimate:
    distribution: Mapping[AffectLabel | str, float]
    valence: float
    arousal: float
    dominance: float | None = None
    dominant_emotion: AffectLabel = field(init=False)

    def __post_init__(self) -> None:
        normalized = normalize_distribution(self.distribution)
        object.__setattr__(self, "distribution", normalized)
        object.__setattr__(self, "dominant_emotion", max(normalized, key=normalized.get))
        if isinstance(self.valence, bool) or not isinstance(self.valence, (int, float)):
            raise TypeError("valence must be numeric")
        if not -1.0 <= float(self.valence) <= 1.0:
            raise ValueError("valence must be between -1 and 1")
        object.__setattr__(self, "valence", float(self.valence))
        object.__setattr__(self, "arousal", _unit("arousal", self.arousal))
        if self.dominance is not None:
            object.__setattr__(self, "dominance", _unit("dominance", self.dominance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_emotion": self.dominant_emotion.value,
            "distribution": {key.value: value for key, value in self.distribution.items()},
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }


@dataclass(frozen=True)
class UncertaintyProfile:
    predictive_entropy: float
    top_two_margin: float
    signal_quality: float
    data_coverage: float
    missingness: float
    ensemble_variance: float = 0.0
    temporal_instability: float = 0.0
    cross_modal_disagreement: float | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "predictive_entropy",
            "top_two_margin",
            "signal_quality",
            "data_coverage",
            "missingness",
            "ensemble_variance",
            "temporal_instability",
        ):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        if self.cross_modal_disagreement is not None:
            object.__setattr__(
                self,
                "cross_modal_disagreement",
                _unit("cross_modal_disagreement", self.cross_modal_disagreement),
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyContext:
    privacy: Privacy = Privacy.PRIVATE
    time_pressure: TimePressure = TimePressure.LOW
    interaction_stage: InteractionStage = InteractionStage.EXPLORING
    receptive: bool = True
    user_initiated: bool = True
    safety_allows_intervention: bool = True
    user_need: UserNeed = UserNeed.UNKNOWN
    history: HistoryRecord = field(default_factory=HistoryRecord)

    def to_dict(self) -> dict[str, Any]:
        return {
            "privacy": self.privacy.value,
            "time_pressure": self.time_pressure.value,
            "interaction_stage": self.interaction_stage.value,
            "receptive": self.receptive,
            "user_initiated": self.user_initiated,
            "safety_allows_intervention": self.safety_allows_intervention,
            "user_need": self.user_need.value,
            "history": self.history.to_dict(),
        }


@dataclass(frozen=True)
class InterventionDecision:
    mode: InterventionMode
    intensity: InterventionIntensity
    support_action: SupportAction
    epistemic_style: EpistemicStyle
    reasons: tuple[str, ...]
    policy_name: str
    policy_version: str
    calibration_status: str

    def __post_init__(self) -> None:
        if not self.reasons:
            raise ValueError("decision must include at least one reason")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("mode", "intensity", "support_action", "epistemic_style"):
            result[key] = getattr(self, key).value
        return result


@dataclass(frozen=True)
class StudyFeedback:
    appropriateness: int
    intrusiveness: int
    perceived_understanding: int
    willingness_to_continue: int
    helpfulness: int
    corrected_system: bool = False
    corrected_emotion: AffectLabel | None = None

    def __post_init__(self) -> None:
        for name in (
            "appropriateness",
            "intrusiveness",
            "perceived_understanding",
            "willingness_to_continue",
            "helpfulness",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"{name} must be an integer from 1 to 5")
