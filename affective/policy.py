"""Experiment-ready policies with explicitly provisional Pilot thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import (
    AffectEstimate,
    EpistemicStyle,
    InterventionDecision,
    InterventionIntensity,
    InterventionMode,
    PolicyContext,
    Privacy,
    SupportAction,
    UncertaintyProfile,
    UserNeed,
)
from .support import select_support_action


@dataclass(frozen=True)
class PilotThresholds:
    """Engineering defaults only; replace with frozen Study 1 calibration."""

    minimum_signal_quality: float = 0.45
    minimum_data_coverage: float = 0.50
    high_entropy: float = 0.72
    moderate_entropy: float = 0.50
    low_margin: float = 0.18
    moderate_margin: float = 0.32
    high_ensemble_variance: float = 0.35
    high_temporal_instability: float = 0.55
    experimental_cross_modal_disagreement: float = 0.35


class InterventionPolicy(Protocol):
    name: str
    version: str

    def decide(
        self,
        affect: AffectEstimate,
        uncertainty: UncertaintyProfile,
        context: PolicyContext,
    ) -> InterventionDecision: ...


_INTENSITY_ORDER = [
    InterventionIntensity.NONE,
    InterventionIntensity.LOW,
    InterventionIntensity.MEDIUM,
    InterventionIntensity.HIGH,
]


def _base_intensity(affect: AffectEstimate) -> InterventionIntensity:
    if affect.arousal >= 0.75:
        return InterventionIntensity.HIGH
    if affect.arousal >= 0.45:
        return InterventionIntensity.MEDIUM
    return InterventionIntensity.LOW


def _cap_intensity(
    intensity: InterventionIntensity,
    cap: InterventionIntensity,
) -> InterventionIntensity:
    return _INTENSITY_ORDER[min(_INTENSITY_ORDER.index(intensity), _INTENSITY_ORDER.index(cap))]


def _reduce_intensity(intensity: InterventionIntensity) -> InterventionIntensity:
    return _INTENSITY_ORDER[max(0, _INTENSITY_ORDER.index(intensity) - 1)]


class UncertaintyAwarePolicy:
    name = "uncertainty_aware"

    def __init__(
        self,
        thresholds: PilotThresholds | None = None,
        version: str = "0.3-pilot-draft",
        calibration_status: str = "PILOT_DRAFT_NOT_VALIDATED",
    ) -> None:
        self.thresholds = thresholds or PilotThresholds()
        self.version = version
        self.calibration_status = calibration_status

    def decide(
        self,
        affect: AffectEstimate,
        uncertainty: UncertaintyProfile,
        context: PolicyContext,
    ) -> InterventionDecision:
        if not context.safety_allows_intervention:
            return self._decision(
                InterventionMode.DEFER,
                InterventionIntensity.NONE,
                SupportAction.OTHERS,
                EpistemicStyle.DEFERRED,
                ["safety boundary blocks autonomous intervention"],
                context,
            )
        if not context.receptive:
            return self._decision(
                InterventionMode.DEFER,
                InterventionIntensity.NONE,
                SupportAction.OTHERS,
                EpistemicStyle.DEFERRED,
                ["the user is currently not receptive"],
                context,
            )
        if context.user_need is UserNeed.PAUSE:
            return self._decision(
                InterventionMode.DEFER,
                InterventionIntensity.NONE,
                SupportAction.OTHERS,
                EpistemicStyle.DEFERRED,
                ["the user explicitly requested a pause"],
                context,
            )

        action, action_reason = select_support_action(affect, context)
        reasons: list[str] = [action_reason]

        t = self.thresholds
        insufficient = (
            uncertainty.signal_quality < t.minimum_signal_quality
            or uncertainty.data_coverage < t.minimum_data_coverage
        )
        highly_ambiguous = (
            uncertainty.predictive_entropy >= t.high_entropy
            or uncertainty.top_two_margin <= t.low_margin
            or uncertainty.ensemble_variance >= t.high_ensemble_variance
        )
        strongly_conflicting = (
            uncertainty.cross_modal_disagreement is not None
            and uncertainty.cross_modal_disagreement >= t.experimental_cross_modal_disagreement
        )
        moderately_uncertain = (
            uncertainty.predictive_entropy >= t.moderate_entropy
            or uncertainty.top_two_margin <= t.moderate_margin
            or uncertainty.temporal_instability >= t.high_temporal_instability
            or strongly_conflicting
        )

        if insufficient:
            reasons.append("available evidence is insufficient for a committed affect claim")
            if context.user_initiated:
                return self._decision(
                    InterventionMode.CONFIRM,
                    _cap_intensity(_base_intensity(affect), InterventionIntensity.MEDIUM),
                    SupportAction.QUESTION,
                    EpistemicStyle.CONFIRMATORY,
                    reasons
                    + [
                        "the user initiated interaction, so clarification is preferable to silence",
                        "CONFIRM temporarily replaces the affect-led content action with a clarification question",
                    ],
                    context,
                )
            return self._decision(
                InterventionMode.WAIT,
                InterventionIntensity.NONE,
                action,
                EpistemicStyle.NON_COMMITTAL,
                reasons + ["passive observation is preferable when no response was requested"],
                context,
            )

        if highly_ambiguous:
            reasons.append("emotion probabilities are highly ambiguous or unstable")
            if context.user_initiated:
                return self._decision(
                    InterventionMode.CONFIRM,
                    _cap_intensity(_base_intensity(affect), InterventionIntensity.MEDIUM),
                    SupportAction.QUESTION,
                    EpistemicStyle.CONFIRMATORY,
                    reasons
                    + [
                        "CONFIRM temporarily replaces the affect-led content action with a clarification question"
                    ],
                    context,
                )
            return self._decision(
                InterventionMode.WAIT,
                InterventionIntensity.NONE,
                action,
                EpistemicStyle.NON_COMMITTAL,
                reasons + ["no user request makes waiting less intrusive than questioning"],
                context,
            )

        intensity = _base_intensity(affect)
        if moderately_uncertain:
            reasons.append("partial uncertainty requires a low-commitment, correctable response")
            intensity = _cap_intensity(intensity, InterventionIntensity.MEDIUM)
            mode, style = InterventionMode.TENTATIVE, EpistemicStyle.TENTATIVE
        else:
            reasons.append("the affect estimate is sufficiently clear for direct support")
            mode, style = InterventionMode.DIRECT, EpistemicStyle.CERTAIN

        return self._decision(mode, intensity, action, style, reasons, context)

    def _decision(
        self,
        mode: InterventionMode,
        intensity: InterventionIntensity,
        action: SupportAction,
        style: EpistemicStyle,
        reasons: list[str],
        context: PolicyContext,
    ) -> InterventionDecision:
        if context.privacy is Privacy.PUBLIC and intensity is not InterventionIntensity.NONE:
            intensity = _reduce_intensity(intensity)
            reasons = reasons + ["public context reduces intervention salience"]
        return InterventionDecision(
            mode=mode,
            intensity=intensity,
            support_action=action,
            epistemic_style=style,
            reasons=tuple(reasons),
            policy_name=self.name,
            policy_version=self.version,
            calibration_status=self.calibration_status,
        )


class EmotionOnlyPolicy:
    """Study 2 baseline: affect selects content; uncertainty is ignored."""

    name = "emotion_only"
    version = "0.3-baseline"

    def decide(
        self,
        affect: AffectEstimate,
        uncertainty: UncertaintyProfile,
        context: PolicyContext,
    ) -> InterventionDecision:
        action, action_reason = select_support_action(affect, context)
        if (
            not context.safety_allows_intervention
            or not context.receptive
            or context.user_need is UserNeed.PAUSE
        ):
            return InterventionDecision(
                mode=InterventionMode.DEFER,
                intensity=InterventionIntensity.NONE,
                support_action=SupportAction.OTHERS,
                epistemic_style=EpistemicStyle.DEFERRED,
                reasons=("shared safety/receptivity/pause boundary applies to both conditions",),
                policy_name=self.name,
                policy_version=self.version,
                calibration_status="BASELINE_FIXED",
            )
        intensity = _base_intensity(affect)
        if context.privacy is Privacy.PUBLIC:
            intensity = _reduce_intensity(intensity)
        return InterventionDecision(
            mode=InterventionMode.DIRECT,
            intensity=intensity,
            support_action=action,
            epistemic_style=EpistemicStyle.CERTAIN,
            reasons=(action_reason, "baseline intentionally ignores perceptual uncertainty"),
            policy_name=self.name,
            policy_version=self.version,
            calibration_status="BASELINE_FIXED",
        )
