"""Deterministic feedback updates for the v4.1 within-session history."""

from __future__ import annotations

from .models import (
    AffectLabel,
    EmotionTrend,
    HistoryEffect,
    HistoryRecord,
    InterventionDecision,
    InterventionIntensity,
    UserNeed,
)


_TREND_FROM_EFFECT = {
    HistoryEffect.IMPROVED: EmotionTrend.IMPROVING,
    HistoryEffect.UNCHANGED: EmotionTrend.STABLE,
    HistoryEffect.WORSENED: EmotionTrend.WORSENING,
    HistoryEffect.UNKNOWN: EmotionTrend.UNKNOWN,
}


def update_history(
    history: HistoryRecord,
    decision: InterventionDecision,
    need: UserNeed,
    effect: HistoryEffect,
    *,
    accepted: bool = True,
    corrected_emotion: AffectLabel | None = None,
) -> HistoryRecord:
    """Return the next state without mutating thresholds or policy rules."""

    repeat_count = (
        history.repeat_count + 1
        if history.previous_action is decision.support_action
        else 1
    )
    rejected = list(history.rejected_actions)
    action_was_rejected = (
        decision.intensity is not InterventionIntensity.NONE
        and (not accepted or effect is HistoryEffect.WORSENED)
    )
    if action_was_rejected and decision.support_action not in rejected:
        rejected.append(decision.support_action)

    preferred_need = history.preferred_need
    if need not in {UserNeed.UNKNOWN, UserNeed.PAUSE}:
        preferred_need = need

    return HistoryRecord(
        previous_mode=decision.mode,
        previous_action=decision.support_action,
        previous_effect=effect,
        repeat_count=repeat_count,
        rejected_actions=tuple(rejected),
        preferred_need=preferred_need,
        corrected_emotion=corrected_emotion or history.corrected_emotion,
        emotion_trend=_TREND_FROM_EFFECT[effect],
    )
