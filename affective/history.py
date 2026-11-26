"""Deterministic feedback updates for the v4.1 within-session history."""

from __future__ import annotations

from .models import (
    AffectLabel,
    EmotionTrend,
    HistoryEffect,
    HistoryRecord,
    InterventionDecision,
    InterventionIntensity,
    InterventionMode,
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

    preferred_need = history.preferred_need
    if need not in {UserNeed.UNKNOWN, UserNeed.PAUSE}:
        preferred_need = need

    # Bug 2 & Bug 4 修复：
    # - WAIT / DEFER (或 intensity == NONE) 未实际输出任何支持动作，绝不应累增 repeat_count 或覆盖上轮动作；
    # - CONFIRM 属于元对话澄清确认，其临时 Question 绝不应污染支持动作历史或把 Question 拉黑。
    non_content_turn = (
        decision.mode in {InterventionMode.WAIT, InterventionMode.DEFER, InterventionMode.CONFIRM}
        or decision.intensity is InterventionIntensity.NONE
    )
    if non_content_turn:
        return HistoryRecord(
            previous_mode=decision.mode,
            previous_action=history.previous_action,
            previous_effect=effect if history.previous_action is not None else history.previous_effect,
            repeat_count=history.repeat_count,
            rejected_actions=history.rejected_actions,
            preferred_need=preferred_need,
            corrected_emotion=corrected_emotion or history.corrected_emotion,
            emotion_trend=_TREND_FROM_EFFECT[effect],
        )

    repeat_count = (
        history.repeat_count + 1
        if history.previous_action is decision.support_action
        else 1
    )
    rejected = list(history.rejected_actions)
    action_was_rejected = not accepted or effect is HistoryEffect.WORSENED
    if action_was_rejected and decision.support_action not in rejected:
        rejected.append(decision.support_action)

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
