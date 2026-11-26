"""Affect-led selection of the secondary ESConv support action."""

from __future__ import annotations

from .models import (
    AffectEstimate,
    AffectLabel,
    HistoryEffect,
    InteractionStage,
    PolicyContext,
    SupportAction,
    TimePressure,
    UserNeed,
)


_NEED_ACTIONS = {
    UserNeed.LISTEN: (SupportAction.REFLECTION, SupportAction.RESTATEMENT),
    UserNeed.COMFORT: (SupportAction.AFFIRMATION, SupportAction.REFLECTION),
    UserNeed.CLARIFY: (SupportAction.QUESTION, SupportAction.RESTATEMENT),
    UserNeed.GUIDANCE: (SupportAction.SUGGESTION, SupportAction.INFORMATION),
    UserNeed.INFORMATION: (SupportAction.INFORMATION, SupportAction.SUGGESTION),
}

_NEED_REASONS = {
    UserNeed.LISTEN: "explicit listening need favors reflection of feelings",
    UserNeed.COMFORT: "explicit comfort need favors affirmation and reassurance",
    UserNeed.CLARIFY: "explicit clarification need favors a question",
    UserNeed.GUIDANCE: "explicit guidance need favors a small suggestion",
    UserNeed.INFORMATION: "explicit information need favors concise information",
}


def _need_action(context: PolicyContext) -> tuple[SupportAction, str] | None:
    need = context.user_need
    inherited = False
    if need is UserNeed.UNKNOWN and context.history.preferred_need is not None:
        need = context.history.preferred_need
        inherited = True
    candidates = _NEED_ACTIONS.get(need)
    if candidates is None:
        return None
    rejected = set(context.history.rejected_actions)
    repeated_without_improvement = (
        context.history.previous_action is not None
        and context.history.repeat_count >= 2
        and context.history.previous_effect is not HistoryEffect.IMPROVED
    )
    if repeated_without_improvement:
        rejected.add(context.history.previous_action)

    preserve_effective = (
        context.history.previous_effect is HistoryEffect.IMPROVED
        and context.history.previous_action in candidates
        and context.history.previous_action not in rejected
    )
    if preserve_effective:
        selected = context.history.previous_action
    else:
        selected = next((item for item in candidates if item not in rejected), SupportAction.QUESTION)
    reason = _NEED_REASONS[need]
    if inherited:
        reason += "; the need is inherited from interaction history"
    if preserve_effective:
        reason += "; history preserves a previously effective support action"
    elif repeated_without_improvement:
        reason += "; history switches after repeated support without improvement"
    elif selected is not candidates[0]:
        reason += "; history excludes a previously rejected support action"
    return selected, reason


_AFFECT_FALLBACK_ORDER: tuple[SupportAction, ...] = (
    SupportAction.REFLECTION,
    SupportAction.AFFIRMATION,
    SupportAction.RESTATEMENT,
    SupportAction.QUESTION,
    SupportAction.SUGGESTION,
    SupportAction.OTHERS,
)


def _apply_history_to_affect_action(
    primary: SupportAction,
    base_reason: str,
    context: PolicyContext,
) -> tuple[SupportAction, str]:
    """Bug 3 修复：情绪驱动分支同样遵循交互历史的避拒与连续无效换招规则。"""
    rejected = set(context.history.rejected_actions)
    repeated_without_improvement = (
        context.history.previous_action is primary
        and context.history.repeat_count >= 2
        and context.history.previous_effect is not HistoryEffect.IMPROVED
    )
    if repeated_without_improvement:
        rejected.add(primary)

    if primary not in rejected:
        return primary, base_reason

    fallback = next((cand for cand in _AFFECT_FALLBACK_ORDER if cand not in rejected), SupportAction.OTHERS)
    if repeated_without_improvement:
        return fallback, f"{base_reason}; history switches after repeated support without improvement"
    return fallback, f"{base_reason}; history excludes a previously rejected support action"


def select_support_action(affect: AffectEstimate, context: PolicyContext) -> tuple[SupportAction, str]:
    """Select response content without consulting perceptual uncertainty."""
    need_result = _need_action(context)
    if need_result is not None:
        return need_result
    if context.time_pressure is TimePressure.HIGH:
        return _apply_history_to_affect_action(
            SupportAction.INFORMATION, "high time pressure favors concise information", context
        )
    if context.interaction_stage in {InteractionStage.OPENING, InteractionStage.EXPLORING}:
        if affect.dominant_emotion in {AffectLabel.FEAR, AffectLabel.SADNESS, AffectLabel.ANGER}:
            return _apply_history_to_affect_action(
                SupportAction.REFLECTION, "early negative affect favors reflection of feelings", context
            )
        return _apply_history_to_affect_action(
            SupportAction.QUESTION, "early interaction favors exploration through a question", context
        )
    if affect.dominant_emotion in {AffectLabel.FEAR, AffectLabel.SADNESS}:
        if affect.arousal >= 0.7:
            return _apply_history_to_affect_action(
                SupportAction.AFFIRMATION, "high-arousal fear or sadness favors reassurance", context
            )
        return _apply_history_to_affect_action(
            SupportAction.REFLECTION, "fear or sadness favors reflection of feelings", context
        )
    if affect.dominant_emotion is AffectLabel.ANGER:
        return _apply_history_to_affect_action(
            SupportAction.RESTATEMENT, "anger favors a non-escalating paraphrase", context
        )
    if affect.dominant_emotion is AffectLabel.JOY:
        return _apply_history_to_affect_action(
            SupportAction.AFFIRMATION, "positive affect favors affirmation", context
        )
    if affect.dominant_emotion is AffectLabel.SURPRISE:
        return _apply_history_to_affect_action(
            SupportAction.QUESTION, "surprise favors clarification before advice", context
        )
    if context.interaction_stage is InteractionStage.CLOSING:
        return _apply_history_to_affect_action(
            SupportAction.OTHERS, "closing stage favors a brief closing response", context
        )
    return _apply_history_to_affect_action(
        SupportAction.SUGGESTION, "stable supporting stage allows a small suggestion", context
    )
