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


def select_support_action(affect: AffectEstimate, context: PolicyContext) -> tuple[SupportAction, str]:
    """Select response content without consulting perceptual uncertainty."""
    need_result = _need_action(context)
    if need_result is not None:
        return need_result
    if context.time_pressure is TimePressure.HIGH:
        return SupportAction.INFORMATION, "high time pressure favors concise information"
    if context.interaction_stage in {InteractionStage.OPENING, InteractionStage.EXPLORING}:
        if affect.dominant_emotion in {AffectLabel.FEAR, AffectLabel.SADNESS, AffectLabel.ANGER}:
            return SupportAction.REFLECTION, "early negative affect favors reflection of feelings"
        return SupportAction.QUESTION, "early interaction favors exploration through a question"
    if affect.dominant_emotion in {AffectLabel.FEAR, AffectLabel.SADNESS}:
        if affect.arousal >= 0.7:
            return SupportAction.AFFIRMATION, "high-arousal fear or sadness favors reassurance"
        return SupportAction.REFLECTION, "fear or sadness favors reflection of feelings"
    if affect.dominant_emotion is AffectLabel.ANGER:
        return SupportAction.RESTATEMENT, "anger favors a non-escalating paraphrase"
    if affect.dominant_emotion is AffectLabel.JOY:
        return SupportAction.AFFIRMATION, "positive affect favors affirmation"
    if affect.dominant_emotion is AffectLabel.SURPRISE:
        return SupportAction.QUESTION, "surprise favors clarification before advice"
    if context.interaction_stage is InteractionStage.CLOSING:
        return SupportAction.OTHERS, "closing stage favors a brief closing response"
    return SupportAction.SUGGESTION, "stable supporting stage allows a small suggestion"
