"""Deterministic Chinese response realization for controlled experiments."""

from __future__ import annotations

from .models import InterventionDecision, InterventionIntensity, InterventionMode, SupportAction


_DIRECT_CONTENT = {
    SupportAction.QUESTION: "你愿意再说说现在最明显的感受吗？",
    SupportAction.RESTATEMENT: "你正在面对一件让你很难平静处理的事情。",
    SupportAction.REFLECTION: "这件事确实让你感到很不好受。",
    SupportAction.SELF_DISCLOSURE: "遇到类似处境时，很多人也会感到不知所措。",
    SupportAction.AFFIRMATION: "你的感受值得被认真对待，我们可以慢慢来。",
    SupportAction.SUGGESTION: "我们可以先选一个最小、最容易完成的步骤。",
    SupportAction.INFORMATION: "我先提供最关键的信息，帮助你明确下一步。",
    SupportAction.OTHERS: "谢谢你愿意告诉我这些。",
}


def realize_response(decision: InterventionDecision) -> str:
    if decision.intensity is InterventionIntensity.NONE:
        if decision.mode is InterventionMode.WAIT:
            return "[保持安静并降低装置显著性，继续观察，不声明用户情绪。]"
        if decision.mode is InterventionMode.DEFER:
            return "[停止主动介入，等待用户重新发起。]"
        return "[不主动发出语言回应。]"
    if decision.mode is InterventionMode.WAIT:
        return "[保持安静并降低装置显著性，继续观察，不声明用户情绪。]"
    if decision.mode is InterventionMode.DEFER:
        return "我先不打扰。如果你希望我介入，可以随时告诉我。"
    if decision.mode is InterventionMode.CONFIRM:
        return "我不确定自己是否理解准确。你现在更接近难过、紧张，还是另一种感受？"

    content = _DIRECT_CONTENT[decision.support_action]
    if decision.mode is InterventionMode.TENTATIVE:
        return f"如果我没有理解错，你现在可能需要一点支持。{content}如果不准确，你可以纠正我。"
    return content
