"""Chinese display labels for the v4.0 research panel."""

from __future__ import annotations


EMOTION_LABELS = {
    "anger": "愤怒",
    "disgust": "厌恶",
    "fear": "恐惧",
    "joy": "愉悦",
    "sadness": "悲伤",
    "surprise": "惊讶",
    "neutral": "中性",
}

MODE_LABELS = {
    "DIRECT": "直接介入",
    "TENTATIVE": "试探介入",
    "CONFIRM": "询问确认",
    "WAIT": "暂时等待",
    "DEFER": "暂缓介入",
}

INTENSITY_LABELS = {"none": "无", "low": "低", "medium": "中", "high": "高"}

ACTION_LABELS = {
    "Question": "提问",
    "Restatement or Paraphrasing": "复述或改述",
    "Reflection of Feelings": "情感反映",
    "Self-disclosure": "自我披露",
    "Affirmation and Reassurance": "肯定与安慰",
    "Providing Suggestions": "提供建议",
    "Information": "提供信息",
    "Others": "其他支持",
}

NEED_LABELS = {
    "LISTEN": "倾听",
    "COMFORT": "安慰",
    "CLARIFY": "梳理确认",
    "GUIDANCE": "寻求建议",
    "INFORMATION": "获取信息",
    "PAUSE": "暂停介入",
    "UNKNOWN": "未明确",
}

REASON_LABELS = {
    "safety boundary blocks autonomous intervention": "安全边界不允许系统自主介入",
    "the user is currently not receptive": "用户当前不愿意接受介入",
    "the user explicitly requested a pause": "用户明确要求暂停介入",
    "available evidence is insufficient for a committed affect claim": "当前证据不足，不能作出确定的情绪判断",
    "the user initiated interaction, so clarification is preferable to silence": "用户主动发起交互，询问确认比保持沉默更合适",
    "CONFIRM temporarily replaces the affect-led content action with a clarification question": "确认模式暂时使用提问代替情绪支持动作",
    "passive observation is preferable when no response was requested": "用户未请求回应，暂时观察比主动介入更合适",
    "emotion probabilities are highly ambiguous or unstable": "情绪概率高度模糊或不稳定",
    "no user request makes waiting less intrusive than questioning": "用户未主动请求，等待比追问更少打扰",
    "partial uncertainty requires a low-commitment, correctable response": "存在部分不确定性，应采用低承诺、可纠正的表达",
    "the affect estimate is sufficiently clear for direct support": "情绪估计较清晰，可以直接提供支持",
    "public context reduces intervention salience": "公共环境下应降低介入显著性",
    "high time pressure favors concise information": "时间压力较高，优先提供简洁信息",
    "early negative affect favors reflection of feelings": "探索阶段的负面情绪优先采用情感反映",
    "early interaction favors exploration through a question": "交互早期优先通过提问进行探索",
    "high-arousal fear or sadness favors reassurance": "高唤醒的恐惧或悲伤优先采用肯定与安慰",
    "fear or sadness favors reflection of feelings": "恐惧或悲伤优先采用情感反映",
    "anger favors a non-escalating paraphrase": "愤怒状态优先采用非升级性的复述",
    "positive affect favors affirmation": "积极情绪优先采用肯定回应",
    "surprise favors clarification before advice": "惊讶状态下先澄清再提供建议",
    "closing stage favors a brief closing response": "结束阶段优先采用简短收束回应",
    "stable supporting stage allows a small suggestion": "稳定支持阶段可以提供小幅建议",
    "explicit listening need favors reflection of feelings": "用户明确希望被倾听，优先采用情感反映",
    "explicit comfort need favors affirmation and reassurance": "用户明确需要安慰，优先采用肯定与安慰",
    "explicit clarification need favors a question": "用户希望梳理感受，优先采用提问确认",
    "explicit guidance need favors a small suggestion": "用户明确希望获得建议，优先提供小幅建议",
    "explicit information need favors concise information": "用户明确需要信息，优先提供简洁信息",
}


def translate_reason(reason: str) -> str:
    parts = [item.strip() for item in reason.split(";")]
    translated = []
    for part in parts:
        if part == "the need is inherited from interaction history":
            translated.append("本轮未重新选择需求，沿用交互历史中的明确需求")
        elif part == "history excludes a previously rejected support action":
            translated.append("交互历史排除了此前被拒绝的支持动作")
        elif part == "history preserves a previously effective support action":
            translated.append("交互历史保留了此前有效的支持动作")
        elif part == "history switches after repeated support without improvement":
            translated.append("同一支持动作连续无改善，交互历史触发动作切换")
        else:
            translated.append(REASON_LABELS.get(part, part))
    return "；".join(translated)
