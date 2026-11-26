"""OpenAI-compatible LLM adapter for Need recognition and Policy-constrained response generation.

Responsibilities (strictly bounded by the Adaptive Policy):
  1. Upstream NLU: Given the ASR transcript, extract explicit UserNeed (defaulting to
     UNKNOWN unless clearly stated) and a 7-class Ekman text emotion distribution.
  2. Downstream Realization: Given the frozen Policy decision (Mode, Action, Intensity,
     EpistemicStyle), generate a natural Chinese utterance that strictly obeys every
     policy constraint without overriding the policy.

Uses Python standard library (urllib.request) so it works cross-platform on macOS and
Windows with zero extra third-party dependencies, and supports Aliyun Bailian (Qwen),
SiliconFlow, DeepSeek, Zhipu, OpenAI, and local Ollama via .env configuration.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import (
    AffectEstimate,
    AffectLabel,
    InterventionDecision,
    InterventionIntensity,
    InterventionMode,
    SupportAction,
    UncertaintyProfile,
    UserNeed,
    normalize_distribution,
)
from .realizer import realize_response

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(env_path: Path | None = None) -> dict[str, str]:
    """Lightweight .env reader (does not override existing os.environ keys)."""
    target = env_path or (ROOT / ".env")
    loaded: dict[str, str] = {}
    if not target.exists():
        return loaded
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            loaded[key] = value
            os.environ.setdefault(key, value)
    return loaded


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 12.0

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "LLMConfig":
        load_dotenv(env_path)
        api_key = os.environ.get("LLM_API_KEY", "").strip()
        base_url = os.environ.get(
            "LLM_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).strip().rstrip("/")
        model = os.environ.get("LLM_MODEL", "qwen-plus").strip()
        try:
            timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "12"))
        except ValueError:
            timeout = 12.0
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and self.api_key != "sk-your-api-key-here"


@dataclass(frozen=True)
class LLMUtteranceAnalysis:
    user_need: UserNeed
    need_confidence: float
    need_reason: str
    text_emotion: dict[AffectLabel, float]
    signal_quality: float
    source: str = "llm"


_NEED_NAME_MAP: dict[str, UserNeed] = {
    "LISTEN": UserNeed.LISTEN,
    "COMFORT": UserNeed.COMFORT,
    "CLARIFY": UserNeed.CLARIFY,
    "GUIDANCE": UserNeed.GUIDANCE,
    "INFORMATION": UserNeed.INFORMATION,
    "PAUSE": UserNeed.PAUSE,
    "UNKNOWN": UserNeed.UNKNOWN,
}

_ACTION_GUIDANCE_ZH: dict[SupportAction, str] = {
    SupportAction.QUESTION: "用温和开放的提问帮用户表达或厘清感受",
    SupportAction.RESTATEMENT: "用平实不评判的话复述/转述用户的处境，传达你听懂了",
    SupportAction.REFLECTION: "聚焦并映照用户此刻的内在情绪感受，给予接纳",
    SupportAction.SELF_DISCLOSURE: "用克制、普适的同理心分享（如“很多人在这种时候也会有类似感受”）拉近距离",
    SupportAction.AFFIRMATION: "肯定用户的感受与努力，提供温暖安心的安抚",
    SupportAction.SUGGESTION: "只给出一个非常小、低负担、当下立刻能做的小建议，绝不说教",
    SupportAction.INFORMATION: "提供简洁、清晰、直接有用的客观信息或步骤",
    SupportAction.OTHERS: "简短温暖地承接并陪伴",
}

_MODE_CONSTRAINT_ZH: dict[InterventionMode, str] = {
    InterventionMode.DIRECT: (
        "【直接介入 (DIRECT) · 确定语气 (certain)】"
        "感知证据清晰，你可以自然、笃定、温暖地直接执行推荐的支持动作。"
    ),
    InterventionMode.TENTATIVE: (
        "【谨慎试探 (TENTATIVE) · 试探语气 (tentative)】"
        "当前感知存在部分不确定性或跨模态冲突。你必须采用低承诺、留有余地、允许用户纠正的口吻"
        "（例如自然带入“如果我没理解错的话……”“如果不准确你可以随时告诉我”），绝不能对用户情绪下武断定论。"
    ),
    InterventionMode.CONFIRM: (
        "【澄清确认 (CONFIRM) · 确认语气 (confirmatory)】"
        "当前感知高度模糊或存在明显冲突。你只能温柔、简短地向用户确认此刻的真实感受或想法，"
        "严禁直接给建议、严禁武断给用户贴情绪标签。"
    ),
    InterventionMode.WAIT: "【静默观察 (WAIT)】不主动发声。",
    InterventionMode.DEFER: "【暂缓介入 (DEFER)】尊重边界，简短退让不打扰。",
}

_INTENSITY_CONSTRAINT_ZH: dict[InterventionIntensity, str] = {
    InterventionIntensity.NONE: "不输出主动安抚话语。",
    InterventionIntensity.LOW: "【低强度 (LOW)】极简轻声陪伴：严格只说 1 句话（约 15~30 字），不追问、不展开长篇大论。",
    InterventionIntensity.MEDIUM: "【中强度 (MEDIUM)】温和适度承接：控制在 1~2 句话（约 30~60 字），清晰共情并留出呼吸空间。",
    InterventionIntensity.HIGH: "【高强度 (HIGH)】充分支持与温和承接：控制在 2~3 句话（约 50~90 字）；若用户处于高唤醒焦虑/愤怒/恐惧，请务必用慢节奏、降唤醒的平稳安抚口吻，绝不说教施压。",
}


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


class LLMAdapter:
    """Bounded LLM client for upstream Need/Text-Affect parsing and downstream Reply realization."""

    def __init__(self, config: LLMConfig | None = None, enabled: bool = True) -> None:
        self.config = config or LLMConfig.from_env()
        self._enabled = enabled and self.config.is_configured
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _chat_completion(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 256,
        json_mode: bool = False,
        disable_thinking: bool = True,
    ) -> str:
        url = f"{self.config.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        # 针对阿里云百炼 Qwen3+ 系列默认关闭长思考模式，保证实时对话秒级响应且最省 1M 额度
        if disable_thinking and "dashscope" in self.config.base_url.lower() and "qwen3" in self.config.model.lower():
            payload["enable_thinking"] = False

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                raw_resp = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="ignore")[:300]
            except Exception:
                pass
            # 若部分模型不支持 response_format=json_object 或 enable_thinking，降级重试一次
            if exc.code == 400 and (json_mode or ("enable_thinking" in payload)):
                return self._chat_completion(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=False,
                    disable_thinking=False,
                )
            raise RuntimeError(f"HTTP {exc.code}: {err_body}") from exc

        data = json.loads(raw_resp)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Empty LLM response choices")
        content = choices[0].get("message", {}).get("content", "")
        return str(content).strip()

    def test_connection(self) -> tuple[bool, str]:
        """Verify live API connectivity with a minimal ping."""
        if not self.enabled:
            return False, "LLM_API_KEY 未配置"
        try:
            reply = self._chat_completion(
                [
                    {"role": "system", "content": "只回复两个字：正常"},
                    {"role": "user", "content": "测试连通性"},
                ],
                temperature=0.0,
                max_tokens=16,
            )
            self.last_error = None
            return True, reply
        except Exception as exc:
            self.last_error = str(exc)
            return False, str(exc)

    def analyze_utterance(
        self,
        transcript: str,
        recent_turns: Sequence[Mapping[str, Any]] | None = None,
        need_confidence_threshold: float = 0.65,
    ) -> LLMUtteranceAnalysis | None:
        """Infer explicit UserNeed (strictly UNKNOWN if unstated) and Ekman 7-class text emotion."""
        if not self.enabled or not transcript or transcript.strip() == "[沉默无言语]":
            return None

        history_lines = []
        if recent_turns:
            for t in list(recent_turns)[-3:]:
                u_txt = t.get("transcript", "")
                s_txt = t.get("system_reply", "")
                if u_txt and u_txt != "[沉默无言语]":
                    history_lines.append(f"用户：{u_txt}")
                if s_txt:
                    history_lines.append(f"装置：{s_txt}")
        history_block = "\n".join(history_lines) if history_lines else "（首轮对话）"

        system_prompt = (
            "你是空间情感交互系统的上游语义分析器，只负责两件事，并严格只返回合法 JSON：\n"
            "1. 判断用户当前这句话是否**明确表达了交互需求 (user_need)**，可选值：\n"
            "   - LISTEN：明确希望被倾听、倾诉、吐槽（如“我就想找人说说”“听我说就好”）\n"
            "   - COMFORT：明确寻求安慰、鼓励、安抚（如“安慰我一下”“感觉撑不住需要鼓励”）\n"
            "   - CLARIFY：明确希望一起梳理感受或想法（如“帮我理一理”“我脑子好乱帮我分析下”）\n"
            "   - GUIDANCE：明确请求建议或行动方案（如“你觉得我该怎么办”“给我点建议”）\n"
            "   - INFORMATION：明确询问客观信息或事实（如“这个怎么操作”“几点结束”）\n"
            "   - PAUSE：明确要求停止打扰或安静（如“别说话了”“让我自己静一静”“先暂停”）\n"
            "   - UNKNOWN：用户只是表达情绪或日常叙述，**没有明确提出上述具体交互诉求**。\n"
            "   【严格规则】绝不要硬猜需求！拿不准或未明说时，user_need 必须填 \"UNKNOWN\"。\n"
            "2. 给出这句话在 Ekman 7 类情绪（anger, disgust, fear, joy, sadness, surprise, neutral）上的概率分布，总和为 1.0。\n"
            "   注意识别中文否定词（如“一点也不开心”属于 sadness 而非 joy）与压力/焦虑（归入 fear）。\n\n"
            "严格按如下 JSON 结构输出，不要输出任何额外解释：\n"
            "{\n"
            '  "user_need": "UNKNOWN",\n'
            '  "need_confidence": 0.0,\n'
            '  "need_reason": "一句话简短说明",\n'
            '  "text_emotion": {"anger": 0.0, "disgust": 0.0, "fear": 0.0, "joy": 0.0, "sadness": 0.0, "surprise": 0.0, "neutral": 1.0}\n'
            "}"
        )
        user_prompt = f"【最近对话上下文】\n{history_block}\n\n【用户本轮发言】\n{transcript}"

        try:
            raw = self._chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=220,
                json_mode=True,
            )
            parsed = _extract_json_object(raw)
            raw_need = str(parsed.get("user_need", "UNKNOWN")).strip().upper()
            need = _NEED_NAME_MAP.get(raw_need, UserNeed.UNKNOWN)
            conf = float(parsed.get("need_confidence", 0.0))
            conf = max(0.0, min(1.0, conf))
            # 置信度不足时强制退回 UNKNOWN，防止大模型乱猜架空情绪驱动策略
            if need not in {UserNeed.UNKNOWN, UserNeed.PAUSE} and conf < need_confidence_threshold:
                need = UserNeed.UNKNOWN
            reason = str(parsed.get("need_reason", "")).strip() or "LLM 语义分析"

            raw_emo = parsed.get("text_emotion") or {}
            clean_emo: dict[AffectLabel, float] = {}
            for label in AffectLabel:
                val = raw_emo.get(label.value, 0.0)
                try:
                    clean_emo[label] = max(0.0, float(val))
                except (TypeError, ValueError):
                    clean_emo[label] = 0.0
            if sum(clean_emo.values()) <= 0:
                clean_emo = {label: 1.0 / len(AffectLabel) for label in AffectLabel}
            dist = normalize_distribution(clean_emo)
            self.last_error = None
            return LLMUtteranceAnalysis(
                user_need=need,
                need_confidence=conf,
                need_reason=reason,
                text_emotion=dist,
                signal_quality=0.85,
                source=f"llm:{self.config.model}",
            )
        except Exception as exc:
            self.last_error = str(exc)
            return None

    def generate_reply(
        self,
        decision: InterventionDecision,
        affect: AffectEstimate,
        uncertainty: UncertaintyProfile,
        transcript: str,
        recent_turns: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        """Generate a policy-constrained Chinese utterance (or silent marker for WAIT/DEFER)."""
        # 硬安全护栏：WAIT 或 NONE 强度绝不调用大模型发声，防止破坏低侵入原则
        if decision.mode is InterventionMode.WAIT or decision.intensity is InterventionIntensity.NONE:
            return realize_response(decision)
        if decision.mode is InterventionMode.DEFER:
            return "好的，我先安静陪着你，不打扰。你想说话的时候随时开口。"

        if not self.enabled:
            return realize_response(decision)

        history_lines = []
        if recent_turns:
            for t in list(recent_turns)[-3:]:
                u_txt = t.get("transcript", "")
                s_txt = t.get("system_reply", "")
                if u_txt and u_txt != "[沉默无言语]":
                    history_lines.append(f"用户：{u_txt}")
                if s_txt and not s_txt.startswith("["):
                    history_lines.append(f"装置：{s_txt}")
        history_block = "\n".join(history_lines) if history_lines else "（首轮对话）"

        conflict_hint = ""
        if uncertainty.cross_modal_disagreement is not None and uncertainty.cross_modal_disagreement >= 0.35:
            conflict_hint = (
                f"注意：系统检测到明显跨模态不一致（冲突度={uncertainty.cross_modal_disagreement:.2f}），"
                "切勿武断下情绪结论，可保持温和留白或轻柔试探。"
            )

        system_prompt = (
            "你是一个面向等待/疗愈空间的短时情感陪伴装置的语言表达层。\n"
            "【人设口吻】安静、温和、克制、有分寸感、像一个懂事不打扰的朋友；"
            "严禁使用居高临下的说教腔、客服套话或生硬的“心理医生/咨询师术语”（不要说“我理解你的深层创伤”“作为AI”等），"
            "也不要输出任何括号动作说明或策略标签。\n"
            "【最高原则】你的回复必须严格服从上游 Adaptive Policy 给出的模式、动作、语气与强度指令，绝对不可越权改变。"
        )
        user_prompt = (
            f"【最近对话】\n{history_block}\n\n"
            f"【用户本轮发言】\n{transcript if transcript else '[用户静默未开口]'}\n\n"
            f"【上游多模态情绪估计（仅供参考，不可当成绝对事实）】\n"
            f"- 融合主导情绪：{affect.dominant_emotion.value}（效价={affect.valence:+.2f}，唤醒度={affect.arousal:.2f}）\n"
            f"- 预测不确定性熵：{uncertainty.predictive_entropy:.2f}\n"
            f"{conflict_hint}\n\n"
            f"【必须严格执行的 Policy 策略指令】\n"
            f"1. 介入模式与认知语气：{_MODE_CONSTRAINT_ZH[decision.mode]}\n"
            f"2. 推荐支持动作：{decision.support_action.value}（{_ACTION_GUIDANCE_ZH[decision.support_action]}）\n"
            f"3. 介入强度控制：{_INTENSITY_CONSTRAINT_ZH[decision.intensity]}\n\n"
            "请直接输出给用户看的最终中文回复（纯文本，不要加引号或前缀）："
        )

        try:
            reply = self._chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.45,
                max_tokens=180,
                json_mode=False,
            )
            reply = reply.strip().strip('"').strip("“”")
            if not reply:
                return realize_response(decision)
            self.last_error = None
            return reply
        except Exception as exc:
            self.last_error = str(exc)
            return realize_response(decision)


if __name__ == "__main__":
    adapter = LLMAdapter()
    ok, msg = adapter.test_connection()
    print(f"[LLM Ping] model={adapter.config.model} ok={ok} msg={msg}")
    if ok:
        sample = "我最近毕设压力好大，你能帮我想想第一步该怎么做吗？"
        analysis = adapter.analyze_utterance(sample)
        print(f"[Need & Affect] {analysis}")
        if analysis is not None:
            from .policy import PolicyContext, UncertaintyAwarePolicy
            aff = AffectEstimate(analysis.text_emotion, -0.5, 0.72)
            unc = UncertaintyProfile(0.28, 0.55, 0.88, 1.0, 0.0, cross_modal_disagreement=0.10)
            dec = UncertaintyAwarePolicy().decide(
                aff, unc, PolicyContext(user_initiated=True, user_need=analysis.user_need)
            )
            reply = adapter.generate_reply(dec, aff, unc, sample)
            print(f"[Policy] mode={dec.mode.value} action={dec.support_action.value} intensity={dec.intensity.value}")
            print(f"[Reply]  {reply}")
