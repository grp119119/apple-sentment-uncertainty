"""Deterministic manual-input path for the v4.0 research interface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from affective_v03.config import load_policy_configuration
from affective_v03.models import (
    AffectEstimate,
    AffectLabel,
    HistoryRecord,
    InterventionDecision,
    PolicyContext,
    UncertaintyProfile,
    UserNeed,
)
from affective_v03.policy import UncertaintyAwarePolicy
from affective_v03.realizer import realize_response
from affective_v03.scenarios import DEMONSTRATION_ONLY
from affective_v03.uncertainty import normalized_entropy, top_two_margin


ROOT = Path(__file__).resolve().parent.parent
POLICY_CONFIG = ROOT / "config" / "policy_thresholds.json"


def _unit(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是数字")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} 必须在 0 到 1 之间")
    return result


@dataclass(frozen=True)
class ManualSimulationInput:
    emotion_distribution: Mapping[AffectLabel | str, float]
    valence: float
    arousal: float
    fatigue: float
    signal_quality: float
    data_coverage: float
    context: PolicyContext
    user_text: str = ""

    def __post_init__(self) -> None:
        parsed: dict[AffectLabel, float] = {}
        for raw_label, raw_value in self.emotion_distribution.items():
            label = raw_label if isinstance(raw_label, AffectLabel) else AffectLabel(raw_label)
            parsed[label] = _unit(f"情绪概率[{label.value}]", raw_value)
        total = sum(parsed.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"七类情绪概率之和必须为 1.00，当前为 {total:.3f}")
        if isinstance(self.valence, bool) or not isinstance(self.valence, (int, float)):
            raise TypeError("效价必须是数字")
        if not -1.0 <= float(self.valence) <= 1.0:
            raise ValueError("效价必须在 -1 到 1 之间")
        object.__setattr__(self, "emotion_distribution", parsed)
        object.__setattr__(self, "valence", float(self.valence))
        object.__setattr__(self, "arousal", _unit("唤醒度", self.arousal))
        object.__setattr__(self, "fatigue", _unit("疲惫程度", self.fatigue))
        object.__setattr__(self, "signal_quality", _unit("信号质量", self.signal_quality))
        object.__setattr__(self, "data_coverage", _unit("数据覆盖率", self.data_coverage))
        object.__setattr__(self, "user_text", str(self.user_text).strip())


@dataclass(frozen=True)
class ManualSimulationResult:
    affect: AffectEstimate
    uncertainty: UncertaintyProfile
    decision: InterventionDecision
    response: str
    fatigue: float
    user_need: UserNeed
    user_text: str
    history_before: HistoryRecord

    def to_dict(self) -> dict[str, object]:
        return {
            "status": DEMONSTRATION_ONLY,
            "input_type": "MANUAL_NUMERIC_SIMULATION",
            "affect": self.affect.to_dict(),
            "auxiliary_state": {"fatigue": self.fatigue},
            "manual_user_input": {
                "text": self.user_text,
                "need": self.user_need.value,
                "source": "MANUAL_SELECTION",
            },
            "history_before": self.history_before.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "decision": self.decision.to_dict(),
            "response": self.response,
        }


def run_manual_simulation(
    data: ManualSimulationInput,
    policy: UncertaintyAwarePolicy | None = None,
) -> ManualSimulationResult:
    """Run the existing adaptive policy on manually supplied research inputs."""
    affect = AffectEstimate(
        distribution=data.emotion_distribution,
        valence=data.valence,
        arousal=data.arousal,
    )
    entropy = normalized_entropy(affect.distribution)
    margin = top_two_margin(affect.distribution)
    uncertainty = UncertaintyProfile(
        predictive_entropy=entropy,
        top_two_margin=margin,
        signal_quality=data.signal_quality,
        data_coverage=data.data_coverage,
        missingness=1.0 - data.data_coverage,
        ensemble_variance=0.0,
        temporal_instability=0.0,
        cross_modal_disagreement=None,
        reasons=(
            f"手动概率分布的归一化预测熵={entropy:.3f}",
            f"最高两类概率差={margin:.3f}",
            f"手动信号质量={data.signal_quality:.3f}",
            f"手动数据覆盖率={data.data_coverage:.3f}",
            "模型方差、时间波动和跨模态冲突未在本轮手动模拟中启用",
        ),
    )
    selected_policy = policy or load_policy_configuration(POLICY_CONFIG).build_policy()
    decision = selected_policy.decide(affect, uncertainty, data.context)
    return ManualSimulationResult(
        affect=affect,
        uncertainty=uncertainty,
        decision=decision,
        response=realize_response(decision),
        fatigue=data.fatigue,
        user_need=data.context.user_need,
        user_text=data.user_text,
        history_before=data.context.history,
    )
