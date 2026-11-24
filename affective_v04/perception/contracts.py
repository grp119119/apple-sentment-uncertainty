"""Contracts for multimodal perception records and raw evidence tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from affective_v03.models import AffectLabel


class PerceptionModality(str, Enum):
    """The three information modalities specified in Framework 2.0."""

    TEXT = "text"
    AUDIO = "audio"
    VISUAL = "visual"


class PerceptionSource(str, Enum):
    """Origin of raw sensor or textual input."""

    TYPED = "typed"                  # 键盘手工输入
    ASR = "asr"                      # 麦克风录音由 ASR 转写
    MICROPHONE = "microphone"        # 麦克风原生声学信号 (SER)
    CAMERA = "camera"                # 摄像头视频帧 / 人脸图片 (FER)
    OFFLINE_DATASET = "dataset"      # 离线验证数据集或基准文件


def softmax(scores: Mapping[str, float]) -> dict[str, float]:
    """Numerically stable softmax for arbitrary logits."""
    if not scores:
        return {}
    max_val = max(scores.values())
    exp_vals = {k: math.exp(v - max_val) for k, v in scores.items()}
    total = sum(exp_vals.values())
    if total <= 0:
        return {k: 1.0 / len(scores) for k in scores}
    return {k: v / total for k, v in exp_vals.items()}


@dataclass(frozen=True)
class PerceptionRecord:
    """Standardized record of raw model inference output before fusion."""

    modality: PerceptionModality
    source: PerceptionSource
    model_name: str
    model_version: str
    timestamp: float
    raw_scores: Mapping[str, float]               # 原始标签与未篡改得分
    mapped_scores: Mapping[AffectLabel, float]    # 映射至 Ekman 7 类标准概率
    signal_quality: float = 1.0                   # 0.0 至 1.0 信号质量 (如人脸置信度、音频SNR)
    available: bool = True                        # 模态是否有效检出
    missing_reason: str | None = None             # 缺失原因 (如 "NO_FACE_DETECTED", "EMPTY_AUDIO")
    transcript: str | None = None                 # 文本内容 (如适用)
    correlation_id: str | None = None             # 同源关联 ID (如文本和声音来自同一段录音)
    latency_ms: float = 0.0                       # 本地模型推理耗时 (毫秒)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.signal_quality) <= 1.0:
            raise ValueError(f"signal_quality 必须在 0 到 1 之间，当前为: {self.signal_quality}")

    def to_dict(self) -> dict[str, object]:
        return {
            "modality": self.modality.value,
            "source": self.source.value,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "raw_scores": dict(self.raw_scores),
            "mapped_scores": {k.value: round(v, 4) for k, v in self.mapped_scores.items()},
            "signal_quality": round(self.signal_quality, 4),
            "available": self.available,
            "missing_reason": self.missing_reason,
            "transcript": self.transcript,
            "correlation_id": self.correlation_id,
            "latency_ms": round(self.latency_ms, 2),
        }

