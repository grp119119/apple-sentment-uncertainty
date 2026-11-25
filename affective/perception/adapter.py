"""Observation Adapter connecting perception records to system fusion layer."""

from __future__ import annotations

import time
from typing import Mapping, Sequence

from affective.fusion import FusionResult, fuse_observation
from affective.models import AffectLabel, normalize_distribution
from affective.multimodal import Modality, ModalityEvidence, MultimodalObservation
from .contracts import (
    PerceptionModality,
    PerceptionRecord,
    PerceptionSource,
    softmax,
)

# -------------------------------------------------------------
# 典型候选模型的标签映射字典 (显式追踪，不可暗箱操作)
# -------------------------------------------------------------

# HSEmotion 8类面部表情 -> Ekman 7类
HSEMOTION_8_MAP: Mapping[str, AffectLabel] = {
    "Anger": AffectLabel.ANGER,
    "Contempt": AffectLabel.DISGUST,    # 蔑视归纳为广义厌恶，在日志中标记
    "Disgust": AffectLabel.DISGUST,
    "Fear": AffectLabel.FEAR,
    "Happiness": AffectLabel.JOY,
    "Neutrality": AffectLabel.NEUTRAL,
    "Sadness": AffectLabel.SADNESS,
    "Surprise": AffectLabel.SURPRISE,
}

# emotion2vec+ 9类语音情绪 -> Ekman 7类
EMOTION2VEC_9_MAP: Mapping[str, AffectLabel] = {
    "angry": AffectLabel.ANGER,
    "disgusted": AffectLabel.DISGUST,
    "fearful": AffectLabel.FEAR,
    "happy": AffectLabel.JOY,
    "neutral": AffectLabel.NEUTRAL,
    "other": AffectLabel.NEUTRAL,       # 其他归于中性基线
    "sad": AffectLabel.SADNESS,
    "surprised": AffectLabel.SURPRISE,
    "unknown": AffectLabel.NEUTRAL,     # 未知归于中性基线
    "unk": AffectLabel.NEUTRAL,
    "<unk>": AffectLabel.NEUTRAL,
    "生气/angry": AffectLabel.ANGER,
    "厌恶/disgusted": AffectLabel.DISGUST,
    "恐惧/fearful": AffectLabel.FEAR,
    "开心/happy": AffectLabel.JOY,
    "中立/neutral": AffectLabel.NEUTRAL,
    "其他/other": AffectLabel.NEUTRAL,
    "难过/sad": AffectLabel.SADNESS,
    "吃惊/surprised": AffectLabel.SURPRISE,
}

# 中文 6类文本情绪模型 -> Ekman 7类
CHINESE_6_TEXT_MAP: Mapping[str, AffectLabel] = {
    "joy": AffectLabel.JOY,
    "sadness": AffectLabel.SADNESS,
    "anger": AffectLabel.ANGER,
    "fear": AffectLabel.FEAR,
    "disgust": AffectLabel.DISGUST,
    "neutral": AffectLabel.NEUTRAL,
    # 注: 部分文本模型缺少 surprise，映射时显式标明
}


def map_raw_scores_to_ekman(
    raw_scores: Mapping[str, float],
    mapping_dict: Mapping[str, AffectLabel],
    apply_softmax: bool = False,
) -> dict[AffectLabel, float]:
    """Deterministically map arbitrary raw model outputs into Ekman 7-class distribution."""
    if not raw_scores:
        return {label: 1.0 / len(AffectLabel) for label in AffectLabel}

    # 过滤以 '_' 开头的内部辅助元数据 (如 _bbox_x, _bbox_y, _bbox_w, _bbox_h 等)，杜绝污染情绪分布
    clean_raw = {k: v for k, v in raw_scores.items() if not str(k).startswith("_")}
    if not clean_raw:
        return {label: 1.0 / len(AffectLabel) for label in AffectLabel}

    scores = softmax(clean_raw) if apply_softmax else dict(clean_raw)

    # 累加映射到 7 类
    ekman_counts: dict[AffectLabel, float] = {label: 0.0 for label in AffectLabel}
    for raw_label, score in scores.items():
        mapped_target = mapping_dict.get(raw_label)
        if mapped_target is None and "/" in str(raw_label):
            mapped_target = mapping_dict.get(str(raw_label).split("/")[-1].lower().strip())
        if mapped_target is None and str(raw_label).lower().strip() in mapping_dict:
            mapped_target = mapping_dict.get(str(raw_label).lower().strip())
        if mapped_target is not None:
            ekman_counts[mapped_target] += float(score)
        else:
            # 未在字典中的类别，均摊或归入中性
            ekman_counts[AffectLabel.NEUTRAL] += float(score)

    total = sum(ekman_counts.values())
    if total <= 0:
        return {label: 1.0 / len(AffectLabel) for label in AffectLabel}
    return {label: val / total for label, val in ekman_counts.items()}


def record_to_evidence(record: PerceptionRecord) -> ModalityEvidence:
    """Convert an individual PerceptionRecord into a standard ModalityEvidence."""
    mod_map = {
        PerceptionModality.TEXT: Modality.TEXT,
        PerceptionModality.AUDIO: Modality.VOICE,
        PerceptionModality.VISUAL: Modality.FACE,
    }
    target_modality = mod_map[record.modality]

    if not record.available or record.signal_quality <= 0:
        return ModalityEvidence(
            modality=target_modality,
            available=False,
            quality=0.0,
            emotion_distribution=None,
        )

    # 确保概率归一化
    dist = normalize_distribution(record.mapped_scores)

    # 粗略估算效价与唤醒度 (基于离散情绪的心理学典型权重)
    # Joy(+0.8), Sadness(-0.6), Fear(-0.7), Anger(-0.6), Disgust(-0.6), Surprise(+0.1), Neutral(0.0)
    valence = (
        dist[AffectLabel.JOY] * 0.8
        - dist[AffectLabel.SADNESS] * 0.6
        - dist[AffectLabel.FEAR] * 0.7
        - dist[AffectLabel.ANGER] * 0.6
        - dist[AffectLabel.DISGUST] * 0.6
    )
    arousal = (
        dist[AffectLabel.ANGER] * 0.9
        + dist[AffectLabel.FEAR] * 0.85
        + dist[AffectLabel.SURPRISE] * 0.8
        + dist[AffectLabel.JOY] * 0.6
        + dist[AffectLabel.SADNESS] * 0.4
        + dist[AffectLabel.NEUTRAL] * 0.2
    )

    return ModalityEvidence(
        modality=target_modality,
        emotion_distribution=dist,
        valence=max(-1.0, min(1.0, valence)),
        arousal=max(0.0, min(1.0, arousal)),
        quality=record.signal_quality,
        available=True,
    )


def build_observation_from_records(
    records: Sequence[PerceptionRecord],
) -> MultimodalObservation:
    """Combine records from multiple modalities into a single MultimodalObservation."""
    evidences: list[ModalityEvidence] = []
    seen_modalities: set[PerceptionModality] = set()

    for rec in records:
        if rec.modality in seen_modalities:
            # 避免同一模态重复传入
            continue
        seen_modalities.add(rec.modality)
        evidences.append(record_to_evidence(rec))

    # 如果没有任何有效记录，提供默认的不可用占位（防止空列表报错）
    if not evidences:
        evidences = [
            ModalityEvidence(modality=Modality.TEXT, available=False, quality=0.0),
            ModalityEvidence(modality=Modality.VOICE, available=False, quality=0.0),
            ModalityEvidence(modality=Modality.FACE, available=False, quality=0.0),
        ]

    return MultimodalObservation(evidence=tuple(evidences))


def safe_fuse_records(
    records: Sequence[PerceptionRecord],
    previous_distribution: Mapping[AffectLabel | str, float] | None = None,
) -> FusionResult:
    """Safe wrapper over fuse_observation that smoothly handles all-missing sensor scenarios."""
    observation = build_observation_from_records(records)
    active = [e for e in observation.evidence if e.available and e.quality > 0]

    # 全缺失防护 (如摄像头未检出人脸且麦克风无人说话)
    if not active:
        uniform_dist = {label: 1.0 / len(AffectLabel) for label in AffectLabel}
        dummy_evidence = (
            ModalityEvidence(
                modality=Modality.TEXT,
                emotion_distribution=uniform_dist,
                quality=0.001,  # 极低质量占位
                available=True,
            ),
        )
        dummy_obs = MultimodalObservation(evidence=dummy_evidence)
        result = fuse_observation(dummy_obs, previous_distribution)
        # 修正覆盖率与质量为 0
        from dataclasses import replace
        zero_uncertainty = replace(
            result.uncertainty,
            data_coverage=0.0,
            signal_quality=0.0,
            missingness=1.0,
        )
        return FusionResult(
            affect=result.affect,
            uncertainty=zero_uncertainty,
            notes=("ALL_MODALITIES_MISSING: smoothly degraded without crash",),
        )

    return fuse_observation(observation, previous_distribution)

