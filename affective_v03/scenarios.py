"""Demonstration scenarios for UI checks, never research evidence."""

from __future__ import annotations

from .multimodal import Modality, ModalityEvidence, MultimodalObservation


DEMONSTRATION_ONLY = "DEMONSTRATION_ONLY_NOT_STUDY_DATA"


def _e(modality, distribution, valence, arousal, quality=0.9, available=True):
    return ModalityEvidence(
        modality,
        distribution if available else None,
        valence if available else None,
        arousal if available else None,
        quality=quality if available else 0.0,
        available=available,
    )


def demonstration_scenarios() -> dict[str, MultimodalObservation]:
    return {
        "清晰一致的恐惧": MultimodalObservation(
            (
                _e(Modality.TEXT, {"fear": 0.82, "neutral": 0.10, "sadness": 0.08}, -0.7, 0.82),
                _e(Modality.VOICE, {"fear": 0.86, "neutral": 0.08, "sadness": 0.06}, -0.72, 0.88),
                _e(Modality.FACE, {"fear": 0.78, "neutral": 0.12, "sadness": 0.10}, -0.62, 0.76),
            )
        ),
        "恐惧与悲伤难区分": MultimodalObservation(
            (
                _e(Modality.TEXT, {"fear": 0.38, "sadness": 0.35, "neutral": 0.27}, -0.55, 0.72),
                _e(Modality.VOICE, {"fear": 0.35, "sadness": 0.36, "neutral": 0.29}, -0.50, 0.78),
                _e(Modality.FACE, {"fear": 0.32, "sadness": 0.31, "neutral": 0.37}, -0.40, 0.65),
            )
        ),
        "语言与非语言冲突": MultimodalObservation(
            (
                _e(Modality.TEXT, {"joy": 0.70, "neutral": 0.20, "fear": 0.10}, 0.55, 0.35),
                _e(Modality.VOICE, {"fear": 0.68, "sadness": 0.17, "neutral": 0.15}, -0.60, 0.82),
                _e(Modality.FACE, {"fear": 0.62, "sadness": 0.20, "neutral": 0.18}, -0.55, 0.74),
            )
        ),
        "仅文本信号可用": MultimodalObservation(
            (
                _e(Modality.TEXT, {"sadness": 0.58, "neutral": 0.27, "fear": 0.15}, -0.52, 0.58),
                _e(Modality.VOICE, {}, 0.0, 0.0, available=False),
                _e(Modality.FACE, {}, 0.0, 0.0, available=False),
            )
        ),
    }
