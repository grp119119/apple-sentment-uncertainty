"""Unit tests for the Stage 1 Perception Record & Adapter layer."""

from __future__ import annotations

import unittest

from affective_v03.models import AffectLabel, PolicyContext
from affective_v03.policy import UncertaintyAwarePolicy
from affective_v04.perception.adapter import (
    EMOTION2VEC_9_MAP,
    HSEMOTION_8_MAP,
    build_observation_from_records,
    map_raw_scores_to_ekman,
    record_to_evidence,
    safe_fuse_records,
)
from affective_v04.perception.contracts import (
    PerceptionModality,
    PerceptionRecord,
    PerceptionSource,
    softmax,
)


class TestPerceptionAdapter(unittest.TestCase):
    def test_softmax_normalizes_arbitrary_logits(self) -> None:
        logits = {"a": -2.0, "b": 0.0, "c": 3.0}
        probs = softmax(logits)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)
        self.assertTrue(probs["c"] > probs["b"] > probs["a"])

    def test_hsemotion_8_mapping_preserves_distribution(self) -> None:
        raw_hsemotion = {
            "Anger": 0.05,
            "Contempt": 0.05,  # 映射至 disgust
            "Disgust": 0.10,
            "Fear": 0.05,
            "Happiness": 0.60,
            "Neutrality": 0.05,
            "Sadness": 0.05,
            "Surprise": 0.05,
        }
        ekman = map_raw_scores_to_ekman(raw_hsemotion, HSEMOTION_8_MAP)
        self.assertAlmostEqual(sum(ekman.values()), 1.0, places=5)
        self.assertAlmostEqual(ekman[AffectLabel.JOY], 0.60, places=3)
        # Disgust 包含 0.10(Disgust) + 0.05(Contempt) = 0.15
        self.assertAlmostEqual(ekman[AffectLabel.DISGUST], 0.15, places=3)

    def test_emotion2vec_9_mapping(self) -> None:
        raw_audio = {
            "angry": 0.0,
            "disgusted": 0.0,
            "fearful": 0.0,
            "happy": 0.70,
            "neutral": 0.10,
            "other": 0.10,
            "sad": 0.0,
            "surprised": 0.10,
            "unknown": 0.0,
        }
        ekman = map_raw_scores_to_ekman(raw_audio, EMOTION2VEC_9_MAP)
        self.assertAlmostEqual(sum(ekman.values()), 1.0, places=5)
        self.assertAlmostEqual(ekman[AffectLabel.JOY], 0.70, places=3)
        # neutral 包含 0.10(neutral) + 0.10(other) = 0.20
        self.assertAlmostEqual(ekman[AffectLabel.NEUTRAL], 0.20, places=3)

    def test_record_to_evidence_conversion(self) -> None:
        rec = PerceptionRecord(
            modality=PerceptionModality.VISUAL,
            source=PerceptionSource.CAMERA,
            model_name="HSEmotionONNX",
            model_version="enet_b0_8_best_afew",
            timestamp=1700000000.0,
            raw_scores={"Happiness": 0.8, "Neutrality": 0.2},
            mapped_scores={
                AffectLabel.JOY: 0.8,
                AffectLabel.NEUTRAL: 0.2,
                AffectLabel.SADNESS: 0.0,
                AffectLabel.ANGER: 0.0,
                AffectLabel.FEAR: 0.0,
                AffectLabel.DISGUST: 0.0,
                AffectLabel.SURPRISE: 0.0,
            },
            signal_quality=0.95,
            available=True,
        )
        evidence = record_to_evidence(rec)
        self.assertTrue(evidence.available)
        self.assertEqual(evidence.quality, 0.95)
        self.assertAlmostEqual(evidence.emotion_distribution[AffectLabel.JOY], 0.8, places=3)
        self.assertTrue(evidence.valence > 0)  # 高喜悦应呈现正效价

    def test_missing_modality_handles_gracefully(self) -> None:
        # 人脸未检出
        missing_face = PerceptionRecord(
            modality=PerceptionModality.VISUAL,
            source=PerceptionSource.CAMERA,
            model_name="YuNet",
            model_version="1.0",
            timestamp=1700000000.0,
            raw_scores={},
            mapped_scores={},
            signal_quality=0.0,
            available=False,
            missing_reason="NO_FACE_DETECTED",
        )
        evidence = record_to_evidence(missing_face)
        self.assertFalse(evidence.available)
        self.assertEqual(evidence.quality, 0.0)

    def test_safe_fuse_records_with_all_missing_never_crashes(self) -> None:
        # 所有传感器全无数据 (如没人说话且人脸出画)
        empty_records = [
            PerceptionRecord(
                modality=PerceptionModality.VISUAL,
                source=PerceptionSource.CAMERA,
                model_name="YuNet",
                model_version="1.0",
                timestamp=1700000000.0,
                raw_scores={},
                mapped_scores={},
                signal_quality=0.0,
                available=False,
                missing_reason="NO_FACE_DETECTED",
            ),
            PerceptionRecord(
                modality=PerceptionModality.AUDIO,
                source=PerceptionSource.MICROPHONE,
                model_name="emotion2vec",
                model_version="plus_base",
                timestamp=1700000000.0,
                raw_scores={},
                mapped_scores={},
                signal_quality=0.0,
                available=False,
                missing_reason="EMPTY_AUDIO_SILENCE",
            ),
        ]
        # 不应抛出 ValueError，而是平稳返回
        fusion_result = safe_fuse_records(empty_records)
        self.assertEqual(fusion_result.uncertainty.data_coverage, 0.0)
        self.assertEqual(fusion_result.uncertainty.signal_quality, 0.0)

        # 喂给 Policy 验证平稳决策
        policy = UncertaintyAwarePolicy()
        context = PolicyContext(user_initiated=False)
        decision = policy.decide(fusion_result.affect, fusion_result.uncertainty, context)
        # 证据全无且用户未主动发起，应知趣地进入 WAIT
        self.assertEqual(decision.mode.value, "WAIT")

    def test_end_to_end_multimodal_records_drive_policy(self) -> None:
        # 构造三模态一致的高置信悲伤记录
        sad_dist = {
            AffectLabel.SADNESS: 0.85,
            AffectLabel.NEUTRAL: 0.10,
            AffectLabel.FEAR: 0.05,
            AffectLabel.JOY: 0.0,
            AffectLabel.ANGER: 0.0,
            AffectLabel.DISGUST: 0.0,
            AffectLabel.SURPRISE: 0.0,
        }
        records = [
            PerceptionRecord(
                modality=PerceptionModality.TEXT,
                source=PerceptionSource.ASR,
                model_name="chinese_text_sentiment",
                model_version="v1",
                timestamp=1700000000.0,
                raw_scores={"sadness": 0.85},
                mapped_scores=sad_dist,
                signal_quality=0.9,
                transcript="我最近真的感到很难过。",
                correlation_id="turn_001",
            ),
            PerceptionRecord(
                modality=PerceptionModality.AUDIO,
                source=PerceptionSource.MICROPHONE,
                model_name="emotion2vec_plus_base",
                model_version="1.0",
                timestamp=1700000000.0,
                raw_scores={"sad": 0.85},
                mapped_scores=sad_dist,
                signal_quality=0.85,
                correlation_id="turn_001",
            ),
            PerceptionRecord(
                modality=PerceptionModality.VISUAL,
                source=PerceptionSource.CAMERA,
                model_name="HSEmotionONNX",
                model_version="enet_b0",
                timestamp=1700000000.0,
                raw_scores={"Sadness": 0.85},
                mapped_scores=sad_dist,
                signal_quality=0.92,
            ),
        ]
        fusion_result = safe_fuse_records(records)
        self.assertEqual(fusion_result.affect.dominant_emotion, AffectLabel.SADNESS)
        self.assertTrue(fusion_result.uncertainty.signal_quality > 0.8)

        policy = UncertaintyAwarePolicy()
        context = PolicyContext(user_initiated=True)
        decision = policy.decide(fusion_result.affect, fusion_result.uncertainty, context)
        # 高置信悲伤状态下，应直接进入 DIRECT 介入
        self.assertEqual(decision.mode.value, "DIRECT")


if __name__ == "__main__":
    unittest.main()

