"""Tests for stream_coordinator (Visual window averaging, VAD state machine, Turn Events)."""

import time
import unittest

import numpy as np

from affective_v03.models import AffectLabel, UserNeed
from affective_v04.perception.contracts import (
    PerceptionModality,
    PerceptionRecord,
    PerceptionSource,
)
from affective_v04.perception.stream_coordinator import (
    InteractiveSessionRunner,
    RealtimeVADTracker,
    TurnEvent,
    TurnEventType,
    VisualWindowBuffer,
    aggregate_visual_records,
)


class TestStreamCoordinator(unittest.TestCase):

    def test_aggregate_visual_records_weighted_mean(self) -> None:
        # 帧 1: 高置信度 (0.9), 快乐 (JOY: 1.0)
        rec1 = PerceptionRecord(
            modality=PerceptionModality.VISUAL,
            source=PerceptionSource.CAMERA,
            model_name="mock",
            model_version="1.0",
            timestamp=100.0,
            raw_scores={},
            mapped_scores={AffectLabel.JOY: 1.0, AffectLabel.NEUTRAL: 0.0},
            signal_quality=0.9,
            available=True,
        )
        # 帧 2: 低置信度 (0.1), 中性 (NEUTRAL: 1.0)
        rec2 = PerceptionRecord(
            modality=PerceptionModality.VISUAL,
            source=PerceptionSource.CAMERA,
            model_name="mock",
            model_version="1.0",
            timestamp=101.0,
            raw_scores={},
            mapped_scores={AffectLabel.JOY: 0.0, AffectLabel.NEUTRAL: 1.0},
            signal_quality=0.1,
            available=True,
        )

        aggregated = aggregate_visual_records([rec1, rec2], now=102.0)
        self.assertTrue(aggregated.available)
        self.assertEqual(aggregated.modality, PerceptionModality.VISUAL)
        # 高置信度的帧权重 0.9 / 1.0 = 0.9
        self.assertAlmostEqual(aggregated.mapped_scores[AffectLabel.JOY], 0.9, delta=0.01)
        self.assertAlmostEqual(aggregated.mapped_scores[AffectLabel.NEUTRAL], 0.1, delta=0.01)

    def test_aggregate_visual_records_all_missing(self) -> None:
        rec_empty = PerceptionRecord(
            modality=PerceptionModality.VISUAL,
            source=PerceptionSource.CAMERA,
            model_name="mock",
            model_version="1.0",
            timestamp=100.0,
            raw_scores={},
            mapped_scores={},
            signal_quality=0.0,
            available=False,
        )
        aggregated = aggregate_visual_records([rec_empty], now=101.0)
        self.assertFalse(aggregated.available)
        self.assertEqual(aggregated.signal_quality, 0.0)

    def test_vad_tracker_speech_onset_and_offset(self) -> None:
        vad = RealtimeVADTracker(
            energy_threshold=0.02,
            silence_cutoff_seconds=0.5,
            idle_timeout_seconds=5.0,
            sample_rate=16000,
        )

        # 1. 模拟静音
        silence_chunk = np.zeros(1600, dtype=np.float32)
        ev1 = vad.process_chunk(silence_chunk, now=1.0)
        self.assertIsNone(ev1)
        self.assertFalse(vad.is_speaking)

        # 2. 模拟人声发音持续 0.6 秒
        voice_chunk = np.ones(1600, dtype=np.float32) * 0.1
        for t in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
            ev = vad.process_chunk(voice_chunk, now=t)
            self.assertIsNone(ev)
            self.assertTrue(vad.is_speaking)

        # 3. 模拟发音后静音超过 0.5 秒 (句尾截断)
        vad.process_chunk(silence_chunk, now=1.7)
        vad.process_chunk(silence_chunk, now=1.9)
        ev_finish = vad.process_chunk(silence_chunk, now=2.2)  # 2.2 - 1.6 = 0.6s >= 0.5s

        self.assertIsNotNone(ev_finish)
        self.assertEqual(ev_finish.event_type, TurnEventType.SPEECH_FINISHED)
        self.assertFalse(vad.is_speaking)
        self.assertIsNotNone(ev_finish.audio_wav_path)

    def test_vad_tracker_prolonged_silence_timeout(self) -> None:
        vad = RealtimeVADTracker(
            energy_threshold=0.02,
            silence_cutoff_seconds=0.5,
            idle_timeout_seconds=3.0,
            sample_rate=16000,
        )
        vad.reset_turn_timer(current_time=10.0)

        silence_chunk = np.zeros(1600, dtype=np.float32)
        # 2.5秒后仍未超时
        ev = vad.process_chunk(silence_chunk, now=12.5)
        self.assertIsNone(ev)

        # 3.1秒后超过 3.0s 超时时间，触发静默事件
        ev_timeout = vad.process_chunk(silence_chunk, now=13.1)
        self.assertIsNotNone(ev_timeout)
        self.assertEqual(ev_timeout.event_type, TurnEventType.PROLONGED_SILENCE)

    def test_interactive_session_runner_handles_turns(self) -> None:
        runner = InteractiveSessionRunner(default_need=UserNeed.COMFORT)

        # 模拟存入人脸记录到 buffer
        rec = PerceptionRecord(
            modality=PerceptionModality.VISUAL,
            source=PerceptionSource.CAMERA,
            model_name="mock",
            model_version="1.0",
            timestamp=100.0,
            raw_scores={},
            mapped_scores={AffectLabel.JOY: 0.9, AffectLabel.NEUTRAL: 0.1},
            signal_quality=0.85,
            available=True,
        )
        runner.visual_buffer.add_record(rec, timestamp=100.0)

        # 模拟触发静默超时轮次
        event = TurnEvent(
            event_type=TurnEventType.PROLONGED_SILENCE,
            timestamp=102.0,
            idle_duration=8.5,
        )
        turn_data = runner.handle_turn_event(event)

        self.assertEqual(turn_data["turn_index"], 1)
        self.assertFalse(turn_data["user_initiated"])
        self.assertIn("policy_mode", turn_data)
        self.assertEqual(len(runner.journey_records), 1)


if __name__ == "__main__":
    unittest.main()

