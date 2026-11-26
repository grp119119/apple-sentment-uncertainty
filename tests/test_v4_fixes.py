"""Unit tests verifying the 3 fixes: Camera detection, VAD spike rejection, and ASR hallucination cleaning."""

import unittest
import numpy as np

from affective.perception.audio_detector import clean_sensevoice_output
from affective.perception.stream_coordinator import (
    RealtimeVADTracker,
    TurnEventType,
    TurnEvent,
)
from main_v4 import find_working_camera


class TestV4Fixes(unittest.TestCase):

    def test_clean_sensevoice_output(self):
        # 1. 噪声生成的纯标点与符号应过滤为空
        self.assertEqual(clean_sensevoice_output("。"), "")
        self.assertEqual(clean_sensevoice_output("，"), "")
        self.assertEqual(clean_sensevoice_output("🎼"), "")
        self.assertEqual(clean_sensevoice_output("   "), "")

        # 2. 带有特殊标签与音乐符号的文本应保留有效言语
        self.assertEqual(
            clean_sensevoice_output("<|zh|><|NEUTRAL|>我好开心呀。"),
            "我好开心呀。",
        )
        self.assertEqual(
            clean_sensevoice_output("🎼相继入心。"),
            "相继入心。",
        )
        self.assertEqual(
            clean_sensevoice_output("你好呀😊"),
            "你好呀",
        )

    def test_vad_rejects_transient_spike(self):
        # 模拟键盘敲击或呼吸：只有 1 个 64ms chunk 超过阈值，随后是静音
        vad = RealtimeVADTracker(
            energy_threshold=0.035,
            silence_cutoff_seconds=0.8,
            min_speech_duration=0.45,
        )

        ambient = np.zeros(1024, dtype=np.float32)
        spike = np.ones(1024, dtype=np.float32) * 0.1

        t0 = 10.0
        # 敲击瞬间
        vad.process_chunk(spike, now=t0)
        # 随后 1 秒静音 (超过 0.8s cutoff)
        ev = None
        for i in range(16):
            t = t0 + 0.064 * (i + 1)
            res = vad.process_chunk(ambient, now=t)
            if res:
                ev = res

        # 因为真实有效时长仅为 0.064s < 0.45s，绝对不触发言语结束事件
        self.assertIsNone(ev)

    def test_vad_accepts_sustained_speech(self):
        vad = RealtimeVADTracker(
            energy_threshold=0.035,
            silence_cutoff_seconds=0.8,
            min_speech_duration=0.45,
        )

        ambient = np.zeros(1024, dtype=np.float32)
        voice = np.ones(1024, dtype=np.float32) * 0.1

        t0 = 20.0
        # 用户持续说话 10 帧 (~0.64s > 0.40s)
        for i in range(10):
            vad.process_chunk(voice, now=t0 + 0.064 * i)

        # 随后静音 1.0s 触发句尾截断
        ev = None
        for i in range(16):
            t = t0 + 0.064 * 10 + 0.064 * (i + 1)
            res = vad.process_chunk(ambient, now=t)
            if res:
                ev = res

        self.assertIsNotNone(ev)
        self.assertEqual(ev.event_type, TurnEventType.SPEECH_FINISHED)

    def test_vad_hysteresis_and_thread_safe_turn_completion(self):
        vad = RealtimeVADTracker(
            energy_threshold=0.020,
            silence_cutoff_seconds=1.0,
            min_speech_duration=0.30,
        )
        t = 50.0
        # 起音帧 (0.030 >= 0.020)
        loud = np.ones(1024, dtype=np.float32) * 0.030
        # 句中弱辅音/句尾降调帧 (0.010 < 0.020, 但 >= sustain_thresh 0.0084)
        soft = np.ones(1024, dtype=np.float32) * 0.010
        silence = np.zeros(1024, dtype=np.float32)

        vad.process_chunk(loud, now=t)
        self.assertTrue(vad.is_speaking)

        # 后台线程完成上一轮推理时调用 mark_turn_completed，绝不能清空当前正在录制的语音
        vad.mark_turn_completed(current_time=t + 0.05)
        self.assertTrue(vad.is_speaking)
        self.assertGreater(len(vad.audio_buffer), 0)

        # 持续 1.2 秒弱音节，因双门限迟滞保持机制，不应在中途被误判为静音截断
        for i in range(20):
            t += 0.064
            ev = vad.process_chunk(soft, now=t)
            self.assertIsNone(ev)
            self.assertTrue(vad.is_speaking)

        # 真正静音超过 1.0s 后完整截断为一整句
        ev_final = None
        for i in range(18):
            t += 0.064
            res = vad.process_chunk(silence, now=t)
            if res:
                ev_final = res
        self.assertIsNotNone(ev_final)
        self.assertEqual(ev_final.event_type, TurnEventType.SPEECH_FINISHED)

    def test_vad_deep_copies_reused_portaudio_buffer(self):
        """Regression test: PortAudio reuses indata buffer in-place across callbacks.
        Ensure process_chunk deep-copies each frame so trailing silence never overwrites speech."""
        import os, wave
        vad = RealtimeVADTracker(
            energy_threshold=0.020,
            silence_cutoff_seconds=0.8,
            min_speech_duration=0.30,
        )
        # 模拟 PortAudio 单一固定复用内存块 indata (1024, 1)
        reused_indata = np.zeros((1024, 1), dtype=np.float32)
        t = 100.0
        for _ in range(10):
            t += 0.064
            reused_indata[:, 0] = 0.25  # 说话期高幅值
            vad.process_chunk(reused_indata[:, 0], now=t)  # 传入切片视图 (view)

        ev = None
        for _ in range(15):
            t += 0.064
            reused_indata[:, 0] = 0.0  # 句尾静音期原地覆写同一块内存为全 0
            res = vad.process_chunk(reused_indata[:, 0], now=t)
            if res:
                ev = res

        self.assertIsNotNone(ev)
        self.assertIsNotNone(ev.audio_wav_path)
        with wave.open(ev.audio_wav_path, "rb") as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        os.remove(ev.audio_wav_path)
        # 导出的 WAV 中必须保留高能量语音帧，峰值绝不能被尾随的 0.0 覆写为空
        self.assertGreater(float(np.max(np.abs(pcm))), 0.20)

    def test_text_lexicon_affect_and_negation(self):
        from affective.models import AffectLabel
        from affective.perception.audio_detector import estimate_text_affect

        dist_joy, q_joy, _ = estimate_text_affect("我现在感觉非常的开心。")
        self.assertEqual(max(dist_joy, key=dist_joy.get), AffectLabel.JOY)
        self.assertGreater(q_joy, 0.7)

        dist_neg, _, _ = estimate_text_affect("我今天一点也不开心。")
        self.assertEqual(max(dist_neg, key=dist_neg.get), AffectLabel.SADNESS)

        dist_fear, _, _ = estimate_text_affect("最近毕设压力大，感觉时间来不及，有点焦虑。")
        self.assertEqual(max(dist_fear, key=dist_fear.get), AffectLabel.FEAR)

        dist_fact, q_fact, _ = estimate_text_affect("开放时间早上9点至下午5点。")
        self.assertEqual(max(dist_fact, key=dist_fact.get), AffectLabel.NEUTRAL)
        self.assertAlmostEqual(q_fact, 0.50, places=2)

    def test_silence_face_only_coverage_triggers_wait_and_temporal_feedback(self):
        """Verify 4 runtime policy fixes:
        1. Face-only prolonged silence has data_coverage=1/3 (< 0.5) -> WAIT (never intrusive DIRECT).
        2. Default UserNeed.UNKNOWN allows affect-led action selection.
        3. temporal_instability > 0 when affect shifts between turns.
        4. observed_feedback_effect dynamically detects IMPROVED on positive valence shift.
        """
        import time
        from unittest.mock import MagicMock
        from affective.models import AffectLabel, UserNeed
        from affective.perception.contracts import PerceptionModality, PerceptionRecord, PerceptionSource
        from affective.perception.stream_coordinator import InteractiveSessionRunner

        mock_audio = MagicMock()
        runner = InteractiveSessionRunner(audio_detector=mock_audio, default_need=UserNeed.UNKNOWN, use_llm=False)
        now = time.time()

        # Turn 1: 静默超时，仅摄像头检测到清晰的悲伤人脸 (1/3 模态覆盖 -> coverage=0.333 < 0.50 -> WAIT)
        sad_face = PerceptionRecord(
            modality=PerceptionModality.VISUAL,
            source=PerceptionSource.CAMERA,
            model_name="mock",
            model_version="1.0",
            timestamp=now,
            raw_scores={},
            mapped_scores={AffectLabel.SADNESS: 0.90, AffectLabel.NEUTRAL: 0.10},
            signal_quality=0.92,
            available=True,
        )
        runner.visual_buffer.add_record(sad_face, timestamp=now)
        turn1 = runner.handle_turn_event(
            TurnEvent(event_type=TurnEventType.PROLONGED_SILENCE, timestamp=now, idle_duration=10.0)
        )
        self.assertAlmostEqual(turn1["data_coverage"], 1.0 / 3.0, places=3)
        self.assertEqual(turn1["policy_mode"], "WAIT")
        self.assertEqual(turn1["temporal_instability"], 0.0)

        # Turn 2: 用户主动开口说“我现在感觉非常开心”，三模态全到齐且情绪由悲转喜
        joy_dist = {AffectLabel.JOY: 0.85, AffectLabel.NEUTRAL: 0.15}
        joy_face = PerceptionRecord(
            modality=PerceptionModality.VISUAL,
            source=PerceptionSource.CAMERA,
            model_name="mock",
            model_version="1.0",
            timestamp=now + 5.0,
            raw_scores={},
            mapped_scores=joy_dist,
            signal_quality=0.90,
            available=True,
        )
        runner.visual_buffer.add_record(joy_face, timestamp=now + 5.0)
        mock_audio.analyze_audio.return_value = (
            PerceptionRecord(
                modality=PerceptionModality.AUDIO,
                source=PerceptionSource.MICROPHONE,
                model_name="mock_ser",
                model_version="1.0",
                timestamp=now + 5.0,
                raw_scores={},
                mapped_scores=joy_dist,
                signal_quality=0.88,
                available=True,
            ),
            PerceptionRecord(
                modality=PerceptionModality.TEXT,
                source=PerceptionSource.ASR,
                model_name="mock_text",
                model_version="1.0",
                timestamp=now + 5.0,
                raw_scores={},
                mapped_scores=joy_dist,
                signal_quality=0.82,
                available=True,
                transcript="我现在感觉非常开心。",
            ),
        )
        turn2 = runner.handle_turn_event(
            TurnEvent(
                event_type=TurnEventType.SPEECH_FINISHED,
                timestamp=now + 5.0,
                speech_start=now + 3.0,
                speech_end=now + 5.0,
                audio_wav_path="/tmp/nonexistent_mock.wav",
            )
        )
        self.assertAlmostEqual(turn2["data_coverage"], 1.0, places=3)
        self.assertGreater(turn2["temporal_instability"], 0.55)
        self.assertEqual(turn2["observed_feedback_effect"], "improved")
        # 情绪由悲转喜触发高时序波动 -> TENTATIVE，且探索阶段以 Question 试探
        self.assertEqual(turn2["policy_mode"], "TENTATIVE")
        self.assertEqual(turn2["policy_action"], "Question")

        # Turn 3: 进入 SUPPORTING 支持阶段，三模态持续稳定愉悦 -> temporal_instability=0 -> DIRECT + Affirmation
        runner.visual_buffer.add_record(joy_face, timestamp=now + 10.0)
        turn3 = runner.handle_turn_event(
            TurnEvent(
                event_type=TurnEventType.SPEECH_FINISHED,
                timestamp=now + 10.0,
                speech_start=now + 8.0,
                speech_end=now + 10.0,
                audio_wav_path="/tmp/nonexistent_mock.wav",
            )
        )
        self.assertEqual(turn3["policy_mode"], "DIRECT")
        self.assertEqual(turn3["policy_action"], "Affirmation and Reassurance")
        self.assertIn("positive affect favors affirmation", turn3["policy_reasons"])

    def test_autonomy_boundary_scenario_available_in_experiment_runner(self):
        from affective.scenarios import demonstration_scenarios
        from experiment_runner import run_comparison

        scenarios = demonstration_scenarios()
        self.assertIn("低不确定静默悲伤（自主权边界）", scenarios)
        obs = scenarios["低不确定静默悲伤（自主权边界）"]
        res_receptive = run_comparison(obs, user_initiated=False, receptive=True)
        res_not_receptive = run_comparison(obs, user_initiated=False, receptive=False)
        self.assertEqual(res_receptive["summary"]["uncertainty_aware"]["mode"], "DIRECT")
        self.assertEqual(res_not_receptive["summary"]["uncertainty_aware"]["mode"], "DEFER")


if __name__ == "__main__":
    unittest.main()
