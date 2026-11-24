"""Unit tests verifying the 3 fixes: Camera detection, VAD spike rejection, and ASR hallucination cleaning."""

import unittest
import numpy as np

from affective_v04.perception.audio_detector import clean_sensevoice_output
from affective_v04.perception.stream_coordinator import (
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


if __name__ == "__main__":
    unittest.main()
