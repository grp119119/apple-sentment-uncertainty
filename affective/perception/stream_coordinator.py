"""Real-time Multimodal Stream Coordinator and Turn-Taking State Machine.

Coordinates:
  1. Live Camera Stream -> Frame-level FER -> Utterance-Synchronized Weighted Window Average
  2. Live Microphone Stream -> RMS Energy VAD -> Speech Chunking (Onset/Offset)
  3. Turn-Taking State Machine:
     - Event A: Speech Finished (VAD Silence Cutoff >= 750ms) -> user_initiated=True
     - Event B: Prolonged Silence (Idle Timeout >= 8~10s)     -> user_initiated=False
  4. Safe Late Fusion -> Uncertainty-Aware JITAI Policy -> Structured Turn Card Output
  5. Session History Tracking (Data Schema for downstream generative art printout)
"""

from __future__ import annotations

import collections
import dataclasses
import os
import tempfile
import threading
import time
import wave
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from affective.history import update_history
from affective.models import (
    AffectLabel,
    HistoryEffect,
    HistoryRecord,
    InteractionStage,
    InterventionDecision,
    UserNeed,
)
from affective.policy import PolicyContext, UncertaintyAwarePolicy
from .adapter import safe_fuse_records
from .audio_detector import AudioEmotionDetector
from .contracts import (
    PerceptionModality,
    PerceptionRecord,
    PerceptionSource,
)
from .visual_detector import VisualEmotionDetector


class TurnEventType(str, Enum):
    """Event triggering an interaction turn."""

    SPEECH_FINISHED = "speech_finished"      # 用户主动说完一句话 (VAD截断)
    PROLONGED_SILENCE = "prolonged_silence"  # 用户长时间未开口 (静默超时)


@dataclasses.dataclass
class TurnEvent:
    event_type: TurnEventType
    timestamp: float
    speech_start: float | None = None
    speech_end: float | None = None
    audio_wav_path: str | None = None
    idle_duration: float = 0.0


def aggregate_visual_records(
    records: Sequence[PerceptionRecord],
    now: float | None = None,
) -> PerceptionRecord:
    """Compute confidence-weighted Ekman distribution average across face frames."""
    current_time = now if now is not None else time.time()
    valid_records = [r for r in records if r.available and r.signal_quality > 0.0]

    if not valid_records:
        return PerceptionRecord(
            modality=PerceptionModality.VISUAL,
            source=PerceptionSource.CAMERA,
            model_name="yunet+hsemotion_windowed",
            model_version="1.0",
            timestamp=current_time,
            raw_scores={"valid_frames": 0.0},
            mapped_scores={label: 1.0 / len(AffectLabel) for label in AffectLabel},
            signal_quality=0.0,
            available=False,
            missing_reason="NO_FACE_IN_WINDOW",
        )

    # 置信度加权 (置信度越高的帧，权重越大)
    weights = [max(0.05, float(r.signal_quality)) for r in valid_records]
    total_weight = sum(weights)

    all_labels = list(AffectLabel)
    weighted_scores: dict[AffectLabel, float] = {label: 0.0 for label in all_labels}

    for rec, w in zip(valid_records, weights):
        for label, score in rec.mapped_scores.items():
            if label in weighted_scores:
                weighted_scores[label] += float(score) * (w / total_weight)

    # 归一化分布
    total_score = sum(weighted_scores.values())
    if total_score > 0:
        normalized_scores = {k: v / total_score for k, v in weighted_scores.items()}
    else:
        normalized_scores = {k: 1.0 / len(all_labels) for k in all_labels}

    avg_signal_quality = float(sum(weights) / len(weights))

    return PerceptionRecord(
        modality=PerceptionModality.VISUAL,
        source=PerceptionSource.CAMERA,
        model_name="yunet+hsemotion_windowed",
        model_version="1.0",
        timestamp=current_time,
        raw_scores={"valid_frames": float(len(valid_records))},
        mapped_scores=normalized_scores,
        signal_quality=avg_signal_quality,
        available=True,
    )


class VisualWindowBuffer:
    """Thread-safe rolling buffer capturing and storing live face perception records."""

    def __init__(
        self,
        visual_detector: VisualEmotionDetector | None = None,
        max_history_seconds: float = 15.0,
    ) -> None:
        self.visual_detector = visual_detector
        self.max_history_seconds = max_history_seconds
        self._lock = threading.Lock()
        # Deque of (timestamp, PerceptionRecord)
        self._buffer: collections.deque[tuple[float, PerceptionRecord]] = collections.deque()
        self._running = False
        self._thread: threading.Thread | None = None

    def add_record(self, record: PerceptionRecord, timestamp: float | None = None) -> None:
        ts = timestamp if timestamp is not None else record.timestamp
        with self._lock:
            self._buffer.append((ts, record))
            self._purge_old_records(ts)

    def _purge_old_records(self, current_time: float) -> None:
        cutoff = current_time - self.max_history_seconds
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def get_records_between(self, start_time: float, end_time: float) -> list[PerceptionRecord]:
        with self._lock:
            return [
                rec
                for ts, rec in self._buffer
                if start_time <= ts <= end_time
            ]

    def get_aggregated_record(self, start_time: float, end_time: float) -> PerceptionRecord:
        records = self.get_records_between(start_time, end_time)
        return aggregate_visual_records(records, now=end_time)

    def get_recent_aggregated_record(self, duration_seconds: float = 3.0) -> PerceptionRecord:
        now = time.time()
        return self.get_aggregated_record(now - duration_seconds, now)


class RealtimeVADTracker:
    """RMS Energy-based Voice Activity Detection and Turn Timing Tracker with Dual-Threshold Hysteresis & AGC."""

    def __init__(
        self,
        energy_threshold: float = 0.018,
        silence_cutoff_seconds: float = 1.15,
        idle_timeout_seconds: float = 10.0,
        sample_rate: int = 16000,
        min_speech_duration: float = 0.30,
    ) -> None:
        self.energy_threshold = energy_threshold
        self.silence_cutoff_seconds = silence_cutoff_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        self.sample_rate = sample_rate
        self.min_speech_duration = min_speech_duration

        self.is_speaking = False
        self.speech_start_time: float | None = None
        self.last_speech_time: float | None = None
        self.last_turn_interaction_time: float = time.time()
        self.noise_floor: float = 0.005
        self.audio_buffer: list[np.ndarray] = []
        # 前置回环缓冲 (~768ms = 12 chunks)，完整保留声母起音，防止首字被切
        self.pre_roll_chunks: collections.deque[np.ndarray] = collections.deque(maxlen=12)

    def reset_turn_timer(self, current_time: float | None = None) -> None:
        """Hard reset VAD state and idle timer (call when starting/restarting a session)."""
        self.last_turn_interaction_time = current_time if current_time is not None else time.time()
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.audio_buffer.clear()
        self.pre_roll_chunks.clear()

    def mark_turn_completed(self, current_time: float | None = None) -> None:
        """Thread-safely refresh idle timer after a turn finishes WITHOUT wiping ongoing speech."""
        now_t = current_time if current_time is not None else time.time()
        if not self.is_speaking:
            self.last_turn_interaction_time = max(self.last_turn_interaction_time, now_t)

    def process_chunk(self, audio_chunk: np.ndarray, now: float | None = None) -> TurnEvent | None:
        """Process an audio chunk (float32, 1D array) and return a TurnEvent if triggered."""
        current_time = now if now is not None else time.time()

        # 关键修复：强制深拷贝音频帧！
        # sounddevice (PortAudio) 的回调缓冲区 indata 会在每次回调时原地覆写 (in-place overwrite)，
        # 若不深拷贝，audio_buffer 中保存的所有历史帧在句尾静音期都会被最后 1 帧静音覆盖，导致识别结果永远变成“我。”！
        if audio_chunk.size == 0:
            return None
        audio_chunk = np.array(audio_chunk, dtype=np.float32, copy=True).reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(audio_chunk))))

        # 双门限迟滞 (Hysteresis): 起音需要达到 energy_threshold，但说话持续中只要高于较低保持门限即可续音
        onset_thresh = max(self.energy_threshold, self.noise_floor * 1.8)
        sustain_thresh = max(self.energy_threshold * 0.45, self.noise_floor * 1.4, 0.006)
        is_voice_active = (rms >= sustain_thresh) if self.is_speaking else (rms >= onset_thresh)

        if is_voice_active:
            if not self.is_speaking:
                # 语音开始 (Voice Onset): 自动拼接前置缓冲，保证首字完整
                self.is_speaking = True
                pre_roll_duration = (len(self.pre_roll_chunks) * len(audio_chunk)) / float(self.sample_rate)
                self.speech_start_time = current_time - pre_roll_duration
                self.audio_buffer = list(self.pre_roll_chunks) + [audio_chunk]
            else:
                self.audio_buffer.append(audio_chunk)
            self.last_speech_time = current_time
            return None

        # 当前块为静音
        if not self.is_speaking:
            # 慢速自适应追踪环境底噪基线，并维护前置滚动缓冲
            if rms < self.energy_threshold:
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
            self.pre_roll_chunks.append(audio_chunk)

        if self.is_speaking:
            self.audio_buffer.append(audio_chunk)
            silence_duration = current_time - (self.last_speech_time or current_time)
            if silence_duration >= self.silence_cutoff_seconds:
                # 事件 A 触发: 说话结束 (Speech Finished)
                self.is_speaking = False
                speech_start = self.speech_start_time or (current_time - silence_duration)
                speech_end = self.last_speech_time or current_time
                active_speech_duration = max(0.0, speech_end - speech_start)

                # 必须达到有效发音时长 (>= min_speech_duration)，排除短于阈值的单次击键、轻微呼气或杂音
                if active_speech_duration >= self.min_speech_duration:
                    concatenated = np.concatenate(self.audio_buffer).astype(np.float32)
                    # 去直流偏置 + 温和增益均衡 (最高放大 4x，避免将纯底噪放大为满幅假语音)
                    if concatenated.size > 0:
                        concatenated = concatenated - float(np.mean(concatenated))
                        peak = float(np.max(np.abs(concatenated)))
                        if 0.02 < peak < 0.70:
                            gain = min(4.0, 0.80 / peak)
                            concatenated = concatenated * gain

                    # 写出临时 WAV 文件供 AudioEmotionDetector 读取
                    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    temp_wav_path = temp_wav.name
                    temp_wav.close()

                    int16_samples = np.clip(concatenated * 32767.0, -32768, 32767).astype(np.int16)
                    with wave.open(temp_wav_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(int16_samples.tobytes())

                    self.audio_buffer.clear()
                    self.pre_roll_chunks.clear()
                    self.last_turn_interaction_time = current_time
                    return TurnEvent(
                        event_type=TurnEventType.SPEECH_FINISHED,
                        timestamp=current_time,
                        speech_start=speech_start,
                        speech_end=speech_end,
                        audio_wav_path=temp_wav_path,
                    )
                else:
                    # 短促杂音被拒绝时，将最近帧回填至 pre_roll_chunks，防止紧随其后的首字被切
                    for c in self.audio_buffer[-12:]:
                        self.pre_roll_chunks.append(c)
                    self.audio_buffer.clear()
                    return None

        # 用户未在说话：检查是否发生静默超时 (事件 B)
        idle_duration = current_time - self.last_turn_interaction_time
        if idle_duration >= self.idle_timeout_seconds:
            # 事件 B 触发: 长时间静默超时 (Prolonged Silence)
            self.last_turn_interaction_time = current_time  # 重置避免连续高频触发
            return TurnEvent(
                event_type=TurnEventType.PROLONGED_SILENCE,
                timestamp=current_time,
                idle_duration=idle_duration,
            )

        return None


class InteractiveSessionRunner:
    """Manages the full end-to-end multi-turn interaction cycle and journey accumulation."""

    def __init__(
        self,
        visual_detector: VisualEmotionDetector | None = None,
        audio_detector: AudioEmotionDetector | None = None,
        policy: UncertaintyAwarePolicy | None = None,
        default_need: UserNeed = UserNeed.UNKNOWN,
    ) -> None:
        self.visual_detector = visual_detector or VisualEmotionDetector()
        self.audio_detector = audio_detector or AudioEmotionDetector()
        self.policy = policy or UncertaintyAwarePolicy()
        self.default_need = default_need

        self.visual_buffer = VisualWindowBuffer(self.visual_detector)
        self.vad_tracker = RealtimeVADTracker()

        self.turn_index: int = 0
        self.history: HistoryRecord = HistoryRecord()
        self.journey_records: list[dict] = []
        # 跨轮追踪：用于真实计算 temporal_instability 与动态效价变化反馈 HistoryEffect
        self._previous_distribution: dict[AffectLabel, float] | None = None
        self._previous_valence: float | None = None
        self._previous_decision = None
        self._previous_need: UserNeed = default_need

    def _infer_observed_effect(self, current_valence: float, data_coverage: float) -> HistoryEffect:
        """Estimate affective feedback effect of the previous turn from observed valence shift."""
        if self._previous_valence is None or data_coverage <= 0.0:
            return HistoryEffect.UNKNOWN
        delta = current_valence - self._previous_valence
        if delta >= 0.15:
            return HistoryEffect.IMPROVED
        if delta <= -0.15:
            return HistoryEffect.WORSENED
        return HistoryEffect.UNCHANGED

    def handle_turn_event(
        self,
        event: TurnEvent,
        manual_need: UserNeed | None = None,
    ) -> dict | None:
        """Execute one complete Turn: Fusion -> Feedback Update -> Policy -> History -> Turn Card."""
        now = event.timestamp
        if manual_need is not None:
            current_need = manual_need
        elif self.history.preferred_need is not None:
            current_need = self.history.preferred_need
        else:
            current_need = self.default_need

        # 1. 提取视觉感知
        if event.event_type == TurnEventType.SPEECH_FINISHED:
            start_t = event.speech_start or (now - 3.0)
            end_t = event.speech_end or now
            vis_rec = self.visual_buffer.get_aggregated_record(start_t, end_t)
            user_initiated = True
        else:
            # 静默超时：取最近 3 秒面部表情均值
            vis_rec = self.visual_buffer.get_recent_aggregated_record(duration_seconds=3.0)
            user_initiated = False

        # 2. 提取音频与文本感知
        if event.event_type == TurnEventType.SPEECH_FINISHED and event.audio_wav_path:
            audio_rec, text_rec = self.audio_detector.analyze_audio(event.audio_wav_path)
            # 清理临时文件
            try:
                if os.path.exists(event.audio_wav_path):
                    os.remove(event.audio_wav_path)
            except Exception:
                pass

            # 过滤非言语环境杂音：对于 SPEECH_FINISHED 事件，如果未识别出任何有效文字，直接忽略，绝不生成“[沉默无言语]”的伪说话轮次
            if not text_rec.available:
                return None
        else:
            # 静默无音频输入
            empty_corr = f"silence_turn_{int(now * 1000)}"
            audio_rec = PerceptionRecord(
                modality=PerceptionModality.AUDIO,
                source=PerceptionSource.MICROPHONE,
                model_name="emotion2vec_plus_base",
                model_version="plus_base",
                timestamp=now,
                raw_scores={},
                mapped_scores={},
                signal_quality=0.0,
                available=False,
                missing_reason="PROLONGED_SILENCE",
                correlation_id=empty_corr,
            )
            text_rec = PerceptionRecord(
                modality=PerceptionModality.TEXT,
                source=PerceptionSource.ASR,
                model_name="SenseVoiceSmall+zh_lexicon_v1",
                model_version="small",
                timestamp=now,
                raw_scores={},
                mapped_scores={},
                signal_quality=0.0,
                available=False,
                missing_reason="PROLONGED_SILENCE",
                transcript="",
                correlation_id=empty_corr,
            )

        self.turn_index += 1

        # 3. 完整三模态后期融合与多源不确定性画像
        # 关键修复：传入全部 3 个通道 [vis, audio, text]（含缺失通道），保证 data_coverage 真实反映模态覆盖率；
        # 同时传入 _previous_distribution 激活跨轮时序波动 temporal_instability
        all_records = [vis_rec, audio_rec, text_rec]
        fusion_result = safe_fuse_records(
            all_records,
            previous_distribution=self._previous_distribution,
        )
        aff = fusion_result.affect
        unc = fusion_result.uncertainty

        # 4. 在本轮策略决策前，用本轮观测效价相对上一轮的变化，动态回填上一轮介入的真实效果 (HistoryEffect)
        observed_effect = self._infer_observed_effect(aff.valence, unc.data_coverage)
        if self._previous_decision is not None:
            self.history = update_history(
                history=self.history,
                decision=self._previous_decision,
                need=self._previous_need,
                effect=observed_effect,
                accepted=True,
            )

        # 5. Uncertainty-Aware Policy 决策 (按 ESConv 交互进程：前 2 轮为探索阶段，第 3 轮起进入支持阶段)
        stage = InteractionStage.EXPLORING if self.turn_index <= 2 else InteractionStage.SUPPORTING
        context = PolicyContext(
            interaction_stage=stage,
            user_initiated=user_initiated,
            user_need=current_need,
            history=self.history,
        )
        decision = self.policy.decide(aff, unc, context)

        # 保存本轮状态供下一轮计算时序波动与反馈效果
        if unc.data_coverage > 0.0:
            self._previous_distribution = dict(aff.distribution)
            self._previous_valence = aff.valence
        self._previous_decision = decision
        self._previous_need = current_need

        # 仅刷新静默倒计时起点，绝不清空音频线程正在录制的说话缓冲 (修复跨线程吞字竞态)
        self.vad_tracker.mark_turn_completed(now)

        # 6. 构造结构化轮次记录 (沉淀为用户旅程数据源)
        top_vis_label = max(vis_rec.mapped_scores.items(), key=lambda x: x[1])[0].value if vis_rec.available and vis_rec.mapped_scores else "N/A"
        top_aud_label = max(audio_rec.mapped_scores.items(), key=lambda x: x[1])[0].value if audio_rec.available and audio_rec.mapped_scores else "N/A"
        top_txt_label = max(text_rec.mapped_scores.items(), key=lambda x: x[1])[0].value if text_rec.available and text_rec.mapped_scores else "N/A"

        top_fused_label = "UNKNOWN"
        if aff.distribution:
            top_key = max(aff.distribution.items(), key=lambda x: x[1])[0]
            top_fused_label = top_key.value if hasattr(top_key, "value") else str(top_key)

        turn_data = {
            "turn_index": self.turn_index,
            "timestamp": now,
            "event_type": event.event_type.value,
            "user_initiated": user_initiated,
            "user_need": current_need.value,
            "transcript": text_rec.transcript if text_rec.available else "[沉默无言语]",
            "face_dominant_emotion": top_vis_label,
            "face_signal_quality": vis_rec.signal_quality,
            "audio_dominant_emotion": top_aud_label,
            "text_dominant_emotion": top_txt_label,
            "data_coverage": unc.data_coverage,
            "temporal_instability": unc.temporal_instability,
            "cross_modal_disagreement": unc.cross_modal_disagreement,
            "predictive_entropy": unc.predictive_entropy,
            "top_two_margin": unc.top_two_margin,
            "overall_signal_quality": unc.signal_quality,
            "fused_dominant_emotion": top_fused_label,
            "fused_valence": aff.valence,
            "fused_arousal": aff.arousal,
            "observed_feedback_effect": observed_effect.value,
            "policy_mode": decision.mode.value,
            "policy_action": decision.support_action.value,
            "policy_intensity": decision.intensity.value,
            "epistemic_style": decision.epistemic_style.value,
            "policy_reasons": list(decision.reasons),
        }
        self.journey_records.append(turn_data)

        # 打印当前轮次策略文字卡
        self._print_turn_card(turn_data)
        return turn_data

    @staticmethod
    def _print_turn_card(data: dict) -> None:
        idx = data["turn_index"]
        ev = "🗣️ 用户主动发言结束" if data["user_initiated"] else "⏳ 长时间静默留白"
        disagree_str = f"{data['cross_modal_disagreement']:.4f}" if data['cross_modal_disagreement'] is not None else "0.0000 (单模态)"

        print("\n" + "=" * 62)
        print(f" 轮次 [ {idx:02d} ] | 触发事件: {ev}")
        print("=" * 62)
        print("【1. 多模态实时感知】")
        print(f"  • 语音转写 (ASR): \"{data['transcript']}\"")
        print(f"  • 面部表情均值:    {data['face_dominant_emotion'].upper()} (人脸置信度: {data['face_signal_quality']:.1%})")
        print(f"  • 声学情绪特征:    {data['audio_dominant_emotion'].upper()}")
        print(f"  • 文本语义情绪:    {data.get('text_dominant_emotion', 'N/A').upper()}")
        print(f"  • 模态覆盖率:      {data.get('data_coverage', 0.0):.1%} | 时序波动: {data.get('temporal_instability', 0.0):.4f}")
        print(f"  • 跨模态冲突度:    {disagree_str} | 预测熵: {data['predictive_entropy']:.4f}")

        print("\n【2. Policy 自适应策略决策】")
        print(f"  • 决策干预模式:   【 {data['policy_mode'].upper()} 】")
        print(f"  • 推荐支持动作:    {data['policy_action']}")
        print(f"  • 干预执行强度:    {data['policy_intensity'].upper()}")
        print(f"  • 语气认知风格:    {data['epistemic_style']}")

        print("\n【3. 策略依据 (Decision Rationale)】")
        for reason in data["policy_reasons"]:
            print(f"  • {reason}")
        print("=" * 62 + "\n")
