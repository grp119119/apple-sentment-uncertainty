"""Local audio perception pipeline: SenseVoiceSmall (ASR) + emotion2vec+ (SER)."""

from __future__ import annotations

import os
import re
import subprocess
import time
import wave
from pathlib import Path
from typing import Mapping, Sequence

# 解决在沙箱环境下 funasr 检查 ffmpeg 时触发 PermissionError 的兼容问题
_orig_check_output = subprocess.check_output
def _safe_check_output(*args, **kwargs):
    try:
        return _orig_check_output(*args, **kwargs)
    except PermissionError:
        raise FileNotFoundError("Sandboxed ffmpeg inaccessible")
subprocess.check_output = _safe_check_output

import numpy as np

from affective_v03.models import AffectLabel
from .adapter import EMOTION2VEC_9_MAP, map_raw_scores_to_ekman
from .contracts import (
    PerceptionModality,
    PerceptionRecord,
    PerceptionSource,
    softmax,
)


def clean_sensevoice_output(raw_text: str) -> str:
    """Sanitize SenseVoice output: strip rich transcription tags, emojis, music icons and dangling punctuation."""
    if not raw_text:
        return ""
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    cleaned = rich_transcription_postprocess(raw_text)
    # 移除音乐与特殊表情符号: 🎼, 🎵, emoji
    cleaned = re.sub(r"[🎼🎵🎶\u2600-\u27bf\U0001f300-\U0001f9ff]", "", cleaned)
    cleaned = cleaned.strip()
    # 检查去除标点与空白后是否还存在真正的语义文本（中文汉字或英文单词）
    has_linguistic_content = bool(re.search(r"[\u4e00-\u9fff\w]", cleaned))
    if not has_linguistic_content:
        return ""
    return cleaned


class AudioEmotionDetector:
    """Offline / local dual-stream audio processor (Speech-to-Text + Acoustic Emotion)."""

    def __init__(
        self,
        asr_model_name: str = "iic/SenseVoiceSmall",
        ser_model_name: str = "iic/emotion2vec_plus_base",
        device: str = "cpu",
    ) -> None:
        self.asr_model_name = asr_model_name
        self.ser_model_name = ser_model_name
        self.device = device
        self._asr_model = None
        self._ser_model = None

    @staticmethod
    def _resolve_model_path(model_id_or_path: str) -> str:
        """Resolve model path locally first to ensure completely offline operation."""
        # 1. Direct path check
        p = Path(model_id_or_path)
        if p.exists():
            return str(p.resolve())

        # 2. Check models/audio/ under workspace
        workspace_audio_dir = Path("models/audio")
        clean_name = model_id_or_path.replace("/", "--").replace("iic--", "")
        candidates = [
            workspace_audio_dir / clean_name,
            workspace_audio_dir / model_id_or_path.split("/")[-1],
        ]
        for c in candidates:
            if c.exists():
                return str(c.resolve())

        # 3. Check ~/.cache/modelscope/models/
        ms_name = model_id_or_path.replace("/", "--")
        cache_snapshot = Path.home() / ".cache" / "modelscope" / "models" / ms_name / "snapshots" / "master"
        if cache_snapshot.exists():
            return str(cache_snapshot.resolve())

        return model_id_or_path

    def _ensure_asr_model(self):
        if self._asr_model is None:
            from funasr import AutoModel
            resolved = self._resolve_model_path(self.asr_model_name)
            self._asr_model = AutoModel(
                model=resolved,
                device=self.device,
                disable_update=True,
            )
        return self._asr_model

    def _ensure_ser_model(self):
        if self._ser_model is None:
            from funasr import AutoModel
            resolved = self._resolve_model_path(self.ser_model_name)
            self._ser_model = AutoModel(
                model=resolved,
                device=self.device,
                disable_update=True,
            )
        return self._ser_model

    def warmup(self) -> None:
        """Pre-load ASR and SER models into memory to avoid cold-start latency on first turn."""
        self._ensure_asr_model()
        self._ensure_ser_model()

    @staticmethod
    def _inspect_audio(audio_path: str) -> tuple[float, float, bool]:
        """Inspect duration (seconds), RMS energy, and whether silence."""
        try:
            import soundfile as sf
            data, sample_rate = sf.read(audio_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            duration = float(len(data)) / float(sample_rate) if sample_rate > 0 else 0.0
            rms = float(np.sqrt(np.mean(np.square(data)))) if data.size > 0 else 0.0
            is_silent = rms < 1e-4 or duration < 0.2
            return duration, rms, is_silent
        except Exception:
            try:
                with wave.open(audio_path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = float(frames) / float(rate) if rate > 0 else 0.0
                    raw_bytes = wf.readframes(frames)
                    pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size > 0 else 0.0
                    return duration, rms, (rms < 1e-4 or duration < 0.2)
            except Exception:
                return 0.0, 0.0, True

    @staticmethod
    def _prepare_normalized_wav(audio_path: str) -> tuple[str, bool]:
        """Ensure a gain-normalized 16kHz mono WAV for optimal ASR/SER accuracy on quiet mic inputs."""
        if not audio_path.lower().endswith(".wav"):
            return audio_path, False
        try:
            with wave.open(audio_path, "rb") as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    return audio_path, False
                rate = wf.getframerate()
                frames = wf.getnframes()
                pcm = np.frombuffer(wf.readframes(frames), dtype=np.int16).astype(np.float32) / 32768.0
            if pcm.size == 0:
                return audio_path, False
            pcm = pcm - float(np.mean(pcm))
            peak = float(np.max(np.abs(pcm)))
            if peak <= 0.02 or peak >= 0.65:
                return audio_path, False
            gain = min(4.0, 0.80 / peak)
            norm_pcm = np.clip(pcm * gain * 32767.0, -32768, 32767).astype(np.int16)
            import tempfile
            tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            norm_path = tf.name
            tf.close()
            with wave.open(norm_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                wf.writeframes(norm_pcm.tobytes())
            return norm_path, True
        except Exception:
            return audio_path, False

    def analyze_audio(
        self,
        audio_path: str | Path,
        source: PerceptionSource = PerceptionSource.MICROPHONE,
        language: str = "zh",
    ) -> tuple[PerceptionRecord, PerceptionRecord]:
        """Process audio file and produce correlated (Audio SER Record, Text ASR Record)."""
        start_time = time.perf_counter()
        now = time.time()
        correlation_id = f"audio_session_{int(now * 1000)}"
        path_str = str(audio_path)

        # 针对 macOS 录音常见格式 (.m4a, .aac 等) 自动无缝转为 16kHz 单声道 wav
        if os.path.exists(path_str) and not path_str.lower().endswith(".wav"):
            wav_candidate = Path(path_str).with_suffix(".wav")
            if not wav_candidate.exists():
                try:
                    _orig_check_output(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", path_str, str(wav_candidate)])
                except Exception:
                    pass
            if wav_candidate.exists():
                path_str = str(wav_candidate)

        if not os.path.exists(path_str):
            empty_audio = PerceptionRecord(
                modality=PerceptionModality.AUDIO,
                source=source,
                model_name=self.ser_model_name,
                model_version="1.0",
                timestamp=now,
                raw_scores={},
                mapped_scores={},
                signal_quality=0.0,
                available=False,
                missing_reason=f"FILE_NOT_FOUND: {path_str}",
                correlation_id=correlation_id,
            )
            empty_text = PerceptionRecord(
                modality=PerceptionModality.TEXT,
                source=PerceptionSource.ASR,
                model_name=self.asr_model_name,
                model_version="1.0",
                timestamp=now,
                raw_scores={},
                mapped_scores={},
                signal_quality=0.0,
                available=False,
                missing_reason=f"FILE_NOT_FOUND: {path_str}",
                correlation_id=correlation_id,
            )
            return empty_audio, empty_text

        duration, rms, is_silent = self._inspect_audio(path_str)

        if is_silent:
            latency = (time.perf_counter() - start_time) * 1000
            silent_audio = PerceptionRecord(
                modality=PerceptionModality.AUDIO,
                source=source,
                model_name=self.ser_model_name,
                model_version="1.0",
                timestamp=now,
                raw_scores={},
                mapped_scores={},
                signal_quality=0.0,
                available=False,
                missing_reason="EMPTY_AUDIO_SILENCE",
                correlation_id=correlation_id,
                latency_ms=latency,
            )
            silent_text = PerceptionRecord(
                modality=PerceptionModality.TEXT,
                source=PerceptionSource.ASR,
                model_name=self.asr_model_name,
                model_version="1.0",
                timestamp=now,
                raw_scores={},
                mapped_scores={},
                signal_quality=0.0,
                available=False,
                missing_reason="EMPTY_AUDIO_SILENCE",
                transcript="",
                correlation_id=correlation_id,
                latency_ms=latency,
            )
            return silent_audio, silent_text

        # 0. 自动增益归一化 (针对轻声或远场录音做峰值均衡，提升 SenseVoice 与 emotion2vec 识别准度)
        infer_path, is_temp_norm = self._prepare_normalized_wav(path_str)
        try:
            # 1. SenseVoiceSmall ASR 语音转写
            asr = self._ensure_asr_model()
            asr_res = asr.generate(input=infer_path, language=language, use_itn=True)
            raw_text = asr_res[0].get("text", "") if asr_res else ""
            clean_text = clean_sensevoice_output(raw_text)

            # 2. emotion2vec+ SER 声学情感分类
            ser = self._ensure_ser_model()
            ser_res = ser.generate(input=infer_path, granularity="utterance", extract_embedding=False)
        finally:
            if is_temp_norm and os.path.exists(infer_path):
                try:
                    os.remove(infer_path)
                except Exception:
                    pass

        raw_ser_scores: dict[str, float] = {}
        if ser_res and isinstance(ser_res[0], dict):
            # 提取 scores 与 labels
            scores_val = ser_res[0].get("scores", [])
            labels_val = ser_res[0].get("labels", [])
            if labels_val and scores_val:
                raw_ser_scores = {str(k): float(v) for k, v in zip(labels_val, scores_val)}

        # 映射至 Ekman 7 类
        mapped_ser_scores = map_raw_scores_to_ekman(raw_ser_scores, EMOTION2VEC_9_MAP)
        latency = (time.perf_counter() - start_time) * 1000

        audio_record = PerceptionRecord(
            modality=PerceptionModality.AUDIO,
            source=source,
            model_name=self.ser_model_name,
            model_version="plus_base",
            timestamp=now,
            raw_scores=raw_ser_scores,
            mapped_scores=mapped_ser_scores,
            signal_quality=min(1.0, max(0.5, rms * 10.0)),
            available=True,
            correlation_id=correlation_id,
            latency_ms=latency,
        )

        # 简单文本先验 (在后续阶段接入独立文本情感模型之前，提供安全基准)
        has_text = len(clean_text) > 0
        text_record = PerceptionRecord(
            modality=PerceptionModality.TEXT,
            source=PerceptionSource.ASR,
            model_name=self.asr_model_name,
            model_version="small",
            timestamp=now,
            raw_scores={"transcript_len": float(len(clean_text)), "raw_text": raw_text},
            mapped_scores={label: 1.0 / len(AffectLabel) for label in AffectLabel},
            signal_quality=0.85 if has_text else 0.0,
            available=has_text,
            missing_reason=None if has_text else "NO_VALID_SPEECH_TEXT",
            transcript=clean_text,
            correlation_id=correlation_id,
            latency_ms=latency,
        )

        return audio_record, text_record

