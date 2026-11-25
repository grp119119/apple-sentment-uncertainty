"""Interactive Audio & Microphone Diagnostic Tool.

Tests microphone input, displays live VU-meter, records a short voice sample,
and verifies SenseVoiceSmall ASR and emotion2vec+ SER locally.

Usage:
  python3.12 debug_audio.py
  (or on Windows: python debug_audio.py)
"""

import os
import sys
import time
import wave
import tempfile
import numpy as np

def main():
    print("=" * 64)
    print(" 🎙️ 麦克风与语音识别本地专项排查诊断工具")
    print("=" * 64)

    # 1. 检测 sounddevice
    try:
        import sounddevice as sd
    except ImportError:
        print("❌ 未安装 sounddevice 库。请执行: pip install sounddevice")
        return 1

    devices = sd.query_devices()
    default_input = sd.default.device[0]
    print(f"\n【步骤 1: 扫描系统音频输入设备】")
    for i, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            is_def = " (当前默认)" if i == default_input else ""
            print(f"  [{i}] {dev.get('name')}{is_def} | 声道: {dev.get('max_input_channels')}")

    # 自动探测优先麦克风
    chosen_device = default_input
    if chosen_device < 0:
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                chosen_device = i
                break

    print(f"\n👉 本次测试绑定输入设备: [{chosen_device}] {devices[chosen_device].get('name')}")

    # 2. 实时声压电平动态侦测 (5 秒)
    print("\n【步骤 2: 实时声压动态侦测 (请保持安静 2 秒，随后正常说话 3 秒)】")
    print("提示: 观察下方的动态能量条是否随你发音产生明显波动：")

    rms_values = []
    def monitor_callback(indata, frames, time_info, status):
        chunk = indata[:, 0]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size > 0 else 0.0
        rms_values.append(rms)
        bars = "█" * min(30, int(rms * 300))
        sys.stdout.write(f"\r  当前实时音量: [{bars:<30}] RMS: {rms:.4f}")
        sys.stdout.flush()

    try:
        stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="float32",
            blocksize=1024,
            device=chosen_device,
            callback=monitor_callback,
        )
        stream.start()
        time.sleep(5.0)
        stream.stop()
        stream.close()
    except Exception as e:
        print(f"\n❌ 启动麦克风监听流失败: {e}")
        print("💡 建议: 请检查系统设置 -> 隐私与安全性 -> 麦克风权限。")
        return 1

    sys.stdout.write("\n")
    if not rms_values:
        print("⚠️ 未收到任何音频数据。")
        return 1

    ambient_rms = float(np.percentile(rms_values[:15], 50))
    peak_rms = float(np.max(rms_values))
    print(f"\n  • 环境底噪能量 (Ambient RMS): {ambient_rms:.5f}")
    print(f"  • 最大说话能量 (Peak RMS):    {peak_rms:.5f}")

    if peak_rms < 0.015:
        print("\n⚠️ [警告] 能量过低！说话时的最大 RMS 低于 0.015。可能原因：")
        print("  1. 系统输入音量被调得过低 (请在系统偏好设置中将麦克风输入音量拉满)");
        print("  2. 选错了麦克风设备 (例如连到了不工作的虚拟麦克风)");
        print("  3. 距离麦克风过远或声音过小。")
    else:
        rec_thresh = max(0.018, ambient_rms * 2.0)
        print(f"  • ✅ 能量正常！系统为您计算的最佳 VAD 门限推荐值: {rec_thresh:.4f}")

    # 3. 录制 4 秒语音并直接测试 SenseVoice 转写
    print("\n【步骤 3: 录制 4 秒音频测试 SenseVoice 转写】")
    input("👉 请按下【回车键 (Enter)】开始录音，倒计时 4 秒内请清晰说一句中文（如：'你好，我最近有点焦虑'）...")

    print("🔴 正在录音中... [4秒]")
    rec_data = sd.rec(int(16000 * 4.0), samplerate=16000, channels=1, dtype="float32", device=chosen_device)
    sd.wait()
    print("⏹️ 录音完成！正在本地调用 SenseVoiceSmall 与 emotion2vec+ 进行推理...\n")

    # 存为 WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name

    int16_samples = np.clip(rec_data[:, 0] * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(int16_samples.tobytes())

    # 调用本地模型推理
    try:
        from affective.perception.audio_detector import AudioEmotionDetector
        det = AudioEmotionDetector()
        aud_rec, txt_rec = det.analyze_audio(wav_path, language="zh")

        print("=" * 64)
        print(" 🎯 本地模型推理结果：")
        print("=" * 64)
        print(f"  • 语音转写字幕 (ASR): \"{txt_rec.transcript}\"")
        print(f"  • 转写状态 (Available): {txt_rec.available}")
        if not txt_rec.available:
            print(f"    未转写原因: {txt_rec.missing_reason}")
        print(f"  • 声学情绪判定 (SER): {max(aud_rec.mapped_scores.items(), key=lambda x: x[1])[0].value if aud_rec.available else 'N/A'}")
        print(f"  • 声学信号质量: {aud_rec.signal_quality:.1%}")
        print("=" * 64)
    except Exception as e:
        print(f"❌ 调用语音模型失败: {e}")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    print("\n💡 诊断完成。")
    return 0

if __name__ == "__main__":
    sys.exit(main())

