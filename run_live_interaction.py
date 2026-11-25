"""Live Multimodal Interactive Installation Runner.

Listens to camera & microphone streams in real time.
Executes dual-event turn segmentation (Speech End / Prolonged Silence).
Runs Uncertainty-Aware Policy on each turn, prints formatted text cards,
and saves the complete user journey trajectory to JSON on exit.

Usage:
  # 真实摄像头与麦克风交互运行:
  python3.12 run_live_interaction.py

  # 指定能量阈值与静默超时:
  python3.12 run_live_interaction.py --energy-thresh 0.02 --silence-timeout 8.0 --need COMFORT
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from affective.models import UserNeed
from affective.perception.stream_coordinator import (
    InteractiveSessionRunner,
    TurnEvent,
    TurnEventType,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Multimodal Interactive Installation Runner")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate (default: 16000)")
    parser.add_argument("--energy-thresh", type=float, default=0.015, help="VAD energy threshold (default: 0.015)")
    parser.add_argument("--silence-cutoff", type=float, default=0.75, help="Speech end silence cutoff (seconds)")
    parser.add_argument("--silence-timeout", type=float, default=8.0, help="Prolonged silence timeout (seconds)")
    parser.add_argument("--need", type=str, default="COMFORT", choices=[n.name for n in UserNeed], help="Default user need")
    args = parser.parse_args()

    user_need = UserNeed[args.need]
    event_queue: queue.Queue[TurnEvent] = queue.Queue()
    stop_event = threading.Event()

    print("=" * 66)
    print(" 🌿 空间人因多模态交互装置 (Live Interactive Installation)")
    print("=" * 66)
    print("  • 轮次规则: 用户每次说完一句话触发分析；超过设定时间未说话触发静默分析")
    print(f"  • VAD 句尾截断: {args.silence_cutoff} 秒 | 静默留白超时: {args.silence_timeout} 秒")
    print(f"  • 预设核心需求: {user_need.value}")
    print("  • 退出交互: 按 Ctrl+C 结束会话并自动导出本次用户旅程数据")
    print("=" * 66)

    # 1. 初始化会话调度器
    print("\n⏳ 正在初始化本地感知与策略引擎...")
    runner = InteractiveSessionRunner(default_need=user_need)
    runner.vad_tracker.energy_threshold = args.energy_thresh
    runner.vad_tracker.silence_cutoff_seconds = args.silence_cutoff
    runner.vad_tracker.idle_timeout_seconds = args.silence_timeout
    runner.vad_tracker.sample_rate = args.sample_rate
    print("✅ 本地引擎加载完成，准备启动实时硬件流...")

    # 2. 启动摄像头后台线程
    def camera_worker() -> None:
        cap = None
        cam_idx = args.camera_id
        # 优先探测指定序号及 macOS FaceTime HD (Index 1)
        for try_idx in [cam_idx, 1, 0, 2]:
            c = cv2.VideoCapture(try_idx)
            if c.isOpened():
                r, f = c.read()
                if r and f is not None and f.size > 0:
                    cap = c
                    cam_idx = try_idx
                    break
                c.release()
        if cap is None or not cap.isOpened():
            print(f"⚠️ [警告] 无法打开可用物理摄像头。视觉模态将保持脱机状态。")
            return

        print(f"📷 摄像头已就绪 (Device {cam_idx})，正在后台捕获人脸表情...")
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            try:
                rec = runner.visual_detector.analyze_image(frame)
                runner.visual_buffer.add_record(rec)
            except Exception:
                pass
            time.sleep(0.04)  # 限制在 ~20 FPS 降低发热

        cap.release()

    cam_thread = threading.Thread(target=camera_worker, daemon=True)
    cam_thread.start()

    # 3. 启动麦克风监听流
    try:
        import sounddevice as sd
    except ImportError:
        print("❌ 缺少 sounddevice 库，无法开启实时麦克风。")
        return 1

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if stop_event.is_set():
            return
        # indata 是 PortAudio 复用的底层 C 缓冲区，必须 .copy() 深拷贝防止历史语音帧被后续静音覆写
        chunk = indata[:, 0].copy()
        event = runner.vad_tracker.process_chunk(chunk)
        if event is not None:
            event_queue.put(event)

    try:
        mic_device = None
        try:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0 and "MacBook" in dev.get("name", ""):
                    mic_device = idx
                    break
        except Exception:
            pass

        audio_stream = sd.InputStream(
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=1024,
            device=mic_device,
            callback=audio_callback,
        )
        audio_stream.start()
        print("🎙️ 麦克风已开启，开始实时倾听...\n")
    except Exception as e:
        print(f"⚠️ [警告] 无法打开麦克风录音设备: {e}")
        print("💡 提示: 请在 macOS 系统设置 -> 隐私与安全性中允许终端/Python 访问麦克风。")
        audio_stream = None

    # 4. 主事件循环
    runner.vad_tracker.reset_turn_timer()
    try:
        while True:
            try:
                # 阻塞等待下一个轮次事件
                event = event_queue.get(timeout=0.5)
                runner.handle_turn_event(event)
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("\n🛑 正在结束交互会话...")
    finally:
        stop_event.set()
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()

    # 5. 沉淀导出本次完整的“用户旅程数据 (User Journey Dataset)”
    os.makedirs("data/journeys", exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    journey_file = Path(f"data/journeys/user_journey_{session_id}.json")

    journey_payload = {
        "session_id": session_id,
        "total_turns": runner.turn_index,
        "default_need": user_need.value,
        "turns": runner.journey_records,
    }

    with open(journey_file, "w", encoding="utf-8") as f:
        json.dump(journey_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 66)
    print(f" 🎨 本次会话已圆满沉淀！")
    print(f"  • 总交互轮次: {runner.turn_index} 轮")
    print(f"  • 用户旅程数据已导出至: {journey_file}")
    print(f"  • 说明: 该 JSON 文件即为你后续‘视觉策略图库组装与画作打印’的标准数据源！")
    print("=" * 66 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

