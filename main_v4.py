"""Interactive Multimodal Affective Panel · v4.2.

Features:
  1. Live Camera Feed with real-time face bounding box & emotion overlay.
  2. Live Microphone VAD status and real-time ASR speech-to-text transcript.
  3. Real-time Uncertainty-Aware Policy Decision Cards (Mode, Action, Intensity, Epistemic Style).
  4. Decision Rationale Box explaining why the system chose this adaptation.
  5. Scrollable Multi-turn Historical Strategy Log.
  6. Minimalist control bar with Start/Stop toggle and UserNeed selector.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from affective.models import AffectLabel, UserNeed
from affective.labels_zh import (
    ACTION_LABELS,
    EMOTION_LABELS,
    INTENSITY_LABELS,
    MODE_LABELS,
    NEED_LABELS,
    translate_reason,
)
from affective.perception.stream_coordinator import (
    InteractiveSessionRunner,
    TurnEvent,
    TurnEventType,
)

# -------------------------------------------------------------
# 视觉设计常数 (Modern Minimal Apple-like Palette)
# -------------------------------------------------------------
BG_COLOR = "#F4F7FB"
SURFACE_COLOR = "#FFFFFF"
TEXT_COLOR = "#182230"
MUTED_COLOR = "#667085"
BORDER_COLOR = "#E2E8F0"
PRIMARY_COLOR = "#315EFB"
PRIMARY_HOVER = "#2449D8"

MODE_THEMES = {
    "DIRECT": {"bg": "#E6F4EA", "fg": "#137333", "border": "#34A853", "zh": "直接介入 (DIRECT)"},
    "TENTATIVE": {"bg": "#F3E8FF", "fg": "#6B21A8", "border": "#A855F7", "zh": "谨慎试探 (TENTATIVE)"},
    "CONFIRM": {"bg": "#E8F0FE", "fg": "#1A73E8", "border": "#4285F4", "zh": "澄清确认 (CONFIRM)"},
    "WAIT": {"bg": "#FEF3C7", "fg": "#92400E", "border": "#F59E0B", "zh": "静默观察 (WAIT)"},
    "DEFER": {"bg": "#FCE8E6", "fg": "#C5221F", "border": "#EA4335", "zh": "暂缓介入 (DEFER)"},
    "IDLE": {"bg": "#F1F5F9", "fg": "#475569", "border": "#94A3B8", "zh": "等待交互启动"},
}


UI_FONT = "PingFang SC" if sys.platform == "darwin" else "Microsoft YaHei"


def find_working_camera() -> tuple[cv2.VideoCapture, int]:
    """Auto-detect working physical camera across macOS, Windows and Linux."""
    cam_order = [1, 0, 2] if sys.platform == "darwin" else [0, 1, 2]
    for idx in cam_order:
        try:
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    return cap, idx
                cap.release()
        except Exception:
            pass
    return cv2.VideoCapture(0), 0


class StrategyAppV4:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("空间人因多模态情感计算策略面板 · v4.2")
        self.root.geometry("1200x720")
        self.root.minsize(1024, 640)
        self.root.configure(bg=BG_COLOR)

        # 核心引擎与队列
        self.runner = InteractiveSessionRunner(default_need=UserNeed.COMFORT)
        self.event_queue: queue.Queue[TurnEvent] = queue.Queue()
        self.turn_result_queue: queue.Queue[dict] = queue.Queue()
        self.stop_event = threading.Event()

        # 硬件与线程句柄
        self.is_running = False
        self.camera_cap: cv2.VideoCapture | None = None
        self.audio_stream = None
        self.mic_device: int | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_face_box: tuple[int, int, int, int] | None = None
        self.latest_face_emotion: str = "—"
        self.latest_face_conf: float = 0.0
        self.current_audio_rms: float = 0.0
        self._manual_recording_active: bool = False
        self._manual_capture_chunks: list[np.ndarray] = []

        # UI 状态绑定变量
        self.current_need_var = tk.StringVar(value="COMFORT")
        self.vad_status_var = tk.StringVar(value="● 装置待机中 (点击下方开启互动)")
        self.transcript_var = tk.StringVar(value="等待用户开口说话...")
        self.fused_summary_var = tk.StringVar(value="面部均值: — ｜ 声音情绪: — ｜ 冲突度: — ｜ 信息熵: —")

        self.mode_theme_key = "IDLE"
        self.mode_title_var = tk.StringVar(value=MODE_THEMES["IDLE"]["zh"])
        self.action_var = tk.StringVar(value="尚未生成动作")
        self.intensity_style_var = tk.StringVar(value="强度: — ｜ 语气: —")
        self.rationale_var = tk.StringVar(value="启动互动后，系统将在此解析自适应策略的判定逻辑...")
        self.status_bar_var = tk.StringVar(value="就绪 ｜ 点击【▶ 开启实时互动】开始体验")

        self._build_styles()
        self._build_layout()
        self._start_ui_loop()

    def _build_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", padding=5, font=(UI_FONT, 10))

    def _build_layout(self) -> None:
        # 1. 顶栏标题与核心操作工具条 (置顶且绝对不被遮挡)
        header = tk.Frame(self.root, bg=SURFACE_COLOR, padx=20, pady=10, relief="solid", bd=1)
        header.pack(fill="x", side="top")

        title_box = tk.Frame(header, bg=SURFACE_COLOR)
        title_box.pack(side="left")
        tk.Label(
            title_box,
            text="空间人因多模态情感交互装置 · 策略大脑",
            font=(UI_FONT, 16, "bold"),
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="v4.2 实时流式感知 ｜ 自动轮次划分 ｜ Uncertainty-Aware Policy",
            font=(UI_FONT, 9),
            bg=SURFACE_COLOR,
            fg=MUTED_COLOR,
        ).pack(anchor="w")

        # 核心主控栏 (移至顶部右上角，最显眼且永不遮挡)
        control_box = tk.Frame(header, bg=SURFACE_COLOR)
        control_box.pack(side="right")

        tk.Label(control_box, text="预设需求:", font=(UI_FONT, 9, "bold"), bg=SURFACE_COLOR, fg=MUTED_COLOR).pack(side="left", padx=(0, 4))
        need_options = [f"{NEED_LABELS.get(n.name, n.name)} ({n.name})" for n in UserNeed if n != UserNeed.UNKNOWN]
        self.need_combo = ttk.Combobox(control_box, values=need_options, state="readonly", width=15)
        self.need_combo.set("安慰 (COMFORT)")
        self.need_combo.pack(side="left", padx=(0, 14))
        self.need_combo.bind("<<ComboboxSelected>>", self._on_need_changed)

        self.toggle_btn = tk.Button(
            control_box,
            text=" ▶ 开启实时互动 ",
            font=(UI_FONT, 11, "bold"),
            bg=PRIMARY_COLOR,
            fg="white",
            activebackground=PRIMARY_HOVER,
            activeforeground="white",
            padx=16,
            pady=6,
            relief="flat",
            cursor="hand2",
            command=self._toggle_interaction,
        )
        self.toggle_btn.pack(side="left")

        # 2. 底部轻量状态条 (先 pack 确保固定在底部)
        self._build_bottom_controls()

        # 3. 主体左右双栏结构
        body = tk.Frame(self.root, bg=BG_COLOR, padx=16, pady=6)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=5, uniform="col")
        body.grid_columnconfigure(1, weight=6, uniform="col")
        body.grid_rowconfigure(0, weight=1)

        left_col = tk.Frame(body, bg=BG_COLOR)
        right_col = tk.Frame(body, bg=BG_COLOR)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right_col.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        # 组装左侧与右侧
        self._build_left_perception_panel(left_col)
        self._build_right_policy_panel(right_col)

    # -------------------------------------------------------------
    # 板块一：左侧 · 实时多模态感知流
    # -------------------------------------------------------------
    def _build_left_perception_panel(self, parent: tk.Widget) -> None:
        # 1. 摄像头视频卡片
        cam_card = tk.LabelFrame(
            parent,
            text=" 📷 实时摄像头视觉捕获 (YuNet + HSEmotion) ",
            font=(UI_FONT, 10, "bold"),
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
            padx=10,
            pady=8,
            relief="solid",
            bd=1,
        )
        cam_card.pack(fill="both", expand=True, pady=(0, 8))

        # 视频渲染标签
        self.video_label = tk.Label(
            cam_card,
            text="摄像头尚未开启\n点击右上角【开启实时互动】唤醒摄像头",
            font=(UI_FONT, 10),
            bg="#0F172A",
            fg="#94A3B8",
            width=50,
            height=12,
        )
        self.video_label.pack(fill="both", expand=True, padx=4, pady=4)

        # 面部表情实时浮动标签
        self.face_status_label = tk.Label(
            cam_card,
            text="人脸表情: — (置信度: —)",
            font=(UI_FONT, 10, "bold"),
            bg="#F8FAFC",
            fg=TEXT_COLOR,
            pady=4,
        )
        self.face_status_label.pack(fill="x", pady=(6, 0))

        # 2. 语音监听与实时字幕卡片
        mic_card = tk.LabelFrame(
            parent,
            text=" 🎙️ 声音监听与实时转写 (VAD + SenseVoice) ",
            font=(UI_FONT, 11, "bold"),
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
            padx=12,
            pady=10,
            relief="solid",
            bd=1,
        )
        mic_card.pack(fill="x", pady=(0, 4))

        # 状态指示条
        self.vad_indicator = tk.Label(
            mic_card,
            textvariable=self.vad_status_var,
            font=(UI_FONT, 10, "bold"),
            bg="#F1F5F9",
            fg="#475569",
            padx=8,
            pady=4,
            anchor="w",
        )
        self.vad_indicator.pack(fill="x", pady=(0, 6))

        # 实时语音转写文本框
        tk.Label(mic_card, text="最新语音转写字幕:", font=(UI_FONT, 9), bg=SURFACE_COLOR, fg=MUTED_COLOR, anchor="w").pack(fill="x")
        self.transcript_box = tk.Label(
            mic_card,
            textvariable=self.transcript_var,
            font=(UI_FONT, 11, "bold"),
            bg="#F8FAFC",
            fg="#0F172A",
            wraplength=480,
            justify="left",
            anchor="w",
            padx=8,
            pady=8,
            relief="groove",
            bd=1,
        )
        self.transcript_box.pack(fill="x", pady=(2, 6))

        # 模态融合参数快照
        self.fused_summary_label = tk.Label(
            mic_card,
            textvariable=self.fused_summary_var,
            font=(UI_FONT, 9),
            bg=SURFACE_COLOR,
            fg=MUTED_COLOR,
            anchor="w",
        )
        self.fused_summary_label.pack(fill="x", pady=(0, 6))

        # 麦克风灵敏度快捷调节与点按试音栏
        mic_tune_box = tk.Frame(mic_card, bg=SURFACE_COLOR)
        mic_tune_box.pack(fill="x", pady=(2, 2))

        self.thresh_label = tk.Label(
            mic_tune_box,
            text=f"VAD 拾音门限: {self.runner.vad_tracker.energy_threshold:.3f}",
            font=(UI_FONT, 8, "bold"),
            bg=SURFACE_COLOR,
            fg=MUTED_COLOR,
        )
        self.thresh_label.pack(side="left")

        self.thresh_slider = ttk.Scale(
            mic_tune_box,
            from_=0.010,
            to=0.045,
            value=self.runner.vad_tracker.energy_threshold,
            command=self._on_thresh_slider_changed,
        )
        self.thresh_slider.pack(side="left", fill="x", expand=True, padx=8)

        self.manual_record_btn = tk.Button(
            mic_tune_box,
            text=" 🎙️ 点按试音(3秒) ",
            font=(UI_FONT, 8, "bold"),
            bg="#E0E7FF",
            fg="#3730A3",
            relief="flat",
            cursor="hand2",
            command=self._trigger_manual_speech_test,
        )
        self.manual_record_btn.pack(side="right")

    # -------------------------------------------------------------
    # 板块二：右侧 · 策略大脑与决策输出
    # -------------------------------------------------------------
    def _build_right_policy_panel(self, parent: tk.Widget) -> None:
        # 1. 当前决策核心大卡
        self.decision_card = tk.LabelFrame(
            parent,
            text=" 🛡️ 当前轮次自适应策略决策 (Policy Decision) ",
            font=(UI_FONT, 11, "bold"),
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
            padx=14,
            pady=12,
            relief="solid",
            bd=1,
        )
        self.decision_card.pack(fill="x", pady=(0, 10))

        # 模式徽章 (醒目大字)
        self.mode_badge = tk.Label(
            self.decision_card,
            textvariable=self.mode_title_var,
            font=(UI_FONT, 16, "bold"),
            bg=MODE_THEMES["IDLE"]["bg"],
            fg=MODE_THEMES["IDLE"]["fg"],
            padx=14,
            pady=8,
            relief="solid",
            bd=1,
        )
        self.mode_badge.pack(fill="x", pady=(0, 8))

        # 动作与强度风格
        action_box = tk.Frame(self.decision_card, bg=SURFACE_COLOR)
        action_box.pack(fill="x", pady=(0, 4))
        tk.Label(action_box, text="推荐干预动作:", font=(UI_FONT, 10, "bold"), bg=SURFACE_COLOR, fg=MUTED_COLOR).pack(side="left")
        self.action_label = tk.Label(action_box, textvariable=self.action_var, font=(UI_FONT, 11, "bold"), bg=SURFACE_COLOR, fg=PRIMARY_COLOR)
        self.action_label.pack(side="left", padx=8)

        tk.Label(
            self.decision_card,
            textvariable=self.intensity_style_var,
            font=(UI_FONT, 10),
            bg=SURFACE_COLOR,
            fg=MUTED_COLOR,
            anchor="w",
        ).pack(fill="x")

        # 2. 策略依据框
        rationale_card = tk.LabelFrame(
            parent,
            text=" 💡 策略决策依据 (Why this adaptation?) ",
            font=(UI_FONT, 10, "bold"),
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
            padx=12,
            pady=8,
            relief="solid",
            bd=1,
        )
        rationale_card.pack(fill="x", pady=(0, 10))

        self.rationale_label = tk.Label(
            rationale_card,
            textvariable=self.rationale_var,
            font=(UI_FONT, 10),
            bg="#F8FAFC",
            fg=TEXT_COLOR,
            wraplength=560,
            justify="left",
            anchor="w",
            padx=10,
            pady=8,
            relief="groove",
            bd=1,
        )
        self.rationale_label.pack(fill="x")

        # 3. 历史策略流转与会话轨迹 (可滚动多轮列表)
        history_card = tk.LabelFrame(
            parent,
            text=" 📜 会话历史策略与旅程脉络 (Session Trajectory) ",
            font=(UI_FONT, 10, "bold"),
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
            padx=10,
            pady=8,
            relief="solid",
            bd=1,
        )
        history_card.pack(fill="both", expand=True)

        history_scroll = ttk.Scrollbar(history_card)
        history_scroll.pack(side="right", fill="y")

        self.history_text = tk.Text(
            history_card,
            font=(UI_FONT, 9),
            bg="#F8FAFC",
            fg=TEXT_COLOR,
            wrap="word",
            yscrollcommand=history_scroll.set,
            relief="flat",
            padx=8,
            pady=6,
        )
        self.history_text.pack(side="left", fill="both", expand=True)
        history_scroll.config(command=self.history_text.yview)
        self.history_text.insert("end", "暂无历史交互轮次。点击下方开启互动后，每轮决策将按时序沉淀在此。\n")
        self.history_text.config(state="disabled")

    # -------------------------------------------------------------
    # 板块三：底部 · 状态与快捷提示条
    # -------------------------------------------------------------
    def _build_bottom_controls(self) -> None:
        bottom_bar = tk.Frame(self.root, bg=SURFACE_COLOR, padx=16, pady=8, relief="solid", bd=1)
        bottom_bar.pack(fill="x", side="bottom")

        tk.Label(
            bottom_bar,
            text="💡 提示: 点击右上角按键开启/结束交互 ｜ 终端按 Ctrl+C 可随时退出并自动导出画作数据",
            font=(UI_FONT, 9),
            bg=SURFACE_COLOR,
            fg=MUTED_COLOR,
        ).pack(side="left")

        self.status_label = tk.Label(
            bottom_bar,
            textvariable=self.status_bar_var,
            font=(UI_FONT, 9, "bold"),
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
        )
        self.status_label.pack(side="right")
        self.status_label.pack(side="right", padx=10)

    # -------------------------------------------------------------
    # 交互调度与线程管理
    # -------------------------------------------------------------
    def _on_need_changed(self, event=None) -> None:
        raw_val = self.need_combo.get()
        need_code = raw_val.split("(")[-1].replace(")", "").strip()
        try:
            self.runner.default_need = UserNeed[need_code]
            if self.runner.history is not None:
                self.runner.history.preferred_need = UserNeed[need_code]
            self.status_bar_var.set(f"核心需求已切换为: {NEED_LABELS.get(need_code, need_code)}")
        except Exception:
            pass

    def _on_thresh_slider_changed(self, val: str) -> None:
        new_val = float(val)
        self.runner.vad_tracker.energy_threshold = new_val
        if hasattr(self, "thresh_label"):
            self.thresh_label.config(text=f"VAD 拾音门限: {new_val:.3f}")

    def _trigger_manual_speech_test(self) -> None:
        """Trigger a 3.5s manual audio capture on the main thread using root.after (100% thread-safe on macOS Cocoa)."""
        if not self.is_running:
            messagebox.showinfo("提示", "请先点击右上角【开启实时互动】，再进行试音测试。")
            return
        if self._manual_recording_active:
            return
        self._manual_capture_chunks.clear()
        self._manual_recording_active = True
        self.manual_record_btn.config(text=" 🔴 正在试音(3.5s)... ", state="disabled", bg="#FEE2E2", fg="#991B1B")
        self.status_bar_var.set("🔴 正在试音录音 (3.5 秒)：请现在对着麦克风清晰说一句话...")
        self.root.after(3500, self._finish_manual_speech_test)

    def _finish_manual_speech_test(self) -> None:
        """Complete manual capture on the main thread and dispatch TurnEvent to background worker."""
        self._manual_recording_active = False
        self.manual_record_btn.config(text=" 🎙️ 点按试音(3秒) ", state="normal", bg="#E0E7FF", fg="#3730A3")
        if not self.is_running:
            return
        try:
            import tempfile, wave
            if not self._manual_capture_chunks:
                self.status_bar_var.set("⚠️ 试音未采集到麦克风音频帧，请检查麦克风权限")
                return
            raw_audio = np.concatenate(self._manual_capture_chunks).astype(np.float32)
            self._manual_capture_chunks.clear()

            if raw_audio.size > 0:
                raw_audio = raw_audio - float(np.mean(raw_audio))
                peak = float(np.max(np.abs(raw_audio)))
                if 0.02 < peak < 0.70:
                    raw_audio = raw_audio * min(4.0, 0.80 / peak)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                wav_path = tf.name

            int16_samples = np.clip(raw_audio * 32767.0, -32768, 32767).astype(np.int16)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(int16_samples.tobytes())

            now = time.time()
            ev = TurnEvent(
                event_type=TurnEventType.SPEECH_FINISHED,
                timestamp=now,
                speech_start=now - 3.5,
                speech_end=now,
                audio_wav_path=wav_path,
            )
            self.event_queue.put(ev)
            self.status_bar_var.set("⏳ 试音采集完成，正在转写识别中...")
        except Exception as e:
            self.status_bar_var.set(f"测试录音异常: {e}")

    def _toggle_interaction(self) -> None:
        if not self.is_running:
            self._start_session()
        else:
            self._stop_session()

    def _start_session(self) -> None:
        self.is_running = True
        self.stop_event.clear()
        self.runner.vad_tracker.reset_turn_timer()
        self.toggle_btn.config(text=" ⏹ 停止互动并导出旅程 ", bg="#DC2626", activebackground="#B91C1C")
        self.status_bar_var.set("互动运行中 ｜ 正在后台预热本地语音模型并监听...")
        self.vad_status_var.set("● 麦克风已就绪 (正在倾听...)")
        self.vad_indicator.config(bg="#E6F4EA", fg="#137333")

        # 0. 后台预热 ASR + SER 模型，注意绝不在子线程直接修改 Tkinter 变量 (避免 macOS Cocoa 线程崩溃)
        def _warmup_worker() -> None:
            try:
                self.runner.audio_detector.warmup()
            except Exception:
                pass
        threading.Thread(target=_warmup_worker, daemon=True).start()

        # 1. 启动摄像头采集线程 (智能探测物理镜头)
        self.camera_cap, cam_idx = find_working_camera()
        if not self.camera_cap.isOpened():
            self.video_label.config(text="⚠️ 摄像头未授权或无法打开\n(可在系统设置中允许终端/Python 访问摄像头)", bg="#1E293B", fg="#F87171")
        else:
            threading.Thread(target=self._camera_loop, daemon=True).start()

        # 2. 启动麦克风录音监听流
        self._start_audio_stream()

        # 3. 启动后台事件监听线程
        threading.Thread(target=self._event_processing_loop, daemon=True).start()

    def _stop_session(self) -> None:
        self.is_running = False
        self.stop_event.set()
        self.toggle_btn.config(text=" ▶ 开启实时互动 ", bg=PRIMARY_COLOR, activebackground=PRIMARY_HOVER)
        self.vad_status_var.set("● 装置待机中 (已停止)")
        self.vad_indicator.config(bg="#F1F5F9", fg="#475569")

        # 释放硬件
        if self.camera_cap is not None:
            self.camera_cap.release()
            self.camera_cap = None
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass
            self.audio_stream = None

        self.video_label.config(image="", text="摄像头已停止\n点击底部【开启实时互动】重新唤醒", bg="#0F172A", fg="#94A3B8")

        # 自动弹出选择框供用户决定是否导出旅程数据
        self._export_journey_file()

    def _export_journey_file(self) -> None:
        if not self.runner.journey_records:
            self.status_bar_var.set("互动已停止 (未产生轮次，未导出数据)")
            return

        # 弹出保存确认框，支持用户自主选择是否留存数据
        save_confirmed = messagebox.askyesno(
            "保存会话旅程数据",
            f"本次互动共完成 {self.runner.turn_index} 轮自适应策略交互。\n\n"
            f"是否保存本次对话的旅程历史数据？\n\n"
            f"• 【是 (Yes)】：将本次交互记录导出为 user_journey_*.json，作为后续视觉图库画作生成的标准数据源\n"
            f"• 【否 (No)】：丢弃本次测试记录，不保存文件",
            default="yes",
        )

        if not save_confirmed:
            self.status_bar_var.set("互动已停止 ｜ 本次测试对话记录已放弃（未保存文件）")
            return

        os.makedirs("data/journeys", exist_ok=True)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        journey_path = Path(f"data/journeys/user_journey_{session_id}.json")

        payload = {
            "session_id": session_id,
            "export_time": datetime.now().isoformat(),
            "total_turns": self.runner.turn_index,
            "default_need": self.runner.default_need.value,
            "turns": self.runner.journey_records,
        }
        with open(journey_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self.status_bar_var.set(f"会话圆满结束！旅程数据已成功保存至: {journey_path.name}")
        messagebox.showinfo(
            "旅程保存成功",
            f"本次互动数据已成功保存至:\n{journey_path}\n\n该数据将作为后续视觉策略图库组装画作的标准数据源！",
        )

    # -------------------------------------------------------------
    # 硬件驱动后台循环
    # -------------------------------------------------------------
    def _camera_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.camera_cap is None:
                break
            ret, frame = self.camera_cap.read()
            if not ret or frame is None:
                time.sleep(0.04)
                continue

            try:
                rec = self.runner.visual_detector.analyze_image(frame)
                self.runner.visual_buffer.add_record(rec)

                # 提取人脸检测框与主导表情
                top_emo = "未检测到人脸"
                box = None
                if rec.available and rec.mapped_scores:
                    top_k, top_v = max(rec.mapped_scores.items(), key=lambda x: x[1])
                    zh_emo = EMOTION_LABELS.get(top_k.value, top_k.value)
                    top_emo = f"{zh_emo} ({top_v:.0%})"
                    if "_bbox_x" in rec.raw_scores:
                        box = (
                            int(rec.raw_scores["_bbox_x"]),
                            int(rec.raw_scores["_bbox_y"]),
                            int(rec.raw_scores["_bbox_w"]),
                            int(rec.raw_scores["_bbox_h"]),
                        )

                self.latest_face_box = box
                self.latest_face_emotion = top_emo
                self.latest_face_conf = rec.signal_quality
                self.latest_frame = frame
            except Exception:
                self.latest_frame = frame

            time.sleep(0.035)  # 限制在 ~25 FPS

    def _start_audio_stream(self) -> None:
        try:
            import sounddevice as sd

            def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
                if self.stop_event.is_set():
                    return
                # 必须 .copy() 深拷贝！否则 PortAudio 下一次回调会原地覆写缓冲区
                chunk = indata[:, 0].copy()
                rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size > 0 else 0.0
                self.current_audio_rms = rms
                if self._manual_recording_active:
                    self._manual_capture_chunks.append(chunk)
                    return
                ev = self.runner.vad_tracker.process_chunk(chunk)
                if ev is not None:
                    self.event_queue.put(ev)

            # 优选 MacBook 麦克风设备
            self.mic_device = None
            try:
                devices = sd.query_devices()
                for idx, dev in enumerate(devices):
                    if dev.get("max_input_channels", 0) > 0 and "MacBook" in dev.get("name", ""):
                        self.mic_device = idx
                        break
            except Exception:
                pass

            self.audio_stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="float32",
                blocksize=1024,
                device=self.mic_device,
                callback=callback,
            )
            self.audio_stream.start()
        except Exception as e:
            self.status_bar_var.set(f"麦克风启动提示: {e}")

    def _event_processing_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                event = self.event_queue.get(timeout=0.3)
                # 执行完整一轮 (Fusion -> Policy -> History)，显式传入界面设定的当前需求
                turn_data = self.runner.handle_turn_event(event, manual_need=self.runner.default_need)
                if turn_data is not None:
                    self.turn_result_queue.put(turn_data)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[Error in turn loop] {e}")

    # -------------------------------------------------------------
    # UI 刷新主循环 (Tkinter 30ms 轮询)
    # -------------------------------------------------------------
    def _start_ui_loop(self) -> None:
        self._update_tick()

    def _update_tick(self) -> None:
        # 1. 刷新摄像头画面与人脸标注
        if self.is_running and self.latest_frame is not None:
            try:
                display_frame = self.latest_frame.copy()
                if self.latest_face_box is not None:
                    bx, by, bw, bh = self.latest_face_box
                    # 绘制人脸绿框与简短标签
                    cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (52, 168, 83), 2)
                    tag = f"{self.latest_face_emotion.split(' ')[0]}"
                    cv2.putText(
                        display_frame,
                        tag,
                        (bx + 4, max(22, by - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (52, 168, 83),
                        2,
                        cv2.LINE_AA,
                    )

                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (460, 345))
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk, text="")
                self.face_status_label.config(
                    text=f"人脸表情: {self.latest_face_emotion} ｜ 置信度: {self.latest_face_conf:.1%}"
                )
            except Exception:
                pass

        # 2. 刷新麦克风 VAD 状态与实时音量动态反馈
        if self.is_running:
            rms_val = getattr(self, "current_audio_rms", 0.0)
            vol_meter = "█" * min(8, int(rms_val * 80)) if rms_val > 0.012 else "·"
            if self.runner.vad_tracker.is_speaking:
                self.vad_status_var.set(f"🎙️ 正在捕获用户说话中... [音量: {vol_meter} {rms_val:.3f}]")
                self.vad_indicator.config(bg="#E8F0FE", fg="#1A73E8")
            else:
                idle = time.time() - self.runner.vad_tracker.last_turn_interaction_time
                remain = max(0.0, self.runner.vad_tracker.idle_timeout_seconds - idle)
                if idle > 2.0:
                    self.vad_status_var.set(f"⏳ 静默留白观察中 (倒计时: {remain:.1f}s / {self.runner.vad_tracker.idle_timeout_seconds:.0f}s) [环境音: {rms_val:.3f}]")
                    self.vad_indicator.config(bg="#FEF3C7", fg="#92400E")
                else:
                    self.vad_status_var.set(f"● 麦克风已就绪 (倾听中... [音量: {vol_meter} {rms_val:.3f}])")
                    self.vad_indicator.config(bg="#E6F4EA", fg="#137333")

        # 3. 检查是否有新轮次决策完成
        try:
            while True:
                turn_data = self.turn_result_queue.get_nowait()
                self._render_turn_to_ui(turn_data)
        except queue.Empty:
            pass

        # 继续下一帧轮询
        self.root.after(35, self._update_tick)

    def _render_turn_to_ui(self, data: dict) -> None:
        turn_idx = data["turn_index"]
        mode = data["policy_mode"]
        theme = MODE_THEMES.get(mode, MODE_THEMES["DIRECT"])
        self.mode_theme_key = mode

        # 更新大字决策徽章
        self.mode_title_var.set(f"轮次 [ {turn_idx:02d} ] · {theme['zh']}")
        self.mode_badge.config(bg=theme["bg"], fg=theme["fg"], highlightbackground=theme["border"])

        # 更新动作与风格
        raw_act = data["policy_action"]
        zh_act = ACTION_LABELS.get(raw_act, raw_act)
        raw_int = data["policy_intensity"]
        zh_int = INTENSITY_LABELS.get(raw_int, raw_int)
        self.action_var.set(f"{zh_act} ({raw_act})")
        self.intensity_style_var.set(f"干预强度: 【 {zh_int} 】 ｜ 认知语气风格: 【 {data['epistemic_style']} 】")

        # 更新左侧字幕与融合摘要
        self.transcript_var.set(data["transcript"])
        disagree_val = f"{data['cross_modal_disagreement']:.4f}" if data['cross_modal_disagreement'] is not None else "0.0000"

        face_raw = data["face_dominant_emotion"]
        zh_face = EMOTION_LABELS.get(face_raw.lower(), face_raw)
        face_str = f"{zh_face} ({face_raw})" if face_raw != "N/A" else "未检出人脸"

        aud_raw = data["audio_dominant_emotion"]
        zh_aud = EMOTION_LABELS.get(aud_raw.lower(), aud_raw)
        aud_str = f"{zh_aud} ({aud_raw})" if aud_raw != "N/A" else "无音频输入"

        top_fused = data.get("fused_dominant_emotion", "UNKNOWN")
        zh_fused = EMOTION_LABELS.get(str(top_fused).lower(), str(top_fused))

        self.fused_summary_var.set(
            f"面部均值: {face_str} ｜ 声音: {aud_str} ｜ 冲突度: {disagree_val} ｜ 熵: {data['predictive_entropy']:.3f}"
        )

        # 更新决策依据 (显式呈现面部与声学情绪感知证据)
        zh_reasons = [
            f"• 【面部线索】人脸表情主导判定为【{face_str}】(人脸置信度: {data['face_signal_quality']:.1%})",
            f"• 【声音线索】语音情绪主导判定为【{aud_str}】",
            f"• 【模态融合】跨模态冲突度 {disagree_val} ｜ 信息熵 {data['predictive_entropy']:.3f} ｜ 综合倾向【{zh_fused}】",
        ]
        for r in data["policy_reasons"]:
            zh_reasons.append(f"• 【策略推理逻辑】{translate_reason(r)}")

        if data["cross_modal_disagreement"] and data["cross_modal_disagreement"] > 0.35:
            zh_reasons.insert(3, f"• ⚠️ 【冲突应对机制】检测到显著跨模态冲突 (冲突度 {disagree_val})，策略采取审慎试探介入")

        self.rationale_var.set("\n".join(zh_reasons))

        # 追加到历史记录文本框
        self.history_text.config(state="normal")
        if turn_idx == 1:
            self.history_text.delete("1.0", "end")

        tag_name = f"turn_{turn_idx}"
        ev_title = "🗣️ 说话响应" if data["user_initiated"] else "⏳ 静默留白"
        card_content = (
            f"【轮次 {turn_idx:02d} ｜ {ev_title}】\n"
            f"  言语内容: \"{data['transcript']}\"\n"
            f"  策略决策: [{theme['zh']}] -> {zh_act}\n"
            f"  感知特征: 面部[{face_str}] 声音[{aud_str}]\n"
            f"  模态参数: 冲突度 {disagree_val} ｜ 熵 {data['predictive_entropy']:.3f}\n"
            f"--------------------------------------------------\n"
        )
        self.history_text.insert("end", card_content)
        self.history_text.see("end")
        self.history_text.config(state="disabled")

        self.status_bar_var.set(f"第 {turn_idx} 轮策略已执行完毕 ｜ 累计交互记录正常沉淀中")

        # 追加到历史记录文本框
        self.history_text.config(state="normal")
        if turn_idx == 1:
            self.history_text.delete("1.0", "end")

        tag_name = f"turn_{turn_idx}"
        ev_title = "🗣️ 说话响应" if data["user_initiated"] else "⏳ 静默留白"
        card_content = (
            f"【轮次 {turn_idx:02d} ｜ {ev_title}】\n"
            f"  言语内容: \"{data['transcript']}\"\n"
            f"  策略决策: [{theme['zh']}] -> {zh_act}\n"
            f"  感知特征: 面部[{face_str}] 声音[{aud_str}]\n"
            f"  模态参数: 冲突度 {disagree_val} ｜ 熵 {data['predictive_entropy']:.3f}\n"
            f"--------------------------------------------------\n"
        )
        self.history_text.insert("end", card_content)
        self.history_text.see("end")
        self.history_text.config(state="disabled")

        self.status_bar_var.set(f"第 {turn_idx} 轮策略已执行完毕 ｜ 累计交互记录正常沉淀中")


def main() -> None:
    root = tk.Tk()
    app = StrategyAppV4(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop_session(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
