"""不确定性情感分析机器人 · 实时多模态策略与对话交互前端 (v4.3).

Features:
  1. 双分页架构 (ttk.Notebook):
     - 分页 1【💬 实时互动与对话】:
       * 左侧: 实时摄像头视觉捕获 (YuNet + HSEmotion) + 语音监听与转写 (VAD + SenseVoice)
       * 右上: 当前轮次自适应策略决策徽章 + 紧凑可滚动决策依据框
       * 右下: 实时文字对话流 (每轮展示用户发言与机器人回复，回复下带轻量策略小标签；WAIT 静默观察时显示浅灰色提示) + 底部测试文字输入框
     - 分页 2【📚 历史对话与保存】:
       * 左侧: 置顶“当前对话会话” + data/journeys/ 下所有已保存历史会话列表
       * 右侧: 点击任意会话即可查阅完整多轮对话与策略明细，支持手动保存、清空当前、删除旧记录
  2. 停止互动后不强制弹窗，由用户自行到【历史对话与保存】分页查阅并按需保存。
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from affective.models import UserNeed
from affective.labels_zh import (
    ACTION_LABELS,
    EMOTION_LABELS,
    INTENSITY_LABELS,
    NEED_LABELS,
    translate_reason,
)
from affective.perception.stream_coordinator import (
    InteractiveSessionRunner,
    TurnEvent,
    TurnEventType,
)

# -------------------------------------------------------------
# 视觉设计常数 (Modern Minimal Palette)
# -------------------------------------------------------------
APP_TITLE = "不确定性情感分析机器人"
JOURNEYS_DIR = Path("data/journeys")

BG_COLOR = "#F4F7FB"
SURFACE_COLOR = "#FFFFFF"
TEXT_COLOR = "#182230"
MUTED_COLOR = "#667085"
SOFT_HINT_COLOR = "#94A3B8"
BORDER_COLOR = "#E2E8F0"
PRIMARY_COLOR = "#315EFB"
PRIMARY_HOVER = "#2449D8"
DANGER_COLOR = "#DC2626"

MODE_THEMES = {
    "DIRECT": {"bg": "#E6F4EA", "fg": "#137333", "border": "#34A853", "zh": "直接介入 (DIRECT)"},
    "TENTATIVE": {"bg": "#F3E8FF", "fg": "#6B21A8", "border": "#A855F7", "zh": "谨慎试探 (TENTATIVE)"},
    "CONFIRM": {"bg": "#E8F0FE", "fg": "#1A73E8", "border": "#4285F4", "zh": "澄清确认 (CONFIRM)"},
    "WAIT": {"bg": "#FEF3C7", "fg": "#92400E", "border": "#F59E0B", "zh": "静默观察 (WAIT)"},
    "DEFER": {"bg": "#FCE8E6", "fg": "#C5221F", "border": "#EA4335", "zh": "暂缓介入 (DEFER)"},
    "IDLE": {"bg": "#F1F5F9", "fg": "#475569", "border": "#94A3B8", "zh": "等待交互启动"},
}

UI_FONT = "PingFang SC" if sys.platform == "darwin" else "Microsoft YaHei"
INPUT_PLACEHOLDER = "输入文字直接测试 Need 分析与自适应回复（回车发送）..."


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


def format_turn_meta_tag(data: dict) -> str:
    """Format a compact subtle strategy tag displayed beneath each robot reply."""
    mode = str(data.get("policy_mode", "DIRECT")).upper()
    raw_act = str(data.get("policy_action", ""))
    zh_act = ACTION_LABELS.get(raw_act, raw_act or "—")
    raw_int = str(data.get("policy_intensity", "low"))
    zh_int = INTENSITY_LABELS.get(raw_int, raw_int)
    style = str(data.get("epistemic_style", "—"))

    parts = [mode, zh_act, f"{zh_int}强度", f"语气:{style}"]
    need_det = str(data.get("llm_detected_need", "UNKNOWN"))
    need_conf = float(data.get("llm_need_confidence", 0.0) or 0.0)
    if need_det and need_det != "UNKNOWN":
        zh_need = NEED_LABELS.get(need_det, need_det)
        parts.append(f"Need:{zh_need}({need_conf:.0%})")
    return " · ".join(parts)


def save_journey_payload(
    turns: list[dict],
    default_need: str,
    journeys_dir: Path = JOURNEYS_DIR,
) -> Path:
    """Persist session turns to data/journeys/user_journey_<timestamp>.json."""
    journeys_dir.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    journey_path = journeys_dir / f"user_journey_{session_id}.json"
    # 防止同一秒连续点击保存覆盖原文件
    suffix = 1
    while journey_path.exists():
        journey_path = journeys_dir / f"user_journey_{session_id}_{suffix}.json"
        suffix += 1

    payload = {
        "session_id": journey_path.stem.replace("user_journey_", ""),
        "export_time": datetime.now().isoformat(),
        "total_turns": len(turns),
        "default_need": default_need,
        "turns": turns,
    }
    journey_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return journey_path


def list_saved_journeys(journeys_dir: Path = JOURNEYS_DIR) -> list[dict]:
    """Scan saved journey JSON files sorted newest-first with safe schema fallback."""
    if not journeys_dir.exists():
        return []
    items: list[dict] = []
    for path in sorted(journeys_dir.glob("user_journey_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            turns = payload.get("turns") or []
            session_id = str(payload.get("session_id") or path.stem.replace("user_journey_", ""))
            export_time = str(payload.get("export_time") or "")
            display_time = export_time[:19].replace("T", " ") if len(export_time) >= 19 else session_id
            items.append(
                {
                    "id": str(path),
                    "path": path,
                    "session_id": session_id,
                    "display_time": display_time,
                    "total_turns": int(payload.get("total_turns", len(turns))),
                    "default_need": str(payload.get("default_need", "UNKNOWN")),
                    "turns": turns,
                }
            )
        except Exception:
            continue
    return items


class StrategyAppV4:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1260x780")
        self.root.minsize(1060, 680)
        self.root.configure(bg=BG_COLOR)

        # 核心引擎与队列 (默认 UserNeed.UNKNOWN：由多模态融合情绪自主驱动支持动作)
        self.runner = InteractiveSessionRunner(default_need=UserNeed.UNKNOWN)
        self.event_queue: queue.Queue[TurnEvent] = queue.Queue()
        self.turn_result_queue: queue.Queue[dict] = queue.Queue()
        self.stop_event = threading.Event()
        self._app_closing = threading.Event()

        # 硬件与会话状态句柄
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

        self._current_session_saved: bool = False
        self._last_saved_filename: str | None = None
        self._chat_has_messages: bool = False
        self._placeholder_active: bool = True
        self._pending_text_requests: int = 0
        self._saved_sessions_map: dict[str, dict] = {}
        self._selected_history_iid: str = "__CURRENT__"

        # UI 状态绑定变量
        self.vad_status_var = tk.StringVar(value="● 装置待机中 (点击右上角开启实时互动，或直接在右下角打字测试)")
        self.transcript_var = tk.StringVar(value="等待用户开口说话或输入文字...")
        self.fused_summary_var = tk.StringVar(value="面部: — ｜ 声音: — ｜ 文本: — ｜ 覆盖率: — ｜ 冲突: — ｜ 熵: —")

        self.mode_theme_key = "IDLE"
        self.mode_title_var = tk.StringVar(value=MODE_THEMES["IDLE"]["zh"])
        self.action_var = tk.StringVar(value="尚未生成动作")
        self.intensity_style_var = tk.StringVar(value="强度: — ｜ 语气: — ｜ 需求: 自动感知")
        self.history_notice_var = tk.StringVar(value="当前会话尚未产生对话轮次")
        self.history_detail_title_var = tk.StringVar(value="⭐ 当前对话会话（尚未开始）")
        self.history_detail_meta_var = tk.StringVar(value="可在左侧切换查看当前会话或任意已保存的历史记录")

        llm_status = (
            f"LLM 已连接 ({self.runner.llm_adapter.config.model})"
            if self.runner.llm_adapter and self.runner.llm_adapter.enabled
            else "LLM 离线回退模式 (未配置 .env)"
        )
        self.status_bar_var = tk.StringVar(value=f"就绪 ｜ {llm_status} ｜ 支持语音实时互动或直接打字测试")

        self._build_styles()
        self._build_layout()

        # 启动常驻单线程事件处理循环 (支持未开摄像头/麦克风时直接打字测试，且语音与文字串行无竞态)
        threading.Thread(target=self._event_processing_loop, daemon=True).start()
        self._refresh_history_list(select_iid="__CURRENT__")
        self._start_ui_loop()

    def _build_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", padding=5, font=(UI_FONT, 10))
        style.configure("Main.TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure(
            "Main.TNotebook.Tab",
            font=(UI_FONT, 11, "bold"),
            padding=(18, 8),
            background="#E8EEF6",
            foreground=MUTED_COLOR,
        )
        style.map(
            "Main.TNotebook.Tab",
            background=[("selected", SURFACE_COLOR)],
            foreground=[("selected", PRIMARY_COLOR)],
        )
        style.configure("History.Treeview", font=(UI_FONT, 10), rowheight=30, background=SURFACE_COLOR, fieldbackground=SURFACE_COLOR)
        style.configure("History.Treeview.Heading", font=(UI_FONT, 10, "bold"), background="#F1F5F9", foreground=TEXT_COLOR)

    def _build_layout(self) -> None:
        # 1. 顶栏标题与核心操作工具条
        header = tk.Frame(self.root, bg=SURFACE_COLOR, padx=20, pady=10, relief="solid", bd=1)
        header.pack(fill="x", side="top")

        title_box = tk.Frame(header, bg=SURFACE_COLOR)
        title_box.pack(side="left")
        tk.Label(
            title_box,
            text=APP_TITLE,
            font=(UI_FONT, 16, "bold"),
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="多模态实时感知 ｜ 不确定性自适应策略 ｜ Need 识别与受控对话",
            font=(UI_FONT, 9),
            bg=SURFACE_COLOR,
            fg=MUTED_COLOR,
        ).pack(anchor="w")

        control_box = tk.Frame(header, bg=SURFACE_COLOR)
        control_box.pack(side="right")

        tk.Label(control_box, text="需求模式:", font=(UI_FONT, 9, "bold"), bg=SURFACE_COLOR, fg=MUTED_COLOR).pack(side="left", padx=(0, 4))
        need_options = ["自动感知·情绪驱动 (UNKNOWN)"] + [
            f"{NEED_LABELS.get(n.name, n.name)} ({n.name})"
            for n in UserNeed
            if n != UserNeed.UNKNOWN
        ]
        self.need_combo = ttk.Combobox(control_box, values=need_options, state="readonly", width=21)
        self.need_combo.set("自动感知·情绪驱动 (UNKNOWN)")
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

        # 3. 顶层双分页 (方案 A: 实时互动与对话 / 历史对话与保存)
        self.notebook = ttk.Notebook(self.root, style="Main.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=14, pady=(8, 4))
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.live_tab = tk.Frame(self.notebook, bg=BG_COLOR)
        self.history_tab = tk.Frame(self.notebook, bg=BG_COLOR)

        self.notebook.add(self.live_tab, text=" 💬 实时互动与对话 ")
        self.notebook.add(self.history_tab, text=" 📚 历史对话与保存 ")

        self._build_live_tab(self.live_tab)
        self._build_history_tab(self.history_tab)

    # -------------------------------------------------------------
    # 分页 1：实时互动与对话
    # -------------------------------------------------------------
    def _build_live_tab(self, parent: tk.Widget) -> None:
        body = tk.Frame(parent, bg=BG_COLOR, padx=4, pady=6)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=5, uniform="col")
        body.grid_columnconfigure(1, weight=6, uniform="col")
        body.grid_rowconfigure(0, weight=1)

        left_col = tk.Frame(body, bg=BG_COLOR)
        right_col = tk.Frame(body, bg=BG_COLOR)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right_col.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._build_left_perception_panel(left_col)
        self._build_right_policy_and_chat_panel(right_col)

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

        self.video_label = tk.Label(
            cam_card,
            text="摄像头尚未开启\n点击右上角【开启实时互动】唤醒摄像头",
            font=(UI_FONT, 10),
            bg="#0F172A",
            fg="#94A3B8",
            width=48,
            height=12,
        )
        self.video_label.pack(fill="both", expand=True, padx=4, pady=4)

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
            font=(UI_FONT, 10, "bold"),
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
            padx=12,
            pady=8,
            relief="solid",
            bd=1,
        )
        mic_card.pack(fill="x")

        self.vad_indicator = tk.Label(
            mic_card,
            textvariable=self.vad_status_var,
            font=(UI_FONT, 9, "bold"),
            bg="#F1F5F9",
            fg="#475569",
            padx=8,
            pady=4,
            anchor="w",
        )
        self.vad_indicator.pack(fill="x", pady=(0, 6))

        tk.Label(mic_card, text="最新用户输入字幕:", font=(UI_FONT, 9), bg=SURFACE_COLOR, fg=MUTED_COLOR, anchor="w").pack(fill="x")
        self.transcript_box = tk.Label(
            mic_card,
            textvariable=self.transcript_var,
            font=(UI_FONT, 10, "bold"),
            bg="#F8FAFC",
            fg="#0F172A",
            wraplength=470,
            justify="left",
            anchor="w",
            padx=8,
            pady=6,
            relief="groove",
            bd=1,
        )
        self.transcript_box.pack(fill="x", pady=(2, 6))

        self.fused_summary_label = tk.Label(
            mic_card,
            textvariable=self.fused_summary_var,
            font=(UI_FONT, 9),
            bg=SURFACE_COLOR,
            fg=MUTED_COLOR,
            anchor="w",
        )
        self.fused_summary_label.pack(fill="x", pady=(0, 6))

        mic_tune_box = tk.Frame(mic_card, bg=SURFACE_COLOR)
        mic_tune_box.pack(fill="x", pady=(2, 0))

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

    def _build_right_policy_and_chat_panel(self, parent: tk.Widget) -> None:
        # 1. 当前轮次策略决策卡 (紧凑清晰，不挤压下方聊天流)
        self.decision_card = tk.LabelFrame(
            parent,
            text=" 🛡️ 当前轮次自适应策略决策 (Policy Decision) ",
            font=(UI_FONT, 10, "bold"),
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
            padx=12,
            pady=8,
            relief="solid",
            bd=1,
        )
        self.decision_card.pack(fill="x", pady=(0, 8))

        self.mode_badge = tk.Label(
            self.decision_card,
            textvariable=self.mode_title_var,
            font=(UI_FONT, 14, "bold"),
            bg=MODE_THEMES["IDLE"]["bg"],
            fg=MODE_THEMES["IDLE"]["fg"],
            padx=12,
            pady=6,
            relief="solid",
            bd=1,
        )
        self.mode_badge.pack(fill="x", pady=(0, 6))

        action_box = tk.Frame(self.decision_card, bg=SURFACE_COLOR)
        action_box.pack(fill="x", pady=(0, 2))
        tk.Label(action_box, text="推荐干预动作:", font=(UI_FONT, 9, "bold"), bg=SURFACE_COLOR, fg=MUTED_COLOR).pack(side="left")
        self.action_label = tk.Label(action_box, textvariable=self.action_var, font=(UI_FONT, 10, "bold"), bg=SURFACE_COLOR, fg=PRIMARY_COLOR)
        self.action_label.pack(side="left", padx=6)

        tk.Label(
            self.decision_card,
            textvariable=self.intensity_style_var,
            font=(UI_FONT, 9),
            bg=SURFACE_COLOR,
            fg=MUTED_COLOR,
            anchor="w",
        ).pack(fill="x")

        # 2. 策略依据框 (定高可滚动只读文本，防止拉长遮挡对话区)
        rationale_card = tk.LabelFrame(
            parent,
            text=" 💡 策略决策依据 (Why this adaptation?) ",
            font=(UI_FONT, 10, "bold"),
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
            padx=10,
            pady=6,
            relief="solid",
            bd=1,
        )
        rationale_card.pack(fill="x", pady=(0, 8))

        rat_scroll = ttk.Scrollbar(rationale_card)
        rat_scroll.pack(side="right", fill="y")
        self.rationale_text = tk.Text(
            rationale_card,
            height=4,
            font=(UI_FONT, 9),
            bg="#F8FAFC",
            fg=TEXT_COLOR,
            wrap="word",
            yscrollcommand=rat_scroll.set,
            relief="flat",
            padx=8,
            pady=4,
        )
        self.rationale_text.pack(side="left", fill="x", expand=True)
        rat_scroll.config(command=self.rationale_text.yview)
        self._set_rationale_text("启动互动或在下方输入文字后，系统将在此解析三模态线索、Need 识别与策略推导逻辑...")

        # 3. 实时对话流卡片 (核心交互区 + 底部测试输入框)
        chat_card = tk.LabelFrame(
            parent,
            text=" 💬 实时对话流 (自适应回复话语) ",
            font=(UI_FONT, 10, "bold"),
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
            padx=10,
            pady=8,
            relief="solid",
            bd=1,
        )
        chat_card.pack(fill="both", expand=True)

        # 对话消息区
        chat_body = tk.Frame(chat_card, bg=SURFACE_COLOR)
        chat_body.pack(fill="both", expand=True, pady=(0, 8))

        chat_scroll = ttk.Scrollbar(chat_body)
        chat_scroll.pack(side="right", fill="y")

        self.chat_text = tk.Text(
            chat_body,
            font=(UI_FONT, 10),
            bg="#F8FAFC",
            fg=TEXT_COLOR,
            wrap="word",
            yscrollcommand=chat_scroll.set,
            relief="flat",
            padx=12,
            pady=10,
            spacing1=2,
            spacing3=2,
        )
        self.chat_text.pack(side="left", fill="both", expand=True)
        chat_scroll.config(command=self.chat_text.yview)
        self._configure_chat_tags(self.chat_text)
        self._reset_chat_view()

        # 底部测试文字输入框
        input_bar = tk.Frame(chat_card, bg=SURFACE_COLOR)
        input_bar.pack(fill="x", side="bottom")

        self.text_input_entry = tk.Entry(
            input_bar,
            font=(UI_FONT, 10),
            bg="#FFFFFF",
            fg=SOFT_HINT_COLOR,
            relief="solid",
            bd=1,
            insertbackground=TEXT_COLOR,
        )
        self.text_input_entry.insert(0, INPUT_PLACEHOLDER)
        self.text_input_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        self.text_input_entry.bind("<FocusIn>", self._on_input_focus_in)
        self.text_input_entry.bind("<FocusOut>", self._on_input_focus_out)
        self.text_input_entry.bind("<Return>", self._on_send_text_clicked)

        self.send_text_btn = tk.Button(
            input_bar,
            text=" 发送测试 ",
            font=(UI_FONT, 10, "bold"),
            bg=PRIMARY_COLOR,
            fg="white",
            activebackground=PRIMARY_HOVER,
            activeforeground="white",
            padx=14,
            pady=5,
            relief="flat",
            cursor="hand2",
            command=self._on_send_text_clicked,
        )
        self.send_text_btn.pack(side="right")

    def _configure_chat_tags(self, widget: tk.Text) -> None:
        widget.tag_config("placeholder", foreground=SOFT_HINT_COLOR, font=(UI_FONT, 10), justify="center", spacing1=24)
        widget.tag_config("user_role", foreground="#1D4ED8", font=(UI_FONT, 9, "bold"), justify="right", spacing1=8)
        widget.tag_config(
            "user_bubble",
            background="#DBEAFE",
            foreground="#0F172A",
            font=(UI_FONT, 10),
            justify="right",
            lmargin1=70,
            lmargin2=70,
            rmargin=6,
            spacing1=4,
            spacing3=6,
        )
        widget.tag_config("bot_role", foreground="#334155", font=(UI_FONT, 9, "bold"), justify="left", spacing1=6)
        widget.tag_config(
            "bot_bubble",
            background="#FFFFFF",
            foreground="#0F172A",
            font=(UI_FONT, 10),
            justify="left",
            lmargin1=6,
            lmargin2=6,
            rmargin=70,
            spacing1=5,
            spacing3=5,
        )
        widget.tag_config(
            "meta_tag",
            foreground="#94A3B8",
            font=(UI_FONT, 8),
            justify="left",
            lmargin1=8,
            lmargin2=8,
            spacing1=1,
            spacing3=8,
        )
        widget.tag_config(
            "wait_hint",
            foreground="#94A3B8",
            font=(UI_FONT, 9),
            justify="center",
            spacing1=8,
            spacing3=8,
        )

    def _reset_chat_view(self) -> None:
        self._chat_has_messages = False
        self.chat_text.config(state="normal")
        self.chat_text.delete("1.0", "end")
        self.chat_text.insert(
            "end",
            "暂无对话内容\n开启实时语音互动，或在下方输入框直接输入一句话即可开始测试。\n",
            "placeholder",
        )
        self.chat_text.config(state="disabled")

    def _append_turn_to_chat(self, data: dict) -> None:
        turn_idx = int(data.get("turn_index", 0))
        mode = str(data.get("policy_mode", "DIRECT")).upper()
        intensity = str(data.get("policy_intensity", "low")).lower()
        transcript = str(data.get("transcript", "")).strip()
        reply_text = str(data.get("system_reply") or "").strip()
        ev_type = str(data.get("event_type", ""))

        self.chat_text.config(state="normal")
        if not self._chat_has_messages:
            self.chat_text.delete("1.0", "end")
            self._chat_has_messages = True

        # 静默观察 (WAIT) 或无强度发声 (NONE) -> 居中浅灰提示，不刷对话气泡
        if mode == "WAIT" or intensity == "none":
            if mode == "DEFER":
                # DEFER 时仍展示温和退让话语（如有）
                if transcript and transcript != "[沉默无言语]":
                    role_suffix = " (文字测试)" if ev_type == TurnEventType.TEXT_INPUT.value else ""
                    self.chat_text.insert("end", f"你{role_suffix} · 轮次 {turn_idx:02d}\n", "user_role")
                    self.chat_text.insert("end", f"  {transcript}  \n", "user_bubble")
                defer_msg = reply_text or "好的，我先安静陪着你，不打扰。你想说话的时候随时开口。"
                self.chat_text.insert("end", "机器人\n", "bot_role")
                self.chat_text.insert("end", f"  {defer_msg}  \n", "bot_bubble")
                self.chat_text.insert("end", f"  {format_turn_meta_tag(data)}\n", "meta_tag")
            else:
                face_raw = str(data.get("face_dominant_emotion", "N/A"))
                zh_face = EMOTION_LABELS.get(face_raw.lower(), face_raw) if face_raw != "N/A" else "未检出人脸"
                hint = f"── 轮次 {turn_idx:02d} · 装置保持安静观察（当前面部倾向: {zh_face}，不主动打扰）──\n"
                self.chat_text.insert("end", hint, "wait_hint")
        else:
            # 正常对话轮次：用户发言 + 机器人回复 + 极淡策略小标签
            if transcript and transcript != "[沉默无言语]":
                role_suffix = " (文字测试)" if ev_type == TurnEventType.TEXT_INPUT.value else ""
                self.chat_text.insert("end", f"你{role_suffix} · 轮次 {turn_idx:02d}\n", "user_role")
                self.chat_text.insert("end", f"  {transcript}  \n", "user_bubble")
            else:
                self.chat_text.insert("end", f"── 轮次 {turn_idx:02d} · 静默观察触发回应 ──\n", "wait_hint")

            self.chat_text.insert("end", "机器人\n", "bot_role")
            self.chat_text.insert("end", f"  {reply_text or '（无主动语言回应）'}  \n", "bot_bubble")
            self.chat_text.insert("end", f"  {format_turn_meta_tag(data)}\n", "meta_tag")

        self.chat_text.see("end")
        self.chat_text.config(state="disabled")

    # -------------------------------------------------------------
    # 分页 2：历史对话与保存
    # -------------------------------------------------------------
    def _build_history_tab(self, parent: tk.Widget) -> None:
        wrapper = tk.Frame(parent, bg=BG_COLOR, padx=4, pady=6)
        wrapper.pack(fill="both", expand=True)

        # 顶部工具条 (保存、清空、删除、刷新 + 内联状态反馈)
        toolbar = tk.Frame(wrapper, bg=SURFACE_COLOR, padx=14, pady=10, relief="solid", bd=1)
        toolbar.pack(fill="x", pady=(0, 8))

        self.save_history_btn = tk.Button(
            toolbar,
            text=" 💾 保存当前对话 ",
            font=(UI_FONT, 10, "bold"),
            bg=PRIMARY_COLOR,
            fg="white",
            activebackground=PRIMARY_HOVER,
            activeforeground="white",
            padx=14,
            pady=5,
            relief="flat",
            cursor="hand2",
            command=self._save_current_session,
        )
        self.save_history_btn.pack(side="left", padx=(0, 8))

        tk.Button(
            toolbar,
            text=" 🧹 清空当前对话 ",
            font=(UI_FONT, 9, "bold"),
            bg="#F1F5F9",
            fg=TEXT_COLOR,
            padx=12,
            pady=5,
            relief="flat",
            cursor="hand2",
            command=self._clear_current_session,
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            toolbar,
            text=" 🗑️ 删除选中记录 ",
            font=(UI_FONT, 9, "bold"),
            bg="#FEE2E2",
            fg="#991B1B",
            padx=12,
            pady=5,
            relief="flat",
            cursor="hand2",
            command=self._delete_selected_history,
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            toolbar,
            text=" 🔄 刷新列表 ",
            font=(UI_FONT, 9, "bold"),
            bg="#F1F5F9",
            fg=TEXT_COLOR,
            padx=10,
            pady=5,
            relief="flat",
            cursor="hand2",
            command=lambda: self._refresh_history_list(select_iid=self._selected_history_iid),
        ).pack(side="left")

        self.history_notice_label = tk.Label(
            toolbar,
            textvariable=self.history_notice_var,
            font=(UI_FONT, 9, "bold"),
            bg=SURFACE_COLOR,
            fg=MUTED_COLOR,
        )
        self.history_notice_label.pack(side="right")

        # 主体左右分栏：左侧会话列表 / 右侧对话与策略详情
        content = tk.Frame(wrapper, bg=BG_COLOR)
        content.pack(fill="both", expand=True)
        content.grid_columnconfigure(0, weight=4, uniform="hcol")
        content.grid_columnconfigure(1, weight=7, uniform="hcol")
        content.grid_rowconfigure(0, weight=1)

        # 左侧会话列表卡片
        list_card = tk.LabelFrame(
            content,
            text=" 📂 会话记录列表 (点击任意项查看详情) ",
            font=(UI_FONT, 10, "bold"),
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
            padx=8,
            pady=8,
            relief="solid",
            bd=1,
        )
        list_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        tree_scroll = ttk.Scrollbar(list_card)
        tree_scroll.pack(side="right", fill="y")

        self.history_tree = ttk.Treeview(
            list_card,
            columns=("title", "turns", "status"),
            show="headings",
            style="History.Treeview",
            selectmode="browse",
            yscrollcommand=tree_scroll.set,
        )
        self.history_tree.heading("title", text="会话 / 时间")
        self.history_tree.heading("turns", text="轮次")
        self.history_tree.heading("status", text="状态")
        self.history_tree.column("title", width=220, anchor="w")
        self.history_tree.column("turns", width=60, anchor="center")
        self.history_tree.column("status", width=80, anchor="center")
        self.history_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.config(command=self.history_tree.yview)
        self.history_tree.bind("<<TreeviewSelect>>", self._on_history_item_selected)

        # 右侧详情卡片
        detail_card = tk.LabelFrame(
            content,
            text=" 📄 会话完整对话与策略回放 ",
            font=(UI_FONT, 10, "bold"),
            bg=SURFACE_COLOR,
            fg=TEXT_COLOR,
            padx=10,
            pady=8,
            relief="solid",
            bd=1,
        )
        detail_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        detail_head = tk.Frame(detail_card, bg="#F8FAFC", padx=10, pady=8, relief="groove", bd=1)
        detail_head.pack(fill="x", pady=(0, 8))
        tk.Label(
            detail_head,
            textvariable=self.history_detail_title_var,
            font=(UI_FONT, 11, "bold"),
            bg="#F8FAFC",
            fg=TEXT_COLOR,
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            detail_head,
            textvariable=self.history_detail_meta_var,
            font=(UI_FONT, 9),
            bg="#F8FAFC",
            fg=MUTED_COLOR,
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

        detail_body = tk.Frame(detail_card, bg=SURFACE_COLOR)
        detail_body.pack(fill="both", expand=True)

        detail_scroll = ttk.Scrollbar(detail_body)
        detail_scroll.pack(side="right", fill="y")

        self.history_detail_text = tk.Text(
            detail_body,
            font=(UI_FONT, 10),
            bg="#F8FAFC",
            fg=TEXT_COLOR,
            wrap="word",
            yscrollcommand=detail_scroll.set,
            relief="flat",
            padx=12,
            pady=10,
        )
        self.history_detail_text.pack(side="left", fill="both", expand=True)
        detail_scroll.config(command=self.history_detail_text.yview)
        self._configure_detail_tags(self.history_detail_text)

    def _configure_detail_tags(self, widget: tk.Text) -> None:
        widget.tag_config("turn_hdr", foreground=PRIMARY_COLOR, font=(UI_FONT, 10, "bold"), spacing1=10, spacing3=2)
        widget.tag_config("u_line", foreground="#0F172A", font=(UI_FONT, 10, "bold"), lmargin1=8, lmargin2=8, spacing1=2)
        widget.tag_config("b_line", foreground="#1E3A8A", font=(UI_FONT, 10), lmargin1=8, lmargin2=8, spacing1=2, spacing3=3)
        widget.tag_config("wait_line", foreground="#94A3B8", font=(UI_FONT, 9), lmargin1=8, lmargin2=8, spacing1=2, spacing3=3)
        widget.tag_config("meta_line", foreground="#64748B", font=(UI_FONT, 9), lmargin1=10, lmargin2=10, spacing1=1)
        widget.tag_config("divider", foreground="#CBD5E1", font=(UI_FONT, 9), spacing1=4, spacing3=4)

    # -------------------------------------------------------------
    # 底部状态条
    # -------------------------------------------------------------
    def _build_bottom_controls(self) -> None:
        bottom_bar = tk.Frame(self.root, bg=SURFACE_COLOR, padx=16, pady=7, relief="solid", bd=1)
        bottom_bar.pack(fill="x", side="bottom")

        tk.Label(
            bottom_bar,
            text="💡 停止互动后不再弹窗，可随时切换到【📚 历史对话与保存】分页查看并保存记录",
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
        self.status_label.pack(side="right", padx=6)

    # -------------------------------------------------------------
    # 输入框与历史分页操作
    # -------------------------------------------------------------
    def _set_rationale_text(self, content: str) -> None:
        self.rationale_text.config(state="normal")
        self.rationale_text.delete("1.0", "end")
        self.rationale_text.insert("end", content)
        self.rationale_text.config(state="disabled")

    def _on_input_focus_in(self, event=None) -> None:
        if self._placeholder_active:
            self.text_input_entry.delete(0, "end")
            self.text_input_entry.config(fg=TEXT_COLOR)
            self._placeholder_active = False

    def _on_input_focus_out(self, event=None) -> None:
        if not self.text_input_entry.get().strip():
            self.text_input_entry.delete(0, "end")
            self.text_input_entry.insert(0, INPUT_PLACEHOLDER)
            self.text_input_entry.config(fg=SOFT_HINT_COLOR)
            self._placeholder_active = True

    def _on_send_text_clicked(self, event=None) -> None:
        if self._placeholder_active:
            return
        raw_text = self.text_input_entry.get().strip()
        if not raw_text:
            return

        # 清空输入框并派发 TEXT_INPUT 事件（复用同一套 Fusion -> Need -> Policy -> LLM 流水线）
        self.text_input_entry.delete(0, "end")
        self._pending_text_requests += 1
        self.send_text_btn.config(text=" 思考中... ", state="disabled", bg="#94A3B8")
        self.status_bar_var.set("⏳ 正在进行 Need 分析与自适应策略生成...")

        now = time.time()
        ev = TurnEvent(
            event_type=TurnEventType.TEXT_INPUT,
            timestamp=now,
            text_input=raw_text,
        )
        self.event_queue.put(ev)

    def _on_tab_changed(self, event=None) -> None:
        current_tab = self.notebook.select()
        if current_tab == str(self.history_tab):
            self._refresh_history_list(select_iid=self._selected_history_iid)

    def _refresh_history_list(self, select_iid: str | None = None) -> None:
        self._saved_sessions_map.clear()
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        cur_turns = len(self.runner.journey_records)
        if cur_turns == 0:
            cur_status = "暂无内容"
        elif self._current_session_saved:
            cur_status = "已保存"
        else:
            cur_status = "未保存"

        self.history_tree.insert(
            "",
            "end",
            iid="__CURRENT__",
            values=("⭐ 当前对话会话", f"{cur_turns} 轮", cur_status),
        )

        saved_items = list_saved_journeys(JOURNEYS_DIR)
        for idx, info in enumerate(saved_items):
            iid = f"saved_{idx}_{info['path'].name}"
            self._saved_sessions_map[iid] = info
            self.history_tree.insert(
                "",
                "end",
                iid=iid,
                values=(f"📄 {info['display_time']}", f"{info['total_turns']} 轮", "已保存"),
            )

        if cur_turns == 0:
            self.history_notice_var.set(f"当前暂无对话 ｜ 磁盘已有 {len(saved_items)} 条历史记录")
            self.history_notice_label.config(fg=MUTED_COLOR)
        elif self._current_session_saved and self._last_saved_filename:
            self.history_notice_var.set(f"✅ 当前对话已保存至 {self._last_saved_filename}")
            self.history_notice_label.config(fg="#137333")
        else:
            self.history_notice_var.set(f"当前对话共 {cur_turns} 轮（尚未保存）｜ 点击左侧按钮可保存")
            self.history_notice_label.config(fg="#B45309")

        target = select_iid if (select_iid and self.history_tree.exists(select_iid)) else "__CURRENT__"
        self._selected_history_iid = target
        self.history_tree.selection_set(target)
        self.history_tree.focus(target)
        self._render_history_detail(target)

    def _on_history_item_selected(self, event=None) -> None:
        sel = self.history_tree.selection()
        if not sel:
            return
        self._selected_history_iid = sel[0]
        self._render_history_detail(sel[0])

    def _render_history_detail(self, iid: str) -> None:
        if iid == "__CURRENT__":
            turns = list(self.runner.journey_records)
            status_txt = "已保存" if self._current_session_saved else ("未保存" if turns else "空会话")
            self.history_detail_title_var.set(f"⭐ 当前对话会话（{status_txt}）")
            self.history_detail_meta_var.set(
                f"累计轮次: {len(turns)} 轮 ｜ 当前需求模式: {self.runner.default_need.value}"
                + (f" ｜ 最近保存文件: {self._last_saved_filename}" if self._last_saved_filename else "")
            )
        else:
            info = self._saved_sessions_map.get(iid)
            if not info:
                return
            turns = info["turns"]
            self.history_detail_title_var.set(f"📄 历史会话 · {info['display_time']}")
            self.history_detail_meta_var.set(
                f"文件: {info['path'].name} ｜ 总轮次: {info['total_turns']} 轮 ｜ 默认需求: {info['default_need']}"
            )

        self.history_detail_text.config(state="normal")
        self.history_detail_text.delete("1.0", "end")

        if not turns:
            self.history_detail_text.insert(
                "end",
                "该会话暂无对话轮次。前往【💬 实时互动与对话】开始互动或输入文字测试后，记录会实时同步到这里。\n",
                "meta_line",
            )
            self.history_detail_text.config(state="disabled")
            return

        for idx, t in enumerate(turns, start=1):
            t_idx = int(t.get("turn_index", idx))
            ev_type = str(t.get("event_type", ""))
            if ev_type == TurnEventType.TEXT_INPUT.value:
                ev_str = "⌨️ 文字测试"
            elif t.get("user_initiated", False):
                ev_str = "🗣️ 语音发言"
            else:
                ev_str = "⏳ 静默观察"

            mode = str(t.get("policy_mode", "UNKNOWN")).upper()
            mode_zh = MODE_THEMES.get(mode, {}).get("zh", mode)
            raw_act = str(t.get("policy_action", "—"))
            zh_act = ACTION_LABELS.get(raw_act, raw_act)
            raw_int = str(t.get("policy_intensity", "—"))
            zh_int = INTENSITY_LABELS.get(raw_int, raw_int)
            style = str(t.get("epistemic_style", "—"))

            transcript = str(t.get("transcript", "[沉默无言语]"))
            reply_text = t.get("system_reply")

            self.history_detail_text.insert("end", f"【轮次 {t_idx:02d} ｜ {ev_str}】\n", "turn_hdr")

            if mode == "WAIT" and not t.get("user_initiated", False):
                self.history_detail_text.insert("end", "── 装置保持安静观察（静默留白，不主动打扰）──\n", "wait_line")
            else:
                if transcript and transcript != "[沉默无言语]":
                    self.history_detail_text.insert("end", f"你：{transcript}\n", "u_line")
                if reply_text:
                    self.history_detail_text.insert("end", f"机器人：{reply_text}\n", "b_line")
                else:
                    self.history_detail_text.insert("end", "机器人：（旧版记录未保存自然语言回复文本）\n", "wait_line")

            # 策略与多模态指标摘要（完全兼容旧版 JSON）
            need_det = str(t.get("llm_detected_need", "UNKNOWN"))
            need_conf = float(t.get("llm_need_confidence", 0.0) or 0.0)
            need_str = (
                f"{NEED_LABELS.get(need_det, need_det)}({need_conf:.0%})"
                if need_det and need_det != "UNKNOWN"
                else "未明说(情绪驱动)"
            )

            face_raw = str(t.get("face_dominant_emotion", "N/A"))
            aud_raw = str(t.get("audio_dominant_emotion", "N/A"))
            txt_raw = str(t.get("text_dominant_emotion", "N/A"))
            cov = t.get("data_coverage")
            cov_str = f"{float(cov):.0%}" if cov is not None else "—"
            dis = t.get("cross_modal_disagreement")
            dis_str = f"{float(dis):.3f}" if dis is not None else "0.000"
            ent = t.get("predictive_entropy")
            ent_str = f"{float(ent):.3f}" if ent is not None else "—"

            self.history_detail_text.insert(
                "end",
                f"  策略: {mode_zh} → {zh_act} ｜ 强度: {zh_int} ｜ 语气: {style} ｜ Need: {need_str}\n",
                "meta_line",
            )
            self.history_detail_text.insert(
                "end",
                f"  感知: 面部[{face_raw}] 声音[{aud_raw}] 文本[{txt_raw}] ｜ 覆盖率 {cov_str} ｜ 冲突 {dis_str} ｜ 熵 {ent_str}\n",
                "meta_line",
            )
            reasons = t.get("policy_reasons") or []
            if reasons:
                zh_r = "；".join(translate_reason(str(r)) for r in reasons[:3])
                self.history_detail_text.insert("end", f"  依据: {zh_r}\n", "meta_line")

            self.history_detail_text.insert("end", "─" * 58 + "\n", "divider")

        self.history_detail_text.config(state="disabled")

    def _save_current_session(self) -> None:
        if not self.runner.journey_records:
            self.history_notice_var.set("⚠️ 当前还没有对话轮次，无需保存")
            self.history_notice_label.config(fg="#B45309")
            self.status_bar_var.set("当前会话为空，暂无可保存的对话记录")
            return

        journey_path = save_journey_payload(
            turns=list(self.runner.journey_records),
            default_need=self.runner.default_need.value,
            journeys_dir=JOURNEYS_DIR,
        )
        self._current_session_saved = True
        self._last_saved_filename = journey_path.name
        self.status_bar_var.set(f"✅ 对话已保存至 {journey_path.name}")
        self._refresh_history_list(select_iid="__CURRENT__")

    def _clear_current_session(self) -> None:
        if not self.runner.journey_records:
            self.history_notice_var.set("当前对话本就是空的")
            self.history_notice_label.config(fg=MUTED_COLOR)
            return
        if not self._current_session_saved:
            ok = messagebox.askyesno(
                "清空当前未保存对话",
                f"当前对话共有 {len(self.runner.journey_records)} 轮且尚未保存。\n确定要清空并开启全新会话吗？",
                default="no",
            )
            if not ok:
                return
        self._reset_current_session_ui()
        self.status_bar_var.set("已清空当前会话，可以开始新的对话")
        self._refresh_history_list(select_iid="__CURRENT__")

    def _reset_current_session_ui(self) -> None:
        self.runner.reset_session()
        self._current_session_saved = False
        self._last_saved_filename = None
        self._reset_chat_view()
        self.mode_title_var.set(MODE_THEMES["IDLE"]["zh"])
        self.mode_badge.config(bg=MODE_THEMES["IDLE"]["bg"], fg=MODE_THEMES["IDLE"]["fg"])
        self.action_var.set("尚未生成动作")
        self.intensity_style_var.set("强度: — ｜ 语气: — ｜ 需求: 自动感知")
        self.transcript_var.set("等待用户开口说话或输入文字...")
        self.fused_summary_var.set("面部: — ｜ 声音: — ｜ 文本: — ｜ 覆盖率: — ｜ 冲突: — ｜ 熵: —")
        self._set_rationale_text("启动互动或在下方输入文字后，系统将在此解析三模态线索、Need 识别与策略推导逻辑...")

    def _delete_selected_history(self) -> None:
        sel = self.history_tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid == "__CURRENT__":
            self.history_notice_var.set("提示：如需重置当前会话，请点击【🧹 清空当前对话】")
            self.history_notice_label.config(fg=MUTED_COLOR)
            return
        info = self._saved_sessions_map.get(iid)
        if not info:
            return
        ok = messagebox.askyesno(
            "删除历史记录",
            f"确定删除历史对话文件吗？\n\n{info['path'].name}（共 {info['total_turns']} 轮）\n删除后无法恢复。",
            default="no",
        )
        if not ok:
            return
        try:
            info["path"].unlink(missing_ok=True)
            self.status_bar_var.set(f"已删除历史记录: {info['path'].name}")
        except Exception as exc:
            self.status_bar_var.set(f"删除失败: {exc}")
        self._refresh_history_list(select_iid="__CURRENT__")

    # -------------------------------------------------------------
    # 交互调度与线程管理
    # -------------------------------------------------------------
    def _on_need_changed(self, event=None) -> None:
        raw_val = self.need_combo.get()
        need_code = raw_val.split("(")[-1].replace(")", "").strip()
        try:
            selected = UserNeed[need_code]
            self.runner.default_need = selected
            if self.runner.history is not None:
                pref = None if selected is UserNeed.UNKNOWN else selected
                self.runner.history = replace(self.runner.history, preferred_need=pref)
            label = "自动感知·情绪驱动" if selected is UserNeed.UNKNOWN else NEED_LABELS.get(need_code, need_code)
            self.status_bar_var.set(f"需求模式已切换为: {label}")
        except Exception:
            pass

    def _on_thresh_slider_changed(self, val: str) -> None:
        new_val = float(val)
        self.runner.vad_tracker.energy_threshold = new_val
        if hasattr(self, "thresh_label"):
            self.thresh_label.config(text=f"VAD 拾音门限: {new_val:.3f}")

    def _trigger_manual_speech_test(self) -> None:
        """Trigger a 3.5s manual audio capture on the main thread using root.after."""
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
        self._manual_recording_active = False
        self.manual_record_btn.config(text=" 🎙️ 点按试音(3秒) ", state="normal", bg="#E0E7FF", fg="#3730A3")
        if not self.is_running:
            return
        try:
            import tempfile
            import wave

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
            self.status_bar_var.set("⏳ 试音采集完成，正在转写识别与策略生成中...")
        except Exception as e:
            self.status_bar_var.set(f"测试录音异常: {e}")

    def _toggle_interaction(self) -> None:
        if not self.is_running:
            self._start_session()
        else:
            self._stop_session()

    def _start_session(self) -> None:
        # 若上一次会话已经保存过，重新开启互动时自动开启干净的新会话
        if self._current_session_saved and self.runner.journey_records:
            self._reset_current_session_ui()

        self.is_running = True
        self.stop_event.clear()
        self.runner.vad_tracker.reset_turn_timer()
        self.toggle_btn.config(text=" ⏹ 停止互动 ", bg=DANGER_COLOR, activebackground="#B91C1C")
        self.status_bar_var.set("互动运行中 ｜ 正在后台预热本地语音模型并监听...")
        self.vad_status_var.set("● 麦克风已就绪 (正在倾听...)")
        self.vad_indicator.config(bg="#E6F4EA", fg="#137333")

        def _warmup_worker() -> None:
            try:
                self.runner.audio_detector.warmup()
            except Exception:
                pass

        threading.Thread(target=_warmup_worker, daemon=True).start()

        self.camera_cap, _ = find_working_camera()
        if not self.camera_cap.isOpened():
            self.video_label.config(
                text="⚠️ 摄像头未授权或无法打开\n(可在系统设置中允许终端/Python 访问摄像头)",
                bg="#1E293B",
                fg="#F87171",
            )
        else:
            threading.Thread(target=self._camera_loop, daemon=True).start()

        self._start_audio_stream()

    def _stop_session(self, closing: bool = False) -> None:
        self.is_running = False
        self.stop_event.set()
        if not closing:
            self.toggle_btn.config(text=" ▶ 开启实时互动 ", bg=PRIMARY_COLOR, activebackground=PRIMARY_HOVER)
            self.vad_status_var.set("● 装置待机中 (已停止)")
            self.vad_indicator.config(bg="#F1F5F9", fg="#475569")

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

        if not closing:
            self.video_label.config(
                image="",
                text="摄像头已停止\n点击右上角【开启实时互动】重新唤醒",
                bg="#0F172A",
                fg="#94A3B8",
            )
            total = len(self.runner.journey_records)
            if total > 0 and not self._current_session_saved:
                self.status_bar_var.set(
                    f"互动已停止（共 {total} 轮，未保存）｜ 可前往【📚 历史对话与保存】分页查看并点击保存"
                )
            elif total > 0:
                self.status_bar_var.set(f"互动已停止（共 {total} 轮，已保存）")
            else:
                self.status_bar_var.set("互动已停止 ｜ 仍可直接在右下角打字测试")
            self._refresh_history_list(select_iid=self._selected_history_iid)

    def _on_window_close(self) -> None:
        self._app_closing.set()
        self._stop_session(closing=True)
        self.root.destroy()

    # -------------------------------------------------------------
    # 硬件驱动与常驻事件循环
    # -------------------------------------------------------------
    def _camera_loop(self) -> None:
        while not self.stop_event.is_set() and not self._app_closing.is_set():
            if self.camera_cap is None:
                break
            ret, frame = self.camera_cap.read()
            if not ret or frame is None:
                time.sleep(0.04)
                continue

            try:
                rec = self.runner.visual_detector.analyze_image(frame)
                self.runner.visual_buffer.add_record(rec)

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

            time.sleep(0.035)

    def _start_audio_stream(self) -> None:
        try:
            import sounddevice as sd

            def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
                if self.stop_event.is_set():
                    return
                chunk = indata[:, 0].copy()
                rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size > 0 else 0.0
                self.current_audio_rms = rms
                if self._manual_recording_active:
                    self._manual_capture_chunks.append(chunk)
                    return
                ev = self.runner.vad_tracker.process_chunk(chunk)
                if ev is not None:
                    self.event_queue.put(ev)

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
        while not self._app_closing.is_set():
            try:
                event = self.event_queue.get(timeout=0.25)
                turn_data = self.runner.handle_turn_event(event, manual_need=self.runner.default_need)
                if turn_data is not None:
                    self.turn_result_queue.put(turn_data)
                elif event.event_type == TurnEventType.TEXT_INPUT:
                    self.turn_result_queue.put({"_text_input_done": True})
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[Error in turn loop] {e}")
                self.turn_result_queue.put({"_text_input_done": True, "_error": str(e)})

    # -------------------------------------------------------------
    # UI 刷新主循环 (Tkinter 主线程安全轮询)
    # -------------------------------------------------------------
    def _start_ui_loop(self) -> None:
        self._update_tick()

    def _update_tick(self) -> None:
        if self._app_closing.is_set():
            return

        # 1. 刷新摄像头画面与人脸标注
        if self.is_running and self.latest_frame is not None:
            try:
                display_frame = self.latest_frame.copy()
                if self.latest_face_box is not None:
                    bx, by, bw, bh = self.latest_face_box
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

        # 2. 刷新麦克风 VAD 状态
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
                    self.vad_status_var.set(
                        f"⏳ 静默留白观察中 (倒计时: {remain:.1f}s / {self.runner.vad_tracker.idle_timeout_seconds:.0f}s) [环境音: {rms_val:.3f}]"
                    )
                    self.vad_indicator.config(bg="#FEF3C7", fg="#92400E")
                else:
                    self.vad_status_var.set(f"● 麦克风已就绪 (倾听中... [音量: {vol_meter} {rms_val:.3f}])")
                    self.vad_indicator.config(bg="#E6F4EA", fg="#137333")

        # 3. 消费已完成轮次并渲染
        try:
            while True:
                turn_data = self.turn_result_queue.get_nowait()
                if turn_data.get("_text_input_done"):
                    self._restore_send_button()
                    if turn_data.get("_error"):
                        self.status_bar_var.set(f"处理异常: {turn_data['_error']}")
                else:
                    self._render_turn_to_ui(turn_data)
        except queue.Empty:
            pass

        self.root.after(35, self._update_tick)

    def _restore_send_button(self) -> None:
        self._pending_text_requests = max(0, self._pending_text_requests - 1)
        if self._pending_text_requests == 0:
            self.send_text_btn.config(text=" 发送测试 ", state="normal", bg=PRIMARY_COLOR)

    def _render_turn_to_ui(self, data: dict) -> None:
        if data.get("event_type") == TurnEventType.TEXT_INPUT.value:
            self._restore_send_button()

        # 新轮次产生，标记当前会话有未保存变更
        self._current_session_saved = False

        turn_idx = data["turn_index"]
        mode = data["policy_mode"]
        theme = MODE_THEMES.get(mode, MODE_THEMES["DIRECT"])
        self.mode_theme_key = mode

        # 1. 更新决策徽章与核心指标
        self.mode_title_var.set(f"轮次 [ {turn_idx:02d} ] · {theme['zh']}")
        self.mode_badge.config(bg=theme["bg"], fg=theme["fg"], highlightbackground=theme["border"])

        raw_act = data["policy_action"]
        zh_act = ACTION_LABELS.get(raw_act, raw_act)
        raw_int = data["policy_intensity"]
        zh_int = INTENSITY_LABELS.get(raw_int, raw_int)
        need_det = data.get("llm_detected_need", "UNKNOWN")
        need_conf = data.get("llm_need_confidence", 0.0)
        need_short = (
            f"{NEED_LABELS.get(need_det, need_det)} ({need_conf:.0%})"
            if need_det and need_det != "UNKNOWN"
            else "自动感知(情绪驱动)"
        )

        self.action_var.set(f"{zh_act} ({raw_act})")
        self.intensity_style_var.set(
            f"强度: 【{zh_int}】 ｜ 语气: 【{data['epistemic_style']}】 ｜ 需求: 【{need_short}】"
        )

        # 2. 更新左侧最新字幕与模态融合快照
        self.transcript_var.set(data["transcript"])
        disagree_val = f"{data['cross_modal_disagreement']:.4f}" if data["cross_modal_disagreement"] is not None else "0.0000"

        face_raw = data["face_dominant_emotion"]
        zh_face = EMOTION_LABELS.get(face_raw.lower(), face_raw)
        face_str = f"{zh_face} ({face_raw})" if face_raw != "N/A" else "未检出人脸"

        aud_raw = data["audio_dominant_emotion"]
        zh_aud = EMOTION_LABELS.get(aud_raw.lower(), aud_raw)
        aud_str = f"{zh_aud} ({aud_raw})" if aud_raw != "N/A" else "无音频输入"

        txt_raw = data.get("text_dominant_emotion", "N/A")
        zh_txt = EMOTION_LABELS.get(txt_raw.lower(), txt_raw)
        txt_str = f"{zh_txt} ({txt_raw})" if txt_raw != "N/A" else "无文本语义"

        top_fused = data.get("fused_dominant_emotion", "UNKNOWN")
        zh_fused = EMOTION_LABELS.get(str(top_fused).lower(), str(top_fused))

        cov_val = data.get("data_coverage", 0.0)
        instab_val = data.get("temporal_instability", 0.0)
        eff_map = {"improved": "情绪改善", "unchanged": "保持平稳", "worsened": "情绪变差", "unknown": "首轮/待观察"}
        eff_str = eff_map.get(data.get("observed_feedback_effect", "unknown"), "待观察")

        self.fused_summary_var.set(
            f"面部: {face_str} ｜ 声音: {aud_str} ｜ 文本: {txt_str} ｜ 覆盖率: {cov_val:.0%} ｜ 冲突: {disagree_val} ｜ 熵: {data['predictive_entropy']:.3f}"
        )

        # 3. 更新策略依据框
        need_reason = data.get("llm_need_reason", "")
        need_note = (
            f"{NEED_LABELS.get(need_det, need_det)} ({need_det}, 置信度 {need_conf:.0%})"
            + (f" — {need_reason}" if need_reason else "")
            if need_det and need_det != "UNKNOWN"
            else "未明说具体诉求（交由情绪驱动）"
        )
        zh_reasons = [
            f"• 【三模态线索】面部[{face_str}](置信度 {data['face_signal_quality']:.0%}) ｜ 声音[{aud_str}] ｜ 文本[{txt_str}]",
            f"• 【需求识别】{need_note}",
            f"• 【多源不确定性】覆盖率 {cov_val:.0%} ｜ 跨模态冲突 {disagree_val} ｜ 时序波动 {instab_val:.3f} ｜ 预测熵 {data['predictive_entropy']:.3f} → 综合倾向【{zh_fused}】",
            f"• 【会话反馈追踪】上轮效价反馈：{eff_str}",
        ]
        for r in data["policy_reasons"]:
            zh_reasons.append(f"• 【策略推理逻辑】{translate_reason(r)}")
        if data["cross_modal_disagreement"] and data["cross_modal_disagreement"] > 0.35:
            zh_reasons.insert(3, f"• ⚠️ 【冲突应对机制】检测到显著跨模态冲突 (冲突度 {disagree_val})，策略采取审慎试探介入")
        self._set_rationale_text("\n".join(zh_reasons))

        # 4. 追加到实时对话流
        self._append_turn_to_chat(data)

        # 5. 同步刷新历史分页里的“当前会话”计数与预览
        self._refresh_history_list(select_iid=self._selected_history_iid)
        self.status_bar_var.set(f"第 {turn_idx} 轮已完成 ｜ 共 {len(self.runner.journey_records)} 轮（未保存）")


def main() -> None:
    root = tk.Tk()
    app = StrategyAppV4(root)
    root.protocol("WM_DELETE_WINDOW", app._on_window_close)
    root.mainloop()


if __name__ == "__main__":
    main()
