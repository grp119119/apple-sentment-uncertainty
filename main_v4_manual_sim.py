"""Chinese manual-simulation interface for Adaptive Policy v4.0."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from affective_v03.history import update_history
from affective_v03.models import (
    AffectLabel,
    HistoryEffect,
    HistoryRecord,
    InteractionStage,
    PolicyContext,
    Privacy,
    TimePressure,
    UserNeed,
)
from affective_v04.labels_zh import (
    ACTION_LABELS,
    EMOTION_LABELS,
    INTENSITY_LABELS,
    MODE_LABELS,
    NEED_LABELS,
    translate_reason,
)
from affective_v04.simulation import ManualSimulationInput, ManualSimulationResult, run_manual_simulation


BG = "#F4F7FB"
SURFACE = "#FFFFFF"
TEXT = "#182230"
MUTED = "#667085"
BORDER = "#D9E1EC"
PRIMARY = "#315EFB"
PRIMARY_DARK = "#2449D8"
SUCCESS = "#147D64"
WARNING = "#B55A13"
MODE_COLORS = {
    "DIRECT": "#147D64",
    "TENTATIVE": "#8B5CF6",
    "CONFIRM": "#315EFB",
    "WAIT": "#B55A13",
    "DEFER": "#B42318",
}

PRIVACY_VALUES = {"私密环境": Privacy.PRIVATE, "公共环境": Privacy.PUBLIC}
TIME_VALUES = {"低时间压力": TimePressure.LOW, "高时间压力": TimePressure.HIGH}
STAGE_VALUES = {
    "开始阶段": InteractionStage.OPENING,
    "探索阶段": InteractionStage.EXPLORING,
    "支持阶段": InteractionStage.SUPPORTING,
    "结束阶段": InteractionStage.CLOSING,
}
NEED_VALUES = {label: UserNeed(value) for value, label in NEED_LABELS.items()}
EFFECT_VALUES = {
    "有所改善": HistoryEffect.IMPROVED,
    "没有变化": HistoryEffect.UNCHANGED,
    "感觉变差": HistoryEffect.WORSENED,
    "暂不评价": HistoryEffect.UNKNOWN,
}


class StrategyAppV4:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.current_result: ManualSimulationResult | None = None
        self.history = HistoryRecord()
        self.probability_vars = {label.value: tk.StringVar() for label in AffectLabel}
        self.total_var = tk.StringVar(value="概率总和  1.00")
        self.valence_var = tk.StringVar(value="-0.45")
        self.arousal_var = tk.StringVar(value="0.62")
        self.fatigue_var = tk.StringVar(value="0.20")
        self.signal_quality_var = tk.StringVar(value="1.00")
        self.coverage_var = tk.StringVar(value="1.00")
        self.privacy_var = tk.StringVar(value="私密环境")
        self.time_pressure_var = tk.StringVar(value="低时间压力")
        self.stage_var = tk.StringVar(value="探索阶段")
        self.user_initiated_var = tk.BooleanVar(value=True)
        self.receptive_var = tk.BooleanVar(value=True)
        self.safety_var = tk.BooleanVar(value=True)
        self.need_var = tk.StringVar(value="倾听")
        self.user_text_var = tk.StringVar(value="我现在只想先说一说。")
        self.feedback_effect_var = tk.StringVar(value="没有变化")
        self.feedback_accepted_var = tk.BooleanVar(value=True)
        self.history_var = tk.StringVar(value="暂无历史反馈")
        self.mode_var = tk.StringVar(value="等待运行")
        self.mode_code_var = tk.StringVar(value="输入模拟数据后生成决策")
        self.dominant_var = tk.StringVar(value="—")
        self.entropy_var = tk.StringVar(value="—")
        self.margin_var = tk.StringVar(value="—")
        self.intensity_var = tk.StringVar(value="—")
        self.action_var = tk.StringVar(value="—")
        self.quality_var = tk.StringVar(value="—")
        self._configure_window()
        self._build_interface()
        self.load_example()

    def _configure_window(self) -> None:
        self.root.title("情感计算策略测试面板 · v4.1")
        self.root.geometry("1280x900")
        self.root.minsize(1100, 780)
        self.root.configure(bg=BG)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", padding=6, font=("Microsoft YaHei UI", 10))
        style.configure("TCheckbutton", background=SURFACE, font=("Microsoft YaHei UI", 10))
        style.map("TCheckbutton", background=[("active", SURFACE)])
        style.configure("Primary.TButton", font=("Microsoft YaHei UI", 11, "bold"), padding=(18, 10))
        style.map(
            "Primary.TButton",
            background=[("!disabled", PRIMARY), ("active", PRIMARY_DARK)],
            foreground=[("!disabled", "white")],
        )
        style.configure("Quiet.TButton", font=("Microsoft YaHei UI", 10), padding=(12, 8))

    def _build_interface(self) -> None:
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=28, pady=22)
        self._build_header(outer)
        body = tk.Frame(outer, bg=BG)
        body.pack(fill="both", expand=True, pady=(18, 0))
        body.grid_columnconfigure(0, weight=5, uniform="body")
        body.grid_columnconfigure(1, weight=6, uniform="body")
        body.grid_rowconfigure(0, weight=1)
        left = tk.Frame(body, bg=BG)
        right = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        right.grid(row=0, column=1, sticky="nsew", padx=(9, 0))
        self._build_input_panel(left)
        self._build_result_panel(right)

    def _build_header(self, parent: tk.Widget) -> None:
        header = tk.Frame(parent, bg=BG)
        header.pack(fill="x")
        title_box = tk.Frame(header, bg=BG)
        title_box.pack(side="left")
        tk.Label(
            title_box,
            text="情感计算策略测试面板",
            bg=BG,
            fg=TEXT,
            font=("Microsoft YaHei UI", 22, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="手动情绪与需求输入 · History反馈闭环 · Adaptive Policy v4.1",
            bg=BG,
            fg=MUTED,
            font=("Microsoft YaHei UI", 10),
        ).pack(anchor="w", pady=(4, 0))
        badge = tk.Label(
            header,
            text="仅限演示 · 非研究数据",
            bg="#E8EEFF",
            fg=PRIMARY_DARK,
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=12,
            pady=7,
        )
        badge.pack(side="right", anchor="n")

    @staticmethod
    def _card(parent: tk.Widget) -> tk.Frame:
        card = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", pady=(0, 12))
        return card

    @staticmethod
    def _card_title(parent: tk.Widget, title: str, subtitle: str = "") -> tk.Frame:
        box = tk.Frame(parent, bg=SURFACE)
        box.pack(fill="x", padx=18, pady=(15, 10))
        tk.Label(box, text=title, bg=SURFACE, fg=TEXT, font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        if subtitle:
            tk.Label(box, text=subtitle, bg=SURFACE, fg=MUTED, font=("Microsoft YaHei UI", 9)).pack(anchor="w", pady=(3, 0))
        return box

    def _build_input_panel(self, parent: tk.Widget) -> None:
        actions = tk.Frame(parent, bg=BG)
        actions.pack(fill="x", side="bottom", pady=(4, 0))
        ttk.Button(
            actions,
            text="运行识别与策略决策",
            style="Primary.TButton",
            command=self.run_decision,
        ).pack(side="left", fill="x", expand=True)

        emotions = self._card(parent)
        title_box = self._card_title(emotions, "01  情绪概率分布", "七类概率之和必须等于 1.00；疲惫作为辅助状态单独输入")
        self.total_label = tk.Label(
            title_box,
            textvariable=self.total_var,
            bg="#EAF8F3",
            fg=SUCCESS,
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=9,
            pady=4,
        )
        self.total_label.place(relx=1.0, y=0, anchor="ne")
        grid = tk.Frame(emotions, bg=SURFACE)
        grid.pack(fill="x", padx=18, pady=(0, 14))
        for index, label in enumerate(AffectLabel):
            row, column = divmod(index, 2)
            cell = tk.Frame(grid, bg=SURFACE)
            cell.grid(row=row, column=column, sticky="ew", padx=(0, 18) if column == 0 else (0, 0), pady=4)
            grid.grid_columnconfigure(column, weight=1)
            tk.Label(
                cell,
                text=f"{EMOTION_LABELS[label.value]}  {label.value}",
                bg=SURFACE,
                fg=TEXT,
                font=("Microsoft YaHei UI", 9),
                width=15,
                anchor="w",
            ).pack(side="left")
            entry = ttk.Entry(cell, textvariable=self.probability_vars[label.value], width=8, justify="center")
            entry.pack(side="right", fill="x", expand=True)
            entry.bind("<KeyRelease>", self._update_total)
        controls = tk.Frame(emotions, bg=SURFACE)
        controls.pack(fill="x", padx=18, pady=(0, 15))
        ttk.Button(controls, text="载入示例", style="Quiet.TButton", command=self.load_example).pack(side="left")
        ttk.Button(controls, text="自动归一化", style="Quiet.TButton", command=self.normalize_distribution).pack(side="left", padx=8)
        ttk.Button(controls, text="清空", style="Quiet.TButton", command=self.clear_distribution).pack(side="left")

        state = self._card(parent)
        self._card_title(state, "02  状态与证据", "基础状态用于计算强度；证据质量用于判断是否具备可靠输入")
        row = tk.Frame(state, bg=SURFACE)
        row.pack(fill="x", padx=18, pady=(0, 15))
        self._labeled_entry(row, "效价  -1 至 1", self.valence_var)
        self._labeled_entry(row, "唤醒度  0 至 1", self.arousal_var)
        self._labeled_entry(row, "疲惫程度  0 至 1", self.fatigue_var, note="仅记录")
        self._labeled_entry(row, "信号质量  0 至 1", self.signal_quality_var)
        self._labeled_entry(row, "数据覆盖  0 至 1", self.coverage_var, last=True)

        context = self._card(parent)
        self._card_title(context, "03  交互情境")
        selectors = tk.Frame(context, bg=SURFACE)
        selectors.pack(fill="x", padx=18)
        self._labeled_combo(selectors, "隐私环境", self.privacy_var, tuple(PRIVACY_VALUES))
        self._labeled_combo(selectors, "时间压力", self.time_pressure_var, tuple(TIME_VALUES))
        self._labeled_combo(selectors, "交互阶段", self.stage_var, tuple(STAGE_VALUES), last=True)
        flags = tk.Frame(context, bg=SURFACE)
        flags.pack(fill="x", padx=18, pady=(12, 15))
        ttk.Checkbutton(flags, text="用户主动发起", variable=self.user_initiated_var).pack(side="left")
        ttk.Checkbutton(flags, text="愿意接受介入", variable=self.receptive_var).pack(side="left", padx=18)
        ttk.Checkbutton(flags, text="安全边界允许", variable=self.safety_var).pack(side="left")
        need_row = tk.Frame(context, bg=SURFACE)
        need_row.pack(fill="x", padx=18, pady=(0, 15))
        self._labeled_combo(need_row, "当前需求（手动选择）", self.need_var, tuple(NEED_VALUES))
        self._labeled_entry(need_row, "用户原始文字（仅记录，不自动识别）", self.user_text_var, last=True)

    def _labeled_entry(self, parent: tk.Widget, label: str, variable: tk.StringVar, note: str = "", last: bool = False) -> None:
        frame = tk.Frame(parent, bg=SURFACE)
        frame.pack(side="left", fill="x", expand=True, padx=(0, 0 if last else 10))
        caption = label if not note else f"{label} · {note}"
        tk.Label(frame, text=caption, bg=SURFACE, fg=MUTED, font=("Microsoft YaHei UI", 8), anchor="w").pack(fill="x")
        ttk.Entry(frame, textvariable=variable, justify="center", width=8).pack(fill="x", pady=(4, 0))

    def _labeled_combo(self, parent: tk.Widget, label: str, variable: tk.StringVar, values: tuple[str, ...], last: bool = False) -> None:
        frame = tk.Frame(parent, bg=SURFACE)
        frame.pack(side="left", fill="x", expand=True, padx=(0, 0 if last else 10))
        tk.Label(frame, text=label, bg=SURFACE, fg=MUTED, font=("Microsoft YaHei UI", 8), anchor="w").pack(fill="x")
        ttk.Combobox(frame, textvariable=variable, values=values, state="readonly").pack(fill="x", pady=(4, 0))

    def _build_result_panel(self, parent: tk.Widget) -> None:
        result = self._card(parent)
        result.pack(fill="both", expand=True, pady=0)
        top = tk.Frame(result, bg=SURFACE)
        top.pack(fill="x", padx=20, pady=(18, 0))
        tk.Label(top, text="策略决策结果", bg=SURFACE, fg=MUTED, font=("Microsoft YaHei UI", 10)).pack(anchor="w")
        self.mode_banner = tk.Frame(result, bg="#EEF2F7")
        self.mode_banner.pack(fill="x", padx=20, pady=(10, 16))
        self.mode_title = tk.Label(
            self.mode_banner,
            textvariable=self.mode_var,
            bg="#EEF2F7",
            fg=TEXT,
            font=("Microsoft YaHei UI", 24, "bold"),
        )
        self.mode_title.pack(anchor="w", padx=18, pady=(14, 2))
        self.mode_code = tk.Label(
            self.mode_banner,
            textvariable=self.mode_code_var,
            bg="#EEF2F7",
            fg=MUTED,
            font=("Microsoft YaHei UI", 10),
        )
        self.mode_code.pack(anchor="w", padx=18, pady=(0, 14))

        stats = tk.Frame(result, bg=SURFACE)
        stats.pack(fill="x", padx=20)
        stat_items = (
            ("主导情绪", self.dominant_var),
            ("预测熵", self.entropy_var),
            ("概率差", self.margin_var),
            ("介入强度", self.intensity_var),
            ("支持动作", self.action_var),
            ("证据质量", self.quality_var),
        )
        for index, (label, variable) in enumerate(stat_items):
            row, column = divmod(index, 3)
            tile = tk.Frame(stats, bg="#F7F9FC", highlightbackground="#E7ECF3", highlightthickness=1)
            tile.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
            stats.grid_columnconfigure(column, weight=1, uniform="stats")
            tk.Label(tile, text=label, bg="#F7F9FC", fg=MUTED, font=("Microsoft YaHei UI", 8)).pack(anchor="w", padx=12, pady=(9, 2))
            tk.Label(tile, textvariable=variable, bg="#F7F9FC", fg=TEXT, font=("Microsoft YaHei UI", 11, "bold"), wraplength=135, justify="left").pack(anchor="w", padx=12, pady=(0, 10))

        tk.Label(result, text="决策理由", bg=SURFACE, fg=TEXT, font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", padx=20, pady=(18, 7))
        self.reason_text = tk.Text(
            result,
            height=5,
            wrap="word",
            bg="#F7F9FC",
            fg=TEXT,
            relief="flat",
            font=("Microsoft YaHei UI", 10),
            padx=12,
            pady=10,
            spacing1=3,
            spacing3=3,
        )
        self.reason_text.pack(fill="x", padx=20)
        self.reason_text.insert("1.0", "运行后，这里会用中文说明触发的判断条件。")
        self.reason_text.configure(state="disabled")

        tk.Label(result, text="模拟回应", bg=SURFACE, fg=TEXT, font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", padx=20, pady=(16, 7))
        self.response_label = tk.Label(
            result,
            text="—",
            bg="#EEF3FF",
            fg=PRIMARY_DARK,
            font=("Microsoft YaHei UI", 11),
            justify="left",
            anchor="w",
            wraplength=570,
            padx=14,
            pady=12,
        )
        self.response_label.pack(fill="x", padx=20, pady=(0, 12))

        feedback = tk.Frame(result, bg="#F7F9FC", highlightbackground="#E7ECF3", highlightthickness=1)
        feedback.pack(fill="x", padx=20, pady=(0, 12))
        feedback_top = tk.Frame(feedback, bg="#F7F9FC")
        feedback_top.pack(fill="x", padx=12, pady=(8, 4))
        tk.Label(
            feedback_top,
            text="本轮反馈与下一轮History",
            bg="#F7F9FC",
            fg=TEXT,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side="left")
        feedback_controls = tk.Frame(feedback_top, bg="#F7F9FC")
        feedback_controls.pack(side="right")
        ttk.Combobox(
            feedback_controls,
            textvariable=self.feedback_effect_var,
            values=tuple(EFFECT_VALUES),
            state="readonly",
            width=9,
        ).pack(side="left")
        ttk.Checkbutton(
            feedback_controls,
            text="接受",
            variable=self.feedback_accepted_var,
        ).pack(side="left", padx=6)
        ttk.Button(
            feedback_controls,
            text="记录反馈",
            style="Quiet.TButton",
            command=self.record_feedback,
        ).pack(side="left")
        ttk.Button(
            feedback_controls,
            text="重置历史",
            style="Quiet.TButton",
            command=self.reset_history,
        ).pack(side="left", padx=(6, 0))
        tk.Label(
            feedback,
            textvariable=self.history_var,
            bg="#F7F9FC",
            fg=MUTED,
            font=("Microsoft YaHei UI", 8),
            justify="left",
            anchor="w",
            wraplength=560,
        ).pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(
            result,
            text="当前阈值为工程演示值，结果不能作为研究结论；疲惫程度仅记录，不参与本轮决策。",
            bg=SURFACE,
            fg=MUTED,
            font=("Microsoft YaHei UI", 8),
            wraplength=600,
            justify="left",
        ).pack(anchor="w", padx=20, pady=(0, 16))

    def load_example(self) -> None:
        example = {label.value: 0.0 for label in AffectLabel}
        example.update({"sadness": 0.60, "fear": 0.20, "neutral": 0.20})
        for label, value in example.items():
            self.probability_vars[label].set(f"{value:.2f}")
        self.valence_var.set("-0.45")
        self.arousal_var.set("0.62")
        self.fatigue_var.set("0.20")
        self.signal_quality_var.set("1.00")
        self.coverage_var.set("1.00")
        self._update_total()

    def clear_distribution(self) -> None:
        for variable in self.probability_vars.values():
            variable.set("0.00")
        self._update_total()

    def normalize_distribution(self) -> None:
        try:
            values = {label: self._read_float(variable.get(), EMOTION_LABELS[label]) for label, variable in self.probability_vars.items()}
            total = sum(values.values())
            if total <= 0:
                raise ValueError("至少输入一个大于 0 的情绪数值")
            for label, value in values.items():
                self.probability_vars[label].set(f"{value / total:.4f}")
        except ValueError as error:
            messagebox.showerror("无法归一化", str(error))
            return
        self._update_total()

    def _update_total(self, _event=None) -> None:
        try:
            total = sum(float(variable.get() or 0) for variable in self.probability_vars.values())
        except ValueError:
            self.total_var.set("概率输入有误")
            self.total_label.configure(bg="#FEECEB", fg="#B42318")
            return
        valid = abs(total - 1.0) <= 1e-6
        self.total_var.set(f"概率总和  {total:.2f}")
        self.total_label.configure(bg="#EAF8F3" if valid else "#FFF3E8", fg=SUCCESS if valid else WARNING)

    @staticmethod
    def _read_float(raw: str, label: str) -> float:
        try:
            return float(raw)
        except ValueError as error:
            raise ValueError(f"{label} 必须是数字") from error

    def run_decision(self) -> None:
        try:
            distribution = {
                label: self._read_float(variable.get(), EMOTION_LABELS[label])
                for label, variable in self.probability_vars.items()
            }
            context = PolicyContext(
                privacy=PRIVACY_VALUES[self.privacy_var.get()],
                time_pressure=TIME_VALUES[self.time_pressure_var.get()],
                interaction_stage=STAGE_VALUES[self.stage_var.get()],
                receptive=self.receptive_var.get(),
                user_initiated=self.user_initiated_var.get(),
                safety_allows_intervention=self.safety_var.get(),
                user_need=NEED_VALUES[self.need_var.get()],
                history=self.history,
            )
            data = ManualSimulationInput(
                emotion_distribution=distribution,
                valence=self._read_float(self.valence_var.get(), "效价"),
                arousal=self._read_float(self.arousal_var.get(), "唤醒度"),
                fatigue=self._read_float(self.fatigue_var.get(), "疲惫程度"),
                signal_quality=self._read_float(self.signal_quality_var.get(), "信号质量"),
                data_coverage=self._read_float(self.coverage_var.get(), "数据覆盖率"),
                context=context,
                user_text=self.user_text_var.get(),
            )
            result = run_manual_simulation(data)
        except (KeyError, TypeError, ValueError) as error:
            messagebox.showerror("无法运行策略", str(error))
            return
        self.current_result = result
        self._show_result(result)

    def _show_result(self, result: ManualSimulationResult) -> None:
        decision = result.decision
        affect = result.affect
        uncertainty = result.uncertainty
        mode = decision.mode.value
        color = MODE_COLORS[mode]
        self.mode_banner.configure(bg=color)
        self.mode_title.configure(bg=color, fg="white")
        self.mode_code.configure(bg=color, fg="#EEF3FF")
        self.mode_var.set(MODE_LABELS[mode])
        self.mode_code_var.set(f"{mode} · Adaptive Policy v4.1")
        dominant = affect.dominant_emotion.value
        self.dominant_var.set(f"{EMOTION_LABELS[dominant]}  {affect.distribution[affect.dominant_emotion]:.2f}")
        self.entropy_var.set(f"{uncertainty.predictive_entropy:.3f}")
        self.margin_var.set(f"{uncertainty.top_two_margin:.3f}")
        self.intensity_var.set(INTENSITY_LABELS[decision.intensity.value])
        self.action_var.set(ACTION_LABELS[decision.support_action.value])
        self.quality_var.set(f"质量 {uncertainty.signal_quality:.2f} · 覆盖 {uncertainty.data_coverage:.2f}")
        reasons = "\n".join(f"• {translate_reason(reason)}" for reason in decision.reasons)
        self.reason_text.configure(state="normal")
        self.reason_text.delete("1.0", tk.END)
        self.reason_text.insert("1.0", reasons)
        self.reason_text.configure(state="disabled")
        self.response_label.configure(text=result.response)

    def record_feedback(self) -> None:
        if self.current_result is None:
            messagebox.showinfo("尚无本轮结果", "请先运行一次策略决策，再记录反馈。")
            return
        self.history = update_history(
            self.history,
            self.current_result.decision,
            self.current_result.user_need,
            EFFECT_VALUES[self.feedback_effect_var.get()],
            accepted=self.feedback_accepted_var.get(),
        )
        self._show_history()

    def reset_history(self) -> None:
        self.history = HistoryRecord()
        self.history_var.set("暂无历史反馈")

    def _show_history(self) -> None:
        action = self.history.previous_action
        rejected = "、".join(ACTION_LABELS[item.value] for item in self.history.rejected_actions) or "无"
        preferred = NEED_LABELS[self.history.preferred_need.value] if self.history.preferred_need else "未记录"
        self.history_var.set(
            f"上一动作：{ACTION_LABELS[action.value] if action else '无'} · "
            f"效果：{self.history.previous_effect.value} · 重复次数：{self.history.repeat_count} · "
            f"偏好需求：{preferred} · 已拒绝动作：{rejected}"
        )


def main() -> None:
    root = tk.Tk()
    StrategyAppV4(root)
    root.mainloop()


if __name__ == "__main__":
    main()
