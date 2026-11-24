"""Complete Tkinter research interface for v0.3."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from affective_v03.config import load_policy_configuration
from affective_v03.models import (
    AffectLabel,
    InteractionStage,
    PolicyContext,
    Privacy,
    ScenarioEmotion,
    StudyFeedback,
    TimePressure,
)
from affective_v03.pipeline import AffectiveInterventionPipeline, PipelineResult
from affective_v03.policy import EmotionOnlyPolicy
from affective_v03.scenarios import DEMONSTRATION_ONLY, demonstration_scenarios
from affective_v03.session import ExperimentSession, StudyMetadata


ROOT = Path(__file__).resolve().parent
POLICY_CONFIG = ROOT / "config" / "policy_thresholds.json"
SESSION_PATH = ROOT / "data" / "v0_3_experiment.json"
LABELS = tuple(label.value for label in AffectLabel)


class StrategyAppV3:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.policy_config = load_policy_configuration(POLICY_CONFIG)
        self.session = ExperimentSession.load(SESSION_PATH)
        self.scenarios = demonstration_scenarios()
        self.current_result: PipelineResult | None = None
        self.current_context: PolicyContext | None = None
        self._build_variables()
        self._configure_window()
        self._build_interface()
        self._show_session_summary()

    def _build_variables(self) -> None:
        self.policy_var = tk.StringVar(value="uncertainty_aware")
        self.scenario_var = tk.StringVar(value=next(iter(self.scenarios)))
        self.participant_var = tk.StringVar(value="P001")
        self.scenario_id_var = tk.StringVar(value="DEMO-01")
        self.scenario_emotion_var = tk.StringVar(value="sadness")
        self.pre_intensity_var = tk.IntVar(value=3)
        self.privacy_var = tk.StringVar(value="private")
        self.time_pressure_var = tk.StringVar(value="low")
        self.stage_var = tk.StringVar(value="exploring")
        self.receptive_var = tk.BooleanVar(value=True)
        self.user_initiated_var = tk.BooleanVar(value=True)
        self.safety_var = tk.BooleanVar(value=True)
        self.appropriateness_var = tk.IntVar(value=3)
        self.intrusiveness_var = tk.IntVar(value=3)
        self.understanding_var = tk.IntVar(value=3)
        self.continue_var = tk.IntVar(value=3)
        self.helpfulness_var = tk.IntVar(value=3)
        self.corrected_var = tk.BooleanVar(value=False)
        self.session_summary_var = tk.StringVar()

    def _configure_window(self) -> None:
        self.root.title("情感感知不确定性介入原型 v0.3")
        self.root.geometry("1180x850")
        self.root.minsize(1020, 720)
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Result.TLabel", font=("Microsoft YaHei UI", 14, "bold"))

    def _build_interface(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="情感感知不确定性 → 介入模式 / 强度 → 支持动作", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text=f"Policy: {self.policy_config.policy_version}｜{self.policy_config.calibration_status}｜演示场景：{DEMONSTRATION_ONLY}",
        ).pack(anchor="w", pady=(2, 10))

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)
        decision_tab = ttk.Frame(notebook, padding=12)
        feedback_tab = ttk.Frame(notebook, padding=12)
        records_tab = ttk.Frame(notebook, padding=12)
        notebook.add(decision_tab, text="决策演示")
        notebook.add(feedback_tab, text="实验反馈")
        notebook.add(records_tab, text="记录与导出")
        self._build_decision_tab(decision_tab)
        self._build_feedback_tab(feedback_tab)
        self._build_records_tab(records_tab)

    def _build_decision_tab(self, tab: ttk.Frame) -> None:
        settings = ttk.LabelFrame(tab, text="1. 条件与演示输入", padding=10)
        settings.pack(fill="x")
        self._combo(settings, "实验条件", self.policy_var, ("uncertainty_aware", "emotion_only"))
        self._combo(settings, "演示场景", self.scenario_var, tuple(self.scenarios))
        self._combo(settings, "环境", self.privacy_var, tuple(item.value for item in Privacy))
        self._combo(settings, "时间压力", self.time_pressure_var, tuple(item.value for item in TimePressure))
        self._combo(settings, "阶段", self.stage_var, tuple(item.value for item in InteractionStage))

        flags = ttk.Frame(settings)
        flags.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(flags, text="用户主动发起", variable=self.user_initiated_var).pack(side="left")
        ttk.Checkbutton(flags, text="当前愿意接受介入", variable=self.receptive_var).pack(side="left", padx=14)
        ttk.Checkbutton(flags, text="安全边界允许", variable=self.safety_var).pack(side="left")
        ttk.Button(flags, text="运行 v0.3 决策", command=self.run_decision).pack(side="right")

        result = ttk.LabelFrame(tab, text="2. Affect / Uncertainty / Policy 输出", padding=10)
        result.pack(fill="both", expand=True, pady=10)
        self.result_title = ttk.Label(result, text="尚未运行", style="Result.TLabel")
        self.result_title.pack(anchor="w")
        self.result_text = tk.Text(result, wrap="word", font=("Consolas", 10), padx=8, pady=8)
        self.result_text.pack(fill="both", expand=True, pady=(8, 0))
        self.result_text.configure(state="disabled")

    def _build_feedback_tab(self, tab: ttk.Frame) -> None:
        metadata = ttk.LabelFrame(tab, text="研究元数据", padding=10)
        metadata.pack(fill="x")
        self._entry(metadata, "参与者编号", self.participant_var)
        self._entry(metadata, "场景编号", self.scenario_id_var)
        self._combo(metadata, "ESConv 场景情绪", self.scenario_emotion_var, tuple(item.value for item in ScenarioEmotion))
        ttk.Label(metadata, text="前测情绪强度 1–5").pack(side="left", padx=(12, 4))
        ttk.Spinbox(metadata, from_=1, to=5, textvariable=self.pre_intensity_var, width=4).pack(side="left")

        ratings = ttk.LabelFrame(tab, text="五级评价", padding=10)
        ratings.pack(fill="x", pady=10)
        self._rating(ratings, "介入合适程度", self.appropriateness_var)
        self._rating(ratings, "打扰程度", self.intrusiveness_var)
        self._rating(ratings, "被理解程度", self.understanding_var)
        self._rating(ratings, "愿意继续交互", self.continue_var)
        self._rating(ratings, "帮助程度", self.helpfulness_var)
        ttk.Checkbutton(ratings, text="用户纠正了系统判断", variable=self.corrected_var).pack(anchor="w", pady=5)
        ttk.Button(tab, text="保存本轮实验记录", command=self.save_feedback).pack(anchor="w")
        ttk.Label(tab, textvariable=self.session_summary_var).pack(anchor="w", pady=8)

    def _build_records_tab(self, tab: ttk.Frame) -> None:
        top = ttk.Frame(tab)
        top.pack(fill="x")
        ttk.Button(top, text="刷新记录", command=self.refresh_records).pack(side="left")
        ttk.Button(top, text="打开 JSON 内容", command=self.show_json).pack(side="left", padx=8)
        ttk.Label(top, text=f"保存位置：{SESSION_PATH}").pack(side="right")
        self.record_table = ttk.Treeview(
            tab,
            columns=("participant", "condition", "scenario", "emotion", "mode", "action", "appropriate"),
            show="headings",
        )
        headings = {
            "participant": "参与者",
            "condition": "条件",
            "scenario": "场景",
            "emotion": "识别情绪",
            "mode": "Mode",
            "action": "Support Action",
            "appropriate": "合适度",
        }
        for key, label in headings.items():
            self.record_table.heading(key, text=label)
            self.record_table.column(key, width=135, anchor="center")
        self.record_table.pack(fill="both", expand=True, pady=10)
        self.refresh_records()

    @staticmethod
    def _combo(parent, label, variable, values) -> None:
        frame = ttk.Frame(parent)
        frame.pack(side="left", padx=(0, 10))
        ttk.Label(frame, text=label).pack(anchor="w")
        ttk.Combobox(frame, textvariable=variable, values=values, state="readonly", width=20).pack()

    @staticmethod
    def _entry(parent, label, variable) -> None:
        frame = ttk.Frame(parent)
        frame.pack(side="left", padx=(0, 10))
        ttk.Label(frame, text=label).pack(anchor="w")
        ttk.Entry(frame, textvariable=variable, width=14).pack()

    @staticmethod
    def _rating(parent, label, variable) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=18).pack(side="left")
        ttk.Spinbox(row, from_=1, to=5, textvariable=variable, width=6, state="readonly").pack(side="left")

    def _policy(self):
        return self.policy_config.build_policy() if self.policy_var.get() == "uncertainty_aware" else EmotionOnlyPolicy()

    def run_decision(self) -> None:
        try:
            context = PolicyContext(
                privacy=Privacy(self.privacy_var.get()),
                time_pressure=TimePressure(self.time_pressure_var.get()),
                interaction_stage=InteractionStage(self.stage_var.get()),
                receptive=self.receptive_var.get(),
                user_initiated=self.user_initiated_var.get(),
                safety_allows_intervention=self.safety_var.get(),
            )
            result = AffectiveInterventionPipeline(self._policy()).run(
                self.scenarios[self.scenario_var.get()], context
            )
        except (KeyError, TypeError, ValueError) as error:
            messagebox.showerror("无法生成决策", str(error))
            return
        self.current_context, self.current_result = context, result
        decision, affect, uncertainty = result.decision, result.fusion.affect, result.fusion.uncertainty
        content = {
            "affect": affect.to_dict(),
            "uncertainty": uncertainty.to_dict(),
            "decision": decision.to_dict(),
            "response": result.response,
        }
        self.result_title.configure(text=f"{decision.mode.value} · {decision.intensity.value} · {decision.support_action.value}")
        self._set_result(json.dumps(content, ensure_ascii=False, indent=2))

    def save_feedback(self) -> None:
        if self.current_result is None or self.current_context is None:
            messagebox.showinfo("尚无决策", "请先在“决策演示”中运行一次决策。")
            return
        try:
            metadata = StudyMetadata(
                participant_id=self.participant_var.get().strip(),
                condition=self.current_result.decision.policy_name,
                scenario_id=self.scenario_id_var.get().strip(),
                scenario_emotion=ScenarioEmotion(self.scenario_emotion_var.get()),
                pre_emotion_intensity=self.pre_intensity_var.get(),
            )
            feedback = StudyFeedback(
                self.appropriateness_var.get(),
                self.intrusiveness_var.get(),
                self.understanding_var.get(),
                self.continue_var.get(),
                self.helpfulness_var.get(),
                corrected_system=self.corrected_var.get(),
            )
            self.session.add_round(metadata, self.current_context, self.current_result, feedback)
            self.session.save(SESSION_PATH)
        except (OSError, TypeError, ValueError) as error:
            messagebox.showerror("保存失败", str(error))
            return
        self._show_session_summary()
        self.refresh_records()
        messagebox.showinfo("已保存", "本轮记录已写入 schema v3 实验日志。")

    def refresh_records(self) -> None:
        for item in self.record_table.get_children():
            self.record_table.delete(item)
        for record in self.session.records:
            metadata, result = record.metadata, record.result
            feedback = record.feedback or {}
            self.record_table.insert("", "end", values=(
                metadata.get("participant_id"),
                metadata.get("condition"),
                metadata.get("scenario_id"),
                result.get("affect", {}).get("dominant_emotion"),
                result.get("decision", {}).get("mode"),
                result.get("decision", {}).get("support_action"),
                feedback.get("appropriateness", "-"),
            ))

    def show_json(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("v0.3 实验日志")
        text = tk.Text(window, wrap="none", font=("Consolas", 10))
        text.pack(fill="both", expand=True)
        payload = {"schema_version": 3, "records": [record.__dict__ for record in self.session.records]}
        text.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2))

    def _show_session_summary(self) -> None:
        self.session_summary_var.set(f"当前会话已记录 {len(self.session.records)} 轮。反馈只写入日志，不修改 Policy。")

    def _set_result(self, content: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", content)
        self.result_text.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    StrategyAppV3(root)
    root.mainloop()


if __name__ == "__main__":
    main()
