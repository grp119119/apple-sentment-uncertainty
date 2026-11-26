"""Tests for Policy/History bug fixes (Bugs 2, 3, 4) and bounded LLMAdapter integration."""

from __future__ import annotations

import json
import time
import unittest
from unittest.mock import MagicMock, patch

from affective.history import update_history
from affective.llm_adapter import LLMAdapter, LLMConfig, LLMUtteranceAnalysis
from affective.models import (
    AffectEstimate,
    AffectLabel,
    EpistemicStyle,
    HistoryEffect,
    HistoryRecord,
    InteractionStage,
    InterventionDecision,
    InterventionIntensity,
    InterventionMode,
    PolicyContext,
    SupportAction,
    UncertaintyProfile,
    UserNeed,
)
from affective.perception.contracts import PerceptionModality, PerceptionRecord, PerceptionSource
from affective.perception.stream_coordinator import InteractiveSessionRunner, TurnEvent, TurnEventType
from affective.support import select_support_action


def _make_decision(
    mode: InterventionMode = InterventionMode.DIRECT,
    intensity: InterventionIntensity = InterventionIntensity.MEDIUM,
    action: SupportAction = SupportAction.REFLECTION,
    style: EpistemicStyle = EpistemicStyle.CERTAIN,
) -> InterventionDecision:
    return InterventionDecision(
        mode=mode,
        intensity=intensity,
        support_action=action,
        epistemic_style=style,
        reasons=("unit test",),
        policy_name="test",
        policy_version="test",
        calibration_status="TEST_ONLY",
    )


class TestPolicyAndLLMAdapter(unittest.TestCase):
    def test_wait_and_defer_do_not_increment_repeat_count_or_overwrite_action(self) -> None:
        # 首轮真正执行了一次 Affirmation
        h1 = update_history(
            HistoryRecord(),
            _make_decision(InterventionMode.DIRECT, InterventionIntensity.MEDIUM, SupportAction.AFFIRMATION),
            UserNeed.COMFORT,
            HistoryEffect.UNCHANGED,
        )
        self.assertEqual(h1.previous_action, SupportAction.AFFIRMATION)
        self.assertEqual(h1.repeat_count, 1)

        # 随后两轮 WAIT（未实际发声）绝不应把 repeat_count 累增到 2 或 3
        wait_dec = _make_decision(
            InterventionMode.WAIT,
            InterventionIntensity.NONE,
            SupportAction.AFFIRMATION,
            EpistemicStyle.NON_COMMITTAL,
        )
        h2 = update_history(h1, wait_dec, UserNeed.COMFORT, HistoryEffect.UNCHANGED)
        h3 = update_history(h2, wait_dec, UserNeed.COMFORT, HistoryEffect.UNCHANGED)
        self.assertEqual(h3.previous_action, SupportAction.AFFIRMATION)
        self.assertEqual(h3.repeat_count, 1)
        self.assertEqual(h3.rejected_actions, ())

    def test_confirm_question_does_not_pollute_history_or_blacklist_question(self) -> None:
        h1 = update_history(
            HistoryRecord(),
            _make_decision(InterventionMode.DIRECT, InterventionIntensity.MEDIUM, SupportAction.REFLECTION),
            UserNeed.UNKNOWN,
            HistoryEffect.UNCHANGED,
        )
        confirm_dec = _make_decision(
            InterventionMode.CONFIRM,
            InterventionIntensity.MEDIUM,
            SupportAction.QUESTION,
            EpistemicStyle.CONFIRMATORY,
        )
        # 即使澄清后情绪变差，也不应把 Question 拉进黑名单，也不应抹掉上轮 Reflection 记忆
        h2 = update_history(h1, confirm_dec, UserNeed.UNKNOWN, HistoryEffect.WORSENED)
        self.assertEqual(h2.previous_action, SupportAction.REFLECTION)
        self.assertNotIn(SupportAction.QUESTION, h2.rejected_actions)

    def test_affect_led_support_action_respects_history_rejection_and_switching(self) -> None:
        sad_affect = AffectEstimate({"sadness": 0.85, "neutral": 0.15}, -0.6, 0.45)

        # 情绪驱动模式下：如果 Reflection 已被历史拒绝，必须自动切换到备选支持动作
        rejected_hist = HistoryRecord(rejected_actions=(SupportAction.REFLECTION,))
        ctx = PolicyContext(
            interaction_stage=InteractionStage.SUPPORTING,
            user_need=UserNeed.UNKNOWN,
            history=rejected_hist,
        )
        action, reason = select_support_action(sad_affect, ctx)
        self.assertNotEqual(action, SupportAction.REFLECTION)
        self.assertIn("history excludes a previously rejected support action", reason)

        # 连续 2 次 Reflection 无改善，也必须自动换招
        repeat_hist = HistoryRecord(
            previous_action=SupportAction.REFLECTION,
            previous_effect=HistoryEffect.UNCHANGED,
            repeat_count=2,
        )
        ctx2 = PolicyContext(
            interaction_stage=InteractionStage.SUPPORTING,
            user_need=UserNeed.UNKNOWN,
            history=repeat_hist,
        )
        action2, reason2 = select_support_action(sad_affect, ctx2)
        self.assertNotEqual(action2, SupportAction.REFLECTION)
        self.assertIn("history switches after repeated support without improvement", reason2)

    def test_llm_adapter_low_confidence_need_coerced_to_unknown(self) -> None:
        adapter = LLMAdapter(
            config=LLMConfig(api_key="sk-test", base_url="https://example.com/v1", model="qwen-plus"),
            enabled=True,
        )
        fake_json = json.dumps({
            "user_need": "GUIDANCE",
            "need_confidence": 0.40,
            "need_reason": "不太确定",
            "text_emotion": {"fear": 0.7, "sadness": 0.2, "neutral": 0.1},
        })
        with patch.object(adapter, "_chat_completion", return_value=fake_json):
            res = adapter.analyze_utterance("我最近毕设压力好大呀")
        self.assertIsNotNone(res)
        # 置信度 0.40 < 0.65，必须强制退回 UNKNOWN，防止大模型乱猜架空情绪
        self.assertEqual(res.user_need, UserNeed.UNKNOWN)
        self.assertEqual(max(res.text_emotion, key=res.text_emotion.get), AffectLabel.FEAR)

    def test_llm_adapter_never_calls_api_on_wait_mode(self) -> None:
        adapter = LLMAdapter(
            config=LLMConfig(api_key="sk-test", base_url="https://example.com/v1", model="qwen-plus"),
            enabled=True,
        )
        wait_dec = _make_decision(
            InterventionMode.WAIT,
            InterventionIntensity.NONE,
            SupportAction.REFLECTION,
            EpistemicStyle.NON_COMMITTAL,
        )
        aff = AffectEstimate({"neutral": 1.0}, 0.0, 0.2)
        unc = UncertaintyProfile(0.8, 0.1, 0.8, 0.33, 0.67)
        with patch.object(adapter, "_chat_completion") as mock_call:
            reply = adapter.generate_reply(wait_dec, aff, unc, "")
            mock_call.assert_not_called()
            self.assertIn("保持安静", reply)

    def test_runner_integrates_llm_need_and_reply(self) -> None:
        mock_audio = MagicMock()
        mock_llm = MagicMock(spec=LLMAdapter)
        mock_llm.enabled = True
        mock_llm.analyze_utterance.return_value = LLMUtteranceAnalysis(
            user_need=UserNeed.GUIDANCE,
            need_confidence=0.92,
            need_reason="用户明确询问下一步该怎么办",
            text_emotion={
                AffectLabel.FEAR: 0.75,
                AffectLabel.SADNESS: 0.15,
                AffectLabel.NEUTRAL: 0.10,
                AffectLabel.ANGER: 0.0,
                AffectLabel.DISGUST: 0.0,
                AffectLabel.JOY: 0.0,
                AffectLabel.SURPRISE: 0.0,
            },
            signal_quality=0.85,
            source="llm:qwen-plus",
        )
        mock_llm.generate_reply.return_value = "听起来时间压力让你很紧绷，不如先挑出今天最重要的一小步写下来？"

        runner = InteractiveSessionRunner(
            audio_detector=mock_audio,
            default_need=UserNeed.UNKNOWN,
            llm_adapter=mock_llm,
        )
        now = time.time()
        fear_dist = {AffectLabel.FEAR: 0.80, AffectLabel.NEUTRAL: 0.20}
        runner.visual_buffer.add_record(
            PerceptionRecord(
                modality=PerceptionModality.VISUAL,
                source=PerceptionSource.CAMERA,
                model_name="mock",
                model_version="1.0",
                timestamp=now,
                raw_scores={},
                mapped_scores=fear_dist,
                signal_quality=0.90,
                available=True,
            ),
            timestamp=now,
        )
        mock_audio.analyze_audio.return_value = (
            PerceptionRecord(
                modality=PerceptionModality.AUDIO,
                source=PerceptionSource.MICROPHONE,
                model_name="mock_ser",
                model_version="1.0",
                timestamp=now,
                raw_scores={},
                mapped_scores=fear_dist,
                signal_quality=0.85,
                available=True,
            ),
            PerceptionRecord(
                modality=PerceptionModality.TEXT,
                source=PerceptionSource.ASR,
                model_name="mock_asr",
                model_version="1.0",
                timestamp=now,
                raw_scores={},
                mapped_scores=fear_dist,
                signal_quality=0.80,
                available=True,
                transcript="我毕设快赶不完了，你觉得我第一步该怎么办？",
            ),
        )
        turn = runner.handle_turn_event(
            TurnEvent(
                event_type=TurnEventType.SPEECH_FINISHED,
                timestamp=now,
                speech_start=now - 2.0,
                speech_end=now,
                audio_wav_path="/tmp/mock_llm.wav",
            )
        )
        self.assertEqual(turn["llm_detected_need"], "GUIDANCE")
        self.assertEqual(turn["policy_action"], "Providing Suggestions")
        self.assertEqual(turn["system_reply"], "听起来时间压力让你很紧绷，不如先挑出今天最重要的一小步写下来？")

    def test_need_short_term_inheritance_decays_after_two_unknown_turns(self) -> None:
        """Turn 1: explicit GUIDANCE -> Turn 2: UNKNOWN inherits GUIDANCE -> Turn 3: UNKNOWN decays back to affect-led."""
        mock_audio = MagicMock()
        mock_llm = MagicMock(spec=LLMAdapter)
        mock_llm.enabled = True
        mock_llm.generate_reply.return_value = "测试回复"

        sad_dist = {
            AffectLabel.SADNESS: 0.85,
            AffectLabel.NEUTRAL: 0.15,
            AffectLabel.ANGER: 0.0,
            AffectLabel.DISGUST: 0.0,
            AffectLabel.FEAR: 0.0,
            AffectLabel.JOY: 0.0,
            AffectLabel.SURPRISE: 0.0,
        }

        runner = InteractiveSessionRunner(
            audio_detector=mock_audio,
            default_need=UserNeed.UNKNOWN,
            llm_adapter=mock_llm,
        )
        now = time.time()

        def run_turn(offset: float, text: str, detected_need: UserNeed) -> dict:
            t = now + offset
            runner.visual_buffer.add_record(
                PerceptionRecord(
                    modality=PerceptionModality.VISUAL,
                    source=PerceptionSource.CAMERA,
                    model_name="mock",
                    model_version="1.0",
                    timestamp=t,
                    raw_scores={},
                    mapped_scores=sad_dist,
                    signal_quality=0.90,
                    available=True,
                ),
                timestamp=t,
            )
            mock_audio.analyze_audio.return_value = (
                PerceptionRecord(
                    modality=PerceptionModality.AUDIO,
                    source=PerceptionSource.MICROPHONE,
                    model_name="mock_ser",
                    model_version="1.0",
                    timestamp=t,
                    raw_scores={},
                    mapped_scores=sad_dist,
                    signal_quality=0.85,
                    available=True,
                ),
                PerceptionRecord(
                    modality=PerceptionModality.TEXT,
                    source=PerceptionSource.ASR,
                    model_name="mock_asr",
                    model_version="1.0",
                    timestamp=t,
                    raw_scores={},
                    mapped_scores=sad_dist,
                    signal_quality=0.80,
                    available=True,
                    transcript=text,
                ),
            )
            mock_llm.analyze_utterance.return_value = LLMUtteranceAnalysis(
                user_need=detected_need,
                need_confidence=0.9 if detected_need is not UserNeed.UNKNOWN else 0.1,
                need_reason="mock",
                text_emotion=sad_dist,
                signal_quality=0.85,
            )
            return runner.handle_turn_event(
                TurnEvent(
                    event_type=TurnEventType.SPEECH_FINISHED,
                    timestamp=t,
                    speech_start=t - 2.0,
                    speech_end=t,
                    audio_wav_path="/tmp/mock_decay.wav",
                ),
                manual_need=UserNeed.UNKNOWN,
            )

        # Turn 1: 明确请求建议 -> Providing Suggestions
        t1 = run_turn(0.0, "帮我想想该怎么办", UserNeed.GUIDANCE)
        self.assertEqual(t1["policy_action"], "Providing Suggestions")

        # Turn 2: 未提新需求 (UNKNOWN) -> 短时沿用上一轮 GUIDANCE (Providing Suggestions)
        t2 = run_turn(5.0, "主要是时间真的太紧了", UserNeed.UNKNOWN)
        self.assertEqual(t2["policy_action"], "Providing Suggestions")
        self.assertTrue(any("inherited from interaction history" in r for r in t2["policy_reasons"]))

        # Turn 3: 连续第 2 轮未提明确需求 -> 自动衰减回落至情绪驱动 (sadness -> Reflection of Feelings)
        t3 = run_turn(10.0, "唉就是觉得心里特别难受", UserNeed.UNKNOWN)
        self.assertEqual(t3["policy_action"], "Reflection of Feelings")
        self.assertIn("fear or sadness favors reflection of feelings", t3["policy_reasons"])

    def test_text_input_turn_and_reset_session(self) -> None:
        """TEXT_INPUT works without audio/camera and reset_session clears turn state cleanly."""
        mock_audio = MagicMock()
        mock_llm = MagicMock(spec=LLMAdapter)
        mock_llm.enabled = True
        mock_llm.analyze_utterance.return_value = LLMUtteranceAnalysis(
            user_need=UserNeed.GUIDANCE,
            need_confidence=0.95,
            need_reason="明确询问建议",
            text_emotion={
                AffectLabel.FEAR: 0.70,
                AffectLabel.SADNESS: 0.15,
                AffectLabel.NEUTRAL: 0.15,
                AffectLabel.ANGER: 0.0,
                AffectLabel.DISGUST: 0.0,
                AffectLabel.JOY: 0.0,
                AffectLabel.SURPRISE: 0.0,
            },
            signal_quality=0.85,
            source="llm:qwen3.7-plus",
        )
        mock_llm.generate_reply.return_value = "先深呼吸一口气，把最急的一件事写下来就行。"

        runner = InteractiveSessionRunner(
            audio_detector=mock_audio,
            default_need=UserNeed.UNKNOWN,
            llm_adapter=mock_llm,
        )
        turn = runner.handle_turn_event(
            TurnEvent(
                event_type=TurnEventType.TEXT_INPUT,
                timestamp=time.time(),
                text_input="毕设卡住了，我第一步该做什么？",
            )
        )
        self.assertIsNotNone(turn)
        self.assertEqual(turn["event_type"], "text_input")
        self.assertEqual(turn["llm_detected_need"], "GUIDANCE")
        self.assertEqual(turn["policy_action"], "Providing Suggestions")
        self.assertEqual(turn["system_reply"], "先深呼吸一口气，把最急的一件事写下来就行。")
        mock_audio.analyze_audio.assert_not_called()

        runner.reset_session()
        self.assertEqual(runner.turn_index, 0)
        self.assertEqual(runner.journey_records, [])
        self.assertIsNone(runner._previous_decision)

    def test_journey_save_list_and_meta_tag_helpers(self) -> None:
        import tempfile
        from pathlib import Path
        from main_v4 import format_turn_meta_tag, list_saved_journeys, save_journey_payload

        sample_turn = {
            "turn_index": 1,
            "policy_mode": "DIRECT",
            "policy_action": "Providing Suggestions",
            "policy_intensity": "low",
            "epistemic_style": "certain",
            "llm_detected_need": "GUIDANCE",
            "llm_need_confidence": 0.95,
            "transcript": "给我个小建议",
            "system_reply": "先花五分钟写个提纲。",
        }
        tag = format_turn_meta_tag(sample_turn)
        self.assertIn("DIRECT", tag)
        self.assertIn("提供建议", tag)
        self.assertIn("Need:", tag)

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            saved = save_journey_payload([sample_turn], "UNKNOWN", journeys_dir=d)
            self.assertTrue(saved.exists())
            items = list_saved_journeys(journeys_dir=d)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["total_turns"], 1)
            self.assertEqual(items[0]["turns"][0]["system_reply"], "先花五分钟写个提纲。")


if __name__ == "__main__":
    unittest.main()
