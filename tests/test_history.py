import unittest

from affective_v03.history import update_history
from affective_v03.models import (
    EpistemicStyle,
    HistoryEffect,
    HistoryRecord,
    InterventionDecision,
    InterventionIntensity,
    InterventionMode,
    SupportAction,
    UserNeed,
)


def decision(action=SupportAction.SUGGESTION):
    return InterventionDecision(
        mode=InterventionMode.DIRECT,
        intensity=InterventionIntensity.MEDIUM,
        support_action=action,
        epistemic_style=EpistemicStyle.CERTAIN,
        reasons=("test decision",),
        policy_name="test",
        policy_version="test",
        calibration_status="TEST_ONLY",
    )


class HistoryTests(unittest.TestCase):
    def test_rejected_action_is_carried_to_next_round(self):
        updated = update_history(
            HistoryRecord(),
            decision(),
            UserNeed.GUIDANCE,
            HistoryEffect.UNCHANGED,
            accepted=False,
        )
        self.assertIn(SupportAction.SUGGESTION, updated.rejected_actions)
        self.assertEqual(updated.preferred_need, UserNeed.GUIDANCE)

    def test_improved_feedback_keeps_action_available(self):
        updated = update_history(
            HistoryRecord(),
            decision(SupportAction.REFLECTION),
            UserNeed.LISTEN,
            HistoryEffect.IMPROVED,
        )
        self.assertNotIn(SupportAction.REFLECTION, updated.rejected_actions)
        self.assertEqual(updated.emotion_trend.value, "improving")

    def test_repeated_action_increments_count(self):
        initial = update_history(
            HistoryRecord(),
            decision(),
            UserNeed.GUIDANCE,
            HistoryEffect.UNCHANGED,
        )
        updated = update_history(
            initial,
            decision(),
            UserNeed.GUIDANCE,
            HistoryEffect.UNCHANGED,
        )
        self.assertEqual(updated.repeat_count, 2)


if __name__ == "__main__":
    unittest.main()
