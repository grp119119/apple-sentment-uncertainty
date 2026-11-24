import unittest

from affective_v03.models import (
    AffectEstimate,
    HistoryEffect,
    HistoryRecord,
    InterventionIntensity,
    InterventionMode,
    PolicyContext,
    Privacy,
    SupportAction,
    UncertaintyProfile,
    UserNeed,
)
from affective_v03.policy import EmotionOnlyPolicy, UncertaintyAwarePolicy
from affective_v03.realizer import realize_response


def affect(arousal=0.6):
    return AffectEstimate({"fear": 0.8, "neutral": 0.2}, -0.6, arousal)


def uncertainty(entropy=0.2, margin=0.7, quality=0.9, coverage=1.0):
    return UncertaintyProfile(
        predictive_entropy=entropy,
        top_two_margin=margin,
        signal_quality=quality,
        data_coverage=coverage,
        missingness=1 - coverage,
    )


class PolicyTests(unittest.TestCase):
    def test_clear_evidence_allows_direct_support(self):
        result = UncertaintyAwarePolicy().decide(affect(), uncertainty(), PolicyContext())
        self.assertEqual(result.mode, InterventionMode.DIRECT)

    def test_high_ambiguity_confirms_when_user_initiated(self):
        result = UncertaintyAwarePolicy().decide(
            affect(), uncertainty(entropy=0.9, margin=0.05), PolicyContext(user_initiated=True)
        )
        self.assertEqual(result.mode, InterventionMode.CONFIRM)

    def test_high_ambiguity_waits_when_passive(self):
        result = UncertaintyAwarePolicy().decide(
            affect(), uncertainty(entropy=0.9, margin=0.05), PolicyContext(user_initiated=False)
        )
        self.assertEqual(result.mode, InterventionMode.WAIT)
        self.assertEqual(result.intensity, InterventionIntensity.NONE)

    def test_moderate_uncertainty_is_tentative(self):
        result = UncertaintyAwarePolicy().decide(
            affect(arousal=0.9), uncertainty(entropy=0.6, margin=0.3), PolicyContext()
        )
        self.assertEqual(result.mode, InterventionMode.TENTATIVE)
        self.assertEqual(result.intensity, InterventionIntensity.MEDIUM)

    def test_non_receptive_user_defers(self):
        result = UncertaintyAwarePolicy().decide(
            affect(), uncertainty(), PolicyContext(receptive=False)
        )
        self.assertEqual(result.mode, InterventionMode.DEFER)

    def test_baseline_ignores_uncertainty(self):
        result = EmotionOnlyPolicy().decide(
            affect(), uncertainty(entropy=1.0, margin=0.0, quality=0.1, coverage=0.2), PolicyContext()
        )
        self.assertEqual(result.mode, InterventionMode.DIRECT)

    def test_public_context_reduces_intensity(self):
        private = UncertaintyAwarePolicy().decide(
            affect(arousal=0.9), uncertainty(), PolicyContext()
        )
        public = UncertaintyAwarePolicy().decide(
            affect(arousal=0.9), uncertainty(), PolicyContext(privacy=Privacy.PUBLIC)
        )
        self.assertEqual(private.intensity, InterventionIntensity.HIGH)
        self.assertEqual(public.intensity, InterventionIntensity.MEDIUM)

    def test_public_context_also_reduces_confirm_intensity(self):
        result = UncertaintyAwarePolicy().decide(
            affect(arousal=0.9),
            uncertainty(entropy=0.9, margin=0.05),
            PolicyContext(privacy=Privacy.PUBLIC, user_initiated=True),
        )
        self.assertEqual(result.mode, InterventionMode.CONFIRM)
        self.assertEqual(result.intensity, InterventionIntensity.LOW)

    def test_none_intensity_never_realizes_active_language(self):
        result = UncertaintyAwarePolicy().decide(
            affect(arousal=0.2),
            uncertainty(),
            PolicyContext(privacy=Privacy.PUBLIC),
        )
        self.assertEqual(result.mode, InterventionMode.DIRECT)
        self.assertEqual(result.intensity, InterventionIntensity.NONE)
        self.assertEqual(realize_response(result), "[不主动发出语言回应。]")

    def test_manual_need_selects_support_action(self):
        result = UncertaintyAwarePolicy().decide(
            affect(), uncertainty(), PolicyContext(user_need=UserNeed.GUIDANCE)
        )
        self.assertEqual(result.support_action, SupportAction.SUGGESTION)

    def test_history_filters_a_rejected_need_action(self):
        history = HistoryRecord(rejected_actions=(SupportAction.SUGGESTION,))
        result = UncertaintyAwarePolicy().decide(
            affect(),
            uncertainty(),
            PolicyContext(user_need=UserNeed.GUIDANCE, history=history),
        )
        self.assertEqual(result.support_action, SupportAction.INFORMATION)

    def test_history_switches_after_repeated_unchanged_action(self):
        history = HistoryRecord(
            previous_action=SupportAction.SUGGESTION,
            previous_effect=HistoryEffect.UNCHANGED,
            repeat_count=2,
        )
        result = UncertaintyAwarePolicy().decide(
            affect(),
            uncertainty(),
            PolicyContext(user_need=UserNeed.GUIDANCE, history=history),
        )
        self.assertEqual(result.support_action, SupportAction.INFORMATION)

    def test_history_preserves_a_previously_effective_action(self):
        history = HistoryRecord(
            previous_action=SupportAction.REFLECTION,
            previous_effect=HistoryEffect.IMPROVED,
            repeat_count=1,
        )
        result = UncertaintyAwarePolicy().decide(
            affect(),
            uncertainty(),
            PolicyContext(user_need=UserNeed.COMFORT, history=history),
        )
        self.assertEqual(result.support_action, SupportAction.REFLECTION)

    def test_pause_need_defers_before_affect_action(self):
        result = UncertaintyAwarePolicy().decide(
            affect(), uncertainty(), PolicyContext(user_need=UserNeed.PAUSE)
        )
        self.assertEqual(result.mode, InterventionMode.DEFER)
        self.assertEqual(result.intensity, InterventionIntensity.NONE)

    def test_policy_context_defaults_to_user_initiated(self):
        self.assertTrue(PolicyContext().user_initiated)


if __name__ == "__main__":
    unittest.main()
