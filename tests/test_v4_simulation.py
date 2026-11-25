import unittest

from affective.models import InterventionMode, PolicyContext, UserNeed
from affective.simulation import ManualSimulationInput, run_manual_simulation


def simulation_input(distribution, **overrides):
    values = {
        "emotion_distribution": distribution,
        "valence": -0.4,
        "arousal": 0.6,
        "fatigue": 0.2,
        "signal_quality": 1.0,
        "data_coverage": 1.0,
        "context": PolicyContext(user_initiated=True, receptive=True),
    }
    values.update(overrides)
    return ManualSimulationInput(**values)


class V4SimulationTests(unittest.TestCase):
    def test_distribution_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "概率之和必须为 1.00"):
            simulation_input({"sadness": 0.6, "fear": 0.2})

    def test_manual_example_computes_uncertainty_and_direct_mode(self):
        result = run_manual_simulation(
            simulation_input({"sadness": 0.6, "fear": 0.2, "neutral": 0.2})
        )

        self.assertAlmostEqual(result.uncertainty.predictive_entropy, 0.4885, places=3)
        self.assertAlmostEqual(result.uncertainty.top_two_margin, 0.4)
        self.assertEqual(result.decision.mode, InterventionMode.DIRECT)
        self.assertEqual(result.fatigue, 0.2)

    def test_high_ambiguity_confirms_when_user_initiated(self):
        result = run_manual_simulation(
            simulation_input({"sadness": 0.34, "fear": 0.33, "neutral": 0.33})
        )
        self.assertEqual(result.decision.mode, InterventionMode.CONFIRM)

    def test_high_ambiguity_waits_when_user_did_not_initiate(self):
        result = run_manual_simulation(
            simulation_input(
                {"sadness": 0.34, "fear": 0.33, "neutral": 0.33},
                context=PolicyContext(user_initiated=False, receptive=True),
            )
        )
        self.assertEqual(result.decision.mode, InterventionMode.WAIT)

    def test_non_receptive_user_defers(self):
        result = run_manual_simulation(
            simulation_input(
                {"sadness": 0.6, "fear": 0.2, "neutral": 0.2},
                context=PolicyContext(user_initiated=True, receptive=False),
            )
        )
        self.assertEqual(result.decision.mode, InterventionMode.DEFER)

    def test_manual_result_remains_demonstration_only(self):
        result = run_manual_simulation(
            simulation_input(
                {"sadness": 0.6, "fear": 0.2, "neutral": 0.2},
                context=PolicyContext(user_need=UserNeed.LISTEN),
                user_text="我现在只想先说一说。",
            )
        ).to_dict()
        self.assertEqual(result["status"], "DEMONSTRATION_ONLY_NOT_STUDY_DATA")
        self.assertEqual(result["input_type"], "MANUAL_NUMERIC_SIMULATION")
        self.assertEqual(result["manual_user_input"]["need"], "LISTEN")
        self.assertEqual(result["manual_user_input"]["source"], "MANUAL_SELECTION")
        self.assertEqual(result["manual_user_input"]["text"], "我现在只想先说一说。")


if __name__ == "__main__":
    unittest.main()
