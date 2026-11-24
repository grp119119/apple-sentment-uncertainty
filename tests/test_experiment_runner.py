import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from affective_v03.scenarios import DEMONSTRATION_ONLY, demonstration_scenarios
from experiment_runner import main, run_all_comparisons, run_comparison


class ExperimentRunnerTests(unittest.TestCase):
    def test_all_runs_every_demonstration_scenario_with_summaries(self):
        result = run_all_comparisons()

        self.assertEqual(result["status"], DEMONSTRATION_ONLY)
        self.assertEqual(set(result["scenarios"]), set(demonstration_scenarios()))
        for scenario_result in result["scenarios"].values():
            self.assertEqual(scenario_result["status"], DEMONSTRATION_ONLY)
            self.assertEqual(
                set(scenario_result["summary"]),
                {"emotion_only", "uncertainty_aware"},
            )
            for summary in scenario_result["summary"].values():
                self.assertIn("mode", summary)
                self.assertIn("intensity", summary)
                self.assertIn("support_action", summary)
                self.assertIn("predictive_entropy", summary["uncertainty"])
                self.assertIn("top_two_margin", summary["uncertainty"])

    def test_user_state_parameters_are_configurable(self):
        observation = next(iter(demonstration_scenarios().values()))
        result = run_comparison(
            observation,
            user_initiated=False,
            receptive=False,
        )

        self.assertFalse(result["context"]["user_initiated"])
        self.assertFalse(result["context"]["receptive"])

    def test_both_conditions_share_observation_and_context_instances(self):
        observation = next(iter(demonstration_scenarios().values()))
        calls = []
        condition_result = {
            "decision": {
                "mode": "WAIT",
                "intensity": "none",
                "support_action": "Question",
            },
            "uncertainty": {
                "predictive_entropy": 0.5,
                "top_two_margin": 0.2,
                "signal_quality": 0.8,
                "data_coverage": 1.0,
                "missingness": 0.0,
                "cross_modal_disagreement": 0.1,
            },
        }

        def record_call(_pipeline, received_observation, received_context):
            calls.append((received_observation, received_context))
            return SimpleNamespace(to_dict=lambda: condition_result)

        with patch(
            "experiment_runner.AffectiveInterventionPipeline.run",
            autospec=True,
            side_effect=record_call,
        ):
            run_comparison(
                observation,
                user_initiated=False,
                receptive=False,
            )

        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][0], observation)
        self.assertIs(calls[1][0], observation)
        self.assertIs(calls[0][1], calls[1][1])
        self.assertFalse(calls[0][1].user_initiated)
        self.assertFalse(calls[0][1].receptive)

    def test_all_cli_exports_json_with_selected_user_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "all_scenarios.json"
            main([
                "--all",
                "--no-user-initiated",
                "--not-receptive",
                "--output",
                str(output),
            ])
            exported = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exported["status"], DEMONSTRATION_ONLY)
        self.assertFalse(exported["context"]["user_initiated"])
        self.assertFalse(exported["context"]["receptive"])
        self.assertEqual(set(exported["scenarios"]), set(demonstration_scenarios()))

    def test_default_and_named_scenario_modes_are_preserved(self):
        scenarios = demonstration_scenarios()
        named_scenario = next(reversed(scenarios))

        default_stdout = io.StringIO()
        with redirect_stdout(default_stdout):
            main([])
        default_result = json.loads(default_stdout.getvalue())

        named_stdout = io.StringIO()
        with redirect_stdout(named_stdout):
            main(["--scenario", named_scenario])
        named_result = json.loads(named_stdout.getvalue())

        self.assertEqual(default_result["scenario"], next(iter(scenarios)))
        self.assertEqual(named_result["scenario"], named_scenario)
        self.assertEqual(default_result["status"], DEMONSTRATION_ONLY)
        self.assertEqual(named_result["status"], DEMONSTRATION_ONLY)


if __name__ == "__main__":
    unittest.main()
