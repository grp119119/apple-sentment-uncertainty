import unittest

from calibration_runner import build_report
from dataset_validator import validate_row
from experiment_runner import run_comparison
from affective.scenarios import demonstration_scenarios


class ToolTests(unittest.TestCase):
    def test_comparison_has_isolated_conditions(self):
        result = run_comparison(next(iter(demonstration_scenarios().values())))
        self.assertEqual(set(result["conditions"]), {"emotion_only", "uncertainty_aware"})

    def test_validator_separates_label_spaces(self):
        self.assertEqual(validate_row({"affect_label": "fear", "scenario_emotion": "anxiety"}, 1), [])
        self.assertTrue(validate_row({"affect_label": "anxiety"}, 1))

    def test_report_does_not_change_thresholds(self):
        report = build_report({"schema_version": 3, "records": []})
        self.assertFalse(report["thresholds_changed"])


if __name__ == "__main__":
    unittest.main()
