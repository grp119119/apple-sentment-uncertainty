import unittest

from affective_v03.models import AffectLabel
from affective_v03.uncertainty import (
    brier_score,
    expected_calibration_error,
    normalized_entropy,
    top_two_margin,
)


class UncertaintyMetricTests(unittest.TestCase):
    def test_concentrated_distribution_has_low_entropy_and_high_margin(self):
        distribution = {"fear": 0.9, "neutral": 0.1}
        self.assertLess(normalized_entropy(distribution), 0.25)
        self.assertAlmostEqual(top_two_margin(distribution), 0.8)

    def test_uniform_distribution_has_maximum_entropy(self):
        distribution = {label: 1 / len(AffectLabel) for label in AffectLabel}
        self.assertAlmostEqual(normalized_entropy(distribution), 1.0)
        self.assertAlmostEqual(top_two_margin(distribution), 0.0)

    def test_perfect_predictions_have_zero_brier(self):
        predictions = [{"fear": 1.0}, {"joy": 1.0}]
        labels = ["fear", "joy"]
        self.assertAlmostEqual(brier_score(predictions, labels), 0.0)
        self.assertAlmostEqual(expected_calibration_error(predictions, labels), 0.0)


if __name__ == "__main__":
    unittest.main()
