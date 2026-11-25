import json
import tempfile
import unittest
from pathlib import Path

from affective.models import PolicyContext, ScenarioEmotion, StudyFeedback
from affective.multimodal import Modality, ModalityEvidence, MultimodalObservation
from affective.pipeline import AffectiveInterventionPipeline
from affective.policy import UncertaintyAwarePolicy
from affective.session import ExperimentSession, StudyMetadata


class SessionTests(unittest.TestCase):
    def test_schema_v3_log_contains_policy_and_feedback(self):
        obs = MultimodalObservation(
            (ModalityEvidence(Modality.TEXT, {"sadness": 0.8, "neutral": 0.2}, -0.6, 0.6),)
        )
        result = AffectiveInterventionPipeline(UncertaintyAwarePolicy()).run(
            obs, PolicyContext(user_initiated=True)
        )
        session = ExperimentSession()
        session.add_round(
            StudyMetadata("P001", "uncertainty_aware", "S01", ScenarioEmotion.SADNESS, 4),
            PolicyContext(user_initiated=True),
            result,
            StudyFeedback(4, 2, 4, 5, 4),
        )
        test_tmp = Path(__file__).resolve().parents[1] / "tmp"
        test_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_tmp) as directory:
            path = Path(directory) / "session.json"
            session.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 3)
        self.assertEqual(data["records"][0]["result"]["decision"]["policy_name"], "uncertainty_aware")
        self.assertEqual(data["records"][0]["feedback"]["appropriateness"], 4)


if __name__ == "__main__":
    unittest.main()
