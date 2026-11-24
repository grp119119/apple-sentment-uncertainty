import unittest

from affective_v03.models import InterventionMode, PolicyContext
from affective_v03.multimodal import Modality, ModalityEvidence, MultimodalObservation
from affective_v03.pipeline import AffectiveInterventionPipeline
from affective_v03.policy import EmotionOnlyPolicy, UncertaintyAwarePolicy


def observation():
    ambiguous = {"fear": 0.36, "sadness": 0.34, "neutral": 0.30}
    return MultimodalObservation(
        (
            ModalityEvidence(Modality.TEXT, ambiguous, -0.5, 0.7, quality=0.9),
            ModalityEvidence(Modality.VOICE, ambiguous, -0.5, 0.8, quality=0.9),
            ModalityEvidence(Modality.FACE, ambiguous, -0.4, 0.6, quality=0.8),
        )
    )


class PipelineTests(unittest.TestCase):
    def test_conditions_share_fusion_but_change_mode(self):
        context = PolicyContext(user_initiated=True)
        baseline = AffectiveInterventionPipeline(EmotionOnlyPolicy()).run(observation(), context)
        proposed = AffectiveInterventionPipeline(UncertaintyAwarePolicy()).run(observation(), context)
        self.assertEqual(baseline.fusion.affect.distribution, proposed.fusion.affect.distribution)
        self.assertEqual(baseline.decision.mode, InterventionMode.DIRECT)
        self.assertEqual(proposed.decision.mode, InterventionMode.CONFIRM)


if __name__ == "__main__":
    unittest.main()
