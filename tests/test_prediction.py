import unittest
import tempfile
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.features import make_pipeline
from src.predict import PredictionError, load_artifacts, predict_text


DOCUMENTS = [
    "government confirms official economic report",
    "verified public report from government sources",
    "secret miracle cure shocking hidden conspiracy",
    "shocking secret rumor reveals hidden miracle",
]
LABELS = [1, 1, 0, 0]


class PredictionTests(unittest.TestCase):
    def test_probability_and_explanations_for_logistic_pipeline(self):
        model = make_pipeline(LogisticRegression(max_iter=1000)).fit(DOCUMENTS, LABELS)
        result = predict_text(model, None, title="Government confirms", body="official report")
        self.assertIn(result["label"], {"REAL NEWS", "FAKE NEWS"})
        self.assertEqual(result["score_type"], "predicted_class_probability")
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 100)
        self.assertIn("positive", result["explanation"])

    def test_svm_decision_score_is_not_reported_as_probability(self):
        model = make_pipeline(LinearSVC()).fit(DOCUMENTS, LABELS)
        result = predict_text(model, None, body="secret hidden rumor")
        self.assertEqual(result["score_type"], "decision_score")
        self.assertIsNone(result["confidence"])
        self.assertIn("not a probability", result["interpretation"])

    def test_empty_and_oversized_inputs_are_rejected(self):
        model = make_pipeline(LogisticRegression(max_iter=1000)).fit(DOCUMENTS, LABELS)
        with self.assertRaises(PredictionError):
            predict_text(model, None)
        with self.assertRaises(PredictionError):
            predict_text(model, None, body="long", max_characters=3)

    def test_missing_model_artifact_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_artifacts(Path(directory) / "missing.pkl")


if __name__ == "__main__":
    unittest.main()
