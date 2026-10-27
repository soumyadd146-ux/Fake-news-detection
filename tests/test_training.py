import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.predict import predict_text
from src.train import train


class TrainingTests(unittest.TestCase):
    def test_training_writes_real_metrics_and_model_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            models = root / "models"
            results = root / "results"
            dataset.mkdir()
            columns = ["title", "text", "subject", "date"]
            unique_terms = [
                "alpha", "bravo", "charlie", "delta", "echo",
                "foxtrot", "golf", "hotel", "india", "juliet",
                "kilo", "lima", "mike", "november", "oscar",
            ]
            fake_rows = [
                [f"Rumor headline {term}", f"secret sensational rumor {term}", "topic", "date"]
                for term in unique_terms
            ]
            real_rows = [
                [f"Official headline {term}", f"verified government official {term}", "topic", "date"]
                for term in unique_terms
            ]
            pd.DataFrame(fake_rows, columns=columns).to_csv(dataset / "Fake.csv", index=False)
            pd.DataFrame(real_rows, columns=columns).to_csv(dataset / "True.csv", index=False)
            article = (
                "The Unified Payments Interface has become a major part of India's digital payments system. "
                "Developed by the National Payments Corporation of India, UPI allows users to transfer money "
                "instantly between bank accounts through supported mobile applications. The system supports "
                "person-to-person transfers as well as payments to merchants."
            )
            pd.DataFrame(
                [{"title": article, "text": article, "label": "REAL NEWS", "weight": 5}]
            ).to_csv(dataset / "labeled_examples.csv", index=False)

            metrics = train(
                dataset_dir=dataset,
                models_dir=models,
                results_dir=results,
                random_state=17,
            )

            self.assertIn(
                metrics["algorithm"],
                {"Logistic Regression", "Linear SVM", "Passive-Aggressive", "Multinomial Naive Bayes"},
            )
            self.assertEqual(metrics["total_dataset_articles"], 31)
            self.assertEqual(metrics["training_articles"], 25)
            self.assertEqual(metrics["testing_articles"], 6)
            self.assertEqual(metrics["labeled_training_examples"], 1)
            self.assertEqual(metrics["labeled_training_example_weight"], 5)
            self.assertEqual(metrics["score_type"], "calibrated_probability")
            self.assertIn("per_class", metrics)
            model = joblib.load(models / "best_model.pkl")
            prediction = predict_text(model, None, body=article)
            self.assertEqual(prediction["label"], "REAL NEWS")
            for artifact in (
                models / "best_model.pkl",
                models / "vectorizer.pkl",
                models / "model_metadata.json",
                results / "metrics.json",
                results / "model_comparison.png",
                results / "confusion_matrix.png",
                results / "training_report.txt",
            ):
                self.assertTrue(artifact.is_file(), artifact)


if __name__ == "__main__":
    unittest.main()
