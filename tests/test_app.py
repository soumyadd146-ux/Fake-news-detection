import sqlite3
import tempfile
import unittest
from pathlib import Path

from sklearn.linear_model import LogisticRegression

import app as webapp
from src.features import make_pipeline


class FlaskApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_pipeline(LogisticRegression(max_iter=1000)).fit(
            [
                "government confirms official report",
                "verified public report from official sources",
                "secret miracle cure shocking rumor",
                "hidden conspiracy rumor miracle cure",
            ],
            [1, 1, 0, 0],
        )

    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.previous_database = webapp.DATABASE
        self.previous_model = webapp._model
        self.previous_vectorizer = webapp._vectorizer
        self.previous_loaded = webapp._model_loaded
        self.previous_error = webapp._model_error
        webapp.DATABASE = str(Path(self.temp_directory.name) / "history.db")
        webapp.init_database()
        webapp._model = self.model
        webapp._vectorizer = None
        webapp._model_error = None
        webapp._model_loaded = True
        webapp.app.config.update(TESTING=True)
        self.client = webapp.app.test_client()

    def tearDown(self):
        webapp.DATABASE = self.previous_database
        webapp._model = self.previous_model
        webapp._vectorizer = self.previous_vectorizer
        webapp._model_error = self.previous_error
        webapp._model_loaded = self.previous_loaded
        self.temp_directory.cleanup()

    def test_routes_and_health(self):
        for path in ("/", "/dashboard", "/history", "/about"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()["status"], "ok")

    def test_browser_prediction_requires_csrf_and_is_stored(self):
        self.client.get("/")
        with self.client.session_transaction() as browser_session:
            token = browser_session["_csrf_token"]
        rejected = self.client.post("/predict", data={"news": "official report"})
        self.assertEqual(rejected.status_code, 400)

        response = self.client.post(
            "/predict",
            data={"csrf_token": token, "title": "Government", "news": "official report"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Predicted-class probability", response.data)
        connection = sqlite3.connect(webapp.DATABASE)
        try:
            count = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(count, 1)

    def test_json_prediction_validation_and_history_search(self):
        invalid = self.client.post("/api/predict", json={"title": "", "text": ""})
        self.assertEqual(invalid.status_code, 400)
        response = self.client.post(
            "/api/predict",
            json={"title": "Government report", "text": "confirmed official report"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["score_type"], "predicted_class_probability")
        history = self.client.get("/history?q=Government")
        self.assertEqual(history.status_code, 200)
        self.assertIn(b"Government", history.data)

    def test_history_paginates_records(self):
        connection = sqlite3.connect(webapp.DATABASE)
        try:
            connection.executemany(
                """
                INSERT INTO predictions (news, result, confidence, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (f"Article record {index}", "REAL NEWS", 80, "2026-10-13 12:00:00")
                    for index in range(12)
                ],
            )
            connection.commit()
        finally:
            connection.close()
        first_page = self.client.get("/history")
        second_page = self.client.get("/history?page=2")
        self.assertIn(b"Page 1 of 2", first_page.data)
        self.assertIn(b"Page 2 of 2", second_page.data)
        self.assertIn(b"Article record 0", second_page.data)


if __name__ == "__main__":
    unittest.main()
