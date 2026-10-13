import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing import clean_text, load_dataset


class PreprocessingTests(unittest.TestCase):
    def test_clean_text_removes_urls_markup_and_punctuation(self):
        self.assertEqual(
            clean_text("<b>Breaking</b> news! https://example.com"),
            "breaking news",
        )

    def test_dataset_source_labels_missing_text_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ["title", "text", "subject", "date"]
            pd.DataFrame(
                [
                    ["Same title", "same article body", "news", "today"],
                    ["", "", "news", "today"],
                ],
                columns=columns,
            ).to_csv(root / "Fake.csv", index=False)
            pd.DataFrame(
                [
                    ["Same title", "same article body", "news", "today"],
                    ["Real report", "another article body", "news", "today"],
                ],
                columns=columns,
            ).to_csv(root / "True.csv", index=False)

            data, report = load_dataset(root)

        self.assertEqual(data["label"].tolist(), [0, 1])
        self.assertEqual(report.rows_read, 4)
        self.assertEqual(report.rows_missing_usable_text, 1)
        self.assertEqual(report.exact_duplicates_removed, 1)
        self.assertEqual(report.rows_used, 2)
        self.assertEqual(report.class_counts, {"FAKE NEWS": 1, "REAL NEWS": 1})


if __name__ == "__main__":
    unittest.main()
