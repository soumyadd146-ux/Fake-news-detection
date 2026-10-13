import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]*>")
NON_LETTER_PATTERN = re.compile(r"[^a-zA-Z\s]")
SPACE_PATTERN = re.compile(r"\s+")


@dataclass
class DatasetReport:
    source_rows: dict[str, int]
    rows_read: int
    malformed_rows_skipped: dict[str, int]
    rows_missing_usable_text: int
    exact_duplicates_removed: int
    rows_used: int
    class_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_text(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = NON_LETTER_PATTERN.sub(" ", text.lower())
    return SPACE_PATTERN.sub(" ", text).strip()


def combine_title_and_body(title: object, body: object) -> str:
    return clean_text(f"{title or ''} {body or ''}")


def content_group(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_duplicate_text(title: str, body: str) -> str:
    return SPACE_PATTERN.sub(" ", f"{title} {body}".casefold()).strip()


def _read_source(path: Path, label: int) -> tuple[pd.DataFrame, int, int]:
    malformed_rows = 0

    def skip_bad_row(_: list[str]) -> None:
        nonlocal malformed_rows
        malformed_rows += 1
        return None

    frame = pd.read_csv(path, engine="python", on_bad_lines=skip_bad_row)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "text" not in frame.columns and "title" not in frame.columns:
        raise ValueError(f"{path} must contain a 'text' or 'title' column")

    for column in ("title", "text"):
        if column not in frame.columns:
            frame[column] = ""

    frame["title"] = frame["title"].fillna("").astype(str)
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["content"] = [
        combine_title_and_body(title, body)
        for title, body in zip(frame["title"], frame["text"])
    ]
    frame["label"] = label
    return frame, len(frame), malformed_rows


def load_dataset(dataset_dir: str | Path) -> tuple[pd.DataFrame, DatasetReport]:
    root = Path(dataset_dir)
    fake, fake_rows, fake_malformed = _read_source(root / "Fake.csv", 0)
    real, real_rows, real_malformed = _read_source(root / "True.csv", 1)
    rows_read = pd.concat([fake, real], ignore_index=True)

    unusable = rows_read["content"].str.len() == 0
    removed_empty = int(unusable.sum())
    usable = rows_read.loc[~unusable].copy()

    usable["group"] = [
        content_group(normalize_duplicate_text(title, body))
        for title, body in zip(usable["title"], usable["text"])
    ]
    duplicate_mask = usable.duplicated("group", keep="first")
    duplicates_removed = int(duplicate_mask.sum())
    data = usable.loc[~duplicate_mask].reset_index(drop=True)
    if data["label"].nunique() < 2:
        raise ValueError("Both Fake.csv and True.csv must contribute usable articles")

    class_counts = {
        "FAKE NEWS": int((data["label"] == 0).sum()),
        "REAL NEWS": int((data["label"] == 1).sum()),
    }
    report = DatasetReport(
        source_rows={"Fake.csv": fake_rows, "True.csv": real_rows},
        rows_read=len(rows_read),
        malformed_rows_skipped={
            "Fake.csv": fake_malformed,
            "True.csv": real_malformed,
        },
        rows_missing_usable_text=removed_empty,
        exact_duplicates_removed=duplicates_removed,
        rows_used=len(data),
        class_counts=class_counts,
    )
    return data, report
