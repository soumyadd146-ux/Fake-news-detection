from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.preprocessing import clean_text


def _clean_documents(documents: Any) -> list[str]:
    return [clean_text(document) for document in documents]


def make_features(
    max_features: int = 50000,
    word_ngram_range: tuple[int, int] = (1, 2),
    include_character_features: bool = False,
) -> Pipeline:
    branches: list[tuple[str, Any]] = [
        (
            "word",
            TfidfVectorizer(
                stop_words="english",
                max_features=max_features,
                ngram_range=word_ngram_range,
                sublinear_tf=True,
            ),
        )
    ]
    if include_character_features:
        branches.append(
            (
                "character",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=max_features // 2,
                    sublinear_tf=True,
                ),
            )
        )
    return Pipeline(
        [
            ("clean", FunctionTransformer(_clean_documents, validate=False)),
            ("tfidf", FeatureUnion(branches)),
        ]
    )


def make_pipeline(classifier: Any, **feature_options: Any) -> Pipeline:
    return Pipeline(
        [
            ("features", make_features(**feature_options)),
            ("classifier", classifier),
        ]
    )


def candidate_models(include_random_forest: bool = False) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    models: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"classifier__C": [0.75, 1.5]},
        ),
        "Linear SVM": (
            LinearSVC(random_state=42),
            {"classifier__C": [0.5, 1.0]},
        ),
        "Passive-Aggressive": (
            SGDClassifier(
                loss="hinge",
                penalty=None,
                learning_rate="pa1",
                eta0=1.0,
                max_iter=1000,
                random_state=42,
            ),
            {"classifier__eta0": [0.5, 1.0]},
        ),
        "Multinomial Naive Bayes": (
            MultinomialNB(),
            {"classifier__alpha": [0.1, 0.5]},
        ),
    }
    if include_random_forest:
        from sklearn.ensemble import RandomForestClassifier

        models["Random Forest"] = (
            RandomForestClassifier(
                n_estimators=150,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced",
            ),
            {"classifier__max_features": ["sqrt", 0.5]},
        )
    return models
