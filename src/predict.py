from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src import LABELS
from src.preprocessing import combine_title_and_body


class PredictionError(ValueError):
    pass


def load_artifacts(model_path: str | Path, legacy_vectorizer_path: str | Path | None = None) -> tuple[Any, Any | None]:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    model = joblib.load(path)
    vectorizer = None
    if legacy_vectorizer_path is not None:
        legacy_path = Path(legacy_vectorizer_path)
        if legacy_path.is_file():
            vectorizer = joblib.load(legacy_path)
    return model, vectorizer


def _feature_matrix(model: Any, vectorizer: Any | None, content: str) -> Any:
    if hasattr(model, "named_steps"):
        return model.named_steps["features"].transform([content])
    if vectorizer is None:
        raise RuntimeError("Legacy model requires its TF-IDF vectorizer")
    return vectorizer.transform([content])


def _explanations(model: Any, vectorizer: Any | None, features: Any, predicted_label: int) -> dict[str, list[dict[str, float | str]]]:
    pipeline_features = model.named_steps["features"] if hasattr(model, "named_steps") else vectorizer
    classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    if not hasattr(classifier, "coef_") and getattr(classifier, "calibrated_classifiers_", None):
        calibrated_pipeline = classifier.calibrated_classifiers_[0].estimator
        if hasattr(calibrated_pipeline, "named_steps"):
            classifier = calibrated_pipeline.named_steps.get("classifier", classifier)
    if pipeline_features is None or not hasattr(classifier, "coef_"):
        return {"positive": [], "negative": []}

    tfidf = pipeline_features.named_steps["tfidf"] if hasattr(pipeline_features, "named_steps") else pipeline_features
    if hasattr(tfidf, "transformer_list"):
        names = tfidf.get_feature_names_out()
    elif hasattr(tfidf, "get_feature_names_out"):
        names = tfidf.get_feature_names_out()
    else:
        return {"positive": [], "negative": []}

    coefficients = np.asarray(classifier.coef_[0]).ravel()
    vector = features.tocsr()
    active_indices = vector.indices
    active_values = vector.data
    direction = 1 if predicted_label == 1 else -1
    contributions = [
        (str(names[index]), float(active_values[position] * coefficients[index] * direction))
        for position, index in enumerate(active_indices)
    ]
    positive = sorted((entry for entry in contributions if entry[1] > 0), key=lambda item: item[1], reverse=True)[:5]
    negative = sorted((entry for entry in contributions if entry[1] < 0), key=lambda item: item[1])[:5]
    return {
        "positive": [{"term": term, "contribution": round(score, 4)} for term, score in positive],
        "negative": [{"term": term, "contribution": round(score, 4)} for term, score in negative],
    }


def predict_text(
    model: Any,
    vectorizer: Any | None,
    title: str = "",
    body: str = "",
    max_characters: int = 20000,
) -> dict[str, Any]:
    if not title.strip() and not body.strip():
        raise PredictionError("Enter a news title or article body.")
    if len(title) + len(body) > max_characters:
        raise PredictionError(f"Input must not exceed {max_characters:,} characters.")

    content = combine_title_and_body(title, body)
    if not content:
        raise PredictionError("Enter text containing at least one letter.")
    features = _feature_matrix(model, vectorizer, content)
    classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    label = int(classifier.predict(features)[0])

    score_type = "decision_score"
    score_value: float | None
    confidence: float | None
    if hasattr(model, "predict_proba"):
        probabilities = classifier.predict_proba(features)[0]
        classes = list(getattr(classifier, "classes_", getattr(model, "classes_", [])))
        score_value = float(probabilities[classes.index(label)])
        score_type = (
            "calibrated_probability"
            if hasattr(classifier, "calibrated_classifiers_")
            else "predicted_class_probability"
        )
        confidence = round(score_value * 100, 2)
    elif hasattr(classifier, "decision_function"):
        decision = np.asarray(classifier.decision_function(features)).ravel()
        score_value = float(decision[0])
        confidence = None
    else:
        score_value = None
        confidence = None

    cautious_message = (
        "Model probability is not the probability that this story is factually true."
        if score_type in {"calibrated_probability", "predicted_class_probability"}
        else "This raw decision score is not a probability or confidence percentage."
    )
    if features.nnz == 0:
        cautious_message = "The submitted text has no recognized TF-IDF features; this prediction may be unreliable."
    elif score_type == "decision_score" and abs(score_value or 0) < 0.2:
        cautious_message = "The model score is close to its decision boundary. Verify the story independently."
    elif score_type in {"calibrated_probability", "predicted_class_probability"} and confidence is not None and confidence < 60:
        cautious_message = "The model's estimated class probability is low. Verify the story independently."

    return {
        "label": LABELS[label],
        "label_id": label,
        "score_type": score_type,
        "score_value": score_value,
        "confidence": confidence,
        "interpretation": cautious_message,
        "explanation": _explanations(model, vectorizer, features, label),
    }
