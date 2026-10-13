import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.evaluate import evaluate_model, save_evaluation, save_model_comparison
from src.features import candidate_models, make_pipeline
from src.preprocessing import load_dataset


BASE_DIR = Path(__file__).resolve().parents[1]


def train(
    dataset_dir: str | Path = BASE_DIR / "dataset",
    models_dir: str | Path = BASE_DIR / "models",
    results_dir: str | Path = BASE_DIR / "results",
    random_state: int = 42,
    include_random_forest: bool | None = None,
    include_character_features: bool | None = None,
) -> dict[str, Any]:
    models_path = Path(models_dir)
    results_path = Path(results_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    data, dataset_report = load_dataset(dataset_dir)
    documents = data["content"].to_numpy()
    labels = data["label"].to_numpy()
    groups = data["group"].to_numpy()
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_indices, test_indices = next(outer_cv.split(documents, labels, groups))
    X_train, X_test = documents[train_indices], documents[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    train_groups = groups[train_indices]

    include_forest = (
        os.environ.get("INCLUDE_RANDOM_FOREST", "").strip().lower() in {"1", "true", "yes"}
        if include_random_forest is None
        else include_random_forest
    )
    candidates = candidate_models(include_random_forest=include_forest)
    use_character_features = (
        os.environ.get("INCLUDE_CHARACTER_FEATURES", "").strip().lower() in {"1", "true", "yes"}
        if include_character_features is None
        else include_character_features
    )
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=random_state)
    inner_splits = list(inner_cv.split(X_train, y_train, train_groups))
    comparison: dict[str, dict[str, Any]] = {}
    best_search: GridSearchCV | None = None
    best_name = ""
    best_score = float("-inf")

    for name, (classifier, parameters) in candidates.items():
        pipeline = make_pipeline(
            classifier,
            include_character_features=use_character_features,
        )
        search = GridSearchCV(
            pipeline,
            param_grid=parameters,
            scoring="f1_macro",
            cv=inner_splits,
            refit=True,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        started = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - started
        mean_score = float(search.best_score_)
        comparison[name] = {
            "cv_macro_f1": round(mean_score * 100, 2),
            "cv_macro_f1_std": round(float(search.cv_results_["std_test_score"][search.best_index_]) * 100, 2),
            "best_parameters": search.best_params_,
            "training_seconds": round(elapsed, 3),
        }
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_search = search

    if best_search is None:
        raise RuntimeError("No model candidate completed training")

    model = best_search.best_estimator_
    selected_classifier = model.named_steps["classifier"]
    model.set_params(
        classifier=CalibratedClassifierCV(
            estimator=selected_classifier,
            method="sigmoid",
            cv=3,
        )
    )
    final_training_started = time.perf_counter()
    model.fit(X_train, y_train)
    final_training_seconds = time.perf_counter() - final_training_started
    prediction_started = time.perf_counter()
    model.predict(X_test)
    prediction_seconds = time.perf_counter() - prediction_started
    metrics = evaluate_model(model, X_test, y_test)
    classifier = model.named_steps["classifier"]
    scores = None
    score_type = metrics["score_type"]
    if score_type in {"calibrated_probability", "predicted_class_probability"}:
        scores = model.predict_proba(X_test)[:, list(classifier.classes_).index(1)]
    elif score_type == "decision_score":
        scores = np.asarray(model.decision_function(X_test)).ravel()
    metrics.update(
        {
            "algorithm": best_name,
            "primary_selection_metric": "3-fold group-aware validation macro F1",
            "validation_macro_f1": comparison[best_name]["cv_macro_f1"],
            "validation_macro_f1_std": comparison[best_name]["cv_macro_f1_std"],
            "selection_parameters": comparison[best_name]["best_parameters"],
            "training_articles": int(len(train_indices)),
            "testing_articles": int(len(test_indices)),
            "total_dataset_articles": dataset_report.rows_used,
            "dataset_report": dataset_report.to_dict(),
            "class_counts": dataset_report.class_counts,
            "training_seconds": round(
                comparison[best_name]["training_seconds"] + final_training_seconds,
                3,
            ),
            "prediction_seconds": round(prediction_seconds, 6),
            "random_state": random_state,
            "sklearn_version": sklearn.__version__,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_comparison": comparison,
        }
    )

    save_evaluation(metrics, y_test, scores, score_type, results_path)
    save_model_comparison(comparison, results_path)
    joblib.dump(model, models_path / "best_model.pkl")
    joblib.dump(model.named_steps["features"], models_path / "vectorizer.pkl")
    metadata = {
        "model_version": metrics["trained_at_utc"],
        "algorithm": best_name,
        "label_mapping": {"0": "FAKE NEWS", "1": "REAL NEWS"},
        "score_type": score_type,
        "calibration": {"method": "sigmoid", "folds": 3},
        "selection_criterion": metrics["primary_selection_metric"],
        "selection_parameters": comparison[best_name]["best_parameters"],
        "random_state": random_state,
        "sklearn_version": sklearn.__version__,
        "dataset_report": dataset_report.to_dict(),
        "training_articles": int(len(train_indices)),
        "testing_articles": int(len(test_indices)),
        "features": {
            "max_features": 50000,
            "word_ngram_range": [1, 2],
            "character_features": use_character_features,
        },
    }
    (models_path / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = [
        "Fake News Detection model training report",
        f"Generated: {metrics['trained_at_utc']}",
        "Dataset source: dataset/Fake.csv and dataset/True.csv",
        "Split: reproducible 80/20 group-aware stratified holdout; exact content duplicates removed before splitting.",
        f"Dataset report: {json.dumps(dataset_report.to_dict(), sort_keys=True)}",
        f"Selection metric: {metrics['primary_selection_metric']}",
        "Validation comparison: " + json.dumps(comparison, sort_keys=True),
        "Untouched test metrics: " + json.dumps(
            {key: metrics[key] for key in ("accuracy", "precision", "recall", "macro_f1", "weighted_f1", "roc_auc", "pr_auc", "score_type")},
            sort_keys=True,
        ),
    ]
    (results_path / "training_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train()
    print(json.dumps(
        {
            "algorithm": result["algorithm"],
            "validation_macro_f1": result["validation_macro_f1"],
            "test_macro_f1": result["macro_f1"],
            "artifacts": ["models/best_model.pkl", "models/vectorizer.pkl", "models/model_metadata.json"],
            "metrics": "results/metrics.json",
        },
        indent=2,
    ))
