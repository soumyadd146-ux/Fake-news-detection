import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _scores(model: Any, documents: Any) -> tuple[np.ndarray | None, str]:
    if hasattr(model, "predict_proba"):
        classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
        probabilities = model.predict_proba(documents) if hasattr(model, "named_steps") else None
        if probabilities is None:
            raise ValueError("Expected a full model pipeline when using predict_proba")
        classes = list(classifier.classes_)
        score_type = "calibrated_probability" if hasattr(classifier, "calibrated_classifiers_") else "predicted_class_probability"
        return np.asarray(probabilities)[:, classes.index(1)], score_type
    if hasattr(model, "decision_function"):
        values = model.decision_function(documents)
        if hasattr(model, "named_steps"):
            return np.asarray(values).ravel(), "decision_score"
    return None, "unavailable"


def evaluate_model(model: Any, X_test: Any, y_test: Any) -> dict[str, Any]:
    predictions = model.predict(X_test)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["FAKE NEWS", "REAL NEWS"],
        output_dict=True,
        zero_division=0,
    )
    result: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, predictions)) * 100, 2),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)) * 100, 2),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro", zero_division=0)) * 100, 2),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)) * 100, 2),
        "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)) * 100, 2),
        "per_class": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }
    scores, score_type = _scores(model, X_test)
    result["score_type"] = score_type
    result["roc_auc"] = None
    result["pr_auc"] = None
    if scores is not None and len(np.unique(y_test)) == 2:
        result["roc_auc"] = round(float(roc_auc_score(y_test, scores)), 6)
        result["pr_auc"] = round(float(average_precision_score(y_test, scores)), 6)
    return result


def save_evaluation(
    metrics: dict[str, Any],
    y_test: Any,
    scores: np.ndarray | None,
    score_type: str,
    results_dir: str | Path,
) -> None:
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["FAKE NEWS", "REAL NEWS"],
        yticklabels=["FAKE NEWS", "REAL NEWS"],
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix (held-out test set)",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(root / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    if scores is not None and len(np.unique(y_test)) == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(false_positive_rate, true_positive_rate, label=f"Test ROC AUC={metrics['roc_auc']}")
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC curve ({score_type})")
        axis.legend()
        figure.tight_layout()
        figure.savefig(root / "roc_curve.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(recall, precision, label=f"Test PR AUC={metrics['pr_auc']}")
        axis.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall curve ({score_type})")
        axis.legend()
        figure.tight_layout()
        figure.savefig(root / "precision_recall_curve.png", dpi=180)
        plt.close(figure)


def save_model_comparison(results: dict[str, dict[str, Any]], results_dir: str | Path) -> None:
    names = list(results)
    scores = [results[name]["cv_macro_f1"] for name in names]
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(names, scores, color="#5267e8")
    axis.bar_label(bars, fmt="%.2f")
    axis.set(ylabel="Validation macro F1 (%)", title="Cross-validation model comparison")
    axis.tick_params(axis="x", rotation=18)
    figure.tight_layout()
    figure.savefig(Path(results_dir) / "model_comparison.png", dpi=180)
    plt.close(figure)
