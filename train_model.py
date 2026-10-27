# ============================================================
# Fake News Detection - Model Training
# TF-IDF + Logistic Regression
# ISOT Fake News Dataset
# ============================================================

import os
import json
import time
import joblib

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.preprocessing import combine_title_and_body, load_labeled_examples

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAKE_PATH = os.path.join(BASE_DIR, "dataset", "Fake.csv")
REAL_PATH = os.path.join(BASE_DIR, "dataset", "True.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")

CONFUSION_PATH = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

ROC_PATH = os.path.join(
    RESULTS_DIR,
    "roc_curve.png"
)


# ============================================================
# Create folders
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Load ISOT dataset
# ============================================================

print("\nLoading ISOT Fake News Dataset...")

fake_df = pd.read_csv(FAKE_PATH)
real_df = pd.read_csv(REAL_PATH)

print("Fake news records:", len(fake_df))
print("Real news records:", len(real_df))


# ============================================================
# Add labels
# 0 = Fake News
# 1 = Real News
# ============================================================

fake_df["label"] = 0
real_df["label"] = 1


# ============================================================
# Combine datasets
# ============================================================

df = pd.concat(
    [fake_df, real_df],
    ignore_index=True
)


# ============================================================
# Combine title and text
# ============================================================

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

df["content"] = [
    combine_title_and_body(title, text)
    for title, text in zip(df["title"], df["text"])
]


# ============================================================
# Remove empty records
# ============================================================

df = df[df["content"].str.strip() != ""]

df = df.reset_index(drop=True)

print("Total records after cleaning:", len(df))


# ============================================================
# Input and target
# ============================================================

X = df["content"]
y = df["label"]


# ============================================================
# Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

base_training_articles = len(X_train)
verified_examples = load_labeled_examples(os.path.join(BASE_DIR, "dataset"))
if not verified_examples.empty:
    weighted_examples = verified_examples.loc[
        verified_examples.index.repeat(verified_examples["weight"])
    ]
    X_train = pd.concat(
        [X_train.reset_index(drop=True), weighted_examples["content"]],
        ignore_index=True,
    )
    y_train = pd.concat(
        [y_train.reset_index(drop=True), weighted_examples["label"]],
        ignore_index=True,
    )

print("\nTraining articles:", base_training_articles + len(verified_examples))
print("Effective training samples (including example weights):", len(X_train))
print("Testing records:", len(X_test))
print("Supplemental labeled training examples:", len(verified_examples))


# ============================================================
# TF-IDF Feature Extraction
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# ============================================================
# Logistic Regression Model
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ============================================================
# Train Model
# ============================================================

print("\nTraining Logistic Regression model...")

training_start = time.time()

model.fit(
    X_train_tfidf,
    y_train
)

training_time = time.time() - training_start

print(
    "Training completed in:",
    round(training_time, 2),
    "seconds"
)


# ============================================================
# Prediction
# ============================================================

print("\nMaking predictions...")

prediction_start = time.time()

y_pred = model.predict(X_test_tfidf)

# Probability of class 1 (Real News)
y_prob = model.predict_proba(X_test_tfidf)[:, 1]

prediction_time = time.time() - prediction_start

print(
    "Prediction completed in:",
    round(prediction_time, 4),
    "seconds"
)


# ============================================================
# Calculate Evaluation Metrics
# ============================================================

print("\nCalculating evaluation metrics...")

accuracy = float(
    accuracy_score(y_test, y_pred) * 100
)

precision = float(
    precision_score(y_test, y_pred) * 100
)

recall = float(
    recall_score(y_test, y_pred) * 100
)

f1 = float(
    f1_score(y_test, y_pred) * 100
)

roc_auc = float(
    roc_auc_score(y_test, y_prob) * 100
)


# ============================================================
# 5-Fold Cross Validation
# ============================================================

print("\nRunning 5-fold cross-validation...")

cv_scores = cross_val_score(
    model,
    vectorizer.transform(X_train.iloc[:base_training_articles]),
    y_train.iloc[:base_training_articles],
    cv=5,
    scoring="f1_macro"
)

cv_macro_f1 = float(
    cv_scores.mean() * 100
)

cv_std = float(
    cv_scores.std() * 100
)

print(
    "Cross-validation Macro F1:",
    round(cv_macro_f1, 2),
    "%"
)

print(
    "Cross-validation Std:",
    round(cv_std, 2),
    "%"
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake News",
            "Real News"
        ]
    )
)


# ============================================================
# Save Confusion Matrix
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Fake News", "Real News"]
)

plt.yticks(
    [0, 1],
    ["Fake News", "Real News"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    CONFUSION_PATH,
    dpi=300
)

plt.close()


# ============================================================
# ROC Curve
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc / 100:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    ROC_PATH,
    dpi=300
)

plt.close()


# ============================================================
# Save Model
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)


# ============================================================
# Save Metrics
# ============================================================

metrics = {

    "accuracy": round(accuracy, 2),

    "precision": round(precision, 2),

    "recall": round(recall, 2),

    "f1_score": round(f1, 2),

    "roc_auc": round(roc_auc, 2),

    "cv_macro_f1": round(cv_macro_f1, 2),

    "cv_std": round(cv_std, 2),

    "training_time": round(training_time, 2),

    "prediction_time": round(prediction_time, 4),

    "confusion_matrix": cm.tolist(),

    "model": "Logistic Regression",

    "feature_extraction": "TF-IDF",

    "dataset": "ISOT Fake News Dataset",

    "cv_folds": 5
}

metrics.update(
    {
        "training_articles": int(base_training_articles + len(verified_examples)),
        "testing_articles": int(len(X_test)),
        "total_dataset_articles": int(len(X) + len(verified_examples)),
        "labeled_training_examples": int(len(verified_examples)),
        "labeled_training_example_weight": int(verified_examples["weight"].sum()),
    }
)


with open(
    METRICS_PATH,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# Final Output
# ============================================================

print("\n" + "=" * 60)

print("MODEL TRAINING COMPLETED")

print("=" * 60)

print(
    f"Accuracy             : {accuracy:.2f}%"
)

print(
    f"Precision            : {precision:.2f}%"
)

print(
    f"Recall               : {recall:.2f}%"
)

print(
    f"F1-Score             : {f1:.2f}%"
)

print(
    f"ROC-AUC              : {roc_auc:.2f}%"
)

print(
    f"CV Macro F1          : {cv_macro_f1:.2f}%"
)

print(
    f"CV Standard Deviation: {cv_std:.2f}%"
)

print(
    f"Training Time        : {training_time:.2f} sec"
)

print(
    f"Prediction Time      : {prediction_time:.4f} sec"
)

print("=" * 60)

print("\nFiles saved successfully:")

print(MODEL_PATH)
print(VECTORIZER_PATH)
print(METRICS_PATH)
print(CONFUSION_PATH)
print(ROC_PATH)

print("\nTraining completed successfully.")