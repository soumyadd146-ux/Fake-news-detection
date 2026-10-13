import os
import re
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


# ============================================================
# 1. CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

fake_data = pd.read_csv("dataset/Fake.csv")
true_data = pd.read_csv("dataset/True.csv")

print("Fake news articles:", len(fake_data))
print("Real news articles:", len(true_data))


# ============================================================
# 3. ADD LABELS
# ============================================================

fake_data["label"] = 0
true_data["label"] = 1

print("\nLabels added:")
print("0 = Fake News")
print("1 = Real News")


# ============================================================
# 4. COMBINE DATA
# ============================================================

data = pd.concat([fake_data, true_data], ignore_index=True)

print("\nTotal articles:", len(data))


# ============================================================
# 5. COMBINE TITLE AND ARTICLE TEXT
# ============================================================

data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

data["content"] = data["title"] + " " + data["text"]


# ============================================================
# 6. TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("\nCleaning text...")

data["content"] = data["content"].apply(clean_text)


# ============================================================
# 7. REMOVE EMPTY ARTICLES
# ============================================================

data = data[data["content"].str.len() > 0]

print("Articles after cleaning:", len(data))


# ============================================================
# 8. PREPARE INPUT AND OUTPUT
# ============================================================

X = data["content"]
y = data["label"]


# ============================================================
# 9. SPLIT DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining articles:", len(X_train))
print("Testing articles:", len(X_test))


# ============================================================
# 10. TF-IDF FEATURE EXTRACTION
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF feature extraction completed.")
print("Number of features:", len(vectorizer.get_feature_names_out()))


# ============================================================
# 11. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed.")


# ============================================================
# 12. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_tfidf)

y_probability = model.predict_proba(X_test_tfidf)[:, 1]


# ============================================================
# 13. CALCULATE PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake", "Real"]
)

disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.close()


# ============================================================
# 15. ROC CURVE AND AUC
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, y_probability)

roc_auc = auc(fpr, tpr)

print(f"\nROC-AUC   : {roc_auc:.4f}")

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.tight_layout()

plt.savefig("results/roc_curve.png", dpi=300)
plt.close()


# ============================================================
# 16. SAVE MODEL PERFORMANCE
# ============================================================

metrics = {
    "algorithm": "Logistic Regression",
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "f1_score": round(f1 * 100, 2),
    "roc_auc": round(roc_auc, 4),
    "training_articles": int(len(X_train)),
    "testing_articles": int(len(X_test)),
    "total_dataset_articles": int(len(data))
}

with open("results/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


# ============================================================
# 17. SAVE MODEL AND VECTORIZER
# ============================================================

joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n========================================")
print("FILES SAVED SUCCESSFULLY")
print("========================================")

print("Model       : models/model.pkl")
print("Vectorizer  : models/vectorizer.pkl")
print("Metrics     : results/metrics.json")
print("Confusion   : results/confusion_matrix.png")
print("ROC Curve   : results/roc_curve.png")

print("\nTraining completed successfully! 🎉")