from flask import Flask, render_template, request
import joblib
import re
import os
import sqlite3
import json
from datetime import datetime


# --------------------------------------------------
# Flask Application
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE = os.path.join(BASE_DIR, "database", "history.db")
METRICS = os.path.join(BASE_DIR, "results", "metrics.json")

# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
METRICS_PATH = "results/metrics.json"

DATABASE_FOLDER = "database"
DATABASE_PATH = os.path.join(DATABASE_FOLDER, "history.db")


# --------------------------------------------------
# Load trained model and TF-IDF vectorizer
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# --------------------------------------------------
# Load model metrics
# --------------------------------------------------

METRICS_PATH = os.path.join("results", "metrics.json")

with open(METRICS_PATH, "r") as file:
    METRICS = json.load(file)


# --------------------------------------------------
# Create database
# --------------------------------------------------

def init_database():

    os.makedirs(DATABASE_FOLDER, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Keep only alphabets and spaces
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# --------------------------------------------------
# Home / Analyze page
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form.get("news", "").strip()

    if not news:
        return render_template(
            "index.html",
            error="Please enter a news article."
        )

    # Clean input
    cleaned_news = clean_text(news)

    # Convert text into TF-IDF features
    news_vector = vectorizer.transform(
        [cleaned_news]
    )

    # Predict
    prediction = model.predict(
        news_vector
    )[0]

    # Get probabilities for both classes
    probabilities = model.predict_proba(
        news_vector
    )[0]

    # Get confidence of the predicted class
    if prediction == 1:
        confidence = round(
            probabilities[1] * 100,
            2
        )
    else:
        confidence = round(
            probabilities[0] * 100,
            2
        )

    # Label
    if prediction == 1:
        result = "REAL NEWS"
    else:
        result = "FAKE NEWS"

    # --------------------------------------------------
    # Save prediction to database
    # --------------------------------------------------

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (news, result, confidence, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        news,
        result,
        confidence,
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    ))

    connection.commit()
    connection.close()

    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    return render_template(
        "result.html",
        news=news,
        result=result,
        confidence=confidence,
        model_name="Logistic Regression"
    )


# --------------------------------------------------
# History page
# --------------------------------------------------

@app.route("/history")
def history():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, news, result, confidence, created_at
        FROM predictions
        ORDER BY id DESC
    """)

    predictions = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        predictions=predictions
    )


# --------------------------------------------------
# Dashboard
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("database/history.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Summary counts
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_articles = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE result = 'REAL NEWS'
    """)
    real_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE result = 'FAKE NEWS'
    """)
    fake_count = cursor.fetchone()[0]

    # Recent predictions
    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
        LIMIT 8
    """)
    recent_predictions = cursor.fetchall()

    # Daily prediction activity (last 7 days)
    cursor.execute("""
        SELECT DATE(created_at) AS day, COUNT(*) AS total
        FROM predictions
        WHERE DATE(created_at) >= DATE('now', '-6 days')
        GROUP BY DATE(created_at)
        ORDER BY day
    """)
    daily_data = cursor.fetchall()

    conn.close()

    # Prepare chart labels and values
    activity_labels = [row["day"] for row in daily_data]
    activity_values = [row["total"] for row in daily_data]

    return render_template(
        "dashboard.html",
        total_articles=total_articles,
        real_count=real_count,
        fake_count=fake_count,
      accuracy=round(METRICS.get("accuracy", 0), 2),
      precision=round(METRICS.get("precision", 0), 2),
      recall=round(METRICS.get("recall", 0), 2),
      f1_score=round(METRICS.get("f1_score", 0), 2),
        recent_predictions=recent_predictions,
        activity_labels=activity_labels,
        activity_values=activity_values
    )

# --------------------------------------------------
# About page
# --------------------------------------------------

@app.route("/about")
def about():

    return render_template(
        "about.html",
          metrics=METRICS
    )


# --------------------------------------------------
# Start application
# --------------------------------------------------

if __name__ == "__main__":

    init_database()

    app.run(
        debug=True
    )