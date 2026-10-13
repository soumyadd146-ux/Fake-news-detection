import json
import logging
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, session

from config import BASE_DIR, MAX_INPUT_CHARACTERS
from src.predict import PredictionError, load_artifacts, predict_text


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=MAX_INPUT_CHARACTERS * 4,
)
app.secret_key = app.config["SECRET_KEY"]

DATABASE = str(BASE_DIR / "database" / "history.db")
METRICS_PATH = BASE_DIR / "results" / "metrics.json"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
NEW_MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
LEGACY_MODEL_PATH = BASE_DIR / "models" / "model.pkl"
LEGACY_VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"

try:
    METRICS = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    logger.exception("Unable to load evaluation metrics from %s", METRICS_PATH)
    METRICS = {}

try:
    MODEL_METADATA = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
except FileNotFoundError:
    MODEL_METADATA = {"algorithm": "Logistic Regression", "model_version": "legacy"}
except json.JSONDecodeError:
    logger.exception("Invalid model metadata at %s", METADATA_PATH)
    MODEL_METADATA = {"algorithm": "Unknown", "model_version": "unknown"}

_model: Any | None = None
_vectorizer: Any | None = None
_model_error: str | None = None
_model_loaded = False


@contextmanager
def database_connection():
    connection = sqlite3.connect(DATABASE)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def clean_text(text: object) -> str:
    from src.preprocessing import clean_text as preprocess

    return preprocess(text)


def init_database() -> None:
    database_path = Path(DATABASE)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with database_connection() as connection:
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                news TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
        }
        migrations = {
            "title": "ALTER TABLE predictions ADD COLUMN title TEXT NOT NULL DEFAULT ''",
            "score_type": (
                "ALTER TABLE predictions ADD COLUMN score_type TEXT NOT NULL "
                "DEFAULT 'predicted_class_probability'"
            ),
            "score_value": "ALTER TABLE predictions ADD COLUMN score_value REAL",
            "model_version": (
                "ALTER TABLE predictions ADD COLUMN model_version TEXT NOT NULL DEFAULT 'legacy'"
            ),
        }
        for column, statement in migrations.items():
            if column not in columns:
                connection.execute(statement)
        connection.execute(
            """
            UPDATE predictions
            SET score_value = confidence / 100.0
            WHERE score_value IS NULL
              AND score_type = 'predicted_class_probability'
            """
        )


init_database()


def _load_model() -> tuple[Any | None, Any | None, str | None]:
    global _model, _vectorizer, _model_error, _model_loaded
    if _model_loaded:
        return _model, _vectorizer, _model_error
    model_path = NEW_MODEL_PATH if NEW_MODEL_PATH.is_file() else LEGACY_MODEL_PATH
    vectorizer_path = None if model_path == NEW_MODEL_PATH else LEGACY_VECTORIZER_PATH
    try:
        _model, _vectorizer = load_artifacts(model_path, vectorizer_path)
        _model_error = None
    except (OSError, ValueError, TypeError, ImportError) as error:
        logger.exception("Model artifact loading failed")
        _model_error = str(error)
    _model_loaded = True
    return _model, _vectorizer, _model_error


def _csrf_token() -> str:
    token = session.get("_csrf_token")
    if token is None:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


app.jinja_env.globals["csrf_token"] = _csrf_token
app.jinja_env.globals["model_name"] = MODEL_METADATA.get("algorithm", "Logistic Regression")
app.jinja_env.globals["max_input_characters"] = MAX_INPUT_CHARACTERS


def _predict_and_store(title: str, body: str) -> dict[str, Any]:
    model, vectorizer, model_error = _load_model()
    if model is None:
        raise RuntimeError("The prediction model is unavailable.") from (
            RuntimeError(model_error) if model_error else None
        )

    started = time.perf_counter()
    outcome = predict_text(
        model,
        vectorizer,
        title=title,
        body=body,
        max_characters=MAX_INPUT_CHARACTERS,
    )
    outcome["prediction_seconds"] = round(time.perf_counter() - started, 6)
    outcome["model_name"] = MODEL_METADATA.get(
        "algorithm",
        getattr(model, "named_steps", {}).get("classifier", model).__class__.__name__,
    )
    outcome["model_version"] = MODEL_METADATA.get("model_version", "legacy")

    news = body.strip() or title.strip()
    if title.strip() and body.strip():
        news = f"{title.strip()}\n\n{body.strip()}"
    confidence = outcome["confidence"] if outcome["confidence"] is not None else 0.0
    with database_connection() as connection:
        connection.execute(
            """
            INSERT INTO predictions
                (news, result, confidence, created_at, title, score_type, score_value, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                news,
                outcome["label"],
                confidence,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                title.strip(),
                outcome["score_type"],
                outcome["score_value"],
                outcome["model_version"],
            ),
        )
    return outcome


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not secrets.compare_digest(request.form.get("csrf_token", ""), session.get("_csrf_token", "")):
        return render_template("index.html", error="Your form session expired. Please try again."), 400

    title = request.form.get("title", "")
    news = request.form.get("news", request.form.get("text", "")).strip()
    try:
        outcome = _predict_and_store(title, news)
    except PredictionError as error:
        return render_template("index.html", error=str(error), title=title, news=news), 400
    except RuntimeError:
        return render_template(
            "index.html",
            error="The prediction model is unavailable. Please contact the administrator.",
        ), 503
    except sqlite3.Error:
        logger.exception("Unable to save prediction history")
        return render_template(
            "index.html",
            error="The prediction could not be saved. Please try again.",
            title=title,
            news=news,
        ), 500
    except Exception:
        logger.exception("Prediction inference failed")
        return render_template(
            "index.html",
            error="The article could not be analyzed. Please try again.",
            title=title,
            news=news,
        ), 500
    return render_template(
        "result.html",
        news=news,
        title=title,
        result=outcome["label"],
        confidence=outcome["confidence"],
        score_type=outcome["score_type"],
        score_value=outcome["score_value"],
        interpretation=outcome["interpretation"],
        explanation=outcome["explanation"],
        model_name=outcome["model_name"],
        prediction_seconds=outcome["prediction_seconds"],
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object."), 400
    title = payload.get("title", "")
    body = payload.get("text", payload.get("body", payload.get("news", "")))
    if not isinstance(title, str) or not isinstance(body, str):
        return jsonify(error="'title' and article text fields must be strings."), 400

    try:
        outcome = _predict_and_store(title, body)
    except PredictionError as error:
        return jsonify(error=str(error)), 400
    except RuntimeError:
        return jsonify(error="The prediction model is unavailable."), 503
    except sqlite3.Error:
        logger.exception("Unable to save API prediction history")
        return jsonify(error="The prediction could not be saved."), 500
    except Exception:
        logger.exception("API prediction inference failed")
        return jsonify(error="The article could not be analyzed."), 500
    return jsonify(
        {
            "prediction": outcome["label"],
            "label_id": outcome["label_id"],
            "score_type": outcome["score_type"],
            "score_value": outcome["score_value"],
            "confidence": outcome["confidence"],
            "interpretation": outcome["interpretation"],
            "explanation": outcome["explanation"],
            "model_name": outcome["model_name"],
            "model_version": outcome["model_version"],
            "prediction_seconds": outcome["prediction_seconds"],
        }
    )


@app.route("/health")
def health():
    model, _, _ = _load_model()
    try:
        with database_connection() as connection:
            connection.execute("SELECT 1")
        database_ok = True
    except sqlite3.Error:
        logger.exception("Health check database query failed")
        database_ok = False
    healthy = model is not None and database_ok
    return jsonify(
        {
            "status": "ok" if healthy else "unavailable",
            "model_loaded": model is not None,
            "database_connected": database_ok,
            "model_version": MODEL_METADATA.get("model_version", "legacy"),
        }
    ), 200 if healthy else 503


@app.route("/history")
def history():
    search = request.args.get("q", "").strip()[:200]
    result_filter = request.args.get("result", "").upper()
    page = request.args.get("page", default=1, type=int)
    page = max(page or 1, 1)
    per_page = 10
    filters: list[str] = []
    parameters: list[Any] = []
    if search:
        filters.append("(news LIKE ? OR title LIKE ?)")
        parameters.extend((f"%{search}%", f"%{search}%"))
    if result_filter in {"REAL NEWS", "FAKE NEWS"}:
        filters.append("result = ?")
        parameters.append(result_filter)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    with database_connection() as connection:
        connection.row_factory = sqlite3.Row
        total = connection.execute(
            f"SELECT COUNT(*) FROM predictions {where_clause}", parameters
        ).fetchone()[0]
        predictions = connection.execute(
            f"""
            SELECT id, news, title, result, confidence, score_type, score_value,
                   model_version, created_at
            FROM predictions {where_clause}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (*parameters, per_page, (page - 1) * per_page),
        ).fetchall()
    total_pages = max((total + per_page - 1) // per_page, 1)
    return render_template(
        "history.html",
        predictions=predictions,
        search=search,
        result_filter=result_filter,
        page=page,
        total_pages=total_pages,
        total_predictions=total,
    )


@app.route("/dashboard")
def dashboard():
    today = datetime.now().date()
    chart_start = today - timedelta(days=29)
    previous_week_start = today - timedelta(days=13)
    current_week_start = today - timedelta(days=6)
    with database_connection() as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        summary = cursor.execute(
            """
            SELECT
                COUNT(*) AS total_articles,
                COALESCE(SUM(CASE WHEN result = 'REAL NEWS' THEN 1 ELSE 0 END), 0) AS real_count,
                COALESCE(SUM(CASE WHEN result = 'FAKE NEWS' THEN 1 ELSE 0 END), 0) AS fake_count,
                AVG(CASE WHEN score_type IN ('predicted_class_probability', 'calibrated_probability') THEN confidence END) AS average_confidence,
                COALESCE(SUM(CASE WHEN DATE(created_at) = ? THEN 1 ELSE 0 END), 0) AS today_count,
                COALESCE(SUM(CASE WHEN score_type IN ('predicted_class_probability', 'calibrated_probability') AND confidence >= 85 THEN 1 ELSE 0 END), 0) AS high_probability_count,
                COALESCE(SUM(CASE WHEN score_type IN ('predicted_class_probability', 'calibrated_probability') AND confidence < 60 THEN 1 ELSE 0 END), 0) AS review_suggested_count,
                COALESCE(SUM(CASE WHEN score_type IN ('predicted_class_probability', 'calibrated_probability') AND confidence >= 60 AND confidence < 85 THEN 1 ELSE 0 END), 0) AS moderate_probability_count
            FROM predictions
            """,
            (today.isoformat(),),
        ).fetchone()
        recent_predictions = cursor.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT 8"
        ).fetchall()
        daily_data = cursor.execute(
            """
            SELECT DATE(created_at) AS day, COUNT(*) AS total
            FROM predictions
            WHERE DATE(created_at) BETWEEN ? AND ?
            GROUP BY DATE(created_at)
            ORDER BY day
            """,
            (chart_start.isoformat(), today.isoformat()),
        ).fetchall()
        weekly_comparison = cursor.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN DATE(created_at) BETWEEN ? AND ? THEN 1 ELSE 0 END), 0) AS previous_week,
                COALESCE(SUM(CASE WHEN DATE(created_at) BETWEEN ? AND ? THEN 1 ELSE 0 END), 0) AS current_week
            FROM predictions
            """,
            (
                previous_week_start.isoformat(),
                (current_week_start - timedelta(days=1)).isoformat(),
                current_week_start.isoformat(),
                today.isoformat(),
            ),
        ).fetchone()

    daily_totals = {row["day"]: row["total"] for row in daily_data}
    activity_days = [chart_start + timedelta(days=offset) for offset in range(30)]
    activity_labels = [f"{day.strftime('%b')} {day.day}" for day in activity_days]
    activity_values = [daily_totals.get(day.isoformat(), 0) for day in activity_days]
    previous_week_count = weekly_comparison["previous_week"]
    current_week_count = weekly_comparison["current_week"]
    weekly_change = (
        round((current_week_count - previous_week_count) / previous_week_count * 100, 1)
        if previous_week_count
        else None
    )
    real_share = round(summary["real_count"] / summary["total_articles"] * 100, 1) if summary["total_articles"] else 0
    fake_share = round(summary["fake_count"] / summary["total_articles"] * 100, 1) if summary["total_articles"] else 0
    model_comparison = METRICS.get("model_comparison", {})
    comparison_entries = [
        {
            "name": name,
            "score": result.get("cv_macro_f1"),
            "selected": name == METRICS.get("algorithm"),
        }
        for name, result in model_comparison.items()
        if isinstance(result, dict) and result.get("cv_macro_f1") is not None
    ]
    comparison_entries.sort(key=lambda entry: entry["score"], reverse=True)
    return render_template(
        "dashboard.html",
        total_articles=summary["total_articles"],
        real_count=summary["real_count"],
        fake_count=summary["fake_count"],
        accuracy=round(METRICS.get("accuracy", 0), 2),
        precision=round(METRICS.get("precision", 0), 2),
        recall=round(METRICS.get("recall", 0), 2),
        f1_score=round(METRICS.get("f1_score", 0), 2),
        roc_auc=round((METRICS.get("roc_auc") or 0) * 100, 2),
        training_articles=METRICS.get("training_articles", 0),
        testing_articles=METRICS.get("testing_articles", 0),
        total_dataset_articles=METRICS.get("total_dataset_articles", 0),
        average_confidence=round(summary["average_confidence"] or 0, 2),
        today_count=summary["today_count"],
        weekly_count=current_week_count,
        previous_week_count=previous_week_count,
        weekly_change=weekly_change,
        real_share=real_share,
        fake_share=fake_share,
        high_probability_count=summary["high_probability_count"],
        moderate_probability_count=summary["moderate_probability_count"],
        review_suggested_count=summary["review_suggested_count"],
        model_comparison=comparison_entries,
        dashboard_date=today.strftime("%A, %B %d, %Y"),
        recent_predictions=recent_predictions,
        activity_labels=activity_labels,
        activity_values=activity_values,
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        metrics=METRICS,
        model_metadata=MODEL_METADATA,
    )


if __name__ == "__main__":
    init_database()
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "").lower() == "true",
    )
