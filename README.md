# Fake News Detection Using Machine Learning

A reproducible NLP project and Flask application for classifying article text using patterns learned from the supplied `Fake.csv` and `True.csv` datasets. It is an educational classifier, not an independent fact-checking service.

> This system classifies news using learned dataset patterns. Predictions are not independent fact-checking or proof that a claim is true.

## Requirements

- Python 3.11 or later (the checked-in dependency pins target the current project environment)
- No GPU is required
- The CSV files in `dataset/`

Create/activate a virtual environment and install the pinned packages:

```powershell
py -3.14 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use the Python version available in your environment if Python 3.14 is not installed.

## Train and evaluate

Training is an explicit command and never runs when the web application starts:

```powershell
python train_model.py
```

The compatibility script delegates to the modular implementation in `src/`. It:

1. Reads and reports the actual source row counts from both CSVs.
2. Assigns labels from source (`Fake.csv` = 0, `True.csv` = 1), combines available title/body text, removes unusable rows and exact duplicates, and records removals/class balance.
3. Makes a reproducible stratified, group-aware 80/20 holdout split. Hyperparameter selection uses only the training portion with three-fold group-aware cross-validation; the held-out test fold is evaluated only after selection.
4. Compares Logistic Regression, Linear SVM, Passive-Aggressive, and Multinomial Naive Bayes using validation macro F1. Set `INCLUDE_RANDOM_FOREST=1` to include the optional Random Forest comparison.
5. Saves the selected pipeline, vectorizer, model metadata, actual test metrics, validation comparison, report, and plots.

Artifacts written by training:

- `models/best_model.pkl` — selected full preprocessing/TF-IDF/classifier pipeline
- `models/vectorizer.pkl` — the fitted feature transformer
- `models/model_metadata.json` — model version, label mapping, split/report details and selection settings
- `results/metrics.json` — separate validation results and untouched test evaluation
- `results/model_comparison.png`, `results/confusion_matrix.png`, `results/roc_curve.png`, and `results/precision_recall_curve.png`
- `results/training_report.txt`

SVM and Passive-Aggressive output raw decision scores; these are not probabilities. Logistic Regression and Multinomial Naive Bayes provide predicted-class probability scores, which should not be interpreted as the probability that a claim is true. The test set is never used for selecting a model.

The included `.pkl` artifacts use Python pickle serialization through joblib. Load only model artifacts you trained yourself or otherwise trust.

## Run the web application

```powershell
python app.py
```

Open <http://127.0.0.1:5000>. The app loads the trained `best_model.pkl` when it exists; before retraining it remains compatible with the original `models/model.pkl` and `models/vectorizer.pkl`.

Optional environment settings:

- `SECRET_KEY` — stable secret for signed browser sessions; a random per-process value is used if unset
- `FLASK_HOST` / `FLASK_PORT` — bind address and port (defaults `127.0.0.1:5000`)
- `FLASK_DEBUG=true` — enable Flask debug mode for local development only
- `MAX_INPUT_CHARACTERS` — maximum combined headline/body length (default 20,000)
- `INCLUDE_RANDOM_FOREST=1` — include the optional Random Forest in a training run
- `INCLUDE_CHARACTER_FEATURES=1` — add the optional character n-gram TF-IDF branch

### Routes

- `/` — headline/body input form with CSRF protection
- `/predict` — browser form prediction
- `/api/predict` — JSON prediction API; accepts `title` and one of `text`, `body`, or `news`
- `/dashboard` — prediction counts, evaluation metrics and Chart.js visualizations
- `/history` — searchable and class-filtered prediction history with pagination
- `/about` — methodology, dataset context and limitations
- `/health` — model/database health status

Example JSON request:

```json
{
  "title": "Example headline",
  "text": "Article body to classify."
}
```

History is stored in `database/history.db`. On startup the app adds the required title, score-type/value, and model-version columns to the existing `predictions` table. Existing rows are preserved and their percentage confidence is backfilled as a legacy predicted-class probability score; no database reset is needed.

## Project layout

```text
.
├── app.py
├── config.py
├── database/
├── dataset/
│   ├── Fake.csv
│   └── True.csv
├── models/
├── results/
├── src/
│   ├── evaluate.py
│   ├── features.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── static/
├── templates/
├── tests/
├── train_model.py
├── requirements.txt
└── README.md
```

## Tests

Run the built-in `unittest` suite:

```powershell
python -m unittest discover -s tests -v
```

The tests cover text cleaning, source labels, duplicate/empty-row handling, prediction score semantics, input validation, CSRF protection, JSON and HTML routes, database persistence, and health checks.

## Data and limitations

The CSVs supplied with this project contain `title`, `text`, `subject`, and `date` columns. The classifier uses the title and article body; subject/date are not model inputs. The source file determines the binary label. Exact normalized-text duplicates are removed before the group-aware holdout and cross-validation splits to reduce duplicate leakage.

This dataset may contain source, topic, time-period and writing-style bias. Split metrics describe performance on a held-out portion of these files, not universal real-world reliability. A high model score does not validate any individual story. Verify claims using multiple independent, reputable sources.
