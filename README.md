<div align="center">

# IncomeIQ — ML Income Predictor

**A production-grade MLOps pipeline that predicts whether someone earns above or below $50K/year.**

Built with real-world engineering practices: modular components, experiment tracking, a feature store, containerised deployment, and an interactive web UI.

[![Python](https://img.shields.io/badge/Python-3.10-3776ab?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-API-black?logo=flask)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-f7931e?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194e2?logo=mlflow)](https://mlflow.org)
[![Feast](https://img.shields.io/badge/Feast-Feature_Store-6b4fbb)](https://feast.dev)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ed?logo=docker&logoColor=white)](https://docker.com)

</div>

---

## What this project does

Most ML tutorials stop at training a model in a notebook. This project goes further — it builds the full pipeline you'd find at a real company:

1. **Raw data in** → ingestion, cleaning, train/test split  
2. **Features engineered** → imputation, scaling, pushed to a Feast feature store  
3. **Three models trained** → RandomForest, DecisionTree, LogisticRegression — all tuned with GridSearchCV, all tracked in MLflow  
4. **Best model saved** → serialised to disk, ready for inference  
5. **REST API deployed** → Flask endpoint returns a prediction + confidence score in milliseconds  
6. **Interactive UI** → anyone can open a browser, fill in a form, and get a prediction without touching code  

---

## Live Demo

> Deploy to Render in 2 minutes — see [Deployment](#deployment) section below.

The app runs in **demo mode** (heuristic predictions) until you train the model with real data. Once `python main.py` is run and artifacts are present, it automatically switches to the real ML model.

---

## Project Structure

```
MLOps_end_to_end_project/
│
├── src/
│   ├── logger.py                    # Timestamped file-based logging
│   ├── exception.py                 # Custom exception — captures file + line number
│   ├── utils.py                     # save_object / load_object helpers
│   │
│   ├── components/
│   │   ├── data_ingestion.py        # Reads CSV → 80/20 train/test split
│   │   ├── data_transformation.py   # Impute + scale + push to Feast feature store
│   │   ├── model_trainer.py         # GridSearchCV across 3 models + MLflow logging
│   │   └── model_monitoring.py      # Drift detection (extensible)
│   │
│   └── pipelines/
│       ├── training_pipeline.py     # Chains: ingest → transform → train
│       └── prediction_pipeline.py   # Loads artifacts → preprocesses → predicts
│
├── artifacts/                       # Generated at training time (git-ignored)
│   ├── data_ingestion/              # raw.csv, train.csv, test.csv
│   ├── data_transformation/         # preprocessor.pkl
│   └── model_trainer/               # model.pkl
│
├── templates/index.html             # Interactive web UI (dark glassmorphism theme)
├── static/css/style.css             # UI styles
│
├── feature_repo/                    # Feast feature store config + parquet files
├── data-source/                     # Place income_cleandata.csv here before training
├── logs/                            # Auto-generated timestamped logs
├── notebooks/                       # EDA and experimentation
├── tests/                           # Unit tests (pytest)
│
├── app.py                           # Flask app — serves UI + prediction API
├── main.py                          # Entry point: run full training pipeline
├── demo_profiles.json               # 3 ready-to-use test profiles with curl examples
├── test-request.py                  # Sends all 3 demo profiles to the running API
│
├── config.yaml                      # Centralised config: paths, features, hyperparams
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package config (pip install -e .)
├── Dockerfile                       # Container image for the Flask API
├── docker-compose.yml               # Flask + MLflow services together
├── Procfile                         # Gunicorn entry point (Render / Heroku)
├── render.yaml                      # One-click Render deploy config
└── Makefile                         # Shortcuts: make train, make serve, make test
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/rafilovestosuffer/MLOps_end_to_end_project.git
cd MLOps_end_to_end_project
pip install -r requirements.txt
pip install -e .
```

### 2. Add your data

Place `income_cleandata.csv` in the `data-source/` folder.  
The dataset should have these columns:

```
age, workclass, education_num, marital_status, occupation,
relationship, race, sex, capital_gain, capital_loss,
hours_per_week, native_country, income
```

### 3. Train the model

```bash
# Optional: start MLflow tracking server in a separate terminal
mlflow server --host 0.0.0.0 --port 5001

# Run the full training pipeline
python main.py
```

This runs data ingestion → feature transformation → model training → saves `model.pkl` and `preprocessor.pkl` to `artifacts/`.

### 4. Start the web app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser — fill in the form, click **Predict**, and get an instant income prediction with a confidence score.

### 5. Test the API

```bash
python test-request.py
```

Or with curl directly:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45, "workclass": 3, "education_num": 14,
    "marital_status": 2, "occupation": 3, "relationship": 0,
    "race": 4, "sex": 1, "capital_gain": 15024,
    "capital_loss": 0, "hours_per_week": 50, "native_country": 38
  }'
```

**Response:**
```json
{
  "status": "success",
  "prediction": 1,
  "income_category": ">50K",
  "confidence": 87.4,
  "demo_mode": false
}
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Interactive prediction UI |
| `GET`  | `/health`| Service health + mode (production / demo) |
| `POST` | `/predict` | Income prediction — returns category + confidence |

### Feature encoding

All categorical inputs are label-encoded (alphabetical order, matching sklearn's `LabelEncoder`):

| Feature | Type | Range / Values |
|---------|------|----------------|
| `age` | int | 17–90 |
| `workclass` | int (encoded) | 0=Federal-gov … 7=Without-pay |
| `education_num` | int | 1=Preschool … 16=Doctorate |
| `marital_status` | int (encoded) | 0=Divorced … 6=Widowed |
| `occupation` | int (encoded) | 0=Adm-clerical … 13=Transport |
| `relationship` | int (encoded) | 0=Husband … 5=Wife |
| `race` | int (encoded) | 0=Amer-Indian … 4=White |
| `sex` | int (encoded) | 0=Female, 1=Male |
| `capital_gain` | int | 0–99999 |
| `capital_loss` | int | 0–4356 |
| `hours_per_week` | int | 1–99 |
| `native_country` | int (encoded) | 0=Cambodia … 40=Yugoslavia |

---

## Run with Docker

```bash
# Runs Flask API + MLflow tracking server together
docker-compose up
```

- Flask API → [http://localhost:5000](http://localhost:5000)
- MLflow UI → [http://localhost:5001](http://localhost:5001)

---

## Run Tests

```bash
pytest tests/ -v
```

Tests cover: custom exception, logger, prediction pipeline DataFrame shape, and utility functions (save/load object).

---

## Deployment

### Deploy to Render (free)

1. Fork or push this repo to your GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect the repo — Render reads `render.yaml` automatically
4. Click **Create Web Service** — live URL in ~3 minutes

The app runs in **demo mode** (heuristic predictions) until model artifacts are present. Train the model locally and commit the artifacts to enable real predictions on the deployed version.

---

## MLOps Pipeline

```
Raw CSV
   │
   ▼
Data Ingestion ──────── 80/20 split → artifacts/data_ingestion/
   │
   ▼
Data Transformation ─── Impute (median) + StandardScaler + Feast push → preprocessor.pkl
   │
   ▼
Model Training ─────── GridSearchCV × 3 models, MLflow logging → model.pkl
   │
   ▼
Flask API ──────────── POST /predict → load artifacts → preprocess → predict → JSON
   │
   ▼
Web UI ─────────────── Interactive form → result card with confidence score
```

---

## Tech Stack

| Tool | Role |
|------|------|
| **scikit-learn** | RandomForest, DecisionTree, LogisticRegression + GridSearchCV + StandardScaler |
| **MLflow** | Experiment tracking, metric logging, model registry |
| **Feast** | Feature store — materialise and serve training features |
| **Flask** | REST API + HTML template serving |
| **Gunicorn** | Production WSGI server |
| **pandas / numpy** | Data manipulation and array ops |
| **Docker / Compose** | Containerised API + MLflow server |
| **pytest** | Unit tests for all pipeline components |

---

## About

Built by [Rafi](https://github.com/rafilovestosuffer) as a full end-to-end MLOps project — covering everything from raw data to a deployed, interactive prediction service.

If you find this useful, give it a ⭐ — it helps a lot.
