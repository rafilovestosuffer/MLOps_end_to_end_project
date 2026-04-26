# MLOps End-to-End Income Prediction Project

A production-grade modular MLOps pipeline that predicts whether a person earns **above or below $50,000/year** based on demographic and employment data.

---

## Project Structure

```
MLOps_end_to_end_project/
├── src/
│   ├── logger.py                    # Timestamped file-based logging
│   ├── exception.py                 # Custom exception with file + line detail
│   ├── utils.py                     # save_object / load_object (pickle)
│   ├── components/
│   │   ├── data_ingestion.py        # Read CSV → split train/test
│   │   ├── data_transformation.py   # Impute + scale + Feast feature store
│   │   ├── model_trainer.py         # GridSearchCV + MLflow tracking
│   │   └── model_monitoring.py      # Drift detection + performance tracking
│   └── pipelines/
│       ├── training_pipeline.py     # Orchestrates ingestion → transform → train
│       └── prediction_pipeline.py   # Load artifacts → predict on new data
├── artifacts/
│   ├── data_ingestion/              # train.csv, test.csv, raw.csv
│   ├── data_transformation/         # preprocessor.pkl
│   └── model_trainer/               # model.pkl
├── data-source/                     # Place income_cleandata.csv here
├── feature_repo/                    # Feast feature store config + parquet
├── logs/                            # Auto-generated timestamped log files
├── notebooks/                       # EDA and experimentation
├── tests/                           # Unit tests (pytest)
├── main.py                          # Run full training pipeline
├── app.py                           # Flask prediction API
├── setup.py                         # Package configuration
├── requirements.txt                 # Python dependencies
├── template.py                      # Python project scaffolding script
├── file-setup.sh                    # Bash project scaffolding script
├── config.yaml                      # Centralized hyperparameter config
├── Dockerfile                       # Container for Flask API
├── docker-compose.yml               # Flask + MLflow services
├── Makefile                         # Shortcuts for common commands
└── .env.example                     # Environment variable template
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## Run Training Pipeline

```bash
# Start MLflow server first
mlflow server --host 0.0.0.0 --port 5000

# Run full pipeline
python main.py
```

---

## Run Prediction API

```bash
python app.py
```

Test it:
```bash
python test-request.py
```

Or with curl:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":39,"workclass":7,"education_num":13,"marital_status":1,"occupation":4,"relationship":1,"race":4,"sex":1,"capital_gain":2174,"capital_loss":0,"hours_per_week":40,"native_country":39}'
```

---

## Run with Docker

```bash
docker-compose up
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| scikit-learn | ML models + preprocessing |
| MLflow | Experiment tracking + model registry |
| Feast | Feature store |
| Flask | Prediction API |
| pandas / numpy | Data manipulation |
| pickle | Model serialization |
| pytest | Unit testing |
| Docker | Containerization |
