import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ModelTrainingConfig:
    train_model_file_path = os.path.join("artifacts/model_trainer", "model.pkl")
    mlflow_uri = "http://localhost:5000"
    experiment_name = "Modular_Workflow_Prediction_Pipeline"


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

        mlflow.set_tracking_uri(self.model_trainer_config.mlflow_uri)
        mlflow.set_experiment(self.model_trainer_config.experiment_name)

        self.client = MlflowClient()
        self.run_name = f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def log_metrics(self, y_true, y_pred, prefix=""):
        metrics = {
            f"{prefix}accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}precision": precision_score(y_true, y_pred, average='weighted'),
            f"{prefix}recall": recall_score(y_true, y_pred, average='weighted'),
            f"{prefix}f1": f1_score(y_true, y_pred, average='weighted'),
        }
        mlflow.log_metrics(metrics)
        return metrics

    def train_model(self, X_train, y_train, X_test, y_test, model_name, model, params):
        try:
            with mlflow.start_run(run_name=f"{model_name}_{self.run_name}") as run:
                logging.info(f"Started training {model_name}")

                mlflow.log_params(params)

                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=5,
                    n_jobs=1,
                    verbose=2,
                    scoring='accuracy'
                )
                grid_search.fit(X_train, y_train)

                best_params = {f"best_{k}": v for k, v in grid_search.best_params_.items()}
                mlflow.log_params(best_params)
                logging.info(f"Best parameters for {model_name}: {best_params}")

                y_train_pred = grid_search.predict(X_train)
                y_test_pred = grid_search.predict(X_test)

                train_metrics = self.log_metrics(y_train, y_train_pred, prefix="train_")
                test_metrics = self.log_metrics(y_test, y_test_pred, prefix="test_")

                mlflow.log_metric("cv_mean_score", grid_search.best_score_)
                mlflow.log_metric(
                    "cv_std_score",
                    grid_search.cv_results_['std_test_score'][grid_search.best_index_]
                )

                if hasattr(grid_search.best_estimator_, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': [f"feature_{i}" for i in range(X_train.shape[1])],
                        'importance': grid_search.best_estimator_.feature_importances_
                    })
                    plt.figure(figsize=(10, 6))
                    plt.bar(feature_importance['feature'], feature_importance['importance'])
                    plt.xticks(rotation=45)
                    plt.title(f'Feature Importance - {model_name}')
                    plt.tight_layout()
                    plot_path = f"feature_importance_{model_name}.png"
                    plt.savefig(plot_path)
                    mlflow.log_artifact(plot_path)
                    os.remove(plot_path)

                mlflow.sklearn.log_model(
                    grid_search.best_estimator_,
                    f"{model_name}_model",
                    registered_model_name=model_name
                )

                logging.info(f"Completed training {model_name}")
                return grid_search.best_estimator_, test_metrics['test_accuracy']

        except Exception as e:
            logging.error(f"Error training {model_name}: {str(e)}")
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training pipeline")

            # Last column is the label, everything before is features
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "RandomForest": {
                    "model": RandomForestClassifier(),
                    "params": {
                        "class_weight": ["balanced"],
                        'n_estimators': [20, 50, 30],
                        'max_depth': [10, 8, 5],
                        'min_samples_split': [2, 5, 10],
                    }
                },
                "DecisionTree": {
                    "model": DecisionTreeClassifier(),
                    "params": {
                        "class_weight": ["balanced"],
                        "criterion": ['gini', "entropy", "log_loss"],
                        "max_depth": [3, 4, 5, 6],
                        "min_samples_split": [2, 3, 4, 5],
                    }
                },
                "LogisticRegression": {
                    "model": LogisticRegression(),
                    "params": {
                        "class_weight": ["balanced"],
                        'C': [0.001, 0.01, 0.1, 1, 10],
                        'solver': ['liblinear', 'saga']
                    }
                }
            }

            model_results = {}
            for model_name, config in models.items():
                logging.info(f"Training {model_name}")
                model, accuracy = self.train_model(
                    X_train, y_train, X_test, y_test,
                    model_name, config['model'], config['params']
                )
                model_results[model_name] = {"model": model, "accuracy": accuracy}

            best_model_name = max(model_results, key=lambda x: model_results[x]["accuracy"])
            best_model = model_results[best_model_name]["model"]
            best_accuracy = model_results[best_model_name]["accuracy"]
            logging.info(f"Best model: {best_model_name} with accuracy: {best_accuracy}")

            with mlflow.start_run(run_name=f"best_model_summary_{self.run_name}"):
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("best_accuracy", best_accuracy)
                comparison_metrics = {
                    f"{name}_accuracy": results["accuracy"]
                    for name, results in model_results.items()
                }
                mlflow.log_metrics(comparison_metrics)

                plt.figure(figsize=(10, 6))
                plt.bar(comparison_metrics.keys(), comparison_metrics.values())
                plt.xticks(rotation=45)
                plt.title('Model Comparison')
                plt.tight_layout()
                plt.savefig("model_comparison.png")
                mlflow.log_artifact("model_comparison.png")
                os.remove("model_comparison.png")

            os.makedirs(os.path.dirname(self.model_trainer_config.train_model_file_path), exist_ok=True)
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            return best_accuracy

        except Exception as e:
            logging.error(f"Error in model training pipeline: {str(e)}")
            raise CustomException(e, sys)
