import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class ModelMonitoring:
    """
    Monitors the deployed model for data drift and performance degradation.
    In production this would run on a schedule, comparing new incoming data
    against the training distribution and alerting if predictions shift.
    """

    def __init__(self, model_path, preprocessor_path):
        self.model = load_object(model_path)
        self.preprocessor = load_object(preprocessor_path)

    def check_data_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        try:
            logging.info("Checking for data drift between reference and current data")
            drift_report = {}

            for col in reference_data.select_dtypes(include=[np.number]).columns:
                ref_mean = reference_data[col].mean()
                cur_mean = current_data[col].mean()
                drift_pct = abs(ref_mean - cur_mean) / (ref_mean + 1e-9) * 100
                drift_report[col] = {
                    "reference_mean": ref_mean,
                    "current_mean": cur_mean,
                    "drift_percentage": drift_pct
                }

            logging.info(f"Drift report generated: {drift_report}")
            return drift_report

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_performance(self, X: pd.DataFrame, y_true):
        try:
            from sklearn.metrics import accuracy_score, classification_report
            scaled = self.preprocessor.transform(X)
            y_pred = self.model.predict(scaled)
            accuracy = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred)
            logging.info(f"Monitoring accuracy: {accuracy}")
            return accuracy, report

        except Exception as e:
            raise CustomException(e, sys)
